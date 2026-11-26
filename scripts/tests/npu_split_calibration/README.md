# NPU split reduction 阈值标定

用实际 `torch.compile → num_splits → create_multilayer → NPU tiling/codegen`，
直接调用编译后的 forward，通过 profiler 采集并解析 `kernel_details.csv`，累加完整 forward 的 device kernel duration，
比较一份固定 S 规则与 S=1。**不搜索 S，不把 S 交给 autotune**；正常 tiling autotune 在两侧保持一致。
所有 hook 限于 worker 进程，不修改安装目录。原始日志、缓存、代码与结果全部放在仓库之外。

产品实现与适用范围见 [NPU Split Reduction 当前实现方案](../../../docs/requirements-design/split-cooperative-reduction-comparison/npu-split-reduction-design.md)。
本套件输出 `npu-split-table-v3` 实验表；产品接收 `npu-split-policy-v1`，当前没有直接转换入口。
即使实验表通过 holdout，也需在实际产品路径验证上下文和环境后再转换，不能只改 schema/status 导入。

**旧 Event 数据无效：** 2026-11-26 整改前的 `device_us` 是 Event 区间，包含任务间空隙，
不能用于本套件的 kernel device 性能阈值。新版使用独立的 `kernel_us` 和 v3 manifest/result/table，
旧结果禁止续跑、合并或生成阈值，须使用新目录重新采集。原始旧数据保留供问题追溯。

## 脚本分工

| 文件 | 职责 |
| --- | --- |
| [run.py](run.py) | 单阶段入口：生成/读取用例全集，筛选百分比区间，预检，指定多卡分配，续跑并汇总 |
| [campaign.py](campaign.py) | 完整实验入口：smoke、初始矩阵、补点、冻结、逐规则 holdout；不接受部分百分比区间 |
| [merge.py](merge.py) | CPU 离线合并：校验全集/环境/配置/来源，保留原始配对样本，重新生成报告并支持后续补测 |
| [cases.py](cases.py) | 定义 op/dtype/X/R/布局/融合矩阵、输入数据、参考计算及工作集预算 |
| [policy.py](policy.py) | 固定 S 候选、IR family 分组、实验准入表验证；不搜索 S |
| [worker.py](worker.py) | 每 case 独立进程：环境采集、编译、精度检查，直接调用并采集 hot/rotate 配对 profiler 数据 |
| [profiling.py](profiling.py) | 复用公共 profiler，按实际 kernel 序列归属样本，计算 kernel duration 总和，核验原始 CSV |
| [hooks.py](hooks.py) | 进程内选择与观测 num_splits/multilayer，记录实际 launch/tiling，按 profile 恢复实验 lowering |
| [analyze.py](analyze.py) | 核验 split/codegen 证据，计算配对收益和裕量，生成 summary、报告及阈值状态 |
| [refinement.py](refinement.py) | 提出 R/上下文补点，形成有限区间，生成并审核逐规则 holdout |
| [common.py](common.py) | 全集与百分比元数据、源码指纹、结果读写、仓库外产物路径校验 |

实际 tiling 记录复用仓库内的 [autotune_tiling.py](../../tools/autotune_tiling.py)。
采集与 CSV 读取直接复用 [npu_profiler.py](../../tools/npu_profiler.py) 的 `TorchNpuProfiler`、`ProfileResultParser`。
各机器应使用同一仓库提交；不需要另行打包或分布式运行环境。

## 常用采集命令

在 `torch-learn` 根目录执行，只需指定卡、用例百分比范围和输出目录：

```bash
python scripts/tests/npu_split_calibration/run.py \
  --devices 0,1,2,3 --range 0,25 \
  --out "$HOME/npu-split-runs/part-000-025"
```

这条命令将固定全集的前 25% 用例均分给物理卡 0/1/2/3，每卡同时运行一个 case。
同一 case 的未拆分和固定 S 候选在同一卡配对测量。只用一张卡写 `--devices 3`；
不传 `--range` 表示 `0,100`，不传 `--devices` 则使用当前可见的第一张卡。
不需要再传逻辑卡号 `--device 0`。

下面这些属于实验口径，已设为默认值，常规采集无需填写。只有明确要改变实验时才覆盖：

| 可选配置及默认值 | 实际测什么 / 为什么保留 |
| --- | --- |
| `--preset coarse,boundary,coverage` | 归约长度粗扫、固定 S 切换边界，以及 dtype/布局/融合代表点。它决定用例全集；临时检查环境可改为 `--preset smoke`。 |
| `--lowering-profile ordinary_ir` | 在 worker 进程内让 prod/any/xor 等进入普通归约 IR，测到 split 本身。某些安装版本默认将它们转给外部算子执行，无法用于 split 标定；如需检查安装默认路径，改为 `installed`。 |
| `--cache hot,rotate` | 分别反复使用同一组输入，以及轮换四组输入地址，观察输入复用对收益的影响。这里指输入数据复用，不是编译缓存开关；rotate 不保证冷 L2。 |
| `--max-working-mib 65536` | 每个 case 的估算工作集上限为 64 GiB，用于排除过大的测试；不会预分配 64 GiB，也不是实测峰值内存保证。目标卡可用内存较少时可降低，例如 `8192`。 |

默认每个 case 直接调用并测 `hot`、`rotate` 两种输入复用条件，各自比较 plain/candidate。
不使用 NPU Graph、Event 或 host wall-clock 作为性能指标；无需传 execution/metric 参数。
默认预算沿用全量数据采集配置，C=56 时初始 5340 例均进入规划；降低预算会显式记录排除项。
多机采集用于合并时，这些实验配置必须一致。

启动时会打印实际使用的配置，并将完整配置写入 plan/manifest；可以加 `--plan --cores 56` 只查看计划。
旧的 `--visible-devices`、`--device`、`--start-percent`、`--end-percent` 作为兼容入口保留，
新版命令统一使用 `--devices` 和 `--range`。新旧区间写法不能混用；旧命令省略的实验参数也会采用本版默认值。

## 从仓库运行

以下命令均在 `torch-learn` 根目录执行，不需要打包。需要目标机器已安装 PyTorch、torch_npu、
triton-ascend 并加载 CANN 环境；套件使用 Python >=3.10 与标准库，不额外安装测试依赖。
内部接口以 PyTorch 2.13 / 默认 NPU Triton backend 为参照，必须先通过目标机器上的 smoke。

```bash
# 仅规划初始矩阵，不需要 torch/NPU。56 替换为目标有效 Vector 核数。
python scripts/tests/npu_split_calibration/campaign.py \
  --plan --cores 56 --out "$HOME/npu-split-runs/plan-v3"

# 自动：smoke → 初始矩阵 → 追加 R/上下文 → 冻结 → 每条规则 holdout。
# --with-dynamic 另收集动态诊断；它不扩大静态准入表的适用范围。
python scripts/tests/npu_split_calibration/campaign.py \
  --devices 0 --with-dynamic \
  --out "$HOME/npu-split-runs/calibration-v3"

# 中断后原命令加 --resume，只测新增/失败/不完整 case。
python scripts/tests/npu_split_calibration/campaign.py \
  --devices 0 --with-dynamic --resume \
  --out "$HOME/npu-split-runs/calibration-v3"
```

campaign 与 run 使用相同的采集默认值，最多 12 轮自动加密。
实际 C 从选定设备的 backend config 读取，遵守 `NPU_DEVICE_LIMIT`。
使用多张空闲卡时传 `--devices 0,1,2,3`，数字为物理卡编号，没有固定的两卡上限。
runner 对每张卡单独设置 `ASCEND_RT_VISIBLE_DEVICES`，每卡同时只运行一个 case；
同一 case 的所有策略始终在同一卡配对，结果记录机器名和卡号。预检要求所选卡软硬件指纹一致。
默认 `--schedule balanced` 按轮询均分待测 case，每卡数量最多差 1；它保证数量均衡，不保证耗时相同。
可用 `--schedule queue` 改为先空闲的卡领取下一个 case。多卡共用同一 manifest、补点与 holdout。
设备列表和输出目录由运行者分配，不自动选择空闲卡；同时运行的命令应使用互不重叠的卡和独立目录。
要求完整收集失败分布时使用 `--collect-all --with-dynamic --with-controls`：单项错误不终止初始矩阵，
继续采集动态诊断和特殊路径控制；存在失败时仍拒绝冻结/验收，报告明确为 INCOMPLETE。
此模式在静态补点之前完成动态诊断；默认模式仍优先 smoke 并在首个执行错误停止。
计划目录和正式测量目录分开。完整矩阵有数千 case，加密/holdout 会继续增加；以 plan 和运行进度为准，不预估设备耗时。

`run.py` 用于单阶段实验；`campaign.py` 调用它并管理追加/冻结/验收：

```bash
python scripts/tests/npu_split_calibration/run.py --preset smoke --fail-fast \
  --out "$HOME/npu-split-runs/smoke-v3"

# 保留安装默认 lowering，并将当前安装的 PR/旧策略作为第三个对照。
python scripts/tests/npu_split_calibration/run.py --preset smoke \
  --lowering-profile installed \
  --modes plain,candidate,policy --policy installed \
  --out "$HOME/npu-split-runs/installed-v3"

# 特殊 reduction 的排除路径控制；不用于普通 split 阈值拟合。
python scripts/tests/npu_split_calibration/run.py --preset controls \
  --out "$HOME/npu-split-runs/controls-v3"
```

`plain` 强制 S=1；`candidate` 使用固定 S；两者统一 read-layout/hint 分类并禁止嵌套 split。
`policy installed` 调用原 selector，包括它自己的 hint/嵌套行为；`policy table` 使用固定 S 加有界准入表。
所有模式统一关闭 cooperative、multi_kernel、mix_order，设置 min_num_split=0，默认启用 max_autotune。
因此 installed 是在这些实验配置下的对照，不等于毫无干预的生产默认行为。

`--lowering-profile installed` 保留当前安装的 lowering，需要显式选择此配置。
若后端把 `aten.prod`、`aten.any.dim/dims` 或 `prims.xor_sum.default` 列在 fallback 表中，
API 正确性通过仍不能说明实际进入了 reduction IR；报告会拒绝将缺失 IR/codegen 证据的点判为 WIN。
`all` 分解后也可能受到 `any` 的 fallback 影响；XOR 没有 eager 实现，fallback 会直接报错。

为标定已实现的普通 IR/codegen，默认使用 `--lowering-profile ordinary_ir`：
worker 在进程内恢复社区 prod/any/xor_sum lowering，再比较相同 lowering 下的 S=1 与固定 S。
此配置不修改安装文件，不改变数学 forward，也不搜索 S；恢复的 overload 列表记录在 environment 中。
**它与安装默认行为属于不同实验域**，表只能用于同样开启这些 lowering 的实现，不能直接套给原安装配置。
两种 profile 的指纹和结果不能混用。使用前先跑对应 smoke，IR 可构造不代表所有 NPU codegen 场景都支持。

```bash
python scripts/tests/npu_split_calibration/campaign.py \
  --devices 4,5 --collect-all --with-dynamic --with-controls \
  --out "$HOME/npu-split-runs/ordinary-ir"
```

## 按百分比采集，结束后离线合并

不需要机器间通信或统一调度。每台机器各运行一次 `run.py`，指定同一用例全集中的一个区间，
结束后将结果目录放到同一机器，用 `merge.py` 重新汇总。需要所有机器使用相同套件源码、
软硬件指纹、有效核数、测量参数、显存规划预算和用例定义。

| 参数 | 含义 |
| --- | --- |
| `--range 0,25` | 用例数量的百分比区间，左闭右开；默认 `0,100`，100 覆盖最后一个 case |
| `--devices 0,1,2,3` | 本进程使用的物理卡，可以指定一张或多张 |
| `--schedule balanced` | 默认值；区间内待测 case 按轮询均分到给出的卡 |

百分位是**用例数量的百分位**。先生成预设或读取 `--cases-json`，按输入顺序去重并应用筛选与 `--limit`，
得到固定顺序、长度为 N 的全集，再取：

```text
cases[floor(N × start_percent / 100) : floor(N × end_percent / 100)]
```

支持 12.5、33.333 等小数，以十进制精确计算索引。`0–25`、`25–50`、`50–75`、`75–100`
无重复、无遗漏；很窄的区间可以包含零个 case。显存预算排除发生在切区间之后，不能改变全集或边界。
百分比区间不能与 `--shards` 混用。各机筛选和 preset 顺序必须相同；改变这些参数会改变全集。
也可将同一份 `plan.json` 通过 `--cases-json` 分发到各机；读取新版计划/manifest 时使用其中记录的完整全集，
即使规划时只选择了部分区间或排除了超预算项，也不会把这些项从全集中丢掉。

例如 C=56、`coarse,boundary,coverage` 共 5340 个静态 case，`0–25` 为 1335 个；
分给四张卡为 334/334/334/333 个。各机可以使用不同卡数，任务多的机器分配更宽的百分比区间即可。

以下命令在各机 `torch-learn` 根目录执行。机器 A 跑 0–25，B 跑 25–50，C 跑 50–100；
改起止参数、卡号和输出目录，其余参数保持一致。先在每台机器跑对应 profile 的 smoke。

```bash
# 可先加 --plan --cores 56 查看数量、索引边界和每张卡的 case 列表，无需 NPU。
python scripts/tests/npu_split_calibration/run.py \
  --devices 0,1,2,3 --range 0,25 \
  --out "$HOME/npu-split-runs/part-000-025"
```

若需要一起收集动态和特殊路径，可在**每台机器**都改成
`--preset coarse,boundary,coverage,dynamic,controls`；这会形成另一份全集，C=56 时为 5498 个 case。
`run.py` 默认单例错误后继续采集，退出状态仍反映失败；原命令加 `--resume` 只重跑失败/不完整项。
续跑按剩余 case 重新均分。`plan.json` 记录完整全集、摘要、区间边界和卡分配；
实际运行的 `manifest.json` 记录相同全集与区间，`schedule.json` 记录本次待测项的分卡情况。

将各机输出复制到同一机器后执行（输入目录可以有任意多个）：

```bash
python scripts/tests/npu_split_calibration/merge.py \
  --inputs "$HOME/npu-split-runs/part-000-025" \
           "$HOME/npu-split-runs/part-025-050" \
           "$HOME/npu-split-runs/part-050-100" \
  --out "$HOME/npu-split-runs/merged"
```

合并保持每个 case 原始的 plain/candidate 配对样本，重新生成 `summary.csv/json`、`report.md`、
补点和阈值建议，并输出 `merge.audit.json`。CSV 带 `hostname` 和 `physical_device`，
manifest 的 `merge.case_sources` 保留来源目录。不同 SoC、软件、源码、配置、全集或重复 case 会拒绝合并，
不会择优挑一份重复结果；未提交的区间和失败项会列为 INCOMPLETE，阻止准入。
单个百分比区间只报告该区间的数据，完整合并前状态为 `PARTIAL_COLLECTION`，不生成可用阈值。

默认复制分析必需的 JSON、worker 日志和每个 profile 的原始 `kernel_details.csv`，
合并及分析会重新解析 CSV 并核对 SHA-256、任务数量、kernel 顺序和样本之和。
加 `--copy-artifacts` 可复制完整 case 目录；两种模式都不会修改输入目录，也不会启动 NPU 任务。
输入必须已经结束，输出必须为空；合并失败或需要加入新输入时使用新的输出目录。

合并后的结果仍可由 `run.py` 续跑。保持原测量参数，使用
`--cases-json <merged>/all_cases.json --resume --extend --out <merged>` 可补齐缺失/失败 case；
使用 `--cases-json <merged>/next_cases.json` 则可追加分析提出的 R/上下文补点。
已完成的 case 会核验身份后复用。每个百分比采集目录自身是固定全集，不能直接 `--extend`。
完整的 `campaign.py` 仍管理单份本地实验，不接受部分百分比；离线合并仅表示数据汇总完成，
不是阈值已经通过冻结和 holdout。

套件源码参与指纹校验；更换源码版本后的新数据不能与旧版本混合合并或续跑，
更早的采集目录还可能缺少完整全集元数据。旧 Event 数据不能作为本版性能标定证据。
本版只采集 profiler kernel device duration；新一轮采集使用新的输出目录和同一份新版源码。

## 类型、dtype、布局与融合覆盖

PR #50315 新的普通 split 入口以 `REDUCTION_COMBINE_FN` 为基础，排除 dot/fmax。
此套件覆盖其六类普通 IR reduction：**sum、prod、min、max、any、xor_sum**。

| forward | 输入 dtype | 说明 |
|---|---|---|
| sum / prod / amax / amin | FP16、BF16、FP32、int32、int64 | 每种组合都有独立粗扫与初始 S 边界；整数 sum/prod 的输出类型由真实 lowering 决定 |
| any / all | bool | all 的分解通常使用取反和 any；以实际 IR 为准 |
| xor_sum | int32、int64 | 直接调用 prims.xor_sum，使用独立 CPU 逐位归约参考 |
| mean / sum_fp32 / add_sum / exp_sum / RMSNorm / softmax / sum_amax | FP16、BF16、FP32 | 融合代表点；缺失上下文在加密阶段自动补测 |

普通 forward 共 24 个 op/dtype 组合。FP64、int8/int16、复数等不在当前矩阵中；没有将这些组合推断为已验证。
候选策略允许的 IR dtype 必须与实际 read/compute 类型一起理解：输入 BF16 经 cast 后可能属于 FP32 IR family。
mean lowering 是 sum 加缩放，不是独立的 IR reduction 类型。

| layout | 构造 | 范围 |
|---|---|---|
| contiguous | `[X,R]`, stride `(R,1)` | 普通类型完整粗扫和边界 |
| stride2 | `[X,R]`, stride `(2R,2)` | 全部 24 个普通 op/dtype 组合的代表点 |
| transpose | `[X,R]`, stride `(1,X)` | 同上；X=1 会退化为 contiguous |
| multi_axis | `[X,R/d,d]`，归约后两轴 | 同上；d 从 64/16/4/2 中选可整除项，否则 1；prime R 会退化 |
| broadcast | `[X,1]` 与 `[1,R]` 组合后归约 | 浮点/整数普通运算用 add，prod 用 mul，bool 用 logical_or，xor_sum 用 bitwise_xor |

另外对 stride2/transpose 测三种浮点 dtype 的 add_sum、RMSNorm。
布局覆盖仍不是任意 stride/storage offset/高维 permutation 或分离归约轴的穷举。
prod 的 multi_axis forward 将相邻归约轴 flatten 后调用单轴 aten.prod；是否保留多轴 IR 需看生成证据。

add_sum/exp_sum/RMSNorm/softmax/sum_amax 显式转 FP32 计算；sum 保留原生语义，sum_fp32 显式要求 FP32 输出。
RMSNorm 没有 affine 权重。softmax 可能走专门 lowering，不能仅凭 API 名称认定普通 split 已实现。

特殊路径的控制 preset 包含 argmax、argmin、var_mean、cumsum，记录：

- `VERIFIED_SPECIAL_UNSPLIT`：实际观察到目标 IR 类型且 S=1。
- `PATH_NOT_EXERCISED`：例如 var_mean 实际改走普通 sum，不能声称验证了 Welford 排除路径。
- `UNEXPECTED_SPECIAL_SPLIT`：出现了不应接管的拆分。

arg、Welford 不属于本次 PR 的普通新策略范围；NPU scan 被明确保留为 S=1；
online-softmax 有不同的复合状态合并语义，也没有在普通 selector 中启用。dot/fmax 明确排除。
控制 preset 不自动强制制造 online-softmax/dot/fmax 专门 IR，也不把没出现的路径算作已覆盖。

## 初始 R 和固定 S

X=输出元素数，R=每个输出归约元素数，C=有效 Vector 核数。

```text
P0_estimate = ceil(X / outputs_per_program)
S_fill      = ceil(C * target_waves / P0_estimate)
S           = max(1, min(S_fill, cap, R-1, floor(R/min_chunk)))
```

默认 outputs_per_program=1、target_waves=1、min_chunk=256，max_s=0 表示 cap=C*target_waves。
这些仍是待实测假设，不是 NPU 已验证资源公式。实际 program 数看 grid，不能用 X*S 替代。
改变 selector 参数必须新开测量目录，不能混用已测结果。

| preset | 矩阵 |
|---|---|
| smoke | 158 个代表 case，提前检查全部普通 op/dtype ×5 布局，以及所有已声明融合的 dtype/布局组合 |
| coarse | 全部 24 组合；X=`{1,2,4,C//4,C//2,C-1,C,C+1,2C}` 去重/正数保护；R=`2^8…2^22`，共 15 点 |
| boundary | 每种组合、每个 X，测固定 S 公式的首次变化和最后一次变化附近 R−1/R/R+1；`--around` 可额外添加中心 |
| coverage | X=`{1,max(2,C//4),C}`；R=4096/65537/262144/1048576；上述布局/融合组合 |
| holdout | 每种普通组合，粗扫相邻 R 的几何中点加 3；用于手动探索，不能替代自动逐规则 holdout |
| dynamic | 全部普通组合及 FP32 RMSNorm；动态 R、动态 X、X/R 双动态，150 个编译种子、675 次 runtime shape 探测 |
| controls | 4 种特殊路径 forward × X=1/2，共 8 个 |

C=56 时 coarse=3240、boundary=720、coverage=1548，合并去重后为 5340 个 case；
默认 65536 MiB 预算保留全部 5340 个；改为 8192 MiB 时保留 5190 个，排除 150 个。
超预算点记录在 manifest，不会当作已测。
初始矩阵之外的精确数量取决于测量结果。`--x/--r/--ops/--dtypes/--layouts/--match/--limit` 都是过滤器，
不生成任意新尺寸；使用 `--cases-json` 可替代预设矩阵，支持现有 forward。

动态 R 用 4096/262144/1048576 三个 seed，运行 4096/65537/262144/1048576/4096。
动态 X 用 2 和 max(3,C//2) 两个 seed，运行 2/max(3,C//2)/C+1/2。
双动态额外交叉改变 X/R。通过 active_compile_id 区分复用、重编译和返回旧图。
动态 table 准入仍为 S=1，直到生产实现拥有真实的范围 guard/dispatch。
实验 candidate 会用编译时 hint 探测动态拆分行为；它属于诊断，
不等于产品 selector 已支持动态 split。当前产品对 symbolic X/R 一律返回 S=1。

## 自动加密如何避免漏点与重跑

每轮分析只从 candidate 的静态 calibration 结果提出追加点：

1. 同一 IR family 的每种已观察上下文，必须在每个拟准入 R 都有数据。
   不能用“sum 在 R=4096 赢，RMSNorm 在 R=65536 赢”代替双方在两点都获益。
2. WIN 与非 WIN 的交界、selected tiling/runtime-block 配置变化的区间，按几何中点加密，
   默认将相邻 R 比缩到 1.05 以内。
3. 两端都 WIN 的区间也加密，默认相邻 R 比不超过 1.25，不假设收益单调。
4. 有获益证据的区间补固定 S 的所有切换邻点；从初始观测的 R tile 配置补局部 tile 边界。
   不递归沿 tile 边界无限展开；这不是对编译器全部切换点的穷举。
5. 两端都不获益的区域不自动做密集搜索，也不导出准入规则；可能漏掉孤立获益岛，但不会因此启用未证实的范围。

加密配置为 `--max-r-gap 1.25 --boundary-r-gap 1.05`。同一 case 的 hot/rotate 两种 profiler 条件都要参与。
至少三个连续通过的 R、上下文齐全且满足间距要求，才形成候选区间。任何未完成 case、待补点或必需补点超预算，
都阻止导出可冻结规则。`provisional_rules` 只用于诊断，不能当作准入表。

使用同一个 manifest 追加 case，保存 manifest 历史，已完成结果按源码/环境/测量参数/输入定义核验后复用。
只提高规划显存预算不会改变既有测量；例如原预算为 8192，可在原命令加 `--resume --max-working-mib 16384`；
增加加密轮数也可续跑，如 `--max-refine-rounds 20`。预算是保守估算，不是设备峰值显存保证。
改变 dtype/布局筛选、S 规则、计时设置、设备/关键软件/套件代码需要新实验；旧 Event 数据不会被误当成 profiler v3 证据。

手动追加也可使用：

```bash
# 必须带上与原 run 相同的测量参数；campaign 已自动处理这些参数。
python scripts/tests/npu_split_calibration/run.py \
  --cases-json "$HOME/npu-split-runs/manual/next_cases.json" \
  --resume --extend --out "$HOME/npu-split-runs/manual"
```

每个 case 独立进程，失败重试保留旧目录至 attempts/；默认超时 1800 秒，可用 `--timeout` 调整。
一个 case 失败只重测该 case，不重跑整个矩阵。运行中还未完成的 case 可能需要重新编译。

## 冻结与逐条 holdout

campaign 在所有补点结束后保存带内容摘要的冻结表，自动生成每条规则的所有上下文：

- 每个通过采样间隔内部的一个新 R，并确保每个 rule/context 至少有两个未参与标定的内部 R。
- 区间两个端点，以及区间外紧邻点；若外侧属于另一条规则，按整张表核对是否应拆分。
- hot/rotate 缓存条件，使用不同输入 seed，重新编译 table 策略。

验收检查实际 IR decision 是否匹配预期准入，所有进入 split 的完整 forward 都必须 WIN。
缺少某条规则、某个上下文、某种缓存条件、内部新点或端点控制，都不能通过；没有实际准入样例也不能通过。
预算排除了 holdout case 同样失败。不会以多数样例或平均收益掩盖退化点。

通过后输出 `validated-policy.json`，状态为 `VALIDATED_MEASURED_DOMAIN`，含审核记录。
不会写回 torch_npu，也不会标成通用生产保证。阈值仍按 IR op/src-dtype/dst-dtype/read-layout/**精确 X** 分组，
不向未测 X/设备外推。图上下文没有成为生产 selector 的 guard，所以任意未知 consumer 仍需单独验收。

## 正确性与计时

所有输入先在 CPU 固定 seed 构造，保留真实 stride，策略共享同一批设备 tensor。
CPU 参考、NPU eager 和 compiled 结果先比较；xor_sum 的 eager 不存在，因此使用独立 CPU bitwise tree，
直接校验两侧 compiled。整数/bool/index 输出精确比较；浮点容差按**输出 dtype**，不会给 FP32 输出套 BF16 容差。
记录最大绝对误差与容差。普通浮点输入为有正负的有界随机值。

prod 使用少量抵消的 2/0.5 因子、非单位尾值和符号，其余为 1，避免长 R 自动下溢、SIMD 分组积溢出。
整数 prod 使用 ±1 和非单位尾值。额外复用 compiled 图做单位元、末尾、分片边界检查，prod 再测末尾零。
bool 检查全 False/全 True 相应单位元及尾部反例；这些都在 profiler 采集外检查。
这是性能准入的正确性前置条件，不是 NaN/Inf、极端消减、任意分布的完整数值认证。

采集前完成编译、autotune、代码/配置采集、精度检查和预热，并卸载观测 hook。
采集区间只直接调用编译后的 no_grad forward；不做输入 H2D、CPU reference、结果 D2H 或重新编译。
无 backward 测试。

- 默认 3 session ×7 round，每 round 随机顺序加逆序；两策略为 ABBA/BAAB。
- 每个策略、每种缓存条件有 42 个批次样本；每批固定 `--repeats 20` 个完整 forward。
  rotate 时 repeats 必须为 buffer 数的整数倍。不会通过 Event pilot 决定重复次数。
- 每个 session 使用一份 Level0 profiler，CPU+NPU activities，关闭 stack、shape、memory 采集。
  编译和算子预热在采集外完成；profiler 的 schedule warmup 为 0，正式采集只执行计划内的批次。
  复用公共 `TorchNpuProfiler`；每批结束同步并 `prof.step()`，同步只确定任务先后，不计为 kernel 时间。
- 复用 `ProfileResultParser.kernel_rows()` 读取 `kernel_details.csv`，支持 `Duration(us)` / `Task Duration(us)`。
  对同一 stream 的完整任务序列排序，逐批核对观测到的 kernel 名称、顺序、次数与 inspected launch 序列一致。
  不按 CPU range 时间戳切片，避免 host/device 时钟偏移；不静默丢弃无法解释的行。
- `kernel_us = Σ该批全部 kernel duration / 实际 forward 数`。split 包括 partial、combine，
  融合/多归约场景包括完整 forward 的所有实际 kernel，不能只取第一阶段。
  每个 launch 单独保存名称、次数、总时长和每次 forward 的平均时长，方便检查两阶段成本。
- 不使用 Event 区间、首尾任务跨度、CPU duration、同步等待或 kernel 间空隙。
  一份 capture 中若出现多设备、CSV 明示的多 stream、额外或缺失任务、名称不符、零/无效 duration，直接报错。
  部分 CANN 的 kernel_details.csv 不含 stream 列，此时明确记录该限制，归属依据为直接调用的串行执行和完整 kernel 序列。
- hot 一组输入；rotate 默认 4 组相同 shape、不同地址的输入，减少同地址缓存复用。
  rotate 不保证冷 L2；记录 working set，不重新编译。
- capture 的 CSV 路径、SHA-256、任务数、实际 duration 列均随结果保存；分析/续跑/合并重新核验 CSV。
  CSV 缺失、被修改或无法复算时，不接受该结果为性能证据。完整 trace 仍保留在原采集目录。

默认 `gain=Tplain-Tsplit`，裕量 `max(Tplain*8%,0.2us)`。
每 round 配对平均，session 内取中位数；所有 session 均超过裕量，且 session 裕量的 2000 次 bootstrap
单侧 99% 经验下界为正才 WIN。至少 3 session、每个 5 paired round。
三个相邻 session 不能算独立实验，也不构成普遍 99% 保证；正式上线还需独立时段/设备与业务子图复测。

## 结果与诊断

| 产物 | 用途 |
|---|---|
| campaign.state.json | 当前阶段、加密轮数、各阶段退出状态 |
| plan.json、schedule.json | 百分比全集/边界和规划分卡；实际待测项及本次调度 |
| calibration/manifest.json、manifest.history/ | 全部测量点、追加历史、预算排除、源代码与环境身份 |
| preflight/environment.json | SoC、有效/物理核数、软件版本、关键源码摘要、CANN 信息、npu-smi |
| coverage.json | 24 个普通 op/dtype 的计划数量和实际证据/收益状态；计划过不代表 split 已实现 |
| refinement.json、next_cases.json | 待补点及原因、预算阻塞、可追加完整矩阵 |
| cases/id/result.json | kernel_us 配对样本、逐 kernel 成本、CSV 指纹，精度/特殊数据、shape/stride、工作集、编译图编号、实际 launch |
| cases/id/profiles/probe-*/session-*/ | profiler 原始 trace/CSV、采集计划 plan.json、解析后的 kernel_samples.json |
| cases/id/mode/lowering.json | num_splits 输入/理由/S、实际 multilayer/depth、partial 逻辑数量与预期 dtype |
| output_code_*.py、fx_*.py、launches_*/ir/ | wrapper/DSL、FX、可取得的 TTIR/adapter/LLIR；每次观测单独保存，避免动态 shape 覆盖旧证据 |
| summary.csv/json、report.md | 时间/收益/证据、S、grid/programs、selected tiling/runtime blocks、图编号、错误清单 |
| merge.audit.json、all_cases.json | 离线合并的区间覆盖、失败清单、来源，以及可续跑的完整用例定义 |
| thresholds.proposed.json | 未完成/待加密/待 holdout 的明确状态及有限规则 |
| tables/、holdout/、holdout.audit.json | 冻结表、逐规则验收计划与覆盖审核 |
| validated-policy.json | 仅在逐规则验收全部通过后产生 |

不能从 X*S 或 partial 逻辑大小推断真实 grid/峰值显存。实际分配看 output_code，
`partial_dtype_expected` 只是按上游 dst_dtype 规则记录的预期值，不控制 partial 分配；
上游 `create_multilayer_helper` 实际决定 intermediate 类型，需结合生成 IR/code 核对。
当前已采集每个实际 kernel 的 profiler device duration；不采集 UB 实际占用、L2 miss、带宽或峰值 workspace。
主指标是设备 kernel 时长总和，不代表包含 host 和任务间隙的端到端延迟；模型端到端验收属于独立指标。

重新分析不重跑设备，但会覆盖报告和候选表；已冻结表单独保存：

```bash
python scripts/tests/npu_split_calibration/analyze.py \
  "$HOME/npu-split-runs/calibration-v3/calibration" \
  --relative-margin 0.10 --absolute-margin-us 0.5
```

退出码：1 为环境/执行/数据或证据问题；2 为没有得到准入区间或 holdout 未通过；3 为补点预算/轮数/覆盖仍待处理。
单独 run.py 的普通采集退出 0 只表示完成，性能退化仍会出现在报告中。
已验证规划、CPU 数据/参考计算、模拟 CSV 的分析/分区合并及损坏证据拒绝。
2026-11-26 在 NPU_A5 的 pta_213_case 环境、单张物理卡 4 上完成 sum/FP32/X=1/R=4096 的
编译、精度检查、hot/rotate profiler 采集及分析流程；另外复用其生成 wrapper，验证
每种缓存条件 3 session ×5 round ×ABBA/BAAB ×4 forward，共 6 份 CSV、720 条 kernel，
全部任务数量、名称顺序及原始 CSV 复算通过。这些是采集链路验证，不能作为全矩阵完成或阈值有效的证据。
