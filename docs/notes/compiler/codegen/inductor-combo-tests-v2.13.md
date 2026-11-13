---
published: false
---

# 2.13 Combo/foreach 最新结果与问题分析

<a id="validation213"></a>

<!-- combo-section:validation213:start -->
## 最新执行结果

2026-11-12 完成 15 个修改源方法的定向验证，包含动态 grid、store mask 和各类 SKIP 条件。

| 范围 | PASS | FAIL | SKIP |
| --- | ---: | ---: | ---: |
| 本轮 15 个修改源方法及 15 个参数实例（最终结果） | 2 | 0 | 13 |
| 合并历史结果的 105 个源方法 | 74 | 14 | 17 |
| 合并历史结果的 314 个已选实例 | 281 | 15 | 18 |
| 2.10 基线 PASS 的 60 个重合方法 | 55 | 0 | 5 |

**11 月 12 日的正式定向验证仅覆盖 15 个修改方法，其余正式状态沿用参考报告。**
11 月 13 日另完成 WRN、FTZ 及两个 foreach 方法的诊断，详见当前问题分析；诊断不改正式统计。
14 个 FAIL 方法、15 个失败实例的正式状态沿用参考报告；本轮诊断不等同于原方法的正式复测。
74 个 PASS 中有 14 个抽样 PASS；原清单 464 个泛化实例的 150 个未选择实例不计为 SKIP 或 PASS。
2.10 重合对照仍是既定的 60 方法适配基线，与 54 方法原生清单口径不同。
当前重合范围的 5 个 SKIP 为 4 个独立分块方法和 1 个 NPU CPU-filter 实例，不是计算错误。

[用例 CSV](inductor-combo-tests-native-v2.13-sync-64dc85c.csv) 保持 116 行原顺序和五列。
11 行嵌套函数随外层方法验证，不另计源方法数。PASS 的当前分析统一为“—”。
本轮 13 个 SKIP 分别是独立分块待实现 8 个、CPU 过滤测试范围不适用 1 个、CUDA TMA/PDL 专属 4 个。
另外 4 个 SKIP 沿用参考报告。SKIP 不计为功能实现或功能修复。

动态 grid 从真实 launcher 观测启动范围，排除 autotune 的计时启动，
检查两组尺寸的 X grid 不同、Y/Z 为 1，并保留数值检查。
store mask 保留原输入数值检查，新增实际双成员 NPU Combo，检查两个分支的尾块边界及全部 store mask。
**新增 store-mask 场景在用例内关闭 dtype 调试断言；相关 codegen 诊断问题仍未修复。**
原标量输入仍被拆成三个普通 kernel，因此尚未覆盖社区原标量 Combo 写竞争路径，详见下文验证前提。

环境：NPU_A5/Ascend950PR，`pta_213_case`，PyTorch `2.13.0+cpu`、Triton `3.6.0`。
基础 wheel 为 `torch_npu 2.13.0+git64dc85c`，实际 NPU Python 为既有 `d12230b556` 覆盖版本。
本轮未修改生产代码或安装包；两阶段执行前后源码哈希一致，并与参考报告的 574 个安装文件哈希逐项一致。
物理卡 1、3，使用独立缓存，执行原生 `python -m pytest -vs`，无自定义插件。

测试分支 `fix/combo-213-npu-case-adaptation` 已推送本轮提交 `e71a557`。
14 个方法的正式报告来自 `4acf77e`，store mask 最终报告来自 `64015d3`；
这 14 个方法在最终提交中未改变。`e71a557` 相对最终验证版本仅修正诊断注释，执行 AST 一致。
逐实例实际提交保存在合并证据 `jobs.json` 的 `validation` 中，源码核对见 `source-validation.json`。

foreach 继续沿用 NPU 功能适配：保留数值、mutation 和梯度检查。
其抽样 PASS 不证明全部参数通过，也不证明 FOREACH 分组融合。

[执行方式](../../../../scripts/tests/combo_kernel_210/README-213.md)；
[本轮 JUnit、日志、精确 node ID、源码哈希和生成代码](../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/)；
[2.10 重合方法对照](../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/baseline-pass-comparison.json)。
未复验方法的实际验证版本保留在合并报告中，来源为对应参考证据。

<!-- combo-section:validation213:end -->

<a id="analysis213"></a>

<!-- combo-section:analysis213:start -->
## 当前问题与优先级

2026-11-13 核对 Ascend `upstream/master` 为 **`44555141711c0c825c5a201e303dcadad1f5fcb4`**，
逐项分析全部 **14 FAIL + 17 SKIP，共 31 个源方法**（15 个失败实例、18 个跳过实例）。
配套社区 Torch 源码为实际 2.13 环境提交 `cf30153c4c131c8164ee7798e5022d810682e2cb`，
测试源码为已推送的 `e71a557`；最新 Triton-Ascend `origin/main` 为
`62f6b54e0f2420d6b718e0d4f89fa7a464d2e78f`。

**最新 master 的源码分析与设备诊断分别记录，未把 master 装包复测。**
11 月 13 日在原 `64dc85c` wheel + `d12230b556` Python 覆盖环境定向诊断 WRN、FTZ 和两个 foreach 方法，
实际使用物理卡 1、3；安装文件 574 项哈希仍与参考报告一致，未修改生产代码或正式测试。
正式统计保持 **74 PASS、14 FAIL、17 SKIP**；诊断 PASS 不直接改写合并 jobs/CSV 的状态。

下表按具体字段或功能列出作用、当前缺口和适配内容，编号链接到后文的源码与验证证据。
一行对应一项字段或功能；同一方法可涉及多行，同一字段也可由多个方法检查。
方法数仍为 **14 FAIL、17 SKIP**，不按表格行数重复计数。

**FAIL：图阶段策略、逐成员归约分类与算子来源映射**

| 具体字段或功能 | 作用 | 当前缺口与适配内容 | 方法 |
| --- | --- | --- | --- |
| `optimize_mem=True` | 推理图向 autotuner 传递内存优化策略 | 未显式导出，缺省 True 与本例期望一致；需补图阶段字段 | [F03](#f03) |
| `optimize_mem=False` | 训练前向图禁用该内存优化策略 | 字段缺失后属性默认 True；需按训练前向阶段显式传 False | [F04](#f04) |
| `reduction_hint_i` | 各成员 INNER/OUTER 归约分类，供候选选择 | 双成员 Combo 已有，仅传首成员 hint；需补真实分类，消费需接 native 搜索 | [F01](#f01) |
| `disable_ftz` | 控制 FP32 非正规数是否冲零 | Combo 未导出字段，Triton-Ascend 无对应选项消费；exp/exp2 数学入口还需跨层适配 | [F05](#f05) |
| kernel → FX 映射 | 将生成 kernel 追溯到 post-grad 算子 | hook 收到 `node=None` 外壳，跳过 origins；需按实际 partition 传成员叶子 | [F14](#f14) |

F03、F04 检查的是**同一个 `optimize_mem` 字段的两种图阶段取值**，不是两种 metadata。
统一取值为 `V.graph.is_inference or V.graph.is_backward`：推理/反向为 True，训练前向为 False。
该策略用于 autotune 重试时保护会被原地修改的输入，并在 clone 可能增加设备峰值时选择 CPU 备份/恢复。
当前 NPU 主 bench 直接 clone，CPU-copy/restore helper 尚未接入；补字段和接入真实搬移路径分别需要处理，
现有证据未显示错误 CPU 搬移或训练输入污染。字段出口见 [E2][E2]，策略与实际调用见 [E5][E5]。
F05 的基本算术已保留可表示的非正规数，指数数学路径的边界冲零需单独处理，不能仅补字段后判定数值能力已实现。

**FAIL：逐成员操作统计、成本估算与报告**

F06–F12 是 **7 个测试方法**，检查的字段和消费路径有重叠，不代表缺少 7 种独立统计。
`inductor_meta_i` 是第 i 个成员的统计字典；当前 NPU 的 grid/split 信息和成员轴约束不能提供这些统计。
下表把每个字段及不同消费出口分别列出。

| 具体字段或出口 | 作用 | 当前缺口与适配内容 | 方法 |
| --- | --- | --- | --- |
| `inductor_meta_i` | 将统计与具体成员一一对应 | 缺逐成员字典；需统一普通 kernel/Combo 出口 | [F06](#f06)–[F09](#f09) |
| `num_load` | 去掉缓存复用后的有效读取操作数 | NPU CSE 已计数，但普通/逐成员统计出口未导出真实值 | [F06](#f06)、[F07](#f07)、[F09](#f09) |
| `num_store` | 有效写回操作数，含归约结果 | 普通/归约 store 漏计；需修计数与移除 store 的扣减 | [F06](#f06)、[F07](#f07)、[F09](#f09) |
| `num_reduction` | 归约操作数，pointwise 成员应为 0 | NPU CSE 已计数，但未分别输出 sum/mean 等成员的实际值 | [F06](#f06)、[F07](#f07)、[F09](#f09) |
| `autotune_hints` | 提示调优器补充特定候选配置 | 普通出口已有，Combo 顶层复用首成员；缺各成员自己的提示集合 | [F06](#f06)、[F09](#f09) |
| `atomic_add_found` | 标记是否有原子加写回，供调优约束判断 | 未输出各成员的真实标记；需按实际 store 模式导出 | [F06](#f06)、[F09](#f09) |
| `has_loadstore_with_contiguous_rdim` | 归约轴 stride=1 访问标志 | load 未更新，默认 False 不可信；需按实际地址推导 | [F08](#f08) |
| `no_x_dim` | 是否没有社区 X 迭代维，影响 grid/分块 | NPU 已有 `no_x_dim_i`，含义却是无 split axis；需定义与普通 kernel 可比的 NPU 口径 | [F09](#f09) |
| `tiling_scores` | 描述各轴访存合并评分，供 tile 候选评价 | 缺与普通 kernel 一致的 NPU 评分出口；需按 native 轴定义，有实际评分时导出 | [F09](#f09) |
| benchmark 的 `kernel_num_gb` | 访问字节估算/1e9，供带宽计算 | 普通 kernel 已有值；Combo 缺成员聚合与 standalone 求和对照 | [F10](#f10) |
| benchmark 的 `kernel_flop` | 浮点操作量估算，用于算力报告与成员成本对照 | 普通 NPU 未导出，Combo 也未聚合；standalone 字段缺失时求和为 0 不是有效基准 | [F10](#f10) |
| profile 的 `kernel_num_gb` | 给带宽 profiler 提供 Combo 总访问量估算 | 缺按 profile 条件的聚合出口；仅 profile 时不输出 FLOP | [F11](#f11) |
| benchmark 代码中的 `num_gb` | 与实测耗时相除，生成 GB/s 报告 | 模板收到整数 0，报告带宽为零；需传真实总估算，改成 0.0 不能修复报告 | [F12](#f12) |

F06 检查三个 pointwise 成员，F07 检查 sum/mean 两个归约成员；两者都要输出各自的 load/store/reduction 计数。
CSE（公共子表达式消除）会复用已计算或已读取的值，`num_load` 只计未复用的有效读取。
主机重放一次有效 store 时，NPU 的 `num_store` 为 0，社区为 1；此处需要修计数，不能只补字段。
F08 的连续归约访问标志用于确定性相关候选筛选；若要消费该标志，需接到 NPU 自身的筛选规则。
F09 另检查同一成员在 standalone/Combo 下的字段一致性，不能用两个空字典相等来证明统计完整。
F11 仅开启 `profile_bandwidth=True`，要求总 GB 估算且不输出 FLOP；F12 则检查生成 benchmark 代码中的估算值。
上述计数是生成操作的计数，GB/FLOP 是编译期成本估算，不是设备实测流量。
例如 F10 原无 alias 的两个 1024 元素 FP32 pointwise 成员，按输入+输出估算共 **16384 bytes，即 1.6384e-5 GB**；
该真实总估算应进入 Combo metadata 和 benchmark，而不是默认填 0。
计数证据见 [E4][E4]，字段契约见 [E11][E11]，估算与报告出口见 [E12][E12]。

**FAIL：需要调整测试检查口径的两项**

| 方法 | 检查的功能及作用 | 当前证据与适配内容 |
| --- | --- | --- |
| [F02](#f02) | `add_persistent_rblock`：为 looped 归约追加大 R 候选 | 本输入已整行 persistent，R1 tile=4096；需查 native 候选与资源 |
| [F13](#f13) | WRN 内存 gate：限制融合带来的峰值增加量 | gate 已存在，原图无 Combo 候选，首次峰值含 192 MiB autotune 缓存；需选有正 delta 的可控图并 warm 测量 |

**SKIP：成员独立分块、默认配置与搜索，以及相关测试入口**

“共享块”指一次 launch 的所有成员复用同名 runtime/constexpr tiling 参数；
“独立分块”要求各成员有真实独立参数、候选与 dispatch 边界。
以下 8 个方法的独立分块目标尚未实现；S05 还涉及独立的 ATen 布局转换与测试观测入口。

| 具体功能 | 作用 | 当前缺口与适配内容 | 方法 |
| --- | --- | --- | --- |
| pointwise 成员独立 X 分块 | 不等长成员分别选择工作块大小 | 当前共用 X 参数；缺逐成员参数、候选和累计 dispatch 边界 | [S01](#s01) |
| persistent 成员独立 X 分块 | 固定各自归约长度，独立选择行方向工作块 | 不同静态 R 已存在，X 仍共享；需增加独立 X 配置及 launch 映射 | [S02](#s02) |
| persistent/regular 混合成员配置 | 在一次 launch 中协调不同归约模式的配置与执行 | 缺独立配置，当前又按模式分区；需设计合法的混合协调与资源校验 | [S03](#s03) |
| 多轴/非连续归约成员独立 block | 各成员按真实归约轴划分 loop 与尾块 | 缺独立归约参数、候选及对应累积/mask；需沿用 NPU 轴语义实现 | [S04](#s04) |
| 成员独立二维 tile | 八个布局转换成员分别选择二维工作块 | 当前 block ABI 共享；缺二维成员配置与地址/dispatch 对齐 | [S05](#s05) |
| `channels_last` 转换 | 把 NCHW 转为所需布局 | 未符合目标格式时仅支持转 Contiguous；需 ATen 适配，已有目标格式可返回 | [S05](#s05) |
| 实际 launch grid 对照 | 验证成员 tile/dispatch 与启动范围一致 | 需用 NPU launcher/profiler 真实记录替换 GPU 固定 grid | [S05](#s05) |
| 无 benchmark 的 pointwise 默认配置 | 不计时时也能拼接各成员默认 X 分块 | 当前仅有共享默认值；缺逐成员 default_config 与 body/grid 的同源定义 | [S06](#s06) |
| 无 benchmark 的 persistent 默认配置 | 保持 R 静态，分别选择各成员默认 X | 静态 R 已有；缺独立 X 默认值及累计 program 边界 | [S07](#s07) |
| 逐成员 coordesc 搜索 | 每步只改变一个成员配置，并优先调最大成员 | 缺成员独立搜索坐标、调优顺序与完整配置评价；需先实现独立 block | [S08](#s08) |

**SKIP：平台范围与旧测试禁用**

| 具体检查目标 | 作用 | 当前排除原因与处理 | 方法 |
| --- | --- | --- | --- |
| CUDA TMA descriptor/block 协议 | 使用 SM90+ 的张量异步搬运及 descriptor 约束 | CUDA 专属协议实例，保留平台排除；NPU 搬运能力需按自身协议定义 | [S09](#s09) |
| CUDA PDL 代码生成 | 生成 `launch_pdl` 与 GDC wait/launch-dependents 顺序 | CUDA SM90+ 的依赖启动协议，保留平台排除 | [S10](#s10) |
| CUDA PDL pointwise 执行 | 检查不同尺寸激活成员的依赖启动与数值 | CUDA PDL 协议实例，保留平台排除 | [S11](#s11) |
| CUDA PDL reduction 执行 | 在 PDL launch 下执行 sum/mean | 类级配置要求 CUDA PDL，保留平台排除；普通归约数值不覆盖该协议 | [S12](#s12) |
| HIP 编译选项与 constexpr 分离 | 避免 `waves_per_eu` 等后端选项混入成员参数 | 两个实例都是 HIP 专属配置，保留平台排除；NPU 应检查自己的选项 | [S13](#s13) |
| XPU 动态 persistent 归约实例 | 检查动态行数与 compile-time autotune | 来自 XPU 门控；对应 NPU 兼容计算已 PASS，无需新增动态归约功能 | [S14](#s14) |
| CPU 节点的 Combo 过滤 | 保证 CPU 节点不参与 Combo 融合 | 当前 NPU 实例没有 CPU 节点，无法验证；由 CPU 实例保留过滤断言 | [S15](#s15) |
| foreach 直接输入的依赖距离 | 检查非空节点的输入距离 min/max 为 0/0 | 旧无条件禁用；保留原断言和数值的诊断已通过，只需测试适配 | [S16](#s16) |
| extern 后 foreach 的依赖距离 | 检查 mm 到消费者的依赖层数，整体 min=0、max≥1 | 旧无条件禁用；非空 schedule 和原断言诊断已通过，只需测试适配 | [S17](#s17) |

“输入距离”是融合前从图输入开始的依赖边层数，group 的 min/max 汇总成员深度。
S16/S17 的诊断不改变正式 SKIP 状态。非正规数与 FTZ 的数值范围见 F05。

### 代码与执行证据

源码摘录包含仓库、完整提交、原行号与 SHA256；
[完整源码索引][source-index]、[31 项结构化分析][case-analysis]、[设备诊断与计数纠正][device-findings]。
原精确实例、FAIL/SKIP 消息及实际验证版本见 [合并 jobs.json][jobs]；
[设备原始报告、脚本/DSL 哈希和脱敏收据][device-manifest] 保留诊断的版本与来源限制。
allocator 的 raw pickle 留在原任务目录，仅归档不含地址的峰值分配与推导，见 F13。

| 证据 | 关键源码与职责 |
| --- | --- |
| [E1][E1] | NPU Combo 创建、轴/模式分区、共享 block、dispatch 与调用 |
| [E2][E2] | NPU 普通/Combo metadata 与 grid/成员 tiling |
| [E3][E3] | native ComboTileGenerator、资源模型、候选与社区大 RBLOCK 条件 |
| [E4][E4] | NPU/社区 CSE load/store/reduction 计数与连续归约访问 |
| [E5][E5] | optimize_mem 图阶段协议、CPU helper 与实际 clone 主 bench |
| [E6][E6] | provenance hook、node=None 外壳与分区叶子展开 |
| [E7][E7] | 内存增加量 gate、NPU 分区与 WRN 原测量范围 |
| [E8][E8] | FTZ metadata 与 CUDA/实际安装编译选项边界 |
| [E9][E9] | TMA/PDL/HIP/XPU/CPU-template 门控与 NPU 兼容层 |
| [E10][E10] | 输入距离算法、group 聚合与旧 foreach skip |
| [E11][E11] | 社区成员 metadata 契约与 NPU contiguous-format 限制 |
| [E12][E12] | 字节/FLOP 估算、Combo 聚合与 benchmark 消费 |
| [E13][E13] | WRN 组合节点过滤、native bench、192 MiB 清缓存分配 |
| [E14][E14] | 安装/最新 Triton Options 与 HMF exp/exp2 数学入口 |

[主机重放][host-results] 确认一次有效 store 的 NPU num_store=0、社区为 1；
provenance 外壳有 handle 而无 mapping，按实际 partition 展开叶子后 mapping 完整。
现有 gate 的 +2 MB 候选被 1 MB 拒绝，delta=0 被 0 阈值接受。
NPU 主 bench 两种 optimize_mem 值均只 clone，没有调用 CPU-copy/restore helper；
补字段与实现真实内存优化是两项能力。

### 14 个 FAIL 的逐项分析

<a id="f01"></a>

#### F01 `test_combo_kernel_per_subkernel_reduction_hint`

源方法：`test_combo_kernels.py::ComboKernelTestsMaxAutotune`；正式状态 **FAIL**，1 个实例。

**测试目标：** 对两个 contiguous 的 128×256 输入分别做 sum(-1) 与 sum(0)，在同一 Combo 内保留 INNER/OUTER 归约分类。

**实际结果：** 数值比较完成后，found_hints 为空，缺 reduction_hint_0，随后对 reduction_hint_1 和 INNER/OUTER 的检查未到达。

**master 因果链：** 失败执行的代码确实只有一个双成员 persistent Combo：行归约加载 r1+256*x0，列归约加载 x0+256*r1，两个成员的 low_dims 分别为 {1}/{0}，不能解释成“混合行列被
拆成普通 kernel”。
master 的 jit_line 只传首成员的 reduction_hint，combo_grid_meta 未导出 reduction_hint_i。
native runtime 使用 combo_subkernel_meta 的轴、尺寸和资源约束生成共享 tile，没有逐成员 hint 消费者；字段缺失不证明没有归约调优，也不等同于 mix_order_reduction 功能。

**需要适配什么：** 需要补齐真实的逐成员归约分类观测，或把用例接到 NPU 的实际成员布局；若要求 hint 影响优化，还需接入 native 候选生成/评价。
共享候选必须同时满足所有分支；不能只打印 INNER/OUTER，也不能把 GPU 的 XBLOCK_i 搜索直接移植。

**是否符合 NPU 设计：** 保留逐成员访问分类符合 NPU 设计；逐成员独立调优属于额外能力。
现有共享块可以承载这两个不同物理 stride 的归约，因此不需要为了此测试取消 ABI 分区或打开 mix-order 纵向融合。

**验证与结论边界：** 以原输入对照成员 FX 和地址公式，确认分类与两个方向一致；观测真实候选和 winner 对两种布局的处理，再保留数值比较。
已归档的 generated code 来自 d12230b556，不是 master 设备复测。

代码依据：[E1][E1]、[E2][E2]、[E3][E3]、[E11][E11]、[本例实际 DSL][hint-output]。

<a id="f02"></a>

#### F02 `test_combo_rms_norm_forwards_add_persistent_rblock`

源方法：`test_combo_kernels.py::ComboKernelMetadataTests`；正式状态 **FAIL**，1 个实例。

**测试目标：** 两个 2048×4096 RMSNorm 构成 Combo，要求社区大 RBLOCK 优化的 add_persistent_rblock 标志逐成员传给候选生成。

**实际结果：** 数值比较完成后，首先缺 inductor_meta_0；并没有执行到两个 add_persistent_rblock 的逐项断言。

**master 因果链：** 社区 add_persistent_rblock 主要针对尚未选择 persistent 的二维 looped reduction，在有显著访存节省且满足资源条件时增添大 RBLOCK 候选。
NPU 此输入已经走 persistent_reduction，失败执行的两个分支都固定 R1BLOCK_SUB=4096，覆盖完整归约行。
master 继续使用 ComboTileGenerator，按最大成员 resident tile、UB/SIMT 资源和每分支 program 数筛选共享候选，未消费社区标志。

**需要适配什么：** 本例应改为验证 NPU standalone/Combo 的完整归约 tile 和合法共享候选可达；不需要为了此输入复制 GPU 的 add_persistent_rblock。
如果原目标扩大为“looped reduction 可选择更大归约块”，需另选当前 NPU 尚未 persistent 的输入证明候选确实被漏掉，再扩展 native 搜索。

**是否符合 NPU 设计：** 完整归约行已进入实际 DSL，符合 native persistent 设计。
当前失败是 GPU 优化标志的观察口径不适配，不能据此宣称 NPU 缺大归约 tile 或该性能优化已经等价验证。
逐成员统计缺口另见 F06–F09。

**验证与结论边界：** 以相同输入比较 standalone/Combo 的归约元素集合、完整 R1 tile、合法 X 子块及 winner；确认 UB 约束和数值。
源码和现存 DSL 能确认整行可达，当前未做 master 的设备候选/性能对照。

代码依据：[E2][E2]、[E3][E3]、[E11][E11]、[本例实际 DSL][rms-output]。

<a id="f03"></a>

#### F03 `test_combo_inductor_meta_has_optimize_mem`

源方法：`test_combo_kernels.py::ComboKernelMetadataTests`；正式状态 **FAIL**，1 个实例。

**测试目标：** 无梯度的 relu/sigmoid 推理 Combo 显式传 optimize_mem=True，与图阶段和 standalone 协议一致。

**实际结果：** 编译、执行和数值比较完成，生成代码没有 optimize_mem=True。

**master 因果链：** master 的 Combo jit_line 复用首成员 create_inductor_meta；后者合并 inductor_meta_common，但没有图阶段字段。
cached_autotune 弹出字段时默认 True，因此本推理例当前属性值与期望相同，失败是生成协议缺口。
CPU-copy helper 在 True 时允许按预算搬移 mutated args、False 时直接返回；但 NPU 主 bench 与 batch 路径直接调用 clone_args，未调用此 helper/restore。
源码重放两种属性值均 clone 一次、CPU-copy/restore 为0，不能称该优化已在主路径生效。

**需要适配什么：** 在 NPU 普通 kernel 与 Combo 的统一元数据出口传 V.graph.is_inference or V.graph.is_backward。
共享 autotuner 可共用此图阶段值，不依赖独立分块。
若要让内存优化实际生效，还需在 native bench 的 runtime launch args 路径接入CPU-copy、排除相应 clone 与逐次恢复，不把补字段当作优化已实现。

**是否符合 NPU 设计：** 图阶段契约适用于 NPU，显式 metadata 可直接补齐；现有主路径通过 clone 保护输入。
CPU 搬移是尚未接入此主路径的可评估内存优化，不能解释成当前推理数值错误。

**验证与结论边界：** 同时覆盖 inference=True、training-forward=False、backward=True，并观测主 bench 的真实 clone/搬移调用。
若接入搬移，需用 mutated args 验证预算、runtime block 位置和恢复；原非 mutation 例只证明字段与数值。

代码依据：[E2][E2]、[E5][E5]、[E11][E11]。

<a id="f04"></a>

#### F04 `test_combo_inductor_meta_optimize_mem_false_in_training_forward`

源方法：`test_combo_kernels.py::ComboKernelMetadataTests`；正式状态 **FAIL**，1 个实例。

**测试目标：** 两个 requires_grad=True 的输入生成训练前向图，要求 optimize_mem=False。

**实际结果：** 数值比较完成，生成代码没有 optimize_mem=False；本例没有检查 backward 或 mutated args 的恢复。

**master 因果链：** 与 F03 是同一个出口遗漏：训练前向的约定值是 False，缺省属性却为 True。
但当前 native 主 bench/batch 调用 clone_args，未进入 CPU-copy helper，不因 True 而启用搬移。
社区 bench 的 copy/maybe_clone/restore 与 NPU 主路径需要分别判断。
原非 mutation 例没有数值污染或梯度错误证据。

**需要适配什么：** 统一图阶段取值，训练前向显式 False；保持现有 clone 输入保护。
不要简单把全部图默认值改成 False。
若以后接入 CPU-copy 内存优化，应以该值控制真实路径，并验证整个 native runtime 参数和恢复协议。

**是否符合 NPU 设计：** 这是生成协议遗漏，修复符合 NPU 图阶段设计；当前主路径的 clone 保护不受这个缺字段影响。
不能将它归为已发生的训练输入污染或已生效的错误 CPU 搬移策略。

**验证与结论边界：** 除原字段和数值断言，还需记录训练主路径实际 clone/搬移调用；涉及新增搬移时验证 mutation/alias、梯度和候选重试后的输入。

代码依据：[E2][E2]、[E5][E5]、[E11][E11]。

<a id="f05"></a>

#### F05 `test_combo_triton_meta_has_disable_ftz`

源方法：`test_combo_kernels.py::ComboKernelMetadataTests`；正式状态 **FAIL**，2 个实例。

**测试目标：** False/True 两个实例要求 Combo triton_meta 继承 eager_numerics.disable_ftz。
FP32 非正规数是绝对值小于 2**-126 的非零可表示值，最小为 2**-149；FTZ 将非正规数冲刷为零。

**实际结果：** 正式两实例均在普通随机值数值比较后缺 disable_ftz 字段，原输入没有验证非正规数。
11 月 13 日的定向诊断分别执行两种 flag、基本算术、归约、指数函数以及真实双成员 Combo。

**master 因果链：** NPU 自定义 jit_line 未合并社区 triton_meta_common，两个 flag 都没有传出。
社区只在 CUDA 分支把 True 映射为 enable_reflect_ftz=False。
实际安装的 Triton parse_options 对默认、disable_ftz=True、enable_reflect_ftz=False 生成相同 hash；最新 Triton-Ascend main 62f6b54 的 112 个
 NPUOptions 字段也无 FTZ 注册项。
设备上 FP32 拷贝、加法、乘 0.5/2 保留可表示的非正规数；归约的 eager/compiled 结果一致。
exp2(-127) 等指数结果却为 0，两种 flag 输出一致；exp2(-126) 的最小正常值边界也为 0，不能把所有边界零都解释为纯 FTZ。
绕过 Inductor 的直接 Triton 在 SIMD、template 模式（实际 parallel_mode=simd）和 pure SIMT 都出现相同行为；TTIR 调用
__hmf_exp2_fp32/__hmf_exp_fp32，SIMD adapter 落到对应 HIVM math intrinsic。
实际安装与最新 main 的 libdevice 文件逐字节一致，A5/A3 FP32 都走这些 HMF 函数。

代表性的 FP32 bit 结果如下；指数列是直接 Triton 的三个模式共同结果：

| 运算 | CPU bits | NPU bits | 结论 |
| --- | --- | --- | --- |
| 最小非正规数相加 | 0x00000002 | 0x00000002 | 保留非正规数 |
| 最小正常数乘 0.5 | 0x00400000 | 0x00400000 | 保留非正规数 |
| exp2(-127) | 0x00400000 | 0x00000000 | 指数输出冲零 |
| exp2(-126) | 0x00800000 | 0x00000000 | 正常值边界也冲零，不能只归为 FTZ |

**需要适配什么：** 统一 NPU 普通/Combo 的数值配置生成与真实编译选项传递。
基本算术已具备非正规数保留能力；指数路径需要 Triton/math library/CANN 定义并实现可验证的保留策略，再让 Inductor 的 True 配置消费该能力。
False 可表示后端默认策略。
只补 metadata 或传入未注册选项无法改变当前指数行为；若存在算子/模式边界，需明确能力范围。

**是否符合 NPU 设计：** 数值语义目标适用于 NPU，并非 CUDA 专属协议。
已证实 Inductor 配置出口遗漏，同时将指数路径的数值能力定界到下层数学入口；不应归因为 NPU 硬件完全无法表示非正规数。
最新源码检查不替代最新编译器/CANN 的设备执行。

**验证与结论边界：** 原始报告、逐 bit 输出、直接 Triton IR 与两份真实双成员 Combo DSL 均已归档。
两种 flag 的独立 exp2/exp Combo 各有 1 个真实 Combo、2 个分支；原诊断脚本误数社区 decorator 得到 0，raw 记录保留，另在 device-findings 中明确按 NPU grid
metadata 纠正。
正常值的 ULP 差异及归约的 CPU 1 ULP 差异不归为 FTZ，也不扩大为本轮生产修复。
设备执行仍使用 d12230b556 覆盖包与已安装 Triton 3.6/CANN，正式两实例仍为 FAIL。

代码依据：[E2][E2]、[E8][E8]、[E11][E11]、[E14][E14]、[逐 bit 数值诊断][ftz-report]、[直接 Triton/真实 Combo][ftz-direct]、
[设备诊断][device-findings]、[原始证据收据][device-manifest]。

<a id="f06"></a>

#### F06 `test_combo_pointwise_combo_grid_meta_has_per_subkernel_fields`

源方法：`test_combo_kernels.py::ComboKernelMetadataTests`；正式状态 **FAIL**，1 个实例。

**测试目标：** 三个 1024 元素的 relu/sigmoid/tanh 成员逐项携带 num_load/store/reduction、autotune_hints、atomic_add_found。

**实际结果：** 数值比较完成后，FileCheck 首先找不到 inductor_meta_0；后续字段的值尚未被检查。

**master 因果链：** NPU combo_grid_meta 只记录 grid 和 split 信息，combo_subkernel_meta 记录轴/tiling 约束，没有统计出口。
普通 NPU create_inductor_meta 也没有导出完整统计。
自定义 CSEProxy 会递增 load 和 reduction，但 store/store_reduction 未递增 num_store；源码重放实际发出一次 store 时 NPU 计数为 0，社区为 1，直接复用基类统计将给出错误
值。

**需要适配什么：** 先同步 CSE 的有效 store 计数及移除 store 的扣减语义，再统一普通 kernel/Combo 的真实统计出口。
按当前 NPU 实际成员导出三组数据，可与轴约束并存，不需要改变共享分块架构。

**是否符合 NPU 设计：** 统计与调试观测是通用编译器能力，符合 NPU 设计；这不是独立分块实现，也不是 relu/sigmoid/tanh 数值错误。

**验证与结论边界：** 检查三组成员和非零有效 load/store，pointwise 的 reduction=0；对 dead-store、mutation/atomic 和复用场景验证计数。
原 FileCheck 只检查字段名，不足以证明数值准确。

代码依据：[E2][E2]、[E4][E4]、[E11][E11]。

<a id="f07"></a>

#### F07 `test_combo_reduction_combo_grid_meta_has_per_subkernel_fields`

源方法：`test_combo_kernels.py::ComboKernelMetadataTests`；正式状态 **FAIL**，1 个实例。

**测试目标：** 两个 32×1024 输入的 sum/mean 成员各自携带 load/store/reduction 统计。

**实际结果：** 数值比较完成，首先缺 inductor_meta_0；没有证明归约本身执行错误。

**master 因果链：** 同 F06 的逐成员出口缺失。
NPU CSEProxy.reduction 会加 num_reduction，store_reduction 不加 num_store；归约后写回不能因为位于 post-loop 就漏计。
axis_names 和 tiling_axis 是资源约束，不是操作统计。

**需要适配什么：** 修普通 store 和 reduction store 的计数，再按真实成员分别导出统计；mean 的后处理算术与归约写回应按共同口径计量。

**是否符合 NPU 设计：** 符合共享候选与 native 归约的现有设计，属于统计协议适配；无需为了导出统计开放 mixed persistent/looped 成员融合。

**验证与结论边界：** 检查两组有效 load/store 和正的 reduction 计数，与 standalone 对齐；覆盖归约结果写回、被消除的中间 store 和广播输出。

代码依据：[E2][E2]、[E4][E4]、[E11][E11]。

<a id="f08"></a>

#### F08 `test_combo_reduction_deterministic_has_contiguous_rdim_per_subkernel`

源方法：`test_combo_kernels.py::ComboKernelMetadataTests`；正式状态 **FAIL**，1 个实例。

**测试目标：** deterministic=True 的 sum/mean 成员携带 has_loadstore_with_contiguous_rdim，用来描述 reduction 方向上是否存在 stride=1 访问。

**实际结果：** 首先缺 inductor_meta_0；未到达连续归约标志检查，原例也没有重复运行验证确定性。

**master 因果链：** NPU 未导出逐成员统计；其自定义 load/indexing 绕过社区更新 has_load_with_contiguous_rdim 的路径。
基类默认 False 不能当作真实 NPU 访问结果。
社区根据地址表达式与 reduction 变量 stride 推导此标志，native runtime 当前未消费同名 GPU 配置过滤字段。

**需要适配什么：** 在 NPU 实际 load/store 地址与归约轴上推导访问标志，在 deterministic/force-filter 范围导出并统一 standalone/Combo。
若用于筛选候选，需接到 NPU 自身规则，不直接拷贝 GPU RBLOCK 过滤。

**是否符合 NPU 设计：** 访问分类与确定性目标符合 NPU 设计，统计可以补齐；字段缺失并不证明 NPU 不支持确定性归约。

**验证与结论边界：** 对原 contiguous 输入与 transpose/stride>1 对照标志，覆盖实际输出 store，并重复运行相同输入核对结果；确认 native 候选选择没有违反既定确定性约束。

代码依据：[E2][E2]、[E4][E4]、[E11][E11]。

<a id="f09"></a>

#### F09 `test_combo_per_kernel_inductor_meta_matches_standalone`

源方法：`test_combo_kernels.py::ComboKernelMetadataTests`；正式状态 **FAIL**，1 个实例。

**测试目标：** 相同 relu/sigmoid 输入分别关闭/开启 Combo，逐成员字段与各 standalone kernel 一致。
原比较范围是 `num_load`、`num_store`、`num_reduction`、`atomic_add_found`、`no_x_dim`、
`autotune_hints`、`tiling_scores` 七个字段；其中 `tiling_scores` 仅在有实际评分时导出。

**实际结果：** standalone 匹配到 2 组，Combo 的 inductor_meta_i 匹配到 0 组，列表长度 2 != 0。

**master 因果链：** Combo 未输出逐成员字典；普通 NPU 元数据也未输出全部字段，因此 standalone 的两组不代表已有完整 load/store 统计。
当前 regex 允许字段缺失，空字典或只含 hints 的字典也可能通过相等检查。
NPU grid 的 no_x_dim_i 表示是否没有 split axis，不能不加核对地替代社区以 x iteration tree 为中心的 no_x_dim。

**需要适配什么：** 用一个 native per-kernel 出口服务普通和 Combo；先修真实计数。
no_x_dim 和 tiling_scores 应定义 NPU 对应含义，不支持的 GPU 专用项用明确的适用条件处理，不以空统计满足列表相等。

**是否符合 NPU 设计：** 同一成员在 standalone/Combo 中保留可比观测符合 NPU 设计；共享 launch 允许优化统计变化时，应解释变化的实际语义。

**验证与结论边界：** 保持 2 组数量检查，额外检查必需字段与实际非零计数，再逐成员与同一 FX/布局对齐；不能只让两个列表长度相等。

代码依据：[E2][E2]、[E4][E4]、[E11][E11]。

<a id="f10"></a>

#### F10 `test_combo_kernel_num_gb_and_flop_match_standalone_sum`

源方法：`test_combo_kernels.py::ComboKernelMetadataTests`；正式状态 **FAIL**，1 个实例。

**测试目标：** benchmark_kernel=True 时，两个 relu/sigmoid 成员的 Combo 字节/FLOP 估算等于相同 standalone 估算之和。

**实际结果：** 首先没有 Combo kernel_num_gb；随后 kernel_flop 存在性和两种求和检查未执行。

**master 因果链：** 普通 NPU codegen_kernel 在 benchmark/profile 时调用继承的 estimate_kernel_num_bytes，但 Combo jit_line 没有聚合出口。
NPU 普通出口也未导出 kernel_flop，原测试对未匹配到的 standalone FLOP 求和会得到 0；这不是已验证的有效 FLOP 基准。
基类有基于真实 node_schedule 的 FLOP 估算入口，但仍须核对 native node/numel 的口径。

**需要适配什么：** 统一两种出口的有效字节/FLOP 估算，并在 Combo 顶层求和。
处理 slice、broadcast、alias/in-place，未知估算须有清晰含义，不能把“缺失/未知”默默当成真实 0 来满足测试。

**是否符合 NPU 设计：** 成本估算是 NPU 可共享的 compiler/profiler 能力，与共享 tile 架构相容。
GB 是编译期访存估算，不是设备实测带宽。

**验证与结论边界：** 原 2×1024 FP32 pointwise 以输入+输出、无 alias 的常规估算为 16384 bytes，即 1.6384e-5 GB；再对真实 standalone 值求和，验证有效 FLOP。
覆盖非连续输入、slice、广播和 mutation 的计量。

代码依据：[E2][E2]、[E4][E4]、[E11][E11]、[E12][E12]。

<a id="f11"></a>

#### F11 `test_combo_inductor_meta_has_kernel_num_gb_under_profile_bandwidth`

源方法：`test_combo_kernels.py::ComboKernelMetadataTests`；正式状态 **FAIL**，1 个实例。

**测试目标：** 仅 profile_bandwidth=True 时，Combo 输出 kernel_num_gb，且不输出 kernel_flop。

**实际结果：** 编译、执行和数值比较完成后缺 kernel_num_gb；不是此前 debug autotuner 构造/run 接口异常。

**master 因果链：** 最新 master 的普通 NPU codegen_kernel 有 profile 字节估算，Combo jit_line 没有聚合。
native runtime 已按 profile_bandwidth 选择 debug autotuner；共享 tile 调优存在并不能自动补齐 profiler 的成本元数据。

**需要适配什么：** 按 profile 条件聚合有效成员字节估算并传给 native profiler；FLOP 只在要求的 benchmark 条件输出，保持无 profile/benchmark 的普通编译不额外携带成本字段。

**是否符合 NPU 设计：** 符合现有 NPU profiler 和共享 autotuner 设计，属于可补齐的统计接入。

**验证与结论边界：** 保留原 profile 数值/字段断言，联验无 profile 的既有 PASS 方法；检查打印或统计消费的是 Combo 总估算而非首成员值。

代码依据：[E2][E2]、[E11][E11]、[E12][E12]。

<a id="f12"></a>

#### F12 `test_benchmark_combo_kernel_emits_real_num_gb`

源方法：`test_combo_kernels.py::ComboKernelMetadataTests`；正式状态 **FAIL**，1 个实例。

**测试目标：** benchmark_combo_kernel=True 的代码包含真实浮点 num_gb 估算和带引号的 NPU device 字符串。

**实际结果：** 数值比较完成后，没有匹配到 num_gb 的非零浮点估算；后续 device 字符串检查未到达。

**master 因果链：** NPU codegen_kernel 调用继承的 codegen_kernel_benchmark(num_gb=0)。
社区模板直接生成 num_gb = 0，而测试要求浮点估算，regex 因此失败；这也使带宽报告为零。
模板自身已经按实际 device.type 生成带引号的设备参数，不能据首个失败推断 device 字符串也有错误。

**需要适配什么：** 复用有效成员字节估算之和，传给 benchmark 模板；保留真实 device.type 和原计时路径。
本例不能用 0.0 仅绕过 regex，须检查估算值与 F10 的口径一致。

**是否符合 NPU 设计：** 符合 native 调优/benchmark 设计，修复的是报告成本，不会因此实现逐成员独立 autotune。

**验证与结论边界：** 核对生成的 num_gb=1.6384e-5（原无 alias pointwise 常规口径）、quoted NPU device，并实际调用 benchmark 入口确认报告使用该值。
共用环境中的本轮记录不构成性能优劣结论。

代码依据：[E2][E2]、[E12][E12]。

<a id="f13"></a>

#### F13 `test_combo_kernel_peak_memory_wide_resnet`

源方法：`test_combo_kernels.py::ComboKernelPeakMemoryTests`；正式状态 **FAIL**，1 个实例。

**测试目标：** 同一 WideResNet-like 模型在禁用门控与 1 MB 增加量阈值下编译执行，要求后者实际峰值严格降低。

**实际结果：** 正式两次 max_memory_allocated 都为 616101888 bytes，严格小于断言失败。
11 月 13 日按原模型、原配置并分别使用 fresh_cache 重放，两次首次峰值仍为 616101888，warm 执行均为 428341248；无 Combo 候选或 codegen 调用。

**master 因果链：** master 沿用社区增加量 gate：delta=max(running_peak, region_peak)-baseline_peak <= limit，约束相对原图的增加量，不承诺峰值严格下降。
实际安装环境两种配置的 schedule 前后均为 259 节点。
额外分组诊断得到 253 个拓扑 group，其中 247 个单节点、6 个双节点；六组双节点都由 ExternKernelSchedulerNode 组成，被 combinable_nodes 过滤。
过滤后 168 组为 0 成员、85 组为 1 成员，没有满足 >=2 的候选，因此不会进入 memory gate；tight 仍正常初始化 baseline_peak=19267584。
相同峰值不是该模型漏接 gate 或 TTIR 后拆分造成。
原测量在 torch.compile 前重置峰值，包含首次 autotune；两份 allocator trace 精确重建：探测前 395507200 + Triton 清缓存张量 201326592 + wrapper 活跃分配
12845056 + mutated args clone 6423040 = 616101888。
192 MiB 来源为 driver.get_empty_cache_for_benchmark，实际安装和最新 Triton driver 都固定此大小。

两种配置各自使用 fresh_cache；峰值包含探测前已分配量，单位为 bytes：

| 配置 | 首次峰值 | warm 峰值 | gate 候选调用 | Combo codegen |
| --- | ---: | ---: | ---: | ---: |
| disabled | 616101888 | 428341248 | 0 | 0 |
| 1 MB 增加量阈值 | 616101888 | 428341248 | 0 | 0 |

首次峰值的四项活跃分配逐份与 allocator 统计相等：

| 分配来源 | bytes |
| --- | ---: |
| 探测前已分配 | 395507200 |
| Triton autotune 清缓存张量 | 201326592 |
| wrapper 活跃张量 | 12845056 |
| mutated args clone | 6423040 |
| 合计 | 616101888 |

**需要适配什么：** 此模型在实际 NPU lowering 下不具备“门控后必须严格降峰”的前提，需适配测试图与测量范围。
选择确有多个可组合成员、正内存 delta 的可控 NPU 图，记录候选接受/拒绝及最终 emitted Combo，warmup 后比较稳定执行峰值，并保留对增加量 gate 的断言。
现有源码和设备证据不要求增加第二套 gate；仅放宽原严格断言无法验证门控。
若新可控图证明正 delta 超阈值仍接受，才修对应内存估算/接口。

**是否符合 NPU 设计：** 现有相对增加量门控符合 NPU 设计；extern 节点过滤后无候选时，相同峰值合法且已经设备确认。
主机测试也确认 +2 MB 候选被 1 MB 拒绝、delta=0 候选可被 0 阈值接受。
首次 autotune 的 192 MiB 缓存主导探测期间新增分配，不能用它判断运行期 Combo 内存优化。

**验证与结论边界：** 两次 fresh-cache 重放的 first/warm 四份 snapshot 均按 512 bytes 分配对齐逆向重建，峰值逐份等于 max_memory_allocated；仅保存脱敏分配大小、栈、推导脚
本及 raw SHA256 收据。
首次两块 wrapper 活跃张量各 6422528 bytes，分别对应 buf25/34；clone 来自实际 native bench。
warm 增量为 32834048，其中 16777728 来自 ATen convolution 调用的额外分配，依据大小和栈推断为工作区，其余为活跃张量。
两种配置均有 85 个生成 kernel，未生成 Combo。
此证据解释 d12230b556 安装环境的正式失败；未将最新 master 装包复测，也未在本轮实现新的门控测试图。

代码依据：[E1][E1]、[E7][E7]、[E13][E13]、[主机重放][host-results]、[独立缓存重放][wrn-fresh]、[真实节点分组][wrn-groups]、
[峰值分配与栈][wrn-allocations]、[设备诊断][device-findings]。

<a id="f14"></a>

#### F14 `test_triton_kernel_to_post_grad_tracing_combo_kernel`

源方法：`test_provenance_tracing.py::TestProvenanceTracingArtifact`；正式状态 **FAIL**，1 个实例。

**测试目标：** 把 Combo 的真实 relu/sigmoid/tanh 成员映射到 post-grad FX nodes，覆盖 Inductor 和 AOTInductor。

**实际结果：** Inductor 子检查的 cppCodeToPost 映射为 0 项，预期 1 项；后面的 AOTInductor 子检查未到达。

**master 因果链：** master 已调用 codegen_comment([combo_kernel_node], kernel_name)，后者调用社区 provenance setter 和 wrapper handle。
根因不是缺 hook：Foreach/Fused 外壳的 node=None，setter 只对 snode.node 非空的条目登记 origins。
主机重放外壳仍产生 handle=1 但 mapping={}；传实际三叶子才得到三个算子。
当前每个分区也把整组外壳传给 define_kernel。

**需要适配什么：** 对当前 node_group 展开每个成员的 get_nodes，把该分区真实叶子列表同时传给 define_kernel 和 codegen_comment；按实际生成/复用的 kernel_name 写
handle。
不要把全 Combo 的所有 origins 登记到每一个被拆分 kernel。

**是否符合 NPU 设计：** 通用追踪能力完全符合 NPU 设计，修复不影响设备 launch 或数值。
hook 已存在，应修传入节点，不是再添加一个重复 hook。
最新命名路径还需按实际名称验证，不能永久硬编码 GPU 名称后缀。

**验证与结论边界：** 覆盖单 Combo、拆分多 partition、源代码缓存复用及三个真实 origins；再分别执行 Inductor/AOTInductor 子检查，验证 wrapper handle 与映射对应。
当前主机重放只确认 setter 协议，没有声称 AOTInductor 设备通过。

代码依据：[E1][E1]、[E6][E6]、[主机重放][host-results]。

### 17 个 SKIP 的逐项分析

<a id="s01"></a>

#### S01 `test_combo_kernel_per_config_subkernel_poi`

源方法：`test_combo_kernels.py::ComboKernelTests`；正式状态 **SKIP**，1 个实例。

**测试目标：** 512 与 524288 元素的 pointwise 成员拥有各自 block 参数与选择空间，并在一个 Combo 内正确 dispatch。

**实际结果：** 显式 NPU 独立分块 SKIP，未执行原输入。
所选共享类的条件分支不检查 XBLOCK_i，不能将共享数值通过当成独立分块通过。

**master 因果链：** master codegen_combo_kernel 没有把 scheduler node 的 per_subkernel_blocks 传给生成函数，默认 False；get_block_args 按轴名
称去重，共用 X0BLOCK/X0BLOCK_SUB，dispatch 使用同一 block 计算每成员 program 数。
因此开关在 NPU 不产生两个独立 block ABI。

**需要适配什么：** 增加逐成员 runtime/constexpr block 名、配置序列、累计 program 边界和 launcher 参数映射，并让候选评价实际使用每个成员的选择；保持共享路径作为另一种受支持模式。

**是否符合 NPU 设计：** 属于通用编译器优化，可在 NPU 设计上扩展，但不是当前共享 ABI 已实现的能力，也不是硬件专属限制。

**验证与结论边界：** 原不等长输入同时验证数值、两个真实独立 tile、dispatch 边界与 winner；切换配置应改变相应成员工作划分。

代码依据：[E1][E1]、[E3][E3]。

<a id="s02"></a>

#### S02 `test_combo_kernel_per_config_subkernel_per`

源方法：`test_combo_kernels.py::ComboKernelTests`；正式状态 **SKIP**，1 个实例。

**测试目标：** 1024×64 与 1024×512 两个 persistent reduction 的 X 分块可独立选择，归约长度在各自 body 中固定。

**实际结果：** 显式 NPU 独立分块 SKIP；原两个归约没有在此方法内重新执行。

**master 因果链：** NPU 已可按成员静态尺寸生成不同的 R*_BLOCK_SUB，但共用 X runtime/tiling block。
不同归约长度的静态常量并不代表每成员都拥有独立 X 分块配置。

**需要适配什么：** 在保留各自完整 R 长度和资源约束的基础上增加 X 子块独立 ABI、候选、默认配置和 program 边界。
无需强迫 NPU 轴名变成 GPU R0_BLOCK_i。

**是否符合 NPU 设计：** persistent 整行归约与成员独立 X 选择可以相容；后者是待实现的优化，不应继续用共享块 PASS 覆盖。

**验证与结论边界：** 验证两个归约集合和 mask，各成员固定 R、独立 X 实际进入 launcher/DSL，并核对最大 resident tile 限制。

代码依据：[E1][E1]、[E3][E3]。

<a id="s03"></a>

#### S03 `test_combo_kernel_per_config_subkernel_red_per`

源方法：`test_combo_kernels.py::ComboKernelTests`；正式状态 **SKIP**，1 个实例。

**测试目标：** 512×128 与 256×2048 输入的 persistent/regular reduction 成员在一个 Combo 内各自配置与执行。

**实际结果：** 显式 NPU 独立分块 SKIP；注释中 GPU 的 persistent/regular 分类没有被当作 NPU 实测分类。

**master 因果链：** 除缺独立 block ABI，NPU horizontal_partition 明确按 persistent_reduction、inside_reduction、kernel type 和轴标记分组，不把不
同模式成员装进同一个共享 autotuner。
是否此输入在 NPU 真产生两个模式还需 native 实际分类。

**需要适配什么：** 实现逐成员 block/heuristic/候选描述后，才能设计不同归约模式的同-launch 协调；同一编译目标和资源上限仍须成立，不能直接删除 mode 分区。

**是否符合 NPU 设计：** 通用优化目标可评估为 NPU 扩展，但直接照搬 GPU 的混合模式违反当前共享 launch 契约。
应先证明 native 模式和收益，再确定允许的组合。

**验证与结论边界：** 记录每成员实际 persistent/looped 分类，验证独立候选、完整归约/loop 累积、尾块和最大资源；要求一个 Combo 时必须确认没有 TTIR 回退成两个普通 kernel。

代码依据：[E1][E1]、[E3][E3]。

<a id="s04"></a>

#### S04 `test_combo_kernel_per_config_subkernel_red`

源方法：`test_combo_kernels.py::ComboKernelTests`；正式状态 **SKIP**，1 个实例。

**测试目标：** 两个 32×64×128 输入做 sum((0,2))，非单一 contiguous reduction 的成员分别拥有 reduction block。

**实际结果：** 显式 NPU 独立分块 SKIP，没有验证原输入的独立 reduction 参数。

**master 因果链：** 两个成员都落在共享 get_block_args/同 mode 的 launch 设计；多轴/非连续归约在 NPU 使用实际轴分析和原生 tiling，GPU 的 XBLOCK_i/R0_BLOCK_i 不能
仅改名就成为对应成员参数。

**需要适配什么：** 为实际 NPU 归约轴分别表示 runtime/constexpr 子块、loop 累积和 masks，并把独立成员候选接到共享设备资源校验。

**是否符合 NPU 设计：** 符合编译器优化扩展方向；需要沿用 NPU reduction 轴语义和 UB 模型，不能用 GPU 扁平 reduction 变量代替。

**验证与结论边界：** 保持原 stride/归约维度和 eager 结果，验证成员独立 block 真的改变 loop/工作划分且最终 store 位于完整归约之后。

代码依据：[E1][E1]、[E3][E3]。

<a id="s05"></a>

#### S05 `test_combo_kernel_per_config_subkernel_block_size`

源方法：`test_combo_kernels.py::ComboKernelTests`；正式状态 **SKIP**，1 个实例。

**测试目标：** 八个 channels-last 转换成员使用独立二维 tile，profiler 启动 grid 与各自 block/dispatch 一致。

**实际结果：** 当前因独立分块未实现提前 SKIP；没有执行到 channels_last eager 或 profiler。

**master 因果链：** 同时存在三个边界：共享 block ABI；master NPUNativeFunctions::contiguous 对尚未满足请求格式的输入只允许转换为 Contiguous，原 NCHW 到
channels_last 转换有独立 eager 阻塞，已有该格式的输入可先返回；测试固定 GPU profiler grid=[3795,1,1] 或 [791,4096,1]，与 NPU 一维
SequentialComboKernelGrid 不同。

**需要适配什么：** 独立分块扩展之外，需要分别确定 ATen channels-last 的 NPU支持策略和 profiler/真实 launch 观测。
若测试目标仅为 tile/grid 对齐，可用具有等价 transpose/stride 意义的 NPU支持输入单独覆盖，不能将其冒充原 channels-last 支持。

**是否符合 NPU 设计：** 独立 tiling 与 grid 一致性适用于 NPU；channels-last eager 属于 ATen tensor/layout 责任，固定 GPU grid 不属于 NPU 必须满足的设计。
此项不能只交给 Inductor 修一个开关。

**验证与结论边界：** 先分别验证 eager layout、真实 NPU launch 记录，再验证八成员独立配置、二维地址/一维 dispatch 对齐；保留数值，原 eager 不支持时明确区分改用输入的覆盖范围。

代码依据：[E1][E1]、[E3][E3]、[E11][E11]。

<a id="s06"></a>

#### S06 `test_combo_kernel_no_bench_stitched_config`

源方法：`test_combo_kernels.py::ComboKernelTests`；正式状态 **SKIP**，1 个实例。

**测试目标：** 关闭 Combo benchmark 时，三个 256 元素 pointwise 成员的 default_config 与 body 中各自 XBLOCK_i 常量一致。

**实际结果：** 显式独立分块 SKIP；不以共享 default_config 的数值执行代替逐成员配置拼接。

**master 因果链：** NPU 有合法共享默认配置，combo_grid_meta 用 default_config={XBLOCK:实际共享 block 参数}，codegen_blocks 写共享常量；没有三份成员默认 tile
 或逐成员 XBLOCK_i 体内常量。

**需要适配什么：** 新增逐成员默认 tile 的生成、拼接、合法性筛选及 body/grid/launcher 同源定义，无 benchmark 时也应完整支持独立配置，而不是让开关退回共享配置。

**是否符合 NPU 设计：** 无计时默认路径与 NPU 设计相容，缺的是独立配置契约；不能解释为 NPU 没有无 benchmark 执行能力。

**验证与结论边界：** 同一来源的三份 block 必须同时匹配生成 body、metadata 和实际 grid，保留三输出数值检查。

代码依据：[E1][E1]、[E2][E2]、[E3][E3]。

<a id="s07"></a>

#### S07 `test_combo_kernel_no_bench_persistent_reduction`

源方法：`test_combo_kernels.py::ComboKernelTests`；正式状态 **SKIP**，1 个实例。

**测试目标：** 关闭 benchmark 的三个 64×1024 persistent sum：default_config 只含各自 XBLOCK_i，归约 block 固定在各自 body。

**实际结果：** 显式独立分块 SKIP，未验证三份成员 X 默认值。

**master 因果链：** NPU 可按成员写静态完整 R*_BLOCK_SUB，但默认 X 配置/dispatch 仍共享。
body 中固定 R 与 metadata 中独立 X 是两个不同契约，已有前者不能推定后者。

**需要适配什么：** 增加三份独立 X default_config 和累计边界；保持 R 为体内常量，不把静态 R 错误加入需要 runtime materialize 的参数。

**是否符合 NPU 设计：** 与 NPU persistent/static reduction 设计相容，属于独立默认配置扩展。

**验证与结论边界：** 验证三成员 default_config 的 X 值与 body/launch 一致，不含不需要的 runtime R，检查完整归约及尾块数值。

代码依据：[E1][E1]、[E2][E2]、[E3][E3]。

<a id="s08"></a>

#### S08 `test_combo_kernel_coordesc_tunes_largest_subkernel_first`

源方法：`test_combo_kernels.py::ComboKernelTestsMaxAutotune`；正式状态 **SKIP**，1 个实例。

**测试目标：** 32×1024、256×256、16×128 三成员中，coordesc 首先只改变最大成员的独立 XBLOCK_1，且探索多份不同配置。

**实际结果：** 独立分块 SKIP，没有逐成员 coordesc 顺序执行证据。

**master 因果链：** 没有 XBLOCK_i 的独立状态空间，就没有“只改最大成员”的搜索坐标。
native 的共享候选搜索和全 Combo 计时仍存在，但它们不提供该成员坐标与顺序契约。

**需要适配什么：** 先实现成员独立 block，再接逐成员 coordesc 维度、完整配置评估与按工作量/计时定义的成员优先级。
不能用共享 tile 的多个候选日志冒充逐成员坐标下降。

**是否符合 NPU 设计：** 可作为 NPU 编译器优化扩展；当前共享搜索符合自身设计，尚不覆盖该独立优化。

**验证与结论边界：** 记录真实 baseline/尝试配置，第一次只改变最大成员参数，后续确实探索不同配置；每次完整 launch 满足资源与数值要求。

代码依据：[E1][E1]、[E3][E3]。

<a id="s09"></a>

#### S09 `test_combo_kernel_tma_descriptor_block_name`

源方法：`test_combo_kernels.py::ComboKernelTests`；正式状态 **SKIP**，1 个实例。

**测试目标：** 对两个行归约成员检查 NVIDIA TMA tensor descriptor 的生成，受 CUDA SM90+ 条件约束。

**实际结果：** 明确 CUDA-only TMA SKIP，未执行原输入或 descriptor 检查。

**master 因果链：** 测试开启 triton.use_tensor_descriptor、assume_aligned_inputs，要求 make_tensor_descriptor。
NPU utils 的 patch_device_supports_tma 会把通用探测改为 NPU 可用性，因此该探测不能代表 NVIDIA TMA 硬件存在，也不能代表 NPU Combo 的 descriptor 路径已验证。
NPU 有自己的 indexing/搬运协议。

**需要适配什么：** 此 NVIDIA 协议实例无需移植。
若要验证 NPU tensor-descriptor/搬运能力，应按实际后端 descriptor 支持、轴名和 block 约束另外定义目标，不能只放宽 SM90 或利用兼容探测宣称 TMA 通过。

**是否符合 NPU 设计：** 保留平台 SKIP 合理。
它不表示 NPU 不支持 sum，也不直接否定通用 tensor descriptor 的可扩展性。

**验证与结论边界：** NVIDIA 原实例由 CUDA 环境验证；NPU descriptor 目标需独立核对实际 compiler/launch 协议，本轮未新增该能力验证。

代码依据：[E9][E9]、[E1][E1]。

<a id="s10"></a>

#### S10 `test_pdl_codegen_in_combo_kernel`

源方法：`test_combo_kernels.py::ComboKernelPDLTests`；正式状态 **SKIP**，1 个实例。

**测试目标：** pointwise Combo 生成 launch_pdl=True 和每分支恰当顺序的 CUDA gdc_wait/gdc_launch_dependents。

**实际结果：** 明确 CUDA-only PDL SKIP。

**master 因果链：** 测试断言直接引用 tl.extra.cuda.gdc_*，并要求 SM90+ 的 grid dependency control 和对应启动协议；NPU Combo 使用自己的 Sequential grid
 与 runtime launcher，没有此 CUDA 协议契约。

**需要适配什么：** 无需为 NPU 实现同名 CUDA intrinsic。
若提出 NPU 异步依赖启动需求，应按实际同步/内存可见性设计单独能力。

**是否符合 NPU 设计：** 平台 SKIP 符合 NPU 当前 launch 设计，不能以打印 launch_pdl=True 替代依赖与同步协议。

**验证与结论边界：** 该实例留在 CUDA 平台验证；NPU 数值和普通 Combo launch 的通过不是 PDL 证据。

代码依据：[E9][E9]。

<a id="s11"></a>

#### S11 `test_pdl_combo_kernel_pointwise`

源方法：`test_combo_kernels.py::ComboKernelPDLTests`；正式状态 **SKIP**，1 个实例。

**测试目标：** 三个不同尺寸 activation 成员在 CUDA PDL launch 下数值正确，每分支有对应 wait/launch_dependents。

**实际结果：** 明确 CUDA-only PDL SKIP，没有以仅数值通过宣称 PDL 实现。

**master 因果链：** 除了数值，还检查 launch_pdl=True、CUDA gdc 调用以及指定分支 dispatch；这些是 CUDA PDL 协议，当前 NPU 共享 dispatch 和设备 launch 接口不同。

**需要适配什么：** 无需将此 PDL 实例移植到 NPU；通用 pointwise Combo 数值与分支 mask 由适用用例覆盖。

**是否符合 NPU 设计：** 平台排除合理，目标中的激活计算适用于 NPU，但不构成 PDL 支持证据。

**验证与结论边界：** CUDA 实例验证协议及数值；NPU 保留适用 Combo 的真实分支/数值验证，本轮不额外改变正式状态。

代码依据：[E9][E9]、[E1][E1]。

<a id="s12"></a>

#### S12 `test_pdl_combo_kernel_reduction`

源方法：`test_combo_kernels.py::ComboKernelPDLTests`；正式状态 **SKIP**，1 个实例。

**测试目标：** 在类级 triton.enable_pdl=True 条件下执行 sum/mean 的 CUDA PDL reduction Combo。

**实际结果：** 明确 CUDA-only PDL SKIP；原方法只比较输出，不能把普通 NPU reduction 数值 PASS 当成 PDL 支持。

**master 因果链：** 协议通过类 setup 和 SM90/CUDA gate 生效，但方法本身没有检查 gdc/launch_pdl。
兼容层把 GPU 入口重定向到 NPU 时，单纯数值检查可能走普通 launch，丢失 PDL 测试目标。

**需要适配什么：** 保留平台 SKIP；不需要移植 CUDA PDL。
若通用 reduction 已有覆盖，不再用此方法重复计为设备依赖启动能力。

**是否符合 NPU 设计：** 符合 NPU 当前普通 reduction launch 设计；数值算子和平台启动协议需要分别判断。

**验证与结论边界：** CUDA 实例应结合 protocol artifact 验证 PDL 真生效；NPU 普通 sum/mean 由适用用例验证。

代码依据：[E9][E9]。

<a id="s13"></a>

#### S13 `test_combo_kernel_amd_special_config_args`

源方法：`test_combo_kernels.py::ComboKernelTests`；正式状态 **SKIP**，2 个实例。

**测试目标：** max_autotune=False/True 两个实例验证 HIP waves_per_eu、matrix_instr_nonkdim、kpack 等选项不被误写成成员 constexpr。

**实际结果：** 两个实例均在 not torch.version.hip 时 SKIP，原方法还要求独立分块开关。

**master 因果链：** 社区 runtime 在 HIP 分支把 signature 外的 cfg kwargs 分离到 backend_options，再作为 compiler options 传入。
这些是 HIP 专用参数，NPU native 有自己的 compile_mode/搬运等选项，不应接收 HIP 配置名。

**需要适配什么：** 本 HIP 实例无需 NPU移植；NPU“编译选项与 kernel/runtime 参数分离”是适用的通用协议，但应使用 NPU真实选项另行验证。

**是否符合 NPU 设计：** 平台 SKIP 合理，不表示 NPU 不支持 max-autotune，也不把两实例计为独立分块实现。

**验证与结论边界：** HIP 验证原选项；NPU 对实际 compile option/runtime block 进行参数归属检查，保持合法 native launcher。

代码依据：[E9][E9]、[E1][E1]。

<a id="s14"></a>

#### S14 `test_dynamic_shapes_persistent_reduction_mixed_x_dim_xpu`

源方法：`test_combo_kernels.py::ComboKernelDynamicShapesTests`；正式状态 **SKIP**，1 个实例。

**测试目标：** XPU 平台下三个动态行数的 sum/mean/max 成员，并打开 compile-time autotune。

**实际结果：** 沿用 requires xpu and triton SKIP，没有执行此 XPU实例。

**master 因果链：** 排除来自 requires_xpu_and_triton，不是 sum/mean/max lowering 错误。
同文件 CUDA 命名的对应方法经 NPU兼容入口执行相同计算与动态范围，已在合并报告 PASS，实际版本见 jobs。

**需要适配什么：** 无需为这个 XPU门控实例新增 NPU算子能力；可把通用动态归约目标抽为设备适用检查，继续验证真实动态 launch 与 compile-time autotune。

**是否符合 NPU 设计：** 平台 SKIP 合理，不能写成 NPU 不支持动态 persistent reduction 或 max。
对应 PASS 的设备/版本和融合模式仍要按实际报告解释，不能外推为 master 全量复测。

**验证与结论边界：** 若统一设备测试，保留动态行维 min/max 标记、不同尺寸和 eager/compiled 输出，对照 NPU实际候选/launch；无需重复统计两个平台别名。

代码依据：[E9][E9]。

<a id="s15"></a>

#### S15 `test_combo_kernel_filter_cpu`

源方法：`test_torchinductor.py::CommonTemplate`；正式状态 **SKIP**，1 个实例。

**测试目标：** CPU/GPU 共用模板：CPU 不做 Combo 融合、GPU 按指定数量融合，验证 CPU 节点过滤。

**实际结果：** 当前 NPU模板实例明确测试范围不适用 SKIP。

**master 因果链：** 所有输入都在 self.device；原 CPU分支检查 4 kernels，CUDA分支检查 2 kernels，NPU分支只有数值比较。
NPU实例并没有 CPU 节点，无法观察 CPU过滤策略。

**需要适配什么：** 此 NPU实例无需生产适配；CPU实例保留过滤断言。
若要验证 NPU/CPU混合图，应实际构造混合设备节点并定义对应过滤目标。

**是否符合 NPU 设计：** 测试范围排除合理。
不是 NPU 不支持 mul/add/min/max，也不是混合设备过滤已验证。

**验证与结论边界：** 由 CPU实例验证此策略；后续混合图测试必须检查实际节点设备与 emitted kernels，不能仅比较输出。

代码依据：[E9][E9]。

<a id="s16"></a>

#### S16 `test_foreach_basic`

源方法：`test_snode_runtime.py::InputDistanceTests`；正式状态 **SKIP**，1 个实例。

**测试目标：** 直接消费图输入的 foreach_add，其 scheduler 节点输入距离 min/max 都为 0。
距离指融合前依赖边的层数。

**实际结果：** 正式方法仍因旧 NPU does not support 无条件 SKIP。
11 月 13 日仅在诊断中 unwrap skip，原距离断言全部保留；非空节点和 FX eager 输出检查均通过。

**master 因果链：** 原方法用 Inductor 收集 metrics.nodes_num_elem，再返回 FX gm 做 eager 执行，并不要求 FOREACH 分组融合。
master 的 compute_input_distances 对无 unmet dependency 节点给 0，并聚合 group 成员的 min/max。
实际 NPU lowering 产生两个 ComputedBuffer SchedulerNode，均无 unmet dependency，min/max 都为 0/0；不是空节点循环造成假 PASS。

**需要适配什么：** 测试适配即可：移除旧无条件 skip，加上非空 scheduler 观测检查，保留原 0/0 距离断言与数值。
不需要为本目标新增 FOREACH 分组融合或硬件能力。

**是否符合 NPU 设计：** 输入距离是现有通用 scheduler 能力，真实 NPU 节点已满足原目标；旧禁用不符合当前事实。
foreach 分解为普通节点仍保持依赖语义，符合现有 NPU lowering 设计。

**验证与结论边界：** 原两组 10/20 元素输入定向执行为 PASS_DIAGNOSTIC，已保存节点类型、非空观测、0/0 距离和FX eager 输出一致证据。
诊断未修改正式方法，CSV/jobs 仍 SKIP；执行为 d12230b556 安装环境。
早期 foreach 运行时未捕获完整脚本哈希，保留报告哈希和恢复脚本的这一限制。

代码依据：[E10][E10]、[主机重放][host-results]、[非空节点与原断言][foreach-report]、[设备诊断][device-findings]。

<a id="s17"></a>

#### S17 `test_foreach_after_extern`

源方法：`test_snode_runtime.py::InputDistanceTests`；正式状态 **SKIP**，1 个实例。

**测试目标：** mm 产生数据后 foreach_add 消费它，检查整个 schedule 的 min distance=0、max distance≥1。

**实际结果：** 正式仍为旧无条件 SKIP。
11 月 13 日 unwrap skip 的定向诊断保留原距离断言，非空 schedule 与 FX eager 输出检查均通过。

**master 因果链：** master 沿用跨 extern 的通用依赖距离算法；原断言取整个 schedule 的 min/max，不要求每个 foreach 成员自身的 min=0。
实际 NPU schedule 中 mm 是 ExternKernelSchedulerNode，距离 0/0；两个 foreach_add 消费者构成 FusedSchedulerNode，读取 buf0，group 和两叶子均为 1/1
。
整体 min=0、max=1 正好满足原目标，未触发算子或 scheduler 不支持。

**需要适配什么：** 移除旧 skip，保留 mm、两个消费者及原整体 0/>=1 断言，补非空观测。
当前图无需新增 NPU 依赖能力，也无需改变 lowering 来制造 FOREACH 节点。

**是否符合 NPU 设计：** extern barrier 后消费者深度加 1 与现有 NPU scheduler 设计一致；设备证据已确认适用。
正式 SKIP 来自测试禁用，不能解释成 mm/foreach 不支持。

**验证与结论边界：** 原 mm 与两个 10x10 消费者输入定向执行为 PASS_DIAGNOSTIC，记录 mm=0/0、消费者=1/1、实际 buf0 依赖及 FX eager 数值一致。
正式测试文件未改，结果仍 SKIP；最新 master 未设备复测。
与 S16 同样不宣称完整脚本在早期 foreach 启动时已记录哈希。

代码依据：[E10][E10]、[主机重放][host-results]、[非空节点与原断言][foreach-report]、[设备诊断][device-findings]。

### 当前 PASS 的验证口径与前提

| 检查内容 | NPU 当前设计与验证要求 | 源码依据 |
| --- | --- | --- |
| persistent reduction size hint | split-axis 尺寸求和，其余轴取最大；不照搬 GPU 的二次幂 hint | `NPUComboKernel.select_combo_heuristics` |
| 普通 kernel 与 Combo grid | 普通 kernel 使用 `GridNpu`；Combo 使用一维 `SequentialComboKernelGrid`。动态尺寸用例从真实 launcher 读取 grid，排除 autotune 启动，检查两组尺寸的 X grid 不同及 Y/Z 为 1，并保留数值检查 | `CompiledKernel.launch_metadata`、`SIMDDispatch`、`SequentialComboKernelGrid` |
| mixed-order 数量 | NPU 关闭 mix_order_reduction 纵向合并；行归约、列归约、消费者及 y/z Combo 共 4 个定义，并检查仅有一个双成员 Combo | `torch_npu/_inductor/config.py` 显式设置 `mix_order_reduction=False` |
| scalar broadcast 数量 | `(x0,r1)` 与 `(r0)` 轴布局使用不同分块接口，生成 2 个普通 kernel；检查数量、两类实际轴布局和数值 | `NPUComboKernel.horizontal_partition` 按 kernel 类型、模式和轴标记细分分区 |
| bucketize 数量 | bucketize 调用 ATen fallback；其余两个逐点成员合为 1 个 Triton Combo。检查 fallback 调用及双成员结构，不能把 Triton 定义数当作全部设备 launch 数 | `lowering_fallback_list.py` 注册 `aten.bucketize.Tensor`；实际 wrapper 保留对应调用 |
| max-autotune 与 grouping | 检查完整共享配置有多个实际候选、执行计时、选出赢家并保留一个 launcher；检查成员数量及输出数值。NPU 不要求 GPU 逐成员分组日志 | `_triton_config_npu_index_combo`、`ComboTileGenerator`、`NPUCachingAutotuner.autotune_to_one_config` |
| persistent RBLOCK 上限 | 不带 HIP 专用的 `max_persistent_rblock=1024`，NPU 按自身资源模型筛选共享候选 | NPU `jit_line` 与共享 tiling 候选生成 |
| store mask | 保留原输入数值检查；额外构造行数为 257/513 的兼容动态 mean，检查一个双成员 Combo、各分支边界和两个 store 的 `x0_mask` | 正式用例 FileCheck/JUnit；`compatible-combo-output.py` 为兼容输入独立探测证据 |

调优检查只在用例内临时包装 `AsyncCompile.wait` 取得真实对象，并调用原函数；
不替换候选、计时或赢家。关闭缓存并将编译线程设为 1 是为了确保本次确实执行待检查路径，退出即恢复。
NPU 的共享 tiling 通过不代表 GPU 的独立逐成员分块或逐成员 coordesc 已实现。
普通 GPU 分支的原断言保留。
动态 grid 的 hook 只记录实际参数并调用原函数，不改 grid 或调优结果。

### store mask 的现存诊断问题

共用模板默认启用 `test_configs.runtime_triton_dtype_assert`。
独立探测日志显示，NPU 对动态整数 `ks0`（Triton signature 为 `i64`）生成了以下断言：

```python
tmp4 = ks0
tl.static_assert(tmp4.dtype == tl.float32)
```

整数在转换为 float32 之前即被要求具有 float32 dtype，
TTIR 检查失败后回退为两个普通 kernel，无法满足本例的真实 Combo 数量断言。
直接证据见 [TTIR
失败片段](../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/dtype-debug-ttir-failure.txt)。
这是仍存在的独立诊断代码生成问题，未作为生产代码修复，也不额外计入本轮源方法 FAIL 数量。

新增 mask 场景仅在用例作用域内关闭该调试项，退出恢复。
随后实际验证一个双成员 Combo、两个分支各自的边界 mask、全部 store mask 和 eager/compiled 数值。
因此该 PASS 有明确的调试配置前提，不代表 dtype 诊断能力已修复。
TMA/PDL 的硬件协议来源保存在
[官方资料记录](../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/cuda-protocol-references.json)。

[E1]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/shared-block-abi.txt

[E2]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/combo-metadata.txt

[E3]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/native-tile-search.txt

[E4]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/cse-statistics.txt

[E5]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/graph-optimize-mem.txt

[E6]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/provenance-chain.txt

[E7]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/peak-memory-gate.txt

[E8]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/ftz-boundary.txt

[E9]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/platform-test-gates.txt

[E10]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/scheduler-input-distance.txt

[E11]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/community-combo-contract.txt

[E12]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/bandwidth-and-flop.txt

[E13]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/wrn-grouping-and-autotune-memory.txt

[E14]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/ftz-compiler-and-math-library.txt

[host-results]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/host-probe-results.json

[hint-output]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/reduction-hint-output.py

[rms-output]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/rms-norm-output.py

[benchmark-output]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/benchmark-output.py

[ftz-report]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/ftz-result.json

[ftz-direct]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/ftz-direct-result.json

[wrn-fresh]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/wrn-fresh-result.json

[wrn-groups]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/wrn-groups-result.json

[wrn-allocations]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/wrn-peak-allocations.json

[foreach-report]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/foreach-result.json

[device-findings]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/device-findings.json

[device-manifest]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/device-evidence-manifest.json

[source-index]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/source-index.json

[case-analysis]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/master-analysis/case-analysis.json

[jobs]: ../../../../artifacts/inductor-combo-tests-native-v2.13-npu-contracts-20261112/jobs.json

<!-- combo-section:analysis213:end -->
