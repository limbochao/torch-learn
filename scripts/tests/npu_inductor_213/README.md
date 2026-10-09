# PTA 2.13 NPU Inductor 分批验证

最新的剩余 100 项全量评审、修复补丁和复测结论见 [剩余用例修复报告](remaining_repairs.md)。
该报告逐项标明必要性、负责模块、通过对应修改和未修原因；下文保留原始批次统计。

2026-10-08 清单共 247 行，保留用户提供的顺序及重名项。
14 个批次已全部完成执行和分析，累计表没有待测试或待分析行。
最终为 **142 通过、87 失败、14 跳过、2 项部分通过且含预期失败、2 项预期失败**。
参数化展开后为 283 个有效实例：162 通过、99 失败、16 跳过、6 xfail。
初始中断、独立补测和机制探针保留为证据，不重复计入有效实例。

PR #47384 指定迁移文件的全量重测也已完成：130 通过、99 失败、14 跳过，另有 4 项含预期失败。
相对上述基线为 5 项转为通过、17 项回归，详见 [PR 重测报告](pr47384_retest.md)。
全部 247 项已补充测试目的、NPU 覆盖建议及源码引用；原基线统计保持不变。

完整 PR 加回原 Triton driver patch 的复测已完成：147 通过、82 失败、14 跳过，另有 4 项含预期失败。
17 项回归全部恢复；14 项适配复测另有 1 项转为通过，本轮未发现新增功能回归。
详见 [driver patch 影响报告](driver_patch_retest.md)，其中列明导入顺序遗漏及未测量的性能影响。

142 个通过项包括 3 项显式 CPU 测试，以及其它主机端、负向和测试适配路径；
它们不等于 142 项 NPU 功能全部得到正向验证，实际覆盖范围逐行注明。
统计见 [progress.json](../../../artifacts/npu-inductor-213-batches-20261008/progress.json)，
完整性检查见 [validation-summary.json](../../../artifacts/npu-inductor-213-batches-20261008/validation-summary.json)。

- [最新总表：各阶段结果及通过对应修改](../../../artifacts/npu-inductor-pr47384-restored-driver-20261009/results.csv)
- [原始基线累计 CSV](../../../artifacts/npu-inductor-213-batches-20261008/results.csv)
- [PR #47384 全量重测与基线对比](pr47384_retest.md)
- [PR #47384 全部提交同步与驱动问题复核](full_pr_review.md)
- [PR 加回原 Triton driver patch 的全量影响验证](driver_patch_retest.md)
- [247 项测试目的与 NPU 覆盖建议](../../../artifacts/npu-inductor-pr47384-20261008/test-objectives.csv)
- [101 个失败及跳过用例的简单修复评审](repair_plan.md)
- [逐用例修复建议 CSV](../../../artifacts/npu-inductor-213-batches-20261008/repair-triage.csv)
- [14 个跳过项的适配建议 CSV](../../../artifacts/npu-inductor-213-batches-20261008/repair-skip-triage.csv)
- [跳过项临时环境验证](skip_validation.md)
- [跳过项临时验证 CSV](../../../artifacts/npu-inductor-213-batches-20261008/skip-validation.csv)
- [第 1 批 CSV](../../../artifacts/npu-inductor-213-batches-20261008/batch-01.csv)
- [本轮完整清单](cases_20261008.tsv)
- [第 1 批精确 nodeid](../../../artifacts/npu-inductor-213-batches-20261008/batch-01-nodeids.json)
- [原始阶段报告（去敏）](../../../artifacts/npu-inductor-213-batches-20261008/batch-01-results.json)
- [测试及安装源码证据](../../../artifacts/npu-inductor-213-batches-20261008/batch-01-source-evidence.json)
- [接口与 libdevice 检查](../../../artifacts/npu-inductor-213-batches-20261008/batch-01-capability-probe.json)
- [launcher 调用跟踪](../../../artifacts/npu-inductor-213-batches-20261008/batch-01-launcher-trace.json)

CSV 使用 UTF-8 BOM，便于 Excel 读取中文。每行包含结果、失败阶段、归因、NPU 机制关系、
能否修复、修复方向、未修复原因、覆盖边界和证据引用。
JSON 引用中 `#` 后的字符串是 nodeid 或 `sources[].id`，用于在文件内定位，不是网页锚点。

最新总表保留基线、PR 原样、PR 加回 driver patch 三阶段状态，并逐行给出“通过变化类型”、
“通过对应修改”和补丁证据。基线已通过项、PR benchmark 修改后通过的 5 项、driver patch
恢复的 17 项分别标注，避免把全部通过都归因于本轮补丁。
14 项适配复测也列在对应原序号行，单独记录测试条件修改、HF32 设置、适配结果和日志；
适配版通过不覆盖原用例的跳过状态，也不重复计入原用例通过数。

各批次均已写入独立 CSV，按清单行统计如下：

| 批次 | 清单行数 | 结果 | 参数化实例数 |
|---|---:|---|---:|
| [第 01 批](../../../artifacts/npu-inductor-213-batches-20261008/batch-01.csv) | 25 | 19 失败；6 通过 | 25 |
| [第 02 批](../../../artifacts/npu-inductor-213-batches-20261008/batch-02.csv) | 22 | 7 通过；12 失败；3 跳过 | 37 |
| [第 03 批](../../../artifacts/npu-inductor-213-batches-20261008/batch-03.csv) | 16 | 8 失败；1 跳过；7 通过 | 19 |
| [第 04 批](../../../artifacts/npu-inductor-213-batches-20261008/batch-04.csv) | 20 | 9 通过；7 失败；2 部分通过/含预期失败；2 预期失败 | 29 |
| [第 05 批](../../../artifacts/npu-inductor-213-batches-20261008/batch-05.csv) | 14 | 3 通过；10 失败；1 跳过 | 16 |
| [第 06 批](../../../artifacts/npu-inductor-213-batches-20261008/batch-06.csv) | 20 | 20 通过 | 20 |
| [第 07 批](../../../artifacts/npu-inductor-213-batches-20261008/batch-07.csv) | 20 | 13 通过；7 失败 | 20 |
| [第 08 批](../../../artifacts/npu-inductor-213-batches-20261008/batch-08.csv) | 20 | 19 通过；1 失败 | 20 |
| [第 09 批](../../../artifacts/npu-inductor-213-batches-20261008/batch-09.csv) | 20 | 16 通过；4 跳过 | 20 |
| [第 10 批](../../../artifacts/npu-inductor-213-batches-20261008/batch-10.csv) | 20 | 10 通过；9 失败；1 跳过 | 20 |
| [第 11 批](../../../artifacts/npu-inductor-213-batches-20261008/batch-11.csv) | 20 | 14 通过；3 跳过；3 失败 | 20 |
| [第 12 批](../../../artifacts/npu-inductor-213-batches-20261008/batch-12.csv) | 20 | 16 通过；4 失败 | 27 |
| [第 13 批](../../../artifacts/npu-inductor-213-batches-20261008/batch-13.csv) | 2 | 2 通过 | 2 |
| [第 14 批](../../../artifacts/npu-inductor-213-batches-20261008/batch-14.csv) | 8 | 7 失败；1 跳过 | 8 |

## 第 1 批结论

全部 25 项都有 setup/call/teardown 报告，19 项均在 call 阶段失败。
不把 skip、无 call、收集失败或 xfail 计为通过；重名方法按测试类区分。
本批没有修改上游测试、断言或已安装的后端代码，也没有将历史结果当作本次结果。

19 个失败的首错分布：

| 数量 | 原因 | 处理方向 |
|---:|---|---|
| 13 | 测试读取 CUDA `cubin`，目标 kernel 只有 NPU 二进制 | NPU loader/参数接口开发，加测试适配 |
| 1 | `uint64` 输入初始化时 `aclnnInplaceZero` 拒绝该 dtype | 算子类型支持；首错不能推断 Triton uint64 参数能力 |
| 1 | 通用 `libdevice.isinf` 占位函数没有映射到 CANN | 待修复的 backend 模块映射/测试集成 bug |
| 1 | strict static 模式没有抛出测试要求的异常 | 接入 NPU static 配置契约 |
| 1 | CANN `isinf` 的类型映射没有 float64 | 保留 fp64 语义的实现或 lowering |
| 1 | CPU torch 构建不导出 `_FastCudaLauncher` | CUDA 专用接口，需 NPU 等价实现和测试 |
| 1 | 数值正确，但未构造 fast launcher | NPU fast 路径接入及正向覆盖 |

`cubin` 和 `npubin` 分别是 CUDA 和 NPU 的编译二进制；ABI 是运行时对参数、句柄等的约定。
当前 static 分派只包含 CUDA/HIP/XPU，因此把字典键改成 `npubin` 仍不能完成适配。
两个 shared-memory 测试的首错也是缺少 `cubin`，并未触发资源耗尽；不能据此写成 NPU UB 溢出。
UB 是 NPU 的片上缓冲区，不能直接套用 CUDA 的 shared-memory 参数及错误预期。

`test_implied_constant` 的实际绑定为 `triton.language.extra.libdevice.isinf`，源码函数体是 `...`，
返回 `None`；Ascend `get_module_map()` 返回空字典，而 CANN 库已有 float32 实现。
另外，原测试输入只有 4 个元素，却传入 `r0_numel=128`，mask 仅约束索引小于 128。
这构成修复编译后需要处理的越界风险；本次尚未运行到该阶段，不能说已经发生越界。

`test_any` 则进入了 CANN `isinf`，因 float64 映射缺失失败。
两种 `isinf` 故障不应合并，也不能以换成 float32 作为原 float64 用例的修复。

6 个通过项中，1 项只测试 FakeKernelOwner 的 Python 生命周期，5 项完成 NPU 数值或禁用路径断言。
原测试没有充分证明 NPU static/fast launcher 被启用。补充跟踪的 4 个编译用例仍为 3 通过、1 失败，
其调用证据单独保存，不重复计入 25 项统计。
“可修复”表示有依据的修复方向，均不代表已实现或已经复测通过。

## 运行方式

目标为 Docker 内的 Inductor 测试目录，使用 `pta_213` Python。
本轮实测版本：torch `2.13.0+cpu`、torch_npu `2.13.0+git5fd5ddf`、Triton `3.6.0`，
device 为 Ascend950PR，`GPU_TYPE=npu`，driver 为 `NPUDriver`，NPU backend 为 `default`。
测试仓提交为 `cf30153c4c131c8164ee7798e5022d810682e2cb`。
安装源码可能包含环境已有修改；具体文件 SHA-256 随源码证据记录，不以 wheel 版本代替源码标识。

先进入容器工作目录并激活上述 Python 环境，将 `LEARN_REPO` 设为容器内 torch-learn 路径。
示例只运行一个完整 nodeid：

```bash
test -f /.dockerenv
mkdir -p npu_batch_20261008
export TORCH_TRANSFER_TO_NPU=1
export TORCH_NPU_DEVICE_CAPABILITY=8.0
export TMPDIR="$PWD/npu_batch_20261008"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/cache"
python "$LEARN_REPO/scripts/tests/npu_inductor_213/run_batch.py" \
  --output "$TMPDIR/result.json" \
  test_static_triton_launcher.py::TestStaticTritonLauncher::test_basic
```

`8.0` 仅兼容社区测试的 CUDA capability 比较，不代表 NPU 硬件能力。
所有缓存、临时输出均位于容器工作目录内。
本轮通过标准输入运行同一执行器，工作目录与上述方式一致。

全批运行时，可用 Python 将 `batch-01-nodeids.json` 的数组作为参数列表传给 `run_batch.py`。
`--collect-only` 只收集；`--trace-launchers` 用透传包装记录 launcher 调用，保持参数、返回值和异常。
补充跟踪选择静态编译的 `test_basic_compile`、`test_static_launch_user_defined_triton_kernels`，
以及 fast 编译类的 `test_basic_compile`、`test_disable_fast_launcher`。
执行器保存请求参数与 pytest 实际 nodeid 的映射，处理仓库 rootdir 导致的 `test/inductor/` 前缀。

旧的 `run_cases.py` 和 `cases.tsv` 对应 2026-09-29 的动态 shape 批次；本轮没有修改它们。

## 批次范围与统计口径

第 2–5 批覆盖 lookup table、utils/inplacing/annotations、alignment/benchmark、padding/extension/subprocess。
第 6–13 批覆盖基础算术、视图、归约、矩阵、卷积、设备迁移；第 14 批覆盖 unbacked/dynamic shapes。
每批在完成后生成 `batch-NN.csv`，累计表始终保留完整 247 行及原始顺序。

同名函数在不同测试类出现时保留原清单中的独立行。参数化用例展开后的实例数在“参数化统计”列中给出，
不能把实例数与清单行数混用。测试输入、dtype、子分支及断言均由源码证据定位。
`test_conv_bn_eval` 是嵌套 helper，通过所属 `test_basic` 执行；首错后的循环组合不视为已覆盖。

普通 `test_torchinductor.py` 的 NPU `common` helper 先比较原始输入，再默认检查 float32 转 float16 的输入；
设置 `check_lowp=False` 的方法除外。默认不检查梯度，`requires_grad=True` 本身不代表已经验证 backward。
“原用例通过”只代表该方法现有断言通过；空张量、纯 CPU、负向检查及仅检查生成代码的覆盖范围需单独看待。

`--case-names` 将 `file.py::source_method` 解析为对应设备类及所有参数化实例；明确 CPU 的方法保留 CPU 范围。
`--resume` 保留有 teardown 的完整实例，仅重跑未完成项，同时保存原环境和中断阶段记录。
必须先确认原测试进程已退出，再使用此选项。
`--precompile-workers 4` 只限制 NPU 编译候选的并行工作数，不修改候选、kernel、dtype 或测试断言。
本轮发生进程退出 137，容器 96 GiB 内存上限及 OOM 计数见 `interruption-resource-evidence.json`；
补测配置、原始中断与合并来源均保留在 JSON 中。并行批次使用不同默认 NPU，多卡测试保持全部设备可见。
`--device-index 1` 只设置默认 NPU 索引，保留其它卡；实际设备索引见各报告 environment。
一次设备可见性重排导致的收集前初始化错误见 `execution-notes.json`，没有计为用例失败。

## 已完成批次中的共同问题

- Lookup table：设备 key 仅识别 CUDA，NPU 正向查表路径没有建立；若只返回空结果，负向用例通过并不证明命中能力。
- Inplacing：部分原输入使用 CPU 全局常量，先在 eager 的 index_put 失败；需要保留别名语义做设备适配。
- Alignment：右移广播分配形状、custom-op 的 PrivateUse1 注册及 stride 检查是不同问题。
- Benchmark/padding：一类失败来自 CUDA/NPU 别名被重复计数；另一类来自 deterministic 模式下未满足基准白名单约定。
- Extension/subprocess：分别发现 worker 模块搜索路径、PrivateUse1 注册冲突、子进程 lowering 恢复类型契约问题。
  自定义 UDTK 独立复测还出现 AIVEC 地址错误，根因尚未锁定，详见该行限制。
- Round：符号标量在 NPU dtype pass 被当作 Tensor；float64 nearbyint 是另一条 dtype 支持问题。
- BMM：小 K 数值已对齐而代码策略不符，与尚未完成编译的 vmap(dot) 基准错误分开记录。
- Gather sparse_grad：AOT 联合图经 NPU gather-backward Meta 构造稀疏 Tensor 后进入设备复制分解递归。
- 大张量：4 项在 `largeTensorTest` 的内存检查中因不识别 `npu` 而跳过，不能归因为实测内存不足。
  无该装饰器的大广播归约已完成原断言，不能将它的通过扩展到被跳过的大存储、stride 和 offset 场景。
- dtype/布局：float64 的算子类型拒绝、Triton libdevice 类型映射缺失、HIR load 验证失败分别记录；
  `channels_last` 的公开 stride 契约也不能由 NPU 私有 format 自动替代。
- Dynamic/unbacked：分别发现空生成代码列表的测试假设、符号整数运算引入 float64、CUDA allocator 接口、
  benchmark 设备别名，以及构造器包装与 Dynamo 对象规则的兼容问题，不能合并为“不支持动态 shape”。

“待修复 bug”“支持缺口”和“测试适配”分别表示实际实现错误、尚未具备被测功能，以及原测试依赖其它后端的接口/约定。
“可修复”是源码支持的方向判断；本轮未修改生产实现，不等于已经提供修复，也不把绕开断言作为修复。
对尚未锁定的根因，CSV 明确保留待定位内容和缺少的验证，而不是填入推测结论。

## 汇总复现

`artifacts/npu-inductor-213-batches-20261008/plan.json` 保存第 2–14 批的行号与方法名，
第 1 批使用单独的 `batch-01-nodeids.json`。按 `plan.json` 的某一批 `cases` 数组传给
`run_batch.py --case-names` 即可复跑该批；输出 JSON 再由汇总脚本合并。

从 torch-learn 根目录重新生成已归档的分批 CSV 和累计表：

```bash
python scripts/tests/npu_inductor_213/summarize_batches.py \
  --raw-dir artifacts/npu-inductor-213-batches-20261008 \
  --output-dir artifacts/npu-inductor-213-batches-20261008
```

`analysis.json` 是逐用例的人工归因，`coverage.json` 是通过项的覆盖说明；
汇总器只对有完成 summary 的报告更新状态，不把收集成功当作执行成功。
如果复跑环境或实现有变化，应重新审核上述归因，不能机械复用旧结论。

多卡覆盖单独核对默认设备：`test_multi_gpu_device` 的目标固定为 NPU:1，
如果默认设备本身为 1，原测试通过并不能证明跨卡复制。因此该项另以默认 NPU:0、全部卡可见复测，
以补测报告作为有效结果，并保留批次原始报告。实际从 0→1 在 eager 的 peer-access 启用阶段失败，
返回 `207001`；关闭 transfer 包装、没有编译任务的 4 元素独立复制仍复现。
探针查询两卡空闲内存均超过 121 GiB，且双向 peer 能力均为 True，不能简单归因为卡内存已满。
问题已限定在 P2P 运行时/驱动路径，具体失败资源尚未锁定；没有完成跨卡 compiled 数值验证。
证据见 `batch-12-multigpu-default-zero.json` 和 `batch-12-p2p-probe.json`。

`test_round` 独立补测以 4 个 NPU 预编译 worker 完成原 float32/float16 断言，主表采用该通过结果。
初轮 worker 退出记录保留于 `batch-07-initial-results.json`，有效复测为 `batch-07-round-isolated.json`。
初轮的内存压力与 OOM 计数不能唯一证明该 worker 被 OOM killer 终止。

第 14 批的 8 个方法同时生成 CUDA/NPU 类。主表优先原生 NPU 实例；额外 CUDA 的 7 个
`_cuda_setStream` 失败和 1 个 FBCODE 跳过保存在 `batch-14-initial-results.json`。
执行器已固定该选择规则；此前各批经检查不存在同方法同时选中 CUDA/NPU 的情况。

常量折叠探针的 6 个组合均为零 kernel、返回原输入且两种 shape 数值相等，但原测试仍因空代码列表索引失败。
unbacked 构造器探针用 `backend="eager"` 隔离 Dynamo：原包装复现数据依赖整数特化错误；
直接使用解包别名仍被标为 skipped；仅在独立进程中恢复 `torch.tensor` 全局绑定并刷新对象规则后捕获成功。
这些均为机制证据，不作为原完整 Inductor 用例的修复通过结果。
