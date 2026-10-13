# PyTorch 2.13 NPU Inductor 失败与 SKIP 重试

本轮接续 [331 行原始复测](../npu-inductor-213-recheck-20261010/README.md)，
执行要求为“失败的全部重试一次，SKIP 检查能否解除后重试”。
原测试文件保持不变，使用同一 PyTorch 2.13 / torch_npu 2.13 环境和输入清单。

共重试 197 行，执行 228 次 pytest 调用。相对上一轮新增 98 行 PASS。
合并保留的上一轮结果后，331 行为 **225 PASS、89 FAIL、9 SKIP、3 XFAIL、2 MIXED、2 TIMEOUT、1 NOT_COLLECTED**。

| 上轮结果 | 本轮结果 |
| --- | --- |
| 38 FAIL | 38 FAIL |
| 58 PROCESS_ERROR | 49 PASS、7 FAIL、2 SKIP |
| 3 TIMEOUT | 1 PASS、2 TIMEOUT |
| 102 SKIP | 48 PASS、44 FAIL、7 SKIP、1 XFAIL、2 MIXED |
| 2 XFAIL | 2 XFAIL |

- [按原始顺序排列的结果 CSV](results.csv)：同时显示上轮结果、本次处理、最新结果、调用次数及原因。
- [汇总 JSON](summary.json)：区分输入行、pytest 节点和各批次结果。
- [各次尝试的节点证据](node-results.json)：包含 pytest 收集、setup/call/teardown、退出码和门控调整记录。
- [102 行 SKIP 逐项评估](skip-audit.json)：保留原跳过原因、解除方式及最终结果。
- [运行脚本与复现方法](../../scripts/tests/npu_inductor_213_retry/README.md)。

## 范围与口径

原 38 行 FAIL、58 行 PROCESS_ERROR、3 行 TIMEOUT 全部重试；原 2 行 XFAIL 也重新执行，保留预期失败标记。
102 行 SKIP 全部检查，其中 96 行尝试解除门控，6 行因 CPU 专属路径或 CUTLASS 依赖保留跳过。

2026-10-13 在 `NPU_A5` 的 `pta_213_case` 环境、标准路径 `v2.13.0/inductor_npu` 复测表中原先仍标记 SKIP 的 6 行（163、165、166、167、197、292）已随 adapter 提交 `ed65776` 修复并全部通过；对应证据见 [`adapter_fix_20261013-summary.json`](adapter_fix_20261013-summary.json) 和 `adapter_fix_20261013/`。98 行重跑结果现为 98 PASS、0 SKIP。
原 127 行 PASS 和 1 行 NOT_COLLECTED 沿用上一轮证据，CSV 明确标记 `PREVIOUS_RESULT`。

每行使用一个独立 pytest 进程，参数化用例仍可能对应多个节点。
PASS 要求该行所有已选节点通过；任何测试断言失败记为 FAIL；setup/teardown 失败记为 ERROR。
同一行包含 PASS 与 XFAIL 等不同结果且没有 FAIL/ERROR 时记为 MIXED，节点计数分别展示。
单行 900 秒超时记为 TIMEOUT，即使已有部分节点完成也不计作整行通过。
最后结果使用该行最后一次尝试，先前尝试仍保存在节点证据中。

## 收集错误与跳过条件

上一轮 58 行 PROCESS_ERROR 的日志均为指定节点不存在。
重新收集确认实际类名为 `TestInductorDynamicPRIVATEUSE1`，原计划使用了 `TestInductorDynamicNPU`。
映射后保留所有原测试名和参数化节点，没有以其他测试替代。
该批次随后遇到 `setUp` 的 `HAS_GPU=False` 门控，插件只在测试模块中解除该门控。

环境中 NPU 可用且安装了 Triton-Ascend，但社区测试辅助代码的 GPU 能力检测为假。
插件按 reason 白名单解除通用 GPU/CUDA/Triton 门控，不修改测试输入、断言、编译器或生成 kernel。
这类通过表示原测试在指定 NPU 迁移环境中通过，不能仅凭门控解除认定 CUDA 专属机制获得支持。
例如静态 launcher 的部分测试实际要求 `kernel.asm['cubin']`，解除入口跳过后会真实报告缺失产物。

最初 30 行仍被 `unittest.case.skip` 保留，补齐该入口后仅补跑这些尚未进入测试逻辑的行。
动态形状第 292 行的函数体不依赖 FBCODE 专用 API，进一步只解除该方法的 FBCODE 装饰器后通过。
原 FAIL 和 TIMEOUT 均只重试一次。

`test_gather1` 从超时转为 PASS，用时约 727 秒；`test_views1`、`test_views2` 仍达到 900 秒上限。
原 SKIP 的 7 行最终仍跳过，分别是第 50、155、156、169、201、202、204 行。
新增的两行动态形状 SKIP 为第 314、315 行，源码明确标记 `onlyCPU`。
第 76、77 行为 MIXED，各含 1 PASS 和 1 XFAIL；第 79 行为 XFAIL。

大张量测试通过 `torch.npu.mem_get_info` 检查真实空闲内存，不直接忽略内存限制。
第 163、165、166、167 行分别需要 4、4、3、2 GiB，检查时各卡空闲内存均超过 122 GiB，四行全部通过。
第 201 行开启 `PYTORCH_TEST_WITH_SLOW=1` 后，仍因函数体明确的 CPU-only 限制跳过。
保留该限制，不改变用例语义。

门控与限制对应的源码、行号及哈希见 [源码证据](gate-source-evidence.json)。
逐行实际门控调整、环境变量和错误见 CSV 与节点结果。

## 环境和复核

基础环境为 `TORCH_TRANSFER_TO_NPU=1`、`TORCH_NPU_DEVICE_CAPABILITY=8.0`，
`TORCHINDUCTOR_COMPILE_THREADS`、`TORCHNPU_PRECOMPILE_THREADS`、`OMP_NUM_THREADS` 均为 2。
每行使用独立 Inductor 缓存，额外环境变量按行记录。
原跨设备失败项第 230 行使用全设备可见性；源码实际访问设备 0、1。

[运行前状态](preflight.json)、[运行后状态](postflight.json) 和 [manifest](manifest.json)
确认 176 个测试文件哈希与上一轮及本轮运行前一致，3 个已安装测试辅助模块也未变化。
所有 driver 已结束，228 次调用均有终态记录，原始 331 行顺序及清单哈希一致。
226 次非超时调用的 JUnit XML 与 JSON 节点计数逐项一致，检查记录见 [validation.json](validation.json)。
测试仓库没有已跟踪文件变化；运行后留下 `export_only_prof_dir/`、`extra-info/` 两个未跟踪产物目录。
归档证据已去除个人目录及进程 ID，原始日志和 JUnit XML 留在测试工作目录。
