# Cooperative barrier 对比：原始证据

报告见[三种 x_grid_barrier 的 A5 与 RTX 4090 验证](../../docs/requirements-design/cooperative-reduction/barrier-comparison.md)。
所有设备运行均在指定容器内进行；NPU 为 NPU_A5 / pta_213_case / 物理卡 2，
GPU 为 GPU_4090 / pytorch-dev-gpu67 / 物理卡 6。

## 最终性能数据

| 数据 | 内容 | 实际提交 |
| --- | --- | --- |
| [NPU bench-shared][npu-bench] | 156 点，三种 helper，同一点复用相同设备地址，全部校验 PASS | `e3522329` |
| [CUDA bench-shared](cuda/bench-shared/bench.json) | 54 点，原始 helper，全部校验 PASS | 同上 |
| [NPU manifest](npu/bench-shared/manifest.json) | 实际版本、设备、脚本/helper 摘要、编译选项、命令 | 同上 |
| [CUDA manifest](cuda/bench-shared/manifest.json) | 同上，含 UUID 和 CUDA 版本 | 同上 |

每点包含 7 个 device Event 样本、中位数、最小/最大值、误差及编译 metadata。
最终性能点的 `same_buffers_across_helpers=True`，实际 `buffer_addresses` 用于检查同地址控制。
图和 CSV 位于报告旁的 `assets/barrier-comparison/`，使用 `bench-shared` 绘制。

`npu/bench/` 与 `cuda/bench/` 是初次各 helper 单独分配输入的控制记录，提交为 `90dc3131`。
它们用于说明大输入的分配/缓存影响，不能与最终同地址数据混为一组。

## 基础精度与重复同步

| 数据 | 结果 | 实际提交 |
| --- | --- | --- |
| [NPU correctness][npu-check] | 三种 helper 各 40 组，共 120 组 PASS；整数严格相等 | `e7826ac9` |
| [CUDA correctness](cuda/check/correctness.json) | 原始 helper 40 组 PASS | `90dc3131ca6d0de95f7fe9c128100ef321b385d9` |

13 个整数 Grid 各用 3 个 epoch，每次 16 phase，phase 内有两次同步。
27 个浮点组合覆盖三类输入 dtype、sum/mean/amax 以及三种尾块/空片/多行配置。
浮点 prologue、累加、输出均为 fp32，CPU 参考为 fp32 加法后 fp64 归约。

## 并发配置结果索引

NPU 下列 11 组全部 PASS，共保留 1024 次浮点归约和 1024 次变化 epoch 的整数交换结果。
每条流连续提交 32 轮，workspace/semaphore 按流独立持有，所有输出在同步后逐一检查。
实际提交为 `e352232967947b1b11a64dbfeb7c8128816e78b3`。

| Helper | 1 进程 × 2 流 | 2 进程 × 1 流 | 2 进程 × 2 流 | 2 进程 × 2 流，S=56 |
| --- | --- | --- | --- | --- |
| original | [PASS][o12] | [PASS][o21] | [PASS][o22] | 未运行 |
| native | [PASS][n12] | [PASS][n21] | [PASS][n22] | [PASS][n56] |
| debug | [PASS][d12] | [PASS][d21] | [PASS][d22] | [PASS][d56] |

未标注 S 的项均为 S=16、X=1、R=1048593；每个子进程目录都有 manifest 和自身结果。
原始 [batch 退出状态](npu/concurrency-batch.log)与 JSON 结果一致。

CUDA 的[两进程双流](cuda/concurrent-epochs/concurrency.json)也通过了 128 次浮点归约和
128 次整数交换，提交为 `4766a40d9b75f2bc151627ad3edaaed7444ac551`。
本轮检查并发提交，没有采集实际设备执行重叠的时间线。

## 分核诊断

| 配置 | native | debug |
| --- | --- | --- |
| 关闭 AutoBlockify，Grid=(7,4,2)，共 56 program | [PASS][map-n-off] | [PASS][map-d-off] |
| 默认 AutoBlockify，Grid=(112,1,1) | [FAIL：错误结果][map-n-auto] | [FAIL：错误结果][map-d-auto] |

关闭项使用独立编译缓存，metadata 的 `auto_blockify_enabled=False`；提交为 `e3522329`。
超核项为 `True`，提交为 `26f4c341a498cd5b69a5351adada3c3529d33dba`。
两个超核项均正常返回，随后严格整数检查发现 1736/1792 个值错误，进程退出状态为 1；
外部 45 秒期限及 NPU 10 秒算子期限均未触发。该失败被保留，不作为正确性能点。

`mapping-observations.json` 保留首末 phase 的实际值、期望值、错误计数和 metadata。
`ir/` 保留实际执行 kernel 的 adapter、TTIR 和 metadata。
[失败配置最终 MLIR](npu/final-mapping112/module.hivm.opt.mlir)显示步长 56 的 program 循环，
全 Vector 屏障位于循环内部。相关编译导出未 launch 新 kernel。

## 同步 lowering 与设备状态

`npu/final-native/`、`npu/final-debug/` 保存 R=1048576、S=32 的最终 MLIR/LLVM IR 与导出 manifest。
两者的 adapter 来自初次运行，与 `bench-shared/ir/` 中对应 adapter 字节一致；
导出是仅编译操作。`debug` 的最终 LLVM IR 比 `native` 多两条 PIPE_ALL 本核屏障。

- [NPU 最终健康状态](npu/device-after-health.txt)：物理卡 2，Health OK。
- [NPU 最终进程状态](npu/device-after-process.txt)：无运行进程。
- [GPU 最终占用快照](cuda/device-final.csv)：UUID 复核和共享占用信息。

`verification-summary.json` 汇总归档数据的计数及一致性检查；`sha256.json` 为本目录其余文件的摘要。
未归档整个编译缓存、设备二进制或 Python 缓存。

[npu-bench]: npu/bench-shared/bench.json
[npu-check]: npu/check/correctness.json
[o12]: npu/concurrent-original-w1-s2/concurrency.json
[o21]: npu/concurrent-original-w2-s1/concurrency.json
[o22]: npu/concurrent-original-w2-s2/concurrency.json
[n12]: npu/concurrent-native-w1-s2/concurrency.json
[n21]: npu/concurrent-native-w2-s1/concurrency.json
[n22]: npu/concurrent-native-w2-s2/concurrency.json
[n56]: npu/concurrent-native-w2-s2-rsplit56/concurrency.json
[d12]: npu/concurrent-debug-w1-s2/concurrency.json
[d21]: npu/concurrent-debug-w2-s1/concurrency.json
[d22]: npu/concurrent-debug-w2-s2/concurrency.json
[d56]: npu/concurrent-debug-w2-s2-rsplit56/concurrency.json
[map-n-off]: npu/mapping-off-native-7x4x2/mapping.json
[map-d-off]: npu/mapping-off-debug-7x4x2/mapping.json
[map-n-auto]: npu/mapping-auto-native-112/mapping.json
[map-d-auto]: npu/mapping-auto-debug-112/mapping.json
