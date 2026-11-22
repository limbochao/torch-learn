# Cooperative reduction：固定 8 Vector 核预算的完整分片实验

2026-11-23（北京时间）在 NPU_A5 Docker 的 `pta_213_case` 中运行，使用本轮明确授权的物理卡 5。
工作树为 `/home/l30023782/PTA_213/torch-learn-core-limit-20261123`；未修改安装包或重置设备。
正文见[同步、性能与限核边界](../../docs/requirements-design/cooperative-reduction/barrier-comparison.md)。

## 数据范围

主矩阵为 `matrix-8/output-axis.json`，固定 fp32 `(a+b).sum(-1)`、SIMD、4 warps、BLOCK=1024，
X=1/2/8、R=32768/1048593、S=1/2/4/6/8/12/16/32/48/56。
三种 helper 各 60 点，plain 每个 X/R 只测 S=1，共 186 个唯一配置。

| 实现 | 配置数 | 通过 | 结果错误 | 超时 | 已计时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| plain | 6 | 6 | 0 | 0 | 6 |
| original | 60 | 48 | 0 | 12 | 48 |
| native | 60 | 26 | 30 | 4 | 56 |
| debug | 60 | 26 | 30 | 4 | 56 |

没有 SKIP，全部结果错误点保留 device Event 耗时。超时是实际尝试后未在墙钟时限内完成，
不是按预算跳过；没有 Event 耗时的调用不补成 0。主矩阵每个超预算 helper 点的进程时限为 30 秒。
`smoke-8/` 的 3 个 native 结果已合入主矩阵，不额外增加配置数。
`control-56/` 是同卡 56 Vector 预算对照：X=1/2、两种 R、plain 及 native S=4/8/16，16 点全部通过。
对照只覆盖这三个 S，不代表 56 核全部候选中的最优配置。

`recheck-native-s12/` 与 `recheck-original-x2-s32/` 分别复核 native X=1/S=12、original X=2/S=32，
R 均为 32768，环境预算仍为 8。两者复用主矩阵已编译缓存，将子进程时限改为 120 秒，
均仍停在 `sum_synchronize`，未取得输出或 Event 耗时；保存了首次发射返回后的编译 IR 和 metadata。
主矩阵保留原 30 秒记录，复核不增加 186 个唯一配置数，也未用假定耗时替换超时。

## 预算、映射与软件栈

主矩阵设置 `NPU_DEVICE_LIMIT=4,8`、`ASCEND_RT_VISIBLE_DEVICES=5`，ACL 反查逻辑 0 对应物理 5。
manifest 同时记录整卡 56 Vector、TA 属性 8、runtime device 与当前 stream 限额 8；
56 核对照设置 `28,56`，使用独立编译缓存，没有验证同缓存中切换限核状态。
环境变量、runtime 限额、提交 Grid 和同时驻留物理核数不是同一个量。

本环境 launcher 的 `prepareGrid` 在 AUTO_MAP 打开时提交 `min(logicalBlocks,8)`，
关闭时提交完整 `logicalBlocks`。native/debug 开启 AutoBlockify；original 的 atomic 使其关闭。
因此 original 的超预算通过不能当作与 native 相同的 8 核 program 循环，更不能用耗时差直接计算屏障净成本。
未采集实际核驻留时间线；源代码快照位于 `compiler-source/`。

包版本为 torch `2.13.0+cpu`、torch_npu `2.13.0+git93d7d01`、Triton `3.6.0`、
TA `3.6.0+dev20261112182052`；实际 NPUIR 为 `395c9c5646f9`、LLVM 19.1.7。
smoke/plain 使用 `4e47f34`，native 后续与 debug 使用 `302c530`，original、56 核对照和超时复核使用 `ed5eef8`。
这些版本只改采集、隔离及保存时机，数学算法与 helper 不变。
每点通过 `run_manifest` / `worker_manifest` 保留实际提交、命令、环境和源码 SHA256。

## 精度、计时与产物

每点检查首次 NaN partial/output、计时后的固定输入，以及同地址换三组随机输入；
换输入时保留上一次的 partial/semaphore，输出重新填 NaN。CPU 先做 fp32 加法，再用 fp64 求和，
容限为 atol=0.02、rtol=2e-4。helper 还检查 4 phase、epoch=11、分片工作量不同的严格整数交换，
读者结束后另加屏障防止下一 phase 覆盖。PASS 要求全部检查通过。
仅测试 fp32 sum，换数值不等于动态 shape；不表示 mean/amax 或 Inductor 集成已通过限核验收。

计时为 NPUGraph + device Event，默认 batch=32、replays=10、7 个样本取中位数。
慢点以 `--max-sample-ms 500` 缩减重复数，保留实际参数；首次、换输入和整数检查不计入 sum 耗时。
本组按 helper 分阶段执行，阶段间及隔离子进程独立分配地址，沿用相同随机种子与算法，
不作为三种 helper 的同地址对照。
时限包括初始化、缓存读取/编译、执行和检查，不等于 kernel 设备时间。
被终止 worker 的 `active-case.json` 可能仍为 RUNNING，最终以主矩阵状态为准。

`ir/` 与 `cases/*/ir/` 是实际编译对象的原始 TTIR、adapter 和 metadata。
`final-*/` 只对已有 adapter/metadata 重新编译，保留最终 MLIR、LLVM IR 和 manifest，**没有 launch**。
临时全量编译输出、Triton 缓存和二进制不归档；设备快照见 `device-*.txt`。
结束时物理卡 5 为 Health OK、利用率 0%，没有本任务设备进程；未操作其他卡或他人进程。

## 复现

先激活 `pta_213_case`，按工作区规定核对时段、占用与授权，再在 Docker 工作树执行。
卡号只描述本轮实验，不构成其他时段的选卡授权。脚本通用参数和使用方法见
[测试脚本说明](../../scripts/tests/cooperative_reductions/README.md#限核后的分片与精度检查)。

先对 X=1/2/8、两种 R、S=1 运行 `--only-kind plain`；再在相同输出目录分别运行
native、debug、original，使用全部十种 S、`--resume`、`--allow-over-core`、
`--time-wrong-results`、`--accuracy-repeats 3`、`--save-kernels`、
`--max-sample-ms 500`、`--case-timeout 30`。各阶段完整命令位于对应 manifest 的 `argv`。
含错误或超时的矩阵以非零退出码结束，JSON 仍逐点写出，不应取消后续 helper。
56 核对照使用另一输出目录、缓存和 `NPU_DEVICE_LIMIT=28,56`。

绘图使用 `plot_output_axis_cases.py`，输入 `matrix-8/output-axis.json`，
输出到 `docs/requirements-design/cooperative-reduction/assets/barrier-comparison/core-limit-8`；
标题为 `A5 cooperative reduction: Vector core budget 8 (physical 56)`。
`verification-summary.json` 记录矩阵、源码、CSV 和文档表格核对，`sha256.json` 为归档文件摘要索引。
原始 generated code 和设备日志保留原格式，不为普通空白检查改写。
