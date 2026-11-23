# fp32 sum / amax 全形状原始记录

用于[同步与性能报告](../../docs/requirements-design/cooperative-reduction/barrier-comparison.md)的
origin、debug、RTX 4090 三组对照。与实际 Inductor split/cooperative 整图实验使用独立结果，
这里测量的是固定 BLOCK 的手写 kernel。

## 范围与选点

计算为 `(a+b).sum(-1)` / `(a+b).amax(-1)`，输入连续 fp32，seed=213。
X=1/7/56，R=4096 起连续翻倍至 1048576，共 9 档；NPU 每组尝试
RS=1/2/4/8/16/32/48/56，original/debug 各 432 个候选；CUDA 另加 RS=64，共 486 个。
原始 JSON 保留所有候选；文档 CSV 按每组最小的 PASS 中位数选择，共 162 行。
WRONG/ERROR/TIMEOUT 不参加最佳选择；数值错误但能完成计时的候选仍保留耗时。

| 场景 | 候选 | PASS | 校验失败 | 执行失败 | 超时 | 最佳结果行数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A5 origin | 432 | 324 | 0 | 0 | 108 | 54 |
| A5 debug | 432 | 324 | 108 | 0 | 0 | 54 |
| RTX 4090 original | 486 | 414 | 0 | 72 | 0 | 54 |

NPU 的 108 个校验失败点均完成计时，96 点出现浮点结果错误，108 点的严格整数交换均失败；
另外 12 点不能仅凭浮点输出判定正确。NPU 超时均在 origin 的 X=7/56、RS=32/48/56；
debug 校验失败均在相同两个 X 的 RS=16/32/48。
CUDA 启动失败均在 X=56、RS=32/48/56/64，原因为 `too many blocks in cooperative launch`。
超时和启动失败没有完成调用，不填设备耗时。

## 环境与源码

| 项目 | NPU | CUDA |
| --- | --- | --- |
| 位置 | NPU_A5 Docker，`pta_213_case` | GPU_4090 的 `pytorch-dev-gpu67` |
| worktree | `/home/l30023782/PTA_213/torch-learn-matrix-20261123` | `/home/gaojg/dev/torch-learn-matrix-20261123` |
| 设备 | Ascend950PR_9579；X=1 用卡 1，X=7/56 用卡 2；56 Vector | RTX 4090 物理卡 6，128 SM |
| PyTorch | `2.13.0+cpu` | `2.13.0+cu126` |
| torch_npu / TA | `2.13.0+git93d7d01` / `3.6.0+dev20261112182052` | — |
| Triton / CUDA | `3.6.0` / — | `3.7.1` / `12.6` |
| 选卡与限额 | 分别设置可见物理卡 1/2；`NPU_DEVICE_LIMIT=28,56` | UUID `GPU-528f4e79-58c0-e843-bf5e-0b40c2671cd7` |
| 采集脚本提交 | `49a6591` / `e6397d4` | `1f833fb` |
| kernel | SIMD、4 warps、BLOCK=1024 | 4 warps、BLOCK=1024 |

原始输出目录分别为：

- NPU：`/home/l30023782/PTA_213/cooperative-results/matrix-20261123/` 下的 `npu-card1`、`npu-card2-x7` 与 `npu-card2-x56`。
- GPU：`/mnt/raid1/gaojg/data/dev/cooperative-matrix-20261123/cuda`。

汇总时按完整形状选择来源，不混选不同卡的 RS；`source-index.json` 记录逐组原始目录和文件摘要。
汇总 `run.json` 显式记录拼接方式，并内嵌各来源最后一次父进程调用，不能当作单次扫描命令。
各组 manifest 是该组最后一次 worker 的环境；续跑前的 RS 仍保留原始 `worker_pid`、参数与 IR 摘要。
各形状 `manifest.json` 保留完整版本、NPUIR 编译器信息、实际限额、设备属性、命令、时间和源码 SHA256。
original 在两端使用相同源码，SHA256 为
`c9fa8ee90b54adc67a11ba5406b5fe4be55178b974cc26a2bde8c2c9f19f5bb3`，
CUDA 为 PyTorch v2.13.0 提交 `cf30153c4c131c8164ee7798e5022d810682e2cb` 的快照。
debug 为两侧各有一次 `tl.debug_barrier()` 的 `sync_block_all("all_vector",0)`。
两端均显式设置 `launch_cooperative_grid=True`；该字段不证明 NPU 的实际同步映射合法。

脚本从 `1f833fb` 到 `e6397d4` 调整子进程看护、共享编译缓存、整数检查阶段记录及 NPU 默认 RS 范围，
计算 kernel、输入、计时方法和数值判定不变。最新入口还在独立平铺脚本目录中，
以默认 9 个 RS 测试 X=1/R=4K 的 sum/amax，共 18 个候选全部通过，记录位于 `gpu-portable-smoke/`。
该目录不位于源码仓库内，证明 CUDA 入口不依赖 PyTorch 源码仓库或 NPU 包。
NPU 先在卡 1 完成 X=1，并取得 X=7/56 的预算内参考。后续卡 1 出现 Alarm、利用率 100%，
不继续在该卡测速；X=7/56 改在卡 2 按完整形状重测，卡 1 的这两组参考不进入最终性能矩阵。
主矩阵的 8 个 RS 均进入最终原始记录。
另有进程用同参数 `warmup()` 预编译后续 kernel，不发射计算；每卡设备性能测试保持串行。
这只提前建立编译缓存，不改变计时内的计算。

设备共享使用。卡 1 小矩阵开始前 Health OK、利用率采样为 0%，有约 121312 MB 的驻留进程，
仍有约 3.9 GiB 可用，足够本实验；GPU 开始时约 10.2 GiB 空闲、利用率 0%。
NPU 卡 2 的早期 RS=64 尝试不进入性能汇总；后续在卡 2 完整重测 X=7/56。
同一形状的 original/debug 与全部 RS 同卡；CSV 保留获选候选的物理卡号。
卡 2 校准时 Health OK、采样利用率 0%，有驻留进程；与卡 1 的 X=7/R=4K sum 耗时接近。
卡 5 虽然 Health OK、利用率 0%、无可见进程，同配置 amax 耗时却高约十几倍，故不采用该卡性能。
校准代表值和原始记录单独归档，不计入 864 个主矩阵候选。
没有修改安装包、复位设备或操作其他任务。

## 选卡校准的实测值

全部为 X=7/R=4K，单位 µs；倍率为校准卡 / 卡 1。

| 操作 | helper | RS | 卡 1 | 校准卡 | 校准值 | 倍率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sum | original | 1 | 31.787 | 2 | 33.377 | 1.05× |
| sum | original | 8 | 1423.647 | 2 | 1407.828 | 0.99× |
| sum | debug | 1 | 3.274 | 2 | 3.242 | 0.99× |
| sum | debug | 8 | 5.590 | 2 | 5.881 | 1.05× |
| amax | original | 1 | 31.557 | 5 | 471.060 | 14.93× |
| amax | original | 8 | 1451.024 | 5 | 18977.863 | 13.08× |
| amax | debug | 1 | 3.946 | 5 | 52.586 | 13.33× |
| amax | debug | 8 | 5.842 | 5 | 98.275 | 16.82× |

卡 5 的原因未定位；这组校准用于排除该卡的性能数据，不把不同卡绝对耗时拼接成一条曲线。
16 个原始候选及对应环境保存在 `npu-calibration/`；卡 2 的 4 个校准候选也属于正式 X=7 记录，
在主矩阵中只计一次，额外校准汇总不增加 864 个候选的计数。

## 计算、计时与检查

每片循环读取本片输入，做 fp32 加法和局部规约；把 partial 写到 GM，经过被测屏障后，
每片读取本行全部 RS 个 partial 并合并，最后只有 shard 0 写行输出。
这保持 all-read 算法，partial 的逻辑读取量随 RS² 增长；不是 owner-only 合并或 persistent 数据保留实现。
RS=1 也调用屏障，没有 plain/split 基线，不能据此计算 cooperative 相对它们的收益。

NPUGraph / CUDAGraph + device Event，7 样本取中位数，保留原始样本及 min/max。
默认每图 batch=16，每样本 replay=10；单调用 pilot 估计每样本调用预算，目标 50 ms，
慢点减少 batch/replays，最少仍调用一次。预热 5 次、Graph 预热 3 次；pilot ≥5 ms 时各降为 1 次。
编译、CPU fp64 参考计算、检查和回传不计入设备耗时。
这属于共享卡上的热数据测量，不能直接作为生产阈值或芯片带宽上限。
热数据指同一组地址反复使用、不主动刷缓存，不表示所有形状的输入都能装入缓存。

每点必须同时通过首次输出、最后一次计时输出、两组变化输入和 4 phase 严格整数交换。
CPU 先做 fp32 加法再以 fp64 sum/amax 作为参考，sum 使用 `atol=0.02,rtol=2e-4`，amax 要求精确一致。
换输入时保留已有 workspace/semaphore，整数交换包括分片工作量差异及下一 phase 写入前的屏障。
随机输入检查不包含 NaN/Inf、动态 shape、并发和完整 Inductor 接入验收。

每个形状/helper 独立子进程，异常后重新建 context。NPU 执行/检查阶段无进展 15 秒则终止该子进程组，
初始化、编译和首次 launch 阶段为 180 秒；完成结果后的退出清理最多等待 15 秒，
退出清理超时不会丢弃已完成结果。CUDA 执行看护为 60 秒，未触发。
阶段进度和已完成样本会在超时记录中保留，超时不填作 kernel 延迟。

## 异常波动组的整组重测

卡 2 的 X=7、amax、R=16K、debug 首次记录中，最佳 RS=2 的 7 样本范围为
6.034～369.104 µs，中位数 105.745 µs；采样时设备利用率曾达到 100%。
相邻形状明显更低，因此停止本轮任务、核对状态后，在同卡相同参数下重新执行该组全部 8 个 RS。
重测最佳仍为 RS=2，中位数为 5.733 µs，范围为 5.725～5.766 µs，最终矩阵采用整组重测值。
前后全部 8 个 RS 的 IR 摘要和编译 metadata 完全相同。
旧组保留在 `npu-contention/amax-x7-r16384-debug/`，不计入 864 个主矩阵候选。
没有跨两次测量挑选最小样本；波动原因未进一步归因，不能将旧组高耗时解释为算子固有成本。

## RS=64 的 NPU 诊断

早期卡 2 的 X=1、sum 代表点中，RS≤56 的 8 个候选均通过后，RS=64 未完成首次同步。
4K/8K、original/debug 共 4 点被 host 看护终止；16K/original 返回 `vector core timeout`。
5 个实际记录及对应 manifest/helper 单独保留在 `npu-rs64-diagnostic/`，不计入 NPU 性能矩阵。
因此本轮 NPU 全形状测速采用 RS≤56；它仍实测 X=7/56 时 X×RS 超过 56 的组合，
没有把“总 program 数超核”整体过滤。CUDA 仍实际扫描 RS=64。

## 文件与刷新

| 文件 | 用途 |
| --- | --- |
| `npu/candidates.json`、`cuda/candidates.json` | 全候选及逐点状态、实际样本、误差、metadata、IR 摘要 |
| 相邻 `run.json`、`source-index.json` | 完整范围、实际采集 run 和逐形状来源 |
| `<op>-x<X>-r<R>-<kind>/candidates.json` | 分形状原始记录，可与总表交叉核对 |
| 各形状 `manifest.json`、`helper-*.py` | 版本、设备、编译器、实际 helper 源码与 SHA256 |
| `cooperative-reduction-matrix-gpu.tar.gz` | 其他 GPU 的便携采集、绘图脚本及 NPU 数据 |

不归档编译缓存、临时 `active.json` 和重复打印结果的正常日志；异常 traceback 已在候选 JSON 中。
CSV 与 SVG/PNG 位于 `docs/requirements-design/cooperative-reduction/assets/reduction-matrix/`。
运行和刷新命令统一维护在[全形状扫描说明](../../scripts/tests/cooperative_reductions/GPU_MATRIX_README.md)。
图表只从原始记录选择结果，不改变原始 JSON。
