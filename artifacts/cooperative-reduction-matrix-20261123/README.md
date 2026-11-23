# fp32 sum / amax 全形状原始记录

用于[同步与性能报告](../../docs/requirements-design/cooperative-reduction/barrier-comparison.md)的
origin、debug、L20 cooperative 及两端 plain 对照。
与实际 Inductor split/cooperative 整图实验使用独立结果，这里测量的是固定 BLOCK 的手写 kernel。

## 范围与选点

计算为 `(a+b).sum(-1)` / `(a+b).amax(-1)`，输入连续 fp32，seed=213。
X=1/7/56，R=4096 起连续翻倍至 1048576，共 9 档；NPU 每组尝试
RS=1/2/4/8/16/32/48/56，original/debug 各 432 个候选；CUDA 另加 RS=64，共 486 个。
两端各增加 54 个 plain：每行一个 program，固定 RS=1，实际 Grid 为 `(X,)`。
合计 1458 个候选；原始 JSON 保留全部记录，文档 CSV 按每组最小的 PASS 中位数选择，共 270 行。
WRONG/ERROR/TIMEOUT 不参加最佳选择；数值错误但能完成计时的候选仍保留耗时。

| 场景 | 候选 | PASS | 校验失败 | 执行失败 | 超时 | 最佳结果行数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A5 plain | 54 | 54 | 0 | 0 | 0 | 54 |
| A5 origin | 432 | 324 | 0 | 0 | 108 | 54 |
| A5 debug | 432 | 324 | 108 | 0 | 0 | 54 |
| L20 plain | 54 | 54 | 0 | 0 | 0 | 54 |
| L20 original | 486 | 414 | 0 | 72 | 0 | 54 |

NPU 的 108 个校验失败点均完成计时，96 点出现浮点结果错误，108 点的严格整数交换均失败；
另外 12 点不能仅凭浮点输出判定正确。NPU 超时均在 origin 的 X=7/56、RS=32/48/56；
debug 校验失败均在相同两个 X 的 RS=16/32/48。
CUDA 启动失败均在 X=56、RS=32/48/56/64，原因为 `too many blocks in cooperative launch`。
超时和启动失败没有完成调用，不填设备耗时。

## 环境与源码

| 项目 | NPU | L20 |
| --- | --- | --- |
| 位置 | NPU_A5 Docker，`pta_213_case` | 用户在其他机器采集，回传目录 `tmp/cooperative/L20/` |
| worktree | `/home/l30023782/PTA_213/torch-learn-matrix-20261123` | 未提供 |
| 设备 | Ascend950PR_9579；统一物理卡 2；56 Vector | NVIDIA L20，92 SM；物理卡号未提供 |
| PyTorch | `2.13.0+cpu` | 未提供 |
| torch_npu / TA | `2.13.0+git93d7d01` / `3.6.0+dev20261112182052` | — |
| Triton / CUDA | `3.6.0` / — | 未提供 |
| 选卡与限额 | 可见物理卡 2；`NPU_DEVICE_LIMIT=28,56` | UUID `bedb56d0-f57b-2286-b96d-3631aa2d4dc9`，全部记录一致 |
| 采集脚本提交 | `910ba83`，沿用组为 `49a6591` / `e6397d4` | 未提供 Git 提交；脚本 SHA256 与原采集脚本一致 |
| kernel | SIMD、4 warps、BLOCK=1024 | 4 warps、BLOCK=1024 |

原始输出目录分别为：

- NPU plain：`/home/l30023782/PTA_213/cooperative-results/matrix-plain-20261123/npu-plain`。
- NPU X=1 origin/debug：同级 `npu-x1`；两种 helper 的全部 8 个 RS 同卡重测。
- NPU X=7/56 origin/debug：`/home/l30023782/PTA_213/cooperative-results/matrix-20261123/`
  下的 `npu-card2-x7`、`npu-card2-x56`，沿用同日卡 2 完整组。
- L20：`/home/qc/Repos/Ascend/tmp/cooperative/L20/`，回传的完整 `run.json` 与 `candidates.json`
  原样归档在 `cuda-l20/`，不从不同来源拼接候选。

汇总时按完整形状选择来源，不跨来源挑选最小 RS；`source-index.json` 记录逐组原始目录和文件摘要。
NPU 汇总 `run.json` 内嵌各来源的实际调用，不能当作单次扫描命令。
NPU 各形状 `manifest.json` 保留完整版本、NPUIR 编译器、实际限额、设备属性、命令、时间和源码 SHA256。
本仓旧采集器的 original helper SHA256 为
`c9fa8ee90b54adc67a11ba5406b5fe4be55178b974cc26a2bde8c2c9f19f5bb3`，
CUDA 使用 PyTorch v2.13.0 提交 `cf30153c4c131c8164ee7798e5022d810682e2cb` 的快照。
L20 的 driver 脚本 SHA256 为
`78ca26c47d3305052679f780364b8db3ef77bca103064575c948f23434ec239e`，与该采集器一致；
回传的 BLOCK、warps、RS 集合、输入、计时和精度检查参数均符合旧方案。
L20 未回传独立 helper 文件、manifest、完整 DSL/IR 或软件版本；候选内保留 metadata 和 IR 摘要。
因此这里只核实到已有记录，不把旧 4090 的版本、helper 文件和生成产物算作 L20 产物。
debug 为两侧各有一次 `tl.debug_barrier()` 的 `sync_block_all("all_vector",0)`。
cooperative 设置 `launch_cooperative_grid=True`，plain 显式设为 False；
该字段不证明 NPU 的实际同步映射合法，仍需看逐点检查。
plain kernel 源码 SHA256 为 `7fcb083c4656fb5c094c3bf602106c359a380aeaf5fd27bc663384c95ae407fa`。

补测开始时，卡 1 告警；卡 2 Health OK、采样利用率 0%、无可见进程，因此在卡 2 测量全部 plain，
并重测 X=1 的 origin/debug，替换旧卡 1 结果。L20 回传未包含占用、频率与空闲显存信息。
每卡测速保持串行，复用已有编译缓存，没有修改安装包、复位设备或操作其他任务。
X=7/56 的 NPU cooperative 与 plain 分时采集；相同物理卡不保证占用、频率完全相同，
接近 1 倍的差异不能作为稳定收益。此前卡 5 的异常慢校准数据未混入主矩阵。

旧 4090 全矩阵保留在 `cuda/`，不再进入当前主表或 SVG；其他 4090 整图和同步实验仍维持原设备标签。
L20 文件与回传源逐字节一致：

| 文件 | SHA256 |
| --- | --- |
| `cuda-l20/run.json` | `a4db5cff2bccccbb545ec4a136dd4a29f94ad24990e094540d981f7b6a763579` |
| `cuda-l20/candidates.json` | `969c8332715a966babe4a7e1ffeeab973c063ecf2500dc1ccc7055e969f7eebb` |

CUDA 旧入口曾在 4090 的独立平铺目录执行 X=1/R=4K 的 sum/amax，包含 2 个 plain 和默认 18 个
cooperative 候选，20 点全部通过，记录位于 `gpu-portable-plain/`。
该目录不在源码仓库内，manifest 的 `git_commit` 为 null；driver/helper SHA256 与已提交脚本一致，
证明新入口不依赖 PyTorch 源码仓库或 NPU 包。

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
在主矩阵中只计一次，额外校准汇总不增加 864 个 cooperative 候选的计数。

## 计算、计时与检查

每片循环读取本片输入，做 fp32 加法和局部规约；把 partial 写到 GM，经过被测屏障后，
每片读取本行全部 RS 个 partial 并合并，最后只有 shard 0 写行输出。
这保持 all-read 算法，partial 的逻辑读取量随 RS² 增长；不是 owner-only 合并或 persistent 数据保留实现。
cooperative 的 RS=1 也调用屏障；独立 plain kernel 每行循环读取全部输入，完成相同的 fp32 加法、
sum/amax 后直接写输出，显式设置 `launch_cooperative_grid=False`，没有 partial workspace、
semaphore 或跨核屏障。plain 固定 BLOCK=1024、4 warps，保存 kernel 源码及编译 IR，
便于核对 ordinary launch 和没有核间交换。这里没有搜索普通归约的其他 BLOCK 或 Inductor 策略。
CSV 的 `plain_us` 与 `speedup_vs_plain` 使用同物理卡及相同 UUID 配对，后者为 plain / cooperative。
这提供固定算法的普通归约收益基线，split 收益仍使用独立的完整图实验。

NPUGraph / CUDAGraph + device Event，7 样本取中位数，保留原始样本及 min/max。
默认每图 batch=16，每样本 replay=10；单调用 pilot 估计每样本调用预算，目标 50 ms，
慢点减少 batch/replays，最少仍调用一次。预热 5 次、Graph 预热 3 次；pilot ≥5 ms 时各降为 1 次。
编译、CPU fp64 参考计算、检查和回传不计入设备耗时。
这属于共享卡上的热数据测量，不能直接作为生产阈值或芯片带宽上限。
热数据指同一组地址反复使用、不主动刷缓存，不表示所有形状的输入都能装入缓存。

每点必须同时通过首次输出、最后一次计时输出和两组变化输入；
cooperative 额外要求 4 phase 严格整数交换通过。plain 没有跨 program 通信，不执行该项。
CPU 先做 fp32 加法再以 fp64 sum/amax 作为参考，sum 使用 `atol=0.02,rtol=2e-4`，amax 要求精确一致。
换输入时保留已有 workspace/semaphore，整数交换包括分片工作量差异及下一 phase 写入前的屏障。
随机输入检查不包含 NaN/Inf、动态 shape、并发和完整 Inductor 接入验收。

每个形状/helper 独立子进程，异常后重新建 context。NPU 执行/检查阶段无进展 15 秒则终止该子进程组，
初始化、编译和首次 launch 阶段为 180 秒；完成结果后的退出清理最多等待 15 秒，
退出清理超时不会丢弃已完成结果。CUDA 执行看护为 60 秒，未触发。
阶段进度和已完成样本会在超时记录中保留，超时不填作 kernel 延迟。

## 异常波动组的整组重测

下列两组在同卡、相同参数下按全部 8 个 RS 重测，采用完整新组；单位 µs。
各组全部候选的 IR 摘要与编译 metadata 在前后保持一致。

| 用例 | 获选 RS | 旧组：中位数（min～max） | 新组：中位数（min～max） |
| --- | ---: | ---: | ---: |
| amax、X=7、R=16K、debug | 2 | 105.745（6.034～369.104） | 5.733（5.725～5.766） |
| sum、X=1、R=1M、original | 4 | 72.368（51.864～180.537） | 46.177（46.040～47.648） |

amax 旧组采样时利用率曾达到 100%；sum 重测前 Health OK、利用率为 0%。
波动原因没有进一步归因，旧组高耗时不视为算子固有成本，没有跨轮次挑选最小样本。
旧组分别保留在 `npu-contention/amax-x7-r16384-debug/` 与
`npu-contention/sum-x1-r1048576-original/`，不计入 918 个 NPU 主矩阵候选。
sum 重测实际调用另存于 `npu/run.json` 的 `cooperative_sum_x1_r1m_repeat`。

## RS=64 的 NPU 诊断

早期卡 2 的 X=1、sum 代表点中，RS≤56 的 8 个候选均通过后，RS=64 未完成首次同步。
4K/8K、original/debug 共 4 点被 host 看护终止；16K/original 返回 `vector core timeout`。
5 个实际记录及对应 manifest/helper 单独保留在 `npu-rs64-diagnostic/`，不计入 NPU 性能矩阵。
因此本轮 NPU 全形状测速采用 RS≤56；它仍实测 X=7/56 时 X×RS 超过 56 的组合，
没有把“总 program 数超核”整体过滤。CUDA 仍实际扫描 RS=64。

## 文件与刷新

| 文件 | 用途 |
| --- | --- |
| `npu/candidates.json`、`cuda-l20/candidates.json` | 当前主表全部候选及逐点状态、样本、误差、metadata、IR 摘要 |
| `cuda/candidates.json` | 旧 4090 矩阵，只作历史证据 |
| 相邻 `run.json`、`source-index.json` | 完整范围、实际采集 run 和逐形状来源 |
| `<op>-x<X>-r<R>-<kind>/candidates.json` | 分形状原始记录，可与总表交叉核对 |
| 各形状 `manifest.json`、`helper-*.py` | 版本、设备、编译器、实际 helper 源码与 SHA256 |
| plain 各形状的 `kernel-plain.py`、`ir/` | 无 cooperative 的 kernel 源码、TTIR、后端 IR 与 metadata |
| `cooperative-reduction-matrix-gpu.tar.gz` | 其他 GPU 的便携采集、绘图脚本及 NPU 数据 |

不归档编译缓存、临时 `active.json` 和重复打印结果的正常日志；异常 traceback 已在候选 JSON 中。
CSV 与 SVG/PNG 位于 `docs/requirements-design/cooperative-reduction/assets/reduction-matrix/`。
运行和刷新命令统一维护在[全形状扫描说明](../../scripts/tests/cooperative_reductions/GPU_MATRIX_README.md)。
图表只从原始记录选择结果，不改变原始 JSON。
