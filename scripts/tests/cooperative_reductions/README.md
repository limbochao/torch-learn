# Cooperative reduction 调研探针

## fp32 sum / amax 全形状最佳 RS 扫描

`reduction_matrix.py` 统一扫描 X=1/7/56、R=4K～1M 的 9 档、sum/amax。
NPU 仅测 original/debug，各 8 个 RS；CUDA 测原始协议、9 个 RS。
完整矩阵为 1350 个候选、162 个最佳正确结果；NPU 的 RS=64 另保留代表点超时证据。
计算、计时、精度、续跑与便携命令见 [GPU 全形状扫描说明](GPU_MATRIX_README.md)。
在指定容器工作目录检查设备占用和映射后运行：

```bash
# CUDA：默认使用固定上游 helper，不依赖 torch_npu。
CUDA_VISIBLE_DEVICES=GPU-实际UUID python -u scripts/tests/cooperative_reductions/reduction_matrix.py \
  --out artifacts/cooperative-reduction-matrix-20261123/cuda

# NPU：选卡须符合当前授权和时段；这里只以物理卡 2 为例。
ASCEND_RT_VISIBLE_DEVICES=2 NPU_DEVICE_LIMIT=28,56 \
  python -u scripts/tests/cooperative_reductions/reduction_matrix.py --backend npu \
  --physical-device 2 --point-timeout 15 --out artifacts/cooperative-reduction-matrix-20261123/npu

# 绘图不需要 GPU/NPU。
python scripts/tests/cooperative_reductions/plot_reduction_matrix.py \
  --npu artifacts/cooperative-reduction-matrix-20261123/npu/candidates.json \
  --cuda artifacts/cooperative-reduction-matrix-20261123/cuda/candidates.json \
  --out docs/requirements-design/cooperative-reduction/assets/reduction-matrix
```

RS=1 仍调用屏障，本扫描用于比较三种实现及寻找已测最佳分片数；它不提供 plain/split 的收益基线。
未通过检查但已完成计时的候选仍保留诊断耗时，不能进入最佳结果。

## Split 与 cooperative 整图对照

`strategy_compare.py` 使用实际 CUDA Inductor，不替换 codegen 或同步 helper。
已测版本是 PyTorch `2.13.0+cu126`（提交 `cf30153c4c131c8164ee7798e5022d810682e2cb`）、
Triton 3.7.1、CUDA 12.6。使用 Inductor 私有接口，其他版本先运行小矩阵核对生成代码。
单独复制该脚本即可在其他 CUDA 环境运行；无需 torch_npu 或本地 PyTorch 源码。
在指定容器工作目录同步代码、检查 UUID 和占用后运行，输出目录建议放在数据挂载盘：

```bash
# 主矩阵：sum、RMSNorm、softmax、4 次依赖 softmax；48 个独立进程 case
CUDA_VISIBLE_DEVICES=GPU-实际UUID python -u scripts/tests/cooperative_reductions/strategy_compare.py \
  --out artifacts/cooperative-strategy-20261123/cuda

# 同一图的不同规约分别命中 cooperative 和 split
CUDA_VISIBLE_DEVICES=GPU-实际UUID python -u scripts/tests/cooperative_reductions/strategy_compare.py \
  --workloads mixed --rows 32 --rnumels 32768 --strategies plain,split,cooperative \
  --out artifacts/cooperative-strategy-20261123/cuda-mixed

# 显式两阶段：外层 4 组，第一阶段内部用 cooperative，第二阶段普通汇总
CUDA_VISIBLE_DEVICES=GPU-实际UUID python -u scripts/tests/cooperative_reductions/strategy_compare.py \
  --workloads sum --rows 1 --rnumels 1048576,8388608 \
  --strategies split,cooperative,hybrid --groups 4 \
  --out artifacts/cooperative-strategy-20261123/cuda-hybrid
```

`--strategies plain,split,cooperative` 分别关闭两种分片、仅开 split、两者开关均开但不强制。
最后一种仍可能回退 split；以 `source_summary.cooperative_decorators` 和生成源码判断。
每图 16 次完整调用、20 次 replay、7 个 device Event 样本；输入/输出、workspace 沿用
compiled wrapper，计时包含其 semaphore 清零等设备工作，排除编译、eager 检查和 Python/CPU 分配。
保留实际设备事件，kernel 数
只统计生成的计算 kernel，不含 semaphore 清零。每点有三组变化输入和计时后的 eager fp32 对照。
CUDA 计时后另调用一次检查结果，未单独保存最后一次 Graph replay 的输出。
`--resume` 读取已有 result，不会重跑已有错误点；针对复测应使用新输出目录。

`npu_strategy_compare.py` 是匹配算法的 A5 SIMD 原型，非 NPU Inductor 集成结果。
两 kernel split 与一 kernel all_vector 使用相同 `BLOCK=1024`、分片和统计量合并，整行输出均重读输入。
在 NPU_A5 Docker、`pta_213_case` 中按当前授权、时段和占用选卡；下例物理卡 1 需要仍满足选卡条件：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 NPU_DEVICE_LIMIT=28,56 \
  python -u scripts/tests/cooperative_reductions/npu_strategy_compare.py \
  --physical-device 1 --out artifacts/cooperative-strategy-20261123/npu
```

默认 X=1、R=32768/1048576、S=1/2/4/8/16/32/56；S=1 是无 workspace/屏障的 plain，
其余分别测 split/cooperative，共 78 条结果。NPU Graph 每图 16 次调用、每样本 10 次 replay、
7 组中位数；同样检查三组输入与计时后精度，参考为 NPU eager fp32。
脚本核对 ACL 物理映射，要求 `X*S` 不超过设备属性所报有效 Vector 核数；
这只是本实验的保守配置约束，不能替代完整的映射、前进性与并发契约。

取回结果后，使用 Matplotlib 在无设备机器上刷新文档同名 SVG/PNG 与 CSV：

```bash
python scripts/tests/cooperative_reductions/plot_strategy_compare.py \
  --cuda artifacts/cooperative-strategy-20261123/cuda/results.json \
  --cuda-mixed artifacts/cooperative-strategy-20261123/cuda-mixed/results.json \
  --cuda-hybrid artifacts/cooperative-strategy-20261123/cuda-hybrid/results.json \
  --npu artifacts/cooperative-strategy-20261123/npu/results.json \
  --npu-recheck artifacts/cooperative-strategy-20261123/npu-recheck-short/results.json \
  --npu-recheck artifacts/cooperative-strategy-20261123/npu-recheck-long/results.json \
  --out docs/requirements-design/cooperative-reduction/assets/strategy-comparison
```

`--npu`、各 CUDA 输入均可单独省略；只画 GPU 时仅需 CUDA JSON。
`--npu-recheck` 为可选的定向复测输入，按 shape/S/策略替换图和 CSV 中的对应点；
原始 JSON 不变。上面两条复测仅针对本轮已观察到的噪声点，没有复测文件时省略这两个参数。
绘图脚本与 `plot_barrier_suite.py` 放在同一目录。保留各 case 的 `result.json`、
`output_code_*.py` / `*.ttir` / `*.ttadapter` 及总表、manifest；不需回传缓存。
结果解释见[整图对照][graph]。

[graph]: ../../../docs/requirements-design/cooperative-reduction/barrier-comparison.md#45-split-与-cooperative整图对照与分层组合

## NPU 基础能力与同步探针

`probe.py` 用于 A5 上的定向研究，结果解释见
[需求分析](../../../docs/requirements-design/cooperative-reduction/requirements.md)。
不修改已安装的 PyTorch、torch_npu 或 Triton。

在 Docker 工作目录激活 `pta_213_case`，按物理卡限制和占用情况设置
`ASCEND_RT_VISIBLE_DEVICES` 后运行。`--device 0` 是可见设备中的逻辑号。

```bash
# 只读版本、源码 SHA256 和对应源码片段，不初始化 NPU。
python scripts/tests/cooperative_reductions/probe.py inspect --out ../cooperative-results/inspect

# 显式指定目标，仅编译，不 launch CUDA semaphore 自旋屏障。
python scripts/tests/cooperative_reductions/probe.py compile --out ../cooperative-results/compile

# 现有 Inductor 路径、开启开关和强制开关的差异，保存实际 selected config。
python scripts/tests/cooperative_reductions/probe.py inductor \
  --shapes 1x1048576 --out ../cooperative-results/inductor

# 动态 shape 编译，并在同一个 callable 上验证缩小 R 和非整除尾块。
python scripts/tests/cooperative_reductions/probe.py inductor --dynamic \
  --shapes 1x1048576 --inductor-modes baseline,enabled,forced \
  --reuse-r 32768,1048593 \
  --out ../cooperative-results/dynamic

# 手写 SIMD 原型：GM 可见性与重复屏障检查，然后比较三种策略。
python scripts/tests/cooperative_reductions/probe.py bench --graph \
  --shapes 1x4096,1x32768,1x1048576,1x16777216,8x1048576,32x1048576 \
  --out ../cooperative-results/bench
```

`compile` 阶段使用固定 target、signature、constexpr 和 compile mode；每个结果明确记录
`launched: false`。`bench` 只运行原生 `all_vector` 屏障，且要求 `X * S` 不超过 vector core 数。
它验证的是手写原型，不是已适配的 Inductor cooperative reduction。
`native_visibility` 在每一阶段写新值、全核同步、读其他 program 的值，并在覆盖 workspace
前再次同步，覆盖 8 个阶段。`probe.py bench` 不运行 CUDA semaphore 协议。

性能结果保留五组原始样本；`--graph` 在 NPU Graph 内捕获 32 次调用，计时 10 次 replay，
先预热三次 replay，以 320 次调用归一化。
未指定 `--graph` 时记录事件加 Python enqueue 时间，不能直接混比。
原型固定 `B=1024`、SIMD，没有 autotune；默认拆分数取 core 数与 X 的比例并限制为 64。
因此结果只用于判断收益区间，不能作为最终触发阈值。

## x_grid_barrier 与原生同步的对照

`atomic_ticket_probe.py` 先检查返回旧值的 atomic：多个 program 对同一个计数器加 1，
要求计数结果正确，且返回值恰好覆盖 `0..N-1`。测试没有自旋等待。

`barrier_bench.py` 按下面的顺序执行，不修改安装包：

1. 使用增加了轮询次数上限的诊断版协议，检查 1/2/8/32 个 program、多个输出组与 8 轮数据交换。
2. 上一步全部通过后，依次调用安装包原始 `x_grid_barrier`、原生 `all_vector` 及 NPU helper 改写版复核交换。
   每项连续运行三次、每次八轮交换；不同调用使用不同输入值，避免旧 workspace 恰好通过检查。
3. 对同一个 sum 算法比较不分片、`x_grid_barrier`、原生 `all_vector` 与 NPU helper 改写版。
   各协作实现保持相同分片、`B=1024`、peer 合并和输出归属，固定 SIMD、4 warps、
   `launch_cooperative_grid=True`；沿用 `event_time` 的 Graph + Event 五组中位数方法。

以下命令以可见物理卡 1 为例，执行前按当前任务的选卡约定设置。脚本通过 CANN API 核对
`--physical-device` 与逻辑设备 0 的映射，并要求 program 数不超过 Vector core 数。

```bash
ASCEND_RT_VISIBLE_DEVICES=1 python scripts/tests/cooperative_reductions/atomic_ticket_probe.py \
  --out ../cooperative-results/atomic-ticket
ASCEND_RT_VISIBLE_DEVICES=1 python scripts/tests/cooperative_reductions/barrier_bench.py \
  --physical-device 1 --out ../cooperative-results/xgrid
```

有限轮询版本只用于功能诊断，不用于性能统计。输出的 `visibility.json` 记录各协议的
交换结果，`bench.json` 只记录原始 helper、原生同步和改写版等实际 kernel 的耗时；功能检查失败
会停止计时。manifest、原始 helper 源码、编译 metadata 与中间 IR 保留在输出目录。

### all_vector 改写版 helper

`npu_triton_helpers.py` 提供 `x_grid_barrier(sem)` 原型，函数体为
`extension.sync_block_all("all_vector", 0)`。保留 `sem` 参数便于替换调用，既不读取也不更新其内容。
它同步整个参与的 Vector 集合；调用者须保证所有参与者按相同顺序、相同次数到达每次屏障，
满足模式/映射约束，并将事件 0 留给该协议。
例如四组各八个 program 统一执行八轮屏障可以作为验证对象；不能让其中一组提前退出或少执行一轮。

改写版的可见性检查将 semaphore 初始化为 123，检查它保持不变；同一 helper 的事件 0
既用于“写后读”，也用于“读完再覆盖”，覆盖同一编号的连续复用。
这是 A5 SIMD 的调研原型；完整 Inductor 接入仍需在选择和 codegen 阶段满足上述调用条件。

## 同步慢例的分量分析

`barrier_components.py` 使用相同 SIMD 配置和 Graph + Event 方法，比较以下分量：

- 普通 load、不使用 atomic 返回值、使用旧值、独立地址与同地址、加 0 的轮询 atomic。
- 不含归约负载的原始 helper 与原生同步，覆盖 1/2/8/32 个 program、1/4 次同步。
- 原生同步关闭自动 block 映射的诊断对照；仅在当前进程临时替换 compiler 的策略函数，
  用不同 constexpr 区分 JIT key，使用独立缓存目录，不修改安装文件。
- 有限轮询诊断版记录每个 program 的轮询次数；该版本不参与性能统计。

独立 atomic 地址间隔为 32 个 int32，即 128 字节。每项在计时前检查写入结果及所用旧值；
atomic 的计数器在计时中继续累加，不把重置操作放入被测 DSL。
各条记录均为一次完整 Triton 调用的设备耗时，未单独剥离 launcher 的隐式状态处理。

```bash
ASCEND_RT_VISIBLE_DEVICES=2 python scripts/tests/cooperative_reductions/barrier_components.py \
  --physical-device 2 --out ../cooperative-results/xgrid-components
```

`dump_adapter.py` 使用已保存的 adapter 和 metadata，仅重新编译，不 launch；编译器的
debug 保存 lowering 后的 MLIR、LLVM IR 等产物，便于确认软件锁调用。
锁体通过模板库链接时，还需结合匹配设备编译器版本的源码分析其实现。

```bash
python scripts/tests/cooperative_reductions/dump_adapter.py \
  ../cooperative-results/xgrid-components/atomic_old_shared-p32.ttadapter \
  ../cooperative-results/xgrid-components/atomic_old_shared-p32.metadata.json \
  --out ../cooperative-results/xgrid-components/dump-atomic
```

## 流水采集与 IR 导出的区别

`dump_adapter.py` 导出编译产物，用于确认后端生成了哪些同步，不运行 kernel，不能获得流水耗时。
上板观测需要 MindStudio 算子工具：`PipeUtilization` 提供忙碌比例等汇总指标，A5 的
`PipeTimeline` 展示各通路运行区间，支持版本的 `InstrTimeline` 可进一步定位具体指令。
`PipeTimeline` 与 `InstrTimeline` 分别采集，通过 MindStudio Insight 打开 `visualize_data.bin`。

以下为命令形式，`repro_one_kernel.py` 表示只运行目标 kernel 的复现脚本占位名，需替换为实际脚本。
先用当前安装工具的 `--help` 确认支持的参数；旧入口可能为 `msprof op`，新入口为 `msopprof`。
示例未指定设备，实际运行仍按工作区规则和本任务约定选卡。

```bash
msopprof --kernel-name=reduction --launch-count=1 \
  --aic-metrics=PipeTimeline --output=profile-pipe python repro_one_kernel.py

msopprof --kernel-name=reduction --launch-count=1 \
  --aic-metrics=InstrTimeline --output=profile-instr python repro_one_kernel.py
```

需选择目标编译配置、shape 和真实 kernel 名称；现有 `barrier_bench.py` 会顺序运行多个同名
配置，不能只用 `--kernel-name=reduction` 就认定已选中目标改写版。
抓取多核结果时，关联 partial 写回、跨核同步、peer 读取，区分本核搬出与等待慢核的时间。
插桩数据用于分析，不直接混入此前未插桩 Graph + Event 的性能表。

仿真可使用 `msprof op simulator` 或 `msopprof simulator`，选择对应 CANN 版本支持的 SoC
仿真器；输出的 `trace.json`/`visualize_data.bin` 是仿真时间线，不能当作 A5 上板时间。
当前调研尚未实际执行这些时间线采集命令。参数与支持范围见
[MindStudio 工具指南](https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md)
和[TA 性能分析方法](https://triton-ascend.readthedocs.io/zh-cn/latest/debug_guide/profiling.html)。

## 三种 helper 的分片扫描与并发验证

`barrier_suite.py` 提供 CUDA/NPU 共用的工作负载，`--kinds` 的三个值为：
`original`（安装包原始协议）、`native`（只调用 all_vector）及 `debug`（两侧各加一次
`tl.debug_barrier` 的 all_vector）。CUDA 只执行 `original`，另两种依赖 CANN。

- `bench`：小整数数据交换和 `(a.float()+b.float()).sum(-1)`，扫描分片数。
  统一 all-read 合并算法、`BLOCK=1024`、4 warps，完整 kernel 以 Graph + device Event 计时。
  同一性能点的 helper 复用相同设备缓冲区，校验前重置。计时前后都检查结果，保存全部样本、
  中位数、精度误差、缓冲区地址及编译 metadata。
- `check`：多输出组与三维 Grid、非二次幂分片、空片、尾块、分片耗时不均、重复 epoch/phase；
  另测 fp32/fp16/bf16 输入的 sum/mean/amax。Prologue 和累加为 fp32，参考为 CPU fp64；输出为 fp32。
- `mapping`：在单独进程中运行一个指定 Grid，观察默认/关闭自动分核的差别，也可诊断超过核数的 Grid。
- `concurrency`：spawn 独立进程，预热后统一起跑；各进程通过独立线程向多个 stream 提交，
  每个 stream 独立持有 workspace/semaphore，保留每次输出。每轮交替执行浮点归约与两 phase
  整数交换，epoch 随进程、流、轮次变化。提交并发不直接等同设备指令并发。

下面的 NPU 命令仅示例物理卡 2，实际运行前确认其仍符合时段限制及选卡约定。
三类命令使用不同结果目录，共享编译缓存以减少重复编译。

```bash
export ASCEND_RT_VISIBLE_DEVICES=2
export TRITON_CACHE_DIR=/home/l30023782/PTA_213/cooperative-results/barrier-suite-cache
python scripts/tests/cooperative_reductions/barrier_suite.py --backend npu --phase bench \
  --physical-device 2 --out ../cooperative-results/barrier-suite/bench
python scripts/tests/cooperative_reductions/barrier_suite.py --backend npu --phase check \
  --physical-device 2 --out ../cooperative-results/barrier-suite/check
python scripts/tests/cooperative_reductions/barrier_suite.py --backend npu --phase concurrency \
  --physical-device 2 --kinds native --workers 2 --streams 2 \
  --out ../cooperative-results/barrier-suite/concurrent-native
```

CUDA 在指定容器内按 UUID 选择物理 GPU 6，使用 `--backend cuda --physical-device 6 --kinds original`。
`--splits` 可显式扩展，常规 bench/check 排除 program 数超过计算核数的配置；超核分核通过独立
`mapping` 命令验证，不把此类配置混入性能图。NPU 设置单算子超时，单项失败保留错误及配置；
并发运行还有进程级期限，不修改安装包或设备全局配置。

这些是 helper 与手写协作算法的验证，不代表已实现 Inductor 自动选择，也不证明按组屏障可通用替换。

`mapping --mapping off` 的诊断使用单独的 `TRITON_CACHE_DIR`，检查输出 metadata 的
`auto_blockify_enabled` 确认实际开关，避免复用另一种映射的编译缓存。
`mapping` 在校验失败时也保留已执行的 IR、metadata 与首末 phase 输出。
超核诊断单独运行并设置进程期限，不将其加入普通性能扫描。

`plot_barrier_suite.py` 使用 Matplotlib 3.8 及以上版本，读取两端 `bench.json`，
导出 SVG/PNG 折线图及 CSV。结果与示例图见
[同步验证文档](../../../docs/requirements-design/cooperative-reduction/barrier-comparison.md)。

## 非规约轴未满核与满核

`output_axis_cases.py` 固定每行一个输出任务，对照 `X=7`（未满 56 核）与 `X=56`（满核），
并扫描 R 轴分片。每个配置检查 fp32 sum 和多轮整数交换；开启以下诊断开关后，数值错误点也采集 device 时间。
额外的 `plain` 基线只在 `S=1` 运行，完全省去 workspace 交换和同步。
设置 `NPU_DEVICE_LIMIT` 时，预算使用 TA 实际返回的有效 Vector 核数，另行记录物理核数和 runtime 查询值。

```bash
python scripts/tests/cooperative_reductions/output_axis_cases.py \
  --backend npu --physical-device 4 --rows 7,56 \
  --rnumels 32768,1048593 --splits 1,2,4,8,16,32,48,56 \
  --allow-over-core --time-wrong-results --max-sample-ms 500 --case-timeout 300 \
  --out artifacts/cooperative-barrier-unfiltered-20261122/npu
```

运行前按工作区约定检查卡号、占用和设备可见性。`--allow-over-core` 放开 original 的核数预算过滤，
三种 helper 的超预算点均逐个启动独立进程；`--case-timeout` 限制单个子进程的墙钟时间。
`--time-wrong-results` 在 sum 校验失败后继续计时，仍保留错误数量、样例和失败状态。
不传这两个开关时保留保守过滤行为。数值错误点的耗时可用于分析性能潜力，不能视为正确实现的收益。
原版的单次 kernel 可能很慢；`--max-sample-ms 500` 根据额外一次 device Event 预测缩减重复数，
每条记录保留实际 batch/replays、预测耗时和 7 个测量样本，短 kernel 沿用 32/10。
至少执行一次调用，因此单个 kernel 本身超过 500 ms 时，样本仍可能超过该目标时间。
中断后可带 `--resume` 继续未完成点；已完成记录保留，其 `run_manifest` 指向原执行版本与参数。
超预算点使用独立分配的地址与相同种子/shape；预算内同一进程的 helper 复用输入/输出及 workspace 地址。
CUDA 可使用同一脚本，默认自动补充当前 GPU 的 SM 数作为满核 X；仅测试 original 和 plain。
便携包运行、选卡、依赖与结果回传见 [GPU 补测说明](GPU_README.md)。

### 限核后的分片与精度检查

以下示例将 TA 编译和启动预算设为 4 Cube / 8 Vector，覆盖限额内、总 program 数超限以及单组分片超限。
卡号仅为示例，执行前仍需核对时段、占用与授权。

```bash
NPU_DEVICE_LIMIT=4,8 ASCEND_RT_VISIBLE_DEVICES=2 \
python scripts/tests/cooperative_reductions/output_axis_cases.py \
  --backend npu --physical-device 2 --rows 1,2,8 \
  --rnumels 32768,1048593 --splits 1,2,4,6,8,12,16,32,48,56 \
  --allow-over-core --time-wrong-results --accuracy-repeats 3 --save-kernels \
  --max-sample-ms 500 --case-timeout 60 \
  --out ../cooperative-results/core-limit-8
```

`--accuracy-repeats 3` 在首次 NaN 哨兵检查和固定输入计时后，原地址连续换三组输入，
保留上一调用的 partial/semaphore，以 CPU fp64 参考检查是否读到旧值。完成后恢复原输入，
确保后续 helper 仍用相同输入比较。严格整数交换另行改变 epoch/phase 和分片工作量。
这些校验都在性能计时之外；最终 PASS 要求首次、计时后、变化输入和整数交换全部通过。
`--save-kernels` 在首次发射返回后、等待结果前保存 sum IR 与完整 metadata；交换调用完成后再保存其产物。
结果未返回的配置也尽量保留编译证据，用于核对核数与映射。
`NPU_DEVICE_LIMIT` 是 TA/Inductor 的环境预算；device/stream API 的 runtime 限额单独记录，不能视为同一个值。

### 结果绘图

`plot_output_axis_cases.py` 按相同 X/R 合并两端 JSON，导出 SVG、PNG 和完整 CSV。
实线表示检查通过的测量，空心虚线表示数值错误或检查未完成但取得的耗时，底部叉号表示无耗时的执行失败。
CSV 同时保留状态、相对本轮 plain 的倍率、计时前后误差及整数交换误差。
限核数据另保留有效/物理核数、环境预算与变化输入错误量，可用 `--title` 指定实验标题。
只有 NPU 数据时可先运行：

```bash
python scripts/tests/cooperative_reductions/plot_output_axis_cases.py \
  --npu artifacts/cooperative-barrier-unfiltered-20261122/npu/output-axis.json \
  --out docs/requirements-design/cooperative-reduction/assets/barrier-comparison
```

增加 `--cuda /实际路径/cuda/output-axis.json` 即可合并 GPU 数据并更新同名 SVG/PNG/CSV；
单设备实验只显示所提供设备，GPU 单独绘图只传 `--cuda`。
