# Cooperative reduction 调研探针

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
并扫描 R 轴分片。每个配置检查 fp32 sum 和多轮整数交换，仅对检查通过的点采集 device 时间。
额外的 `plain` 基线只在 `S=1` 运行，完全省去 workspace 交换和同步。

```bash
python scripts/tests/cooperative_reductions/output_axis_cases.py \
  --backend npu --physical-device 1 --rows 7,56 \
  --rnumels 32768,1048593 --splits 1,2,4,8,16,32,48,56 \
  --out artifacts/cooperative-barrier-output-axis-20261121/npu
```

运行前按工作区约定检查卡号、占用和设备可见性。超过核数预算的 original 点记录为
`SKIP_CORE_BUDGET`；native/debug 的超核点仅用于分核诊断，不能视作上游自动选择的候选。
结果记录数值错误的数量和样例；错误点没有性能值。输入和输出地址在各 helper 之间保持一致。
CUDA 可使用同一脚本，`--rows` 应按该设备的 SM 数设置；CUDA 仅测试 original 和 plain。
