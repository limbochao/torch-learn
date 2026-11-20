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
2. 上一步全部通过后，调用安装包中的原始 `x_grid_barrier` 复核相同的交换过程。
3. 对同一个 sum 算法比较不分片、`x_grid_barrier` 与原生 `all_vector` 三种实现。
   两种协作实现保持相同分片、`B=1024`、peer 合并和输出归属，固定 SIMD、4 warps、
   `launch_cooperative_grid=True`；沿用 `event_time` 的 Graph + Event 五组中位数方法。

以下命令以可见物理卡 1 为例，执行前按当前任务的选卡约定设置。脚本通过 CANN API 核对
`--physical-device` 与逻辑设备 0 的映射，并要求 program 数不超过 Vector core 数。

```bash
ASCEND_RT_VISIBLE_DEVICES=1 python scripts/tests/cooperative_reductions/atomic_ticket_probe.py \
  --out ../cooperative-results/atomic-ticket
ASCEND_RT_VISIBLE_DEVICES=1 python scripts/tests/cooperative_reductions/barrier_bench.py \
  --physical-device 1 --out ../cooperative-results/xgrid
```

有限轮询版本只用于功能诊断，不用于性能统计。输出的 `visibility.json` 记录三种协议的
交换结果，`bench.json` 只记录原始 helper 和原生同步等实际 kernel 的耗时；功能检查失败
会停止计时。manifest、原始 helper 源码、编译 metadata 与中间 IR 保留在输出目录。

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
