---
title: 三种 x_grid_barrier 的 A5 与 RTX 4090 验证
---

# 三种 x_grid_barrier 的 A5 与 RTX 4090 验证

实验日期：2026-11-21，北京时间。本文评估同步 helper 的改写，作为
[Cooperative Reduction 需求分析](requirements.md)和[方案设计](design.md)的验证依据。

**结论：两种 all_vector 改写在本轮核数内、统一参与的配置中精度正常，同地址耗时中位数几乎一致；
它们都不能作为通用 x_grid_barrier 的完整替换。** 默认自动分核将 112 个逻辑 program
映射到 56 核时，两种版本均正常返回却产生错误结果；两侧本核屏障不能解决这一问题。
多进程多流提交检查通过，设备指令真正重叠时的事件隔离尚未确认。

## 1. 评估对象

三种实现采用相同的调用形式 `x_grid_barrier(sem)`：

| 脚本名称 | NPU 实现 | CUDA 对照 | 同步范围 |
| --- | --- | --- | --- |
| `original` | 安装包中的上游协议 | 同一协议 | 相同 y/z 下的全部 x program；每组独立 semaphore |
| `native` | 此前改写的直接 all_vector 版本 | 无 CANN 对应实现 | 本次 kernel 的全部参与 Vector 核 |
| `debug` | 用户提供的两侧本核屏障版本 | 无 CANN 对应实现 | 同上；额外增加两次本核屏障 |

`original` 在本核屏障后，用带 release 的 atomic 登记到达，保留 atomic 返回的旧值，
再用带 acquire 的 atomic 轮询 semaphore 的最高位，直到本组全部到达，最后执行本核屏障。
两端安装包内该函数的源码 SHA256 相同，均以 `c9fa8ee90b54adc6` 开头。

两个 NPU 改写的实际代码如下，均未读取或修改 `sem`：

```python
@triton.jit
def x_grid_barrier(sem):
    extension.sync_block_all("all_vector", 0)


@triton.jit
def x_grid_barrier_debug(sem):
    tl.debug_barrier()
    extension.sync_block_all("all_vector", 0)
    tl.debug_barrier()
```

其中，`tl.debug_barrier()` 表达 block 内同步；`all_vector` 表达参与 Vector 核之间的会合。
本核屏障无法恢复上游的按 y/z 分组语义。完整代码见
[被测 helper](../../../scripts/tests/cooperative_reductions/npu_triton_helpers.py)。

## 2. 环境与版本

| 项目 | NPU | GPU |
| --- | --- | --- |
| 服务器 | NPU_A5 | GPU_4090 |
| 实验位置 | Docker 内 `/home/l30023782/PTA_213/torch-learn` | 唯一指定容器 `pytorch-dev-gpu67` 内 `/home/gaojg/dev/torch-learn` |
| Python 环境 | `pta_213_case` | 容器原有 Python 3.12 |
| 设备 | 物理 NPU 2，Ascend950PR_9579，56 Vector / 28 Cube | 物理 GPU 6，RTX 4090，128 SM |
| 设备选择 | `ASCEND_RT_VISIBLE_DEVICES=2`，ACL 反查物理编号 | `CUDA_VISIBLE_DEVICES=GPU-528f4e79-58c0-e843-bf5e-0b40c2671cd7` |
| 占用情况 | 开始前无运行进程，符合夜间使用限制 | 共享使用；其他进程常驻约 13.4 GiB，检查时利用率 0% |
| PyTorch | `2.13.0+cpu`，通过 torch_npu 使用 NPU | `2.13.0+cu126` |
| torch_npu | `2.13.0+git93d7d01` | — |
| Triton / TA | `3.6.0` / `3.6.0+dev20261112182052` | `3.7.1` |
| NPUIR / CUDA | bishengir-compile 2.0.0，提交 `395c9c5646f9`，LLVM 19.1.7 | CUDA 12.6，驱动 535.183.01 |
| 编译选项 | `compile_mode="simd"`、4 warps | 4 warps |

两端均传入 `launch_cooperative_grid=True`，它是被测配置的一部分；NPU 上该字段存在不等于
自动证明了驻留、分核和全核屏障条件。性能采集代码提交为
`e352232967947b1b11a64dbfeb7c8128816e78b3`，两端脚本 SHA256 均以
`506523767a3a98b4` 开头。功能扩展与并发的各次提交、参数和源码摘要保留在各自 manifest 中。

环境明细：[NPU manifest][npu-manifest]、[GPU manifest][gpu-manifest]。
GPU 数据反映本次共享环境及固定 kernel 的结果，不能直接作为设备峰值或生产准入阈值。

## 3. 用例和计时方法

### 3.1 固定算法后比较 helper

`S` 表示 R 轴分片数，`X` 表示输出行数，`R` 表示每行归约长度。每个 program 负责一片。
本次固定以下算法，三种 NPU 实现及 CUDA 对照仅切换同步 helper：

1. 每片循环加载自己的数据，计算 fp32 局部和。
2. 将 partial 写到 GM workspace。
3. 调用被测 helper。
4. 每片读取本行全部 partial，计算完整结果，仅第 0 片写最终输出。

这是 **all-read 合并**：每个分片都读取最终合并所需的数据。每行有 `S` 个读取者，每人读取
`S` 个 partial，总读取量随 `S²` 增长。选择该算法是为了覆盖同步后的跨核读取，并与上游
各分片可能继续使用完整归约结果的场景对应；owner-only、只由一个分片合并的优化另行评估。

| 工作负载 | 数据和规模 | 目的 |
| --- | --- | --- |
| 小整数交换 | 每片写一个 int32，屏障后读取并求和 | 减少归约计算的干扰，精确检查可见性 |
| fp32 sum | `(a.float()+b.float()).sum(-1)`；`X=1`，`R=4096/32768/1048576/1048593/16777216` | 短轴、收益转折、长轴、尾块和大输入 |
| 多行 sum | 同一公式，`X=4,R=1048576` | 检查多个 y 组共享全 Vector 屏障时的行为 |

分片数为 `1,2,4,8,16,32,48,56`，局部 `BLOCK=1024`。常规性能点要求 `X*S` 不超过
本设备核数，因此 NPU 有 156 点，CUDA 有 54 点；多行曲线的终点随设备核数变化。
手写 kernel 读取 partial 时使用 `next_power_of_2(S)` 大小的 tile，并 mask `peer<S`，
因此可以测 48/56 等分片数；Inductor 生成这些候选仍需检查自身 codegen 的要求。
`S=1` 仍调用 helper，表示单分片的同算法参考，不能当作已经优化掉屏障的普通归约性能。

### 3.2 Device 侧计时

每点先做结果校验，再预热 5 次，将连续 32 次 kernel 调用捕获为一个 Graph。
Graph 再预热 3 次，随后采样 7 组，每组用 device Event 包围 10 次 Graph replay：

```text
每个样本的单 kernel 耗时 = Event 测得的设备时间 / (32 × 10)
展示值 = 7 个样本的中位数
```

NPU 使用 `NPUGraph`，CUDA 使用 `CUDAGraph`。每点计时结束后再次校验结果。
图中的纵轴是上述 **完整 kernel 的 device 侧平均耗时**，包含加载、加法、局部归约、
partial 交换、同步、合并和输出；编译时间、CPU 参考计算不计入。
小整数交换也包含写入和读取，不能用其数值直接命名为“纯 barrier 延迟”。

同一 `X/R/S` 下三种 NPU 实现复用完全相同的输入、workspace、输出与 semaphore 分配，
每个 helper 校验前重置 workspace、输出及 semaphore；记录实际地址，确认没有重新分配输入。
这样避免把各版本输入地址和缓存状态的差异当成同步实现的效果。
每个分片配置轮换 NPU helper 的执行顺序。全部样本、最小/最大值及精度误差均保存，
图中浅色区域表示该点的样本最小值到最大值。

### 3.3 功能检查的判定

整数交换使用严格相等；workspace 先填入哨兵值，制造先读后写时可识别的错误。
重复同步检查每次执行 16 个 phase，每 phase 写入不同值、交换结果，再增加一次屏障保护
读者结束后才能覆盖 workspace；还使用 3 个不同 epoch 和按分片变化的计算量。

浮点检查覆盖 fp32/fp16/bf16 输入的 `sum/mean/amax`，prologue、partial 和合并均使用 fp32，
输出为 fp32；CPU 对同一组输入先做 fp32 加法，再以 fp64 归约作为参考。
`mean` 先合并局部和，再除以真实 `R`，尾块及空片不增加分母。
这套低精度输入用例显式定义了 fp32 prologue，不等同于所有 ATen 半精度加法的舍入规则。

sum 容限为 `atol=0.02,rtol=2e-4`；mean/amax 为 `atol=2e-5,rtol=2e-4`。
同时记录实际误差，不将容限本身当作测量结果。整数交换为同步正确性的独立检查。

## 4. 性能结果

NPU 156 点和 CUDA 54 点全部通过计时前后校验。所有曲线按实际测量值绘制，没有插补缺失点。

![三种 NPU helper 与 CUDA 原始协议：不同 R 的分片扫描](assets/barrier-comparison/barrier-comparison.svg)

总图使用对数纵轴，以同时显示微秒级改写与毫秒级原始协议。下面用线性纵轴放大两种改写
和 CUDA 对照；长轴子图展示 `S>=8`，完整分片范围保留在总图中。

![改写版本的线性坐标细节](assets/barrier-comparison/barrier-rewrites-detail.svg)

### 4.1 代表性测量值

单位均为 µs，表示完整 kernel 耗时。

| 负载 | S | NPU original | NPU native | NPU debug | CUDA original |
| --- | ---: | ---: | ---: | ---: | ---: |
| 小整数交换 | 1 | 5.044 | 0.851 | 0.848 | 1.545 |
| 小整数交换 | 32 | 532.121 | 7.373 | 7.372 | 1.581 |
| 小整数交换 | 56 | 1562.487 | 7.526 | 7.559 | 1.721 |
| X=1,R=4096 | 1 | 5.691 | 1.429 | 1.431 | 2.595 |
| X=1,R=4096 | 56 | 1564.536 | 5.495 | 5.497 | 2.160 |
| X=1,R=1048576 | 1 | 106.528 | 102.270 | 102.266 | 173.674 |
| X=1,R=1048576 | 16 | 152.731 | 9.830 | 9.834 | 12.562 |
| X=1,R=1048576 | 32 | 534.695 | 8.216 | 8.219 | 7.584 |
| X=1,R=1048576 | 56 | 1548.014 | 7.520 | 7.527 | 5.421 |
| X=1,R=16777216 | 32 | 588.863 | 141.618 | 141.609 | 190.441 |
| X=1,R=16777216 | 56 | 1595.645 | 88.558 | 88.576 | 147.686 |
| X=4,R=1048576 | 8 | 495.194 | 18.295 | 18.291 | 23.942 |

原始数据：[NPU 全部性能点][npu-bench]、[CUDA 全部性能点][gpu-bench]；
便于比较的 [CSV](assets/barrier-comparison/timings.csv)包含各点中位数、最小值和最大值。

### 4.2 对分片选择的含义

**NPU 原始 semaphore 协议的开销随分片数明显增加。** 小整数交换从 1 片的 5.044 µs
增至 32 片的 532.121 µs、56 片的 1562.487 µs；两种改写在后两个点约为 7～8 µs。
此前的分量实验已发现当前 SIMD 对使用返回旧值的 atomic 采用软件锁和扫描，
代码及归因见[需求分析中的 atomic 分量证据](requirements.md#61-已有性能证据及边界)。
本次改写同时改变了同步原语及其后端编译路径，不能将前后差值全部解释为某一条指令的耗时。

**两种改写仍需按每片工作量控制 S。** `R=4096` 时 native 在 `S=1` 为 1.429 µs，
拆至 56 片为 5.495 µs；`R=32768` 时最小测量值出现在 `S=4`，约 2.666 µs；
`R=1048576` 时 32～56 片进入约 7.5～8.3 µs 区间；`R=16777216` 时 native 的最小值
为 `S=56` 的 88.558 µs。同步和 all-read 合并的成本随 S 增加，局部归约成本随分片减小，
二者共同决定转折位置。

**CUDA 对照说明两端策略不能直接共用。** 本次小整数交换的 CUDA 原始协议约 1.5～1.7 µs，
在本范围内随 S 变化较小；NPU 改写在大 S 下仍有几微秒的交换开销。对 `R=1048576,S=56`，
CUDA 为 5.421 µs、NPU native 为 7.520 µs；对更大输入 `R=16777216,S=56`，两者分别为
147.686 µs、88.558 µs。这里同时受访存、局部 reduction、资源映射和共享环境影响，
不能用整段 kernel 的设备差值单独判断屏障或芯片的优劣。

这些数据支持 NPU 独立建立小输出/长轴的收益判断，并对合法的 `S/BLOCK` 候选测优。
它们尚不足以直接确定生产阈值：当前只有连续输入、固定 BLOCK 和手写 all-read 算法，
GPU 也为共享使用，正式接入还需要真实 Inductor 图和普通归约对照。

## 5. 本核屏障在编译产物中的体现

对性能用例 `X=1,R=1048576,S=32` 的实际 adapter 和 metadata 做仅编译导出，
native 的 partial store 与 peer load 之间已经存在以下操作，地址和类型在此省略：

```text
call @store_ubuf_to_gm_1d_float(...)
hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
hivm.hir.pipe_barrier[<PIPE_ALL>]
hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0
hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0
hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
call @load_gm_to_ubuf_1d_float_nocache(...)
```

MTE3 搬出本片 partial，`PIPE_ALL` 等待本核前序流水，跨核 set/wait 等待所有参与者，
随后 MTE2 读取 peer 数据。这是当前 native 已实现本核完成和跨核会合的设备证据。

两种版本的 adapter 差异是 `sync_block` 两侧各增加一条 `gpu.barrier`。
到最终 LLVM IR 仍保留了两条额外指令，核心差异如下：

```diff
 call void @llvm.hivm.BARRIER(i64 6)
+call void @llvm.hivm.BARRIER(i64 6)
 call void @llvm.hivm.SET.CROSS.CORE(i64 5, i64 1)
 call void @llvm.hivm.WAIT.FLAG.DEV.PIPE.IMM(i64 0, i64 0)
+call void @llvm.hivm.BARRIER(i64 6)
```

其中 native 原有的本核屏障由同步原语的 lowering 生成，当前编译路径不依赖 DSL 再补写
`tl.debug_barrier()`。接口契约、固定 lowering 规则及 Inductor/TA/NPUIR 的责任说明见
[方案中的同步机制分析](design.md#13-kernel-内协议与运行流程)。
完整证据：[native 最终 MLIR][native-mlir]、[debug 最终 MLIR][debug-mlir]、
[native LLVM IR][native-llvm]、[debug LLVM IR][debug-llvm]。

`debug` 的额外屏障没有恢复按组同步、改变事件编号或解决任意 program 映射问题。
最终同地址对照中两者几乎重合，例如 `R=16777216,S=2` 分别为 1697.866/1697.706 µs，
`S=8` 为 515.560/515.581 µs，`S=32` 为 141.618/141.609 µs。
初次各版本单独分配输入时，前两个配置曾分别得到 1510.476/1761.919 µs 和
514.315/436.185 µs；调整为同地址控制后，差异显著收敛。初次记录仅作为
[分配控制实验][allocation-control]保留，正文曲线和表格全部使用最终同地址测量。
目前没有缓存/访存计数器证据进一步拆分地址和缓存效应，也没有为两条额外指令单独测出净耗时。
这组证据支持“当前用例增加本核屏障的收益不明显”，不支持挑选个别点宣称某版本总是更快。

## 6. 功能、分核与并发结果

### 6.1 基础与重复同步

| 检查 | NPU original / native / debug | CUDA original |
| --- | --- | --- |
| 性能点计时前后校验 | 52 / 52 / 52 点全部 PASS | 54 点全部 PASS |
| 重复整数交换 | 每种 13 个 Grid、3 个 epoch、每次 16 phase，全部 PASS | 同一矩阵 PASS |
| 整数校验量 | 三种共 54,000 个值，严格相等 | 18,000 个值，严格相等 |
| sum/mean/amax 扩展 | 每种 27 组，每组连续运行 3 次，全部 PASS | 27 组全部 PASS |
| 尾块与空分片 | `R=1048593,S=32`、`R=17,S=32`、`X=4,R=32771,S=8` | 同一矩阵 PASS |
| 输入 dtype | fp32、fp16、bf16，均使用 fp32 prologue/累加/输出 | 同一矩阵 PASS |

整数 Grid 包含 `(S,1,1)` 的全部 8 个 S，以及 `(4,2,1)`、`(8,4,1)`、`(7,4,2)`、
`(14,2,2)`、`(28,2,1)`。最后三种 Grid 均有 56 个 program；覆盖了非二次幂分片、
多输出组、三维 program ID 展开和满核参与。本轮各组按相同次数进入屏障。

下表是这批 27 组浮点扩展的最大绝对误差，相对于同输入的 CPU fp64 归约参考：

| 算子 | NPU original | NPU native | NPU debug | CUDA original |
| --- | ---: | ---: | ---: | ---: |
| sum | 2.8302e-4 | 2.8302e-4 | 2.8302e-4 | 4.6612e-4 |
| mean | 4.9086e-8 | 4.9086e-8 | 4.9086e-8 | 1.9284e-8 |
| amax | 0 | 0 | 0 | 0 |

包含更大输入的性能扫描中，NPU sum 最大绝对误差为 `0.0056521`，也在本次容限内。
相同配置下两种 NPU 改写的已记录浮点输出一致。mean 使用分片和与全局元素数，
本轮未发现错误计数或额外的精度损失；这不代替方差、极端值和抵消输入的独立数值验收。
完整记录：[NPU 功能检查][npu-check]、[CUDA 功能检查][gpu-check]。

### 6.2 分核边界

| 配置 | native | debug | 实际 metadata |
| --- | --- | --- | --- |
| 默认分核，核数内的 13 种 Grid | PASS | PASS | `auto_blockify_enabled=True` |
| 关闭自动分核，Grid=(7,4,2)，共 56 program | PASS | PASS | `auto_blockify_enabled=False` |
| 默认分核，Grid=(112,1,1)，共 112 program | **错误结果** | **错误结果** | `auto_blockify_enabled=True` |

关闭分核的两项使用独立编译缓存；开关仅在诊断进程内临时设置，metadata 确认实际关闭。
112 用于显式构造两批等长 program 的分核诊断，不代表上游默认会选择 112 片；
其他超核数量尚未运行，本轮不将这项结果外推为全部超核 Grid 的实际表现。
两项超核诊断均完成编译和运行，随后因整数校验失败以状态 1 退出，没有触发超时。
每项均有 **1736/1792 个值不相等，最大绝对差 9717988**。关键观测如下：

| 输出位置 | 期望值 | native 实际值 | debug 实际值 |
| --- | ---: | ---: | ---: |
| phase 0，program 0 | 1250200 | -67788 | -67788 |
| phase 0，program 56 | 1250200 | 9650200 | 9650200 |
| phase 15，program 0 | 18050200 | 8332212 | 8332212 |
| phase 15，program 56 | 18050200 | 18050200 | 18050200 |

**原因是物理核的会合不能代替尚未执行的逻辑 program。** 失败配置的最终 MLIR 中，
AutoBlockify 生成了步长为 56 的 program 循环，屏障保留在循环内部。下面是实际循环骨架，
省略了地址、计算及同步指令：

```text
%19 = llvm.mlir.constant(56 : i32) : i32
%36 = hivm.hir.get_block_idx -> i64
%37 = llvm.trunc %36 : i64 to i32
cf.br ^bb1(%37 : i32)
^bb1(%39: i32):
  %40 = arith.cmpi slt, %39, %35 : i32
  // 本 logical program 的全部 phase、partial 写入、全 Vector 屏障和 peer 读取
^bb8:
  %89 = arith.addi %39, %19 overflow<nsw> : i32
  cf.br ^bb1(%89 : i32)
```

也就是每个核先完成 program `p` 的全部 16 个 phase，之后才执行 `p+56`。
前 56 个 program 在 phase 0 会合时，后 56 个尚未写入 partial；读取 112 片时仍会读到
workspace 初始哨兵。此时前半部分的和为 623532，后半部分仍是 `56*(-12345)`，
相加正好为观测到的 `-67788`。后半部分 program 随后执行时，又会读到前半部分较晚 phase
留下的值。两条本核屏障只能等待已经发出的本核操作，无法使后一批 program 提前完成。

这把风险从推测收敛为可复现边界：**当前全 Vector helper 与这种 program 循环不能直接组合。**
Inductor/TA 必须在发射、预运行和 autotune 前确认映射合法，限制或拒绝跨批次参与的配置；
不能仅靠 `launch_cooperative_grid=True`、编译成功或额外本核屏障保证正确性。
本轮没有执行超核 original 协议，也没有将错误点纳入性能图。
设备检查结束时物理卡 2 为 Health OK、无运行进程。

证据：[native 失败及错误计数][mapping-native]、[debug 失败及错误计数][mapping-debug]、
[实际输出与 metadata][mapping-values]、[失败配置最终 MLIR][mapping-mlir]。

### 6.3 多进程与多流

使用 spawn 创建独立进程，各进程初始化自己的 NPU/CUDA context。每条流独占输入、
workspace 和 semaphore。初始化完成后统一起跑，每条流用一个线程提交，连续运行 32 轮；
每轮先运行长 sum，再运行两 phase 的整数交换，所有输出保留到同步后逐一校验。
整数 epoch 按进程、流和轮次变化，用于识别旧 partial 或跨流干扰。

| 设备与 helper | 进程数 × 每进程流数 | S | 结果 | 校验的 sum / 整数交换调用数 |
| --- | --- | ---: | --- | ---: |
| NPU original | 1×2、2×1、2×2 | 16 | 全部 PASS | 256 / 256 |
| NPU native | 1×2、2×1、2×2 | 16 | 全部 PASS | 256 / 256 |
| NPU debug | 1×2、2×1、2×2 | 16 | 全部 PASS | 256 / 256 |
| NPU native | 2×2 | 56 | PASS | 128 / 128 |
| NPU debug | 2×2 | 56 | PASS | 128 / 128 |
| CUDA original | 2×2 | 16 | PASS | 128 / 128 |

NPU 共 11 组并发配置，校验 1024 次 sum 和 1024 次变化 epoch 的整数交换，
整数误差为 0，sum 最大绝对误差为 `4.1861e-4`。CUDA 校验 128 次 sum 和 128 次整数交换，
整数误差为 0，sum 最大绝对误差为 `1.9427e-4`。
各子进程均正常退出，没有修改安装包、设备全局状态或其他用户进程。

这说明当前运行时下，多进程、多流提交以及各自临时空间的使用通过了本轮检查。
本轮没有采集设备时间线，不能证明多个 kernel 的设备指令发生了重叠；
运行时可能串行调度。因此“跨 kernel 真正重叠执行时事件 0 的隔离”仍需另行确认。
同一 kernel 内另一跨核协议占用核间事件 0 的情况也未覆盖，不能由进程/流测试代替。
前述本核流水的 `EVENT_ID0` 属于不同同步通路，不能只按数值同为 0 判断冲突。
明细见[并发配置结果索引][concurrency-index]、[GPU 并发结果][gpu-concurrency]。

## 7. 接入范围判断

**两个改写均不具备上游 helper 的通用等价语义。** 上游允许不同 y/z 组独立会合，
all_vector 要求参与 Vector 核执行相同的屏障序列。例如：

```python
if tl.program_id(1) == 0:
    x_grid_barrier(sem_for_this_row)
```

只要第 0 行的全部 x 分片到达，上游按组屏障可完成；全 Vector 屏障还会等待其他没有调用
屏障的行。给两侧增加本核屏障也不改变这一点。这个反例由同步范围直接推出，
本轮不在设备上故意运行这一不满足 all_vector 前提的分支。

建议将改写保留为专用 NPU 全 Vector helper，由 choices/schedule/codegen 和 TA/NPUIR
共同保证适用条件；普通归约和未准入场景继续沿用原有路径。需要明确的条件包括：

- 所有参与者按相同次数、相同顺序调用屏障，尾块只 mask 数据访问；
- 物理分核、program 循环与屏障参与者一致，不能只凭 `X*S` 或 launch 字段推断；
- 同一 kernel 内事件 0 的复用没有与另一同步协议冲突；
- workspace/semaphore 的存储与生命周期满足流、Graph 和重复调用的要求。

当前手写探针还没有覆盖完整 Inductor 自动选择、动态 shape、非连续输入、Welford、索引归约、
NaN/Inf、混合 Cube/Vector、其他编译模式、C++/AOT 等组合。这些仍按需求矩阵独立验收。
本次结论限定于记录的软件栈、A5 SIMD 和实际运行的配置。

由本轮结果得到的周边依赖如下，属于现有接入责任的具体化：

| 组件 | 需要适配或确认的内容 | 本轮依据 |
| --- | --- | --- |
| Inductor choices/schedule/autotune | 先检查全参与、统一 phase 和合法映射，再形成 S 候选；force 仍遵守条件 | 核数内通过，112 program 错误 |
| TA 分核与 launcher | 识别全核同步与 program 循环的关系；给出合法映射或明确拒绝不合法组合 | native/debug 仍开启 AutoBlockify，循环步长为 56 |
| NPUIR 同步 lowering | 保留本核完成、跨核会合及读取依赖；优化后维持语义 | 当前 PIPE_ALL、MTE3 set/S wait 已有实现证据 |
| TA/NPUIR 性能 | 分析大 S 的全核交换与 all-read 合并开销，分别优化访存和同步 | 32 片整数交换 NPU native 7.373 µs，CUDA 1.581 µs |
| Runtime/同步接口契约 | 明确事件在 task/context/stream 间的隔离和实际重叠执行条件 | 提交并发通过，设备重叠与同核事件竞争未覆盖 |

## 8. 复现入口

共用测试脚本为 [barrier_suite.py](../../../scripts/tests/cooperative_reductions/barrier_suite.py)，
阶段、设备选择、结果目录和命令说明见
[实验脚本 README](../../../scripts/tests/cooperative_reductions/README.md#三种-helper-的分片扫描与并发验证)。
绘图使用标准 Matplotlib，脚本为
[plot_barrier_suite.py](../../../scripts/tests/cooperative_reductions/plot_barrier_suite.py)。

```bash
python scripts/tests/cooperative_reductions/plot_barrier_suite.py \
  --npu artifacts/cooperative-barrier-comparison-20261121/npu/bench-shared/bench.json \
  --cuda artifacts/cooperative-barrier-comparison-20261121/cuda/bench-shared/bench.json \
  --out docs/requirements-design/cooperative-reduction/assets/barrier-comparison
```

[npu-manifest]: ../../../artifacts/cooperative-barrier-comparison-20261121/npu/bench-shared/manifest.json
[gpu-manifest]: ../../../artifacts/cooperative-barrier-comparison-20261121/cuda/bench-shared/manifest.json
[npu-bench]: ../../../artifacts/cooperative-barrier-comparison-20261121/npu/bench-shared/bench.json
[gpu-bench]: ../../../artifacts/cooperative-barrier-comparison-20261121/cuda/bench-shared/bench.json
[native-mlir]: ../../../artifacts/cooperative-barrier-comparison-20261121/npu/final-native/module.hivm.opt.mlir
[debug-mlir]: ../../../artifacts/cooperative-barrier-comparison-20261121/npu/final-debug/module.hivm.opt.mlir
[native-llvm]: ../../../artifacts/cooperative-barrier-comparison-20261121/npu/final-native/kernel.ll
[debug-llvm]: ../../../artifacts/cooperative-barrier-comparison-20261121/npu/final-debug/kernel.ll
[allocation-control]: ../../../artifacts/cooperative-barrier-comparison-20261121/npu/bench/bench.json
[npu-check]: ../../../artifacts/cooperative-barrier-comparison-20261121/npu/check/correctness.json
[gpu-check]: ../../../artifacts/cooperative-barrier-comparison-20261121/cuda/check/correctness.json
[concurrency-index]: ../../../artifacts/cooperative-barrier-comparison-20261121/README.md
[gpu-concurrency]: ../../../artifacts/cooperative-barrier-comparison-20261121/cuda/concurrent-epochs/concurrency.json
[mapping-native]: ../../../artifacts/cooperative-barrier-comparison-20261121/npu/mapping-auto-native-112/mapping.json
[mapping-debug]: ../../../artifacts/cooperative-barrier-comparison-20261121/npu/mapping-auto-debug-112/mapping.json
[mapping-values]:
  ../../../artifacts/cooperative-barrier-comparison-20261121/npu/mapping-auto-native-112/mapping-observations.json
[mapping-mlir]: ../../../artifacts/cooperative-barrier-comparison-20261121/npu/final-mapping112/module.hivm.opt.mlir
