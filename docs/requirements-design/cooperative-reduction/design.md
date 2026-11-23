---
title: NPU Cooperative Reduction 初步方案（PyTorch 2.13）
---

# NPU Cooperative Reduction 初步方案（PyTorch 2.13）

状态：方案草案，待评审。目标、支持范围与验收条件见[需求分析](requirements.md)。
下述 A5 SIMD、Index、静态 fp32 sum 与 owner-only 是候选实现路线，尚未确认为产品首期范围。
性能缺口 G1～G8 的定义见[需求分析第 3.4 节](requirements.md#34-性能缺口)，设备依据见[同步验证](barrier-comparison.md)。

## 1. 候选实现路线

### 1.1 首期路线与编译流程

首期在 **NPU Index 路径正向实现 cooperative**，复用现有轴分析、`BLOCK_SUB` 循环、
tile generator、autotuner 和 wrapper。No-linear 先显式使用普通归约，不同时实现两套协议。
原有 `KeyError: 'R'` 要通过 NPU 自身的 cooperative 初始化和索引处理解决；只增加一个字典项
不能证明上游后续 workspace/phase 假设适合 NPU。实现入口见[NPU schedule][N1]与
[Index 初始化][N3]，当前断点见[失败记录][E7]。

下图是建议流程；虚线表示编译期失败后的候选淘汰或普通路径回退。

![cooperative 编译流程](assets/compile-flow.svg)

<details markdown="1">
<summary>流程图 Mermaid 源码</summary>

```mermaid
flowchart TD
    A["ATen reduction 与输入信息"] --> B{"NPU 准入：A5 / SIMD / 静态 / fp32 sum"}
    B -->|通过| C["IR 标记可选；schedule 复核融合与单 phase"]
    B -->|不通过| O["当前普通 reduction"]
    C -->|通过| D["选择 S/B；校验 Grid、映射与资源"]
    C -->|超出范围| O
    D -->|S 大于 1 且有合法候选| E["Index codegen：partial / barrier / owner combine"]
    D -->|S 等于 1 或无候选| O
    E --> F["TA：保留同步依赖；确定 program 映射"]
    F --> G["NPUIR：设备 lowering 与资源 / 同步 metadata"]
    G --> H["合法候选 autotune；缓存 winner"]
    G -.->|候选失败| J["淘汰该候选；无剩余候选则回退"]
    J -.->|全失败| O
    J -.->|其余合法候选| H
    H --> I["Python wrapper：partial workspace 与隐式 lock ABI"]
    I --> K["A5 SIMD cooperative kernel"]
```

</details>

IR 与 schedule 共用同一套能力和必要条件判断，但 IR 阶段无法获知全部后续融合信息，所以
schedule 仍需复核。复核不通过时清除 cooperative 状态、workspace/候选及失败尝试的副作用，
构造当前普通归约，沿用现有全图规划机制。
Force 在本方案中只绕过收益判断，不绕过 SoC、mode、算子、参与者或内存安全条件。

选择原生同步的依据是固定 all-read 算法的对照：X=1/R=1048576/S=32 时，NPU original、native、debug
分别为 534.695/8.216/8.219 µs。分量实验与最终 IR 进一步确认，返回旧值 atomic 的软件锁是 semaphore
路径的主要性能负担；32 个 program 的“不使用返回值/使用旧值”为 4.012/278.324 µs。
因此原生同步作为受限实现路线，参与者与映射作为准入条件；直接复用 semaphore 则依赖 TA/NPUIR 解决该成本。
受测 native/debug 均有同组跨批错误，不能全局替换上游 helper，也不能靠额外本核屏障扩大适用范围。
方法与完整数值见[性能证据](requirements.md#61-已有性能证据及边界)。

### 1.2 最少需要传递的计划信息

沿用已有 choices、kernel kwargs、`inductor_meta` 和配置对象，不新增独立框架。
以下是语义字段，具体名字在实施时对齐现有接口；已由其他 metadata 表达的信息不重复存储。

| 信息 | 作用与首期约束 |
| --- | --- |
| cooperative 准入状态 | IR 与 Index kernel 一致；普通/no-linear 路径明确为 False |
| `RSPLIT=S`、输出 program 数 `Q` | 将归约分片与输出分核分开；首批 `X=1`，扩展小 `X` 时保持输出 tile 为 1 |
| 有效 Vector 核数 `C_eff` | 使用本次启动的限核预算；choices、编译映射与 launcher 一致，不能只读取整卡属性 |
| `BLOCK_SUB=B`、compile mode | 控制分片内循环与局部资源；首期固定 SIMD，非 persistent |
| combine 方式 | 首期 `terminal_owner`：每个输出只有一个分片读取 partial 并输出 |
| partial stride / workspace 大小 | 按候选最大 `S` 分配，行 stride 固定，实际只读当前有效分片 |
| 同步/映射契约 | 原生 `all_vector`、统一单 phase；实际 program 映射由 TA/NPUIR 保证并验证 |

`S/B`、combine 方式、有效核数、影响映射的 options 及相关能力版本须进入对应代码/候选缓存标识，
cache hit 后恢复同样的 Grid、workspace stride 与 compile options；不能只恢复一个 `R` block。
运行时 device/stream/scope 限核状态改变时，先重新校验计划及编译映射，不能直接复用旧的 cooperative 候选。
首期可限定已验证的软件栈和目标架构，不将新增通用 capability 查询服务作为开工前提。

### 1.3 Kernel 内协议与运行流程

设输出 program 数为 `Q=ceil_div(X,XBLOCK)`，Grid 为 `(S, Q, 1)`；首期 `XBLOCK=1`，所以 `Q=X`。
首期仅接 TA 确认的单轮映射，要求 `Q*S<=C_eff`，并确认没有额外 program 循环或 sub-block 改写参与者。
数量检查是必要条件，**不代替后端对物理映射与屏障参与者的确认**。

例如有效预算为 8 个 Vector 核，`X=1` 时 `S<=8`，`X=2` 时 `S<=4`。
在本卡 1:2 配比下，8 个 Vector 核对应 `NPU_DEVICE_LIMIT=4,8`；`8,16` 代表 16 个 Vector 核。
编译器的 `--custom-aiv-number`、launcher 和候选过滤必须使用一致限额；环境变量之外的
device/stream/scope 限核需单独确认如何传递。8 核实测中，`X=1,S=8`、`X=2,S=4` 均通过；
`X=1,S=16` 的两种 all_vector 写法均出现首次 sum、换输入和严格整数交换错误。
`X=2,S=8` 的 16 个 program 则通过全部检查，说明组完整的多批映射也能出现正确结果。
因此 X=1/2 时分别采用 `S<=8`、`S<=4`，这是首期单批方案的限制，不是所有算法的分片上限；
多批映射不在首期默认准入。
详见[限核依据与多批边界](barrier-comparison.md#73-限核后如何约束分片数)。

11 月 21 日已实测：默认 AutoBlockify 将单行的 112 个分片映射到 56 核时，两种 all_vector
helper 均产生 1736/1792 个错误整数结果；增加两侧本核屏障不能修正。实际编译产物按步长 56
循环执行完整 program，第一批屏障返回时后一批尚未写入 partial，见
[分核边界及产物](barrier-comparison.md#62-分核边界)。接入时必须在预运行/autotune 前校验
并拒绝这类映射，force 也不绕过；不能只因 TA 支持该同步 API 就放开任意 Grid。

保留上游调用形式的 NPU helper 原型可以写为：

```python
import triton
from triton.language.extra.cann import extension


@triton.jit
def x_grid_barrier(sem):
    extension.sync_block_all("all_vector", 0)
```

`sem` 在此仅用于兼容调用签名，不读写同步状态；到达登记、atomic 返回旧值和轮询均被
原生全 Vector 屏障替代。完整原型见[NPU helper](../../../scripts/tests/cooperative_reductions/npu_triton_helpers.py)。
该改写要求所有参与者按相同顺序、相同次数执行屏障，并将事件 0 留给此协议。
同一事件编号的连续复用由专门的数据交换探针验证，不能仅根据单次 sum 成功推定。

**本核 partial 的写回如何完成。** `tl.store` 发出写回操作，不代表异步搬出已完成；
这里仍需要本核流水完成与跨核会合两层同步。A5 SIMD 的本次实际产物已包含这两层，
因此 DSL 无需再单独写一个 `tl.debug_barrier()` 来补充本核等待。

对已验证的 `(1,1048576),S=32` 改写版 adapter 和 metadata 进行仅编译导出，
使用受测 NPUIR 提交 `395c9c5646f94119a591794a02961b5801f057c0`。
以下是 partial store 与 peer load 之间的实际关键 IR，仅省略地址、类型和部分属性：

```text
call @store_ubuf_to_gm_1d_float(...)
hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
hivm.hir.pipe_barrier[<PIPE_ALL>]
hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0
hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 0
hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
call @load_gm_to_ubuf_1d_float_nocache(...)
```

| 操作 | 本次产物中的作用 |
| --- | --- |
| `store_ubuf_to_gm` | 通过 MTE3 将本片结果从核内 UB 搬到其他核可读取的 GM；发出与完成是两件事 |
| `pipe_barrier[PIPE_ALL]` | 约束本核所有流水的前序读写完成，包括前面的 MTE3 搬出；不是等待其他核 |
| `sync_block_set`，生产流水为 MTE3 | 本核前序 MTE3 任务完成后才上报到达，不能只根据标量指令执行位置上报 |
| `sync_block_wait` | 等待所有参与 Vector 核上报完成，之后才允许本核继续下发后续指令 |
| MTE3→MTE2 的 `set_flag/wait_flag` | 维护本核搬出与后续搬入的依赖；这一对本核事件不能单独代替跨核会合 |
| `load_gm_to_ubuf_nocache` | 本次实际生成的 peer 读取路径；随后再同步 MTE2 与 Vector 合并计算 |

流水异步执行：V 负责向量计算，MTE3 负责 UB→GM，MTE2 负责 GM→UB，S 负责标量控制。
标量已经发出 store 时，MTE3 仍可能在搬运，故不能用源码中的先后顺序证明数据已就绪。
`PIPE_ALL` 的本核等待和绑定 MTE3 的跨核通知一起保证“写回完成后才上报”；
全体等待再保证“所有分片上报后才读取”。例如核 0 已写完、核 1 仍在搬出时，
核 0 必须停在跨核 wait，不能因为自己的本核屏障已完成就读取核 1 的 partial。
相关硬件语义见[流水与同步说明][sync-overview]、[PipeBarrier 参数说明][pipe-barrier]和
[CrossCoreSetFlag/WaitFlag 说明][cross-core-sync]。

LLVM IR 中也保留了 `llvm.hivm.BARRIER`、`llvm.hivm.SET.CROSS.CORE` 与
`llvm.hivm.WAIT.FLAG.DEV.PIPE.IMM`，不是只有 adapter 上的抽象 `PIPE_ALL` 字段。
完整证据见[后端 MLIR][native-mlir]、[LLVM IR][native-llvm]与[仅编译记录][native-compile]。
上述导出仅提供编译证据；运行精度、性能与变化数据交换结果见[同步验证](barrier-comparison.md)。
两类证据共同限定当前 A5 SIMD 路径的支持范围。

`tl.debug_barrier()` 的 Triton 接口语义是同步一个 block 内的线程，见[接口定义][debug-barrier]；
它不是唯一能表达本核同步的方式，也不能单独同步其他 block。
当前结论限定于已检查的 A5 SIMD 生成路径，不推导其他编译模式具有相同 lowering。

**Inductor 依赖什么保障，而不是如何猜测产物。** 同步原语的接口语义是上层和后端之间的契约。
已安装 TA 的 `sync_block_all` 明确说明，它处理指定核类型之间共享 GM 的 RAW、WAR、WAW
依赖，即写后读、读后写、写后写；不是仅让各核的标量指令到达同一源码位置。
接口定义及参数见[已安装接口片段][installed-sync-api]。

受测 NPUIR 的 `HIVMDecomposeOp.cpp::insertBlockAllVector()` 有如下固定 lowering 规则，
这里仅保留关键语句：

```cpp
auto tpipeAttr = op.getTvectorPipeAttr();
auto pipeAttr = op.getVectorPipeAttr() ? op.getVectorPipeAttr()
                                       : op.getTvectorPipeAttr();
rewriter.create<PipeBarrierOp>(loc, tpipeAttr);
if (tpipeAttr.getPipe() == PIPE::PIPE_ALL) {
  tpipeAttr = PipeAttr::get(ctx, PIPE::PIPE_MTE3);
}
if (pipeAttr.getPipe() == PIPE::PIPE_ALL) {
  pipeAttr = PipeAttr::get(ctx, PIPE::PIPE_S);
}
```

随后 `insertBlockSyncOperations()` 生成跨核 set/wait。当前 adapter 的输入 pipe 为
`PIPE_ALL`，所以这里明确生成本核全流水屏障，再使用 MTE3 通知、S 等待；不是 shape
启发式恰好插入了一条屏障。完整源码见[对应编译器版本的 lowering][sync-lowering-source]。
同步 op 还携带 `HIVMSyncOpTrait`，供分析识别不能随意跨越的同步边界，见
[同步 op 定义][sync-op-source]与[同步边界 trait][sync-trait-source]。
这些是现成的接口、IR 和 lowering 机制；不能据此声称所有自定义优化 pass 都已完成验证。

| 层次 | 应负责的保证 |
| --- | --- |
| Inductor choices/schedule/codegen | 选择合法核集合和模式；partial 先写、同步后读；所有参与者执行一致的屏障序列 |
| TA | 将 DSL 同步原语保留为有同步语义的 IR，维护依赖与执行映射 |
| NPUIR | 按接口语义实现流水完成、跨核通知与等待，优化后仍保持相关内存顺序 |
| launcher/runtime | 正确启动参与者、传递所需 ABI，并满足跨核等待的执行前进条件 |

Inductor 不应在正常编译时解析 LLVM IR 来判断“这次有没有某条 barrier”，也不应把普通
`tl.store` 后的自动依赖分析当成任意跨核通信的保证；应显式发射适用的同步原语。
对接阶段核对 IR 和数据交换探针用于确认实现符合契约，后续后端也可以用等价机制实现该语义，
无需永久保持同一条指令。现有接口不会替 Inductor 自动证明任意 Grid、分支和循环均满足全核会合条件。

**当前改写改变了 helper 的通用语义。** 上游定义是相同 y/z 下的所有 x program 会合，
每组有自己的 semaphore；改写后 `sem` 不参与协议，所有参与 Vector 核共用事件 0。
因此本原型验证的是受限 cooperative 路径的结果交换，不能作为上游 helper 的通用等价替换。
正式接入宜使用表达 NPU 全 Vector 语义的专用 helper，只在已满足条件的 codegen 路径调用，
不全局替换 `triton_helpers.x_grid_barrier`。
全仓调用核对表明，PyTorch `v2.13.0` 只有 cooperative 的一个生产发射点，但它覆盖 mean、
var/std、softmax 和多阶段融合等场景；连续 8 次 softmax 的测试会产生 8 或 16 次屏障。
split scan 与 combo/foreach 显式关闭该路径，见[实际调用范围](barrier-comparison.md#71-v2130-中哪些路径实际调用-x_grid_barrier)。

例如 `Q=4,S=8` 时，上游四个八分片组可以独立同步；改写后 32 个参与者一起等待。
各组都执行同样八轮交换时可以验证结果等价；若一组提前退出或少执行一轮，全体同步条件
就不再成立。因此这是受约束的 NPU 实现原型，完整接入仍需 schedule/codegen 保证调用条件。

一个说明 helper 语义差异的例子是：只有 `row==0` 的组调用屏障，其他组继续执行。
只要该组所有 x 分片均到达，上游的按组屏障允许这样调用；改成全 Vector 屏障则会等待
根本不调用它的其他组。这是通用替换不成立的反例，不表示当前上游调用点已经生成了该分支。
上游当前在 cooperative 的归约循环结束处、`HAS_RSPLIT` 为真时统一调用，见
[helper 定义][upstream-x-grid]与[codegen 调用点][upstream-barrier-call]。

| 需要保留的条件 | 失去条件的后果与处理 |
| --- | --- |
| 统一的屏障次数与顺序 | 提前退出、按输出组决定是否调用、屏障所在循环次数不同会破坏全体会合；尾块只 mask 数据访问 |
| program 与物理参与者映射明确 | 自动分核、sub-block 或一个核循环处理多个 program 可能改变参与者；仅检查 `Q*S` 不够 |
| 核间事件 0 不被另一协议占用 | 不同 semaphore 原本区分的实例现在合并，需隔离或统一规划核间 flag；本核 event 0 不是同一命名空间 |
| workspace 读者完成后再复用 | 写后读屏障不代表所有核已读完；多 phase 复用需另有读后写依赖保障 |
| 启动与并发满足执行前进条件 | 部分核等待、其余核因资源占用无法启动时可能死锁；CUDA launch 标记不能代替 NPU 启动契约 |
| 评估全局等待与 PIPE_ALL 成本 | 各输出组互相等待，全流水屏障也会约束重叠；即使结果相同，性能也不等价 |

这里要求的是屏障序列一致，各组屏障之前的计算耗时可以不同；归约 tile 循环若不含屏障，
其迭代次数不同本身不构成会合错误。首期 `Q=1`、单 phase、映射已确认时，全 Vector 集合
可对应唯一的分片组；扩展 `Q>1` 则必须复核全体会合条件。并发启动约束见
[跨核同步的执行前进说明][cross-core-sync]，多 phase 的空间复用还要与上游
[workspace 两轮复用规则][upstream-workspace]对齐。

增加本核屏障不能修正尚未执行的后续分片。若以后支持同组跨批，需要将“全部 partial 任务”
与“同步后合并”分为两个执行阶段，并由 TA/NPUIR 保证屏障位于任务循环之外；不是仅替换 helper。
该结构尚未实施，首期仍按上述单批协议接入；见[补充逻辑与正确性边界](barrier-comparison.md#74-sync_block_all-版增加什么逻辑才能保证正确)。

![cooperative 单 phase 运行流程](assets/kernel-flow.svg)

<details markdown="1">
<summary>流程图 Mermaid 源码</summary>

```mermaid
flowchart TD
    A["所有 program：计算 row / shard / 输入区间"] --> B["本片 prologue、BLOCK_SUB 循环与局部 sum"]
    B --> C["每个分片写 W[row, shard]；空片写 identity 0"]
    C --> D["所有参与者无条件执行原生 all_vector barrier"]
    D --> E{"shard 等于 0？"}
    E -->|是| F["读取同组有效 partial；合并与标量 epilogue；唯一写回"]
    E -->|否| G["本 program 完成"]
    F --> H["整个 kernel 完成；stream 顺序保护 workspace 生命周期"]
    G --> H
```

</details>

分片区间沿用已通过原型的形式：`chunk=ceil_div(R, S*B)*B`，分片 `s` 处理
`[s*chunk, min((s+1)*chunk, R))`，load 使用有效元素 mask。每个分片将局部 sum 写入自己的槽；
空片仍写 0 并到达屏障。候选生成尽量排除工作过小或过多空片的配置，mask/identity 保证尾部语义。

**Owner 判断必须位于屏障之后，并包住 peer load、combine 和最终 store。** 只 mask store 会
保留 G3 的重复工作。单 phase 的终结归约在 combine 后没有其他分片继续读取/覆盖 workspace，
因此不额外增加一次全体 barrier；wrapper 在整个 kernel 结束后按 stream 管理空间。

此协议支持纯 prologue 与仅依赖最终归约值的标量 epilogue。Softmax 等消费者还需每个分片获得
最终值并继续处理自己的输入片段；后续再接 all-read/广播以及多 phase 的读者完成屏障，
不能直接使用当前 owner-only 协议。该优化尚未实施或测速，已有原型仍使用 all-read。

### 1.4 Tiling、autotune 与运行时

收益选择先看实际输出任务数 `Q=ceil_div(X,XBLOCK)`、有效 AIV 数 `C_eff` 与每片工作量。
输出并行度扫描中，X=7 长轴的 A5 native 最优已测 S=8，CUDA original 为 S=32；
A5 的 X=56、GPU 的 X=128 已有足够输出任务，继续拆 R 没有收益。X=56 在 GPU 仍未满核，需按设备分别判断。
NPU 的 24 个错误点也已计时，但它们不能加入合法 autotune 候选，修正协议后必须重测。
场景、普通基线、统一扫描及耗时限制见[性能结果](barrier-comparison.md#42-非规约轴未满核与满核)。

限制为 8 个 Vector 核后，`X=1/2` 长轴的 native 最优已测 S 分别为 8/4，相对各自 plain
为 6.74/3.61 倍。`X=2,S=8` 虽然正确，但短/长轴均慢于 S=4，不能将可执行上限当作最优分片数。
`X=8` 长轴的正确多批配置 S=2 只观察到约 1.03 倍差异，短轴为 0.78 倍；
这不足以作为满核时默认拆 R 的依据，仍需实际 Inductor 的重复性能验收。

因此先以 `Q*S<=C_eff` 约束首期单批候选，再校准 S/B 的收益；不能直接采用 CUDA 的最优 S，
也不能把 helper S=1 当作普通归约基线。正式比较应保留普通路径和可用 split 的实际选中配置，
不能将固定 BLOCK 的手写测量或尚未定位的物理卡耗时差异固化为生产阈值。

候选先从 `S` 的小集合与保守 `B` 组合开始，按有效核数、输入区间、UB/活跃值和已确认的映射
契约过滤；数量与范围通过第 4 节实验校准。`S=1` 在准入阶段回到普通 kernel，不分配协议
workspace，也不发出屏障。

映射检查同时存在组完整时通过、组跨批时出错的多批配置。首期仍保持第 1.3 节的单批
保守约束；不能只因为某个 S 整除核数且样例通过，就开放所有超核配置。

Cooperative 候选在相同算法/合法参与者约束下编译和计时；编译或资源检查不通过时淘汰候选，
全失败时走普通路径。Partial workspace 按候选最大需求分配，固定 stride，确保候选预运行和
winner 使用一致的布局。全部有效 partial 由本次写入，无需为 sum 额外清零整个 workspace。

Inductor 管理显式 partial workspace；TA/NPUIR 管理原生屏障所需的隐式 scratch/lock metadata。
两者通过现有 wrapper/launcher ABI 传递，Inductor 不自行猜测锁个数或增加 CUDA semaphore。
Python wrapper 和 NPU Graph 作为首期运行验收，C++/AOT、多 stream 并发在后续阶段单独验收。

### 1.5 对 TA / NPUIR 的周边依赖

下表中“契约确认”表示已有基础但还需明确边界，并非要求从零新增 API；“性能优化”表示需先
定位和建立基线，再决定是否修改下层。TA 指 Triton-Ascend，NPUIR 指实际设备编译器。

| 依赖 | 需要提供或确认的内容 | 责任接口 | 是否阻塞首期 |
| --- | --- | --- | --- |
| D1：参与者与映射契约 | 对齐限核后的编译/启动预算；确认 Grid/AIV、auto-map/sub-block 与屏障一致性 | TA launcher + NPUIR 映射 | 是；合法性未明确时不能发射 |
| D2：GM/event 契约 | 屏障保证 partial 可见；明确重排、event 分配/重用与自动同步的约束 | TA 同步依赖 + NPUIR lowering | 是；受限运行通过后仍需明确边界 |
| D3：owner-only 控制流 | 屏障后 owner 分支读/合并/输出，其他参与者完成；屏障不移入分支 | TA 控制流 + NPUIR 优化 | 是；功能与 G3 优化共同依赖 |
| D4：资源与 ABI | 区分 UB 用量/容量/unknown；返回同步 scratch/lock，保持缓存/Graph ABI | NPUIR metadata → TA → runtime | ABI 影响功能；用量影响配置 |
| D5：同步与小 partial 成本 | 定界 G1/G2/G4/G7/G8；直接复用 semaphore 时解决返回值 atomic 的软件锁成本 | NPUIR 主导，TA 提供固定对照 | 原生同步原型可继续；所选协议的性能影响默认启用 |
| D6：模式、数据保留与流水 | 验收 SIMD multibuffer；评估 persistent 片内值跨同步保留的资源成本与连续归约融合；后续补齐 pure SIMT 同步 | TA options + NPUIR pipeline/资源反馈 | SIMD 组合需验收；persistent 与 pure SIMT 不阻塞首期 |
| D7：分组同步或更轻的归约交换 | 若全 Grid phase 在多输出/多 phase 形态中成为瓶颈，再评估分组或其他交换方式 | TA API/IR + NPUIR 同步实现 | 不阻塞首期；仅作为后续按证据提出的优化 |

交付给 TA/NPUIR 的最小输入是：固定 target/mode/constexpr 的 DSL、参与者/Grid、输入与输出
语义、TTIR/adapter、编译 metadata，以及同算法分阶段计时。已有[探针脚本][P1]、
[编译矩阵][E3]和[adapter IR][E13]可作为起点。
下层返回应包含支持组合、最终映射/资源信息、定位结论与对齐基线的性能数据，不能只返回
“能编译”或一个未经对齐的算子耗时。未确认的能力先采用明确范围保护，不据此默认支持所有 NPU。

## 2. 组件修改清单

“必须改”指实现 A5 SIMD、静态、单归约 phase 的最小范围；“检查/扩展”按验收范围决定。
文件名均相对 `pytorch_new/torch_npu/_inductor/`，上游复用点另行注明。

| 组件 | 需要完成的工作 | 范围判断 |
| --- | --- | --- |
| `choices.py`、配置与 backend capability | 增加 NPU 准入和收益判定；架构/模式/算子不支持时回退；IR 与 codegen 共用判定 | 必须改，中等 |
| `ir.py::num_splits` | cooperative 准入与 IR capability 一致；未准入时保留当前普通路径 | 必须改，中等，影响方案选择 |
| `codegen/npu_combined_scheduling.py` | Index 接 cooperative；no-linear 显式关闭；失败尝试不污染普通回退 | 必须改，中等 |
| `codegen/scheduling.py` | kernel choices、fusion 准入、reduction/pointwise phase 边界、unsupported 组合屏蔽 | 必须改，中到大 |
| `codegen/split_tiling.py` | 为分片安排 program 维度；区分输出分核与归约分片；保证参与者和循环次数合法 | 必须改，大 |
| `codegen/triton.py` | Index 接 shard 区间、workspace、原生屏障、owner combine、尾块；no-linear 先仅做能力保护 | 必须改，大，主工作量 |
| `runtime/tile_generator.py` | 联合选择 `S`、输出 tile 和 `BLOCK_SUB`，按可用核数/局部资源筛选 | 必须改，中到大 |
| `runtime/triton_heuristics.py` | cooperative 候选、Grid、编译参数、实际 winner、cache 恢复；对性能比较明确候选层级 | 必须改，大 |
| 同步/combine helper（`runtime/triton_helpers.py`） | 原生屏障、空片 identity、peer 合并；也可由 codegen 直接发出已有扩展调用 | 协议必须接入；文件按方案改 |
| `codegen/wrapper.py` | 复用 `WorkspaceArg` 分配与释放，核对 stream 生命周期、参数和 runtime Grid | 必须核对，已有基础 |
| `codegen/cpp_wrapper_npu.py` | 同样传递计划、Grid、partial workspace 与编译器隐式 lock ABI | 完整 AOT/C++ 范围需要适配与验证 |
| 测试与 cache/config 接入 | 准入/回退、精度、动态、图捕获、候选一致性；配置进入图缓存与候选缓存键 | 必须改，贯穿各层 |

上游 `WorkspaceArg`、reduction combine helper、schedule marker、autotuner 编译/计时框架可以
复用。现有 NPU wrapper 已有 workspace 和编译器 lock 的处理，不应笼统认定需要重写整个
wrapper，见[C++ wrapper 实现][N7]。同样，正常 `torch.sum` 等 decomposition/ATen lowering 并非首期必须修改项；
只有既有 lowering 已将目标融合链提前拆开时，再按具体图评估。

**工作量是跨多个组件的编译策略与运行时功能，超过局部 codegen patch。** 单阶段 static sum
的受限版本与“上游测试全量通过、动态/多 phase/AOT 全兼容”是不同交付范围；现有证据不足以
给出可靠的行数或人天估算。

## 3. 性能模型与 NPU 触发条件建议

### 3.1 相对现有归约路径建立收益模型

以下是决策模型，不是已测得的硬件公式。记 `Q` 为实际输出方向 program 数，`C` 为该模式下
可用于该 kernel 的物理并行单元数，`L` 为一次 launch 的相关开销。

```text
T_ordinary ≈ L + local_reduce(R, B, live_values) with Q-way parallelism
T_split    ≈ partial_kernel + combine_kernel + consumer_kernels
                 + intermediate traffic and launch costs
T_coop     ≈ L + local_reduce(R/S, B, live_values) with Q*S-way parallelism
                 + partial_store + native_barrier + combine + resource/fusion effects
```

方案要求 cooperative 相对现有可用路径缩短完整图时间，基线包含已有 split；不能只与每行
单 program 的普通归约比较。分片计算减少的时间必须覆盖 partial、屏障、合并和资源开销。
产品收益比较须对齐相同图、输入、dtype 和运行条件，保留基线实际选中配置。
连续归约融合还会改变访存和活跃值，需另外测量，不能将单 sum 的收益直接推广到 softmax。

实际 CUDA Inductor 对照已说明这个区别：`(1,1048576)` sum 的 split 为 4.128 µs、cooperative
为 5.344 µs；`(1,32768)` 连续 4 次 softmax 分别为 18.170/7.210 µs。
因此静态 sum 可以作为协议接入入口，其默认启用价值尚未确立；整行消费者和连续归约融合
是必须单独评价的性能方向。单 kernel 长轴 softmax 仍会重读输入，并可能慢于 split。
流程、生成代码与完整数据见[整图策略对照](barrier-comparison.md#45-split-与-cooperative整图对照与分层组合)。
NPU 同算法原型中，短轴 RMSNorm 相对 split 约为 1.10 倍，长轴约持平；该原型会重读输入，
尚未验证跨同步数据保留的价值。融合阶段应先用完整图与资源证据确定收益，再决定扩大默认准入。

还应区分：

- `R` 很小：即使 `Q` 小，也可能是同步开销主导；
- `Q >= C`：输出方向已有并行度，再分 `R` 的价值变小；
- `S=1`：没有 cooperative 并行收益，应省掉同步和 peer 协议；
- `R/S` 太小：过多分片导致空片或小工作量 program；
- UB/寄存器不够：persistent 的资源风险与 `R`、dtype、融合输入和活跃值相关；
- 全局屏障：不同输出组互相等待，耗时由最慢参与者决定，跨组不均匀会增加尾部代价。

### 3.2 建议的准入顺序

```text
1. capability: SoC、compile mode、原生同步契约已验证
2. legality: 单归约形态、支持的 combine、统一 phase、限核后的合法 Grid/映射和 workspace
3. parallelism: 当前 Q 不足，存在 S > 1 且每片有足够工作量
4. resources: B、S、persistent/looping 的 UB 与寄存器资源合法
5. profitability: 预测或实测 cooperative 优于现有可用路径，包括已有 split
6. 否则回退；force 只绕过第 5 项，不能绕过前四项
```

首期可以静态 shape 和 SIMD 为边界，用保守的配置集比较 `S` 与 `BLOCK_SUB`，逐步建立
按 `X/R/dtype/layout/fusion/graph mode` 区分的收益表。`S` 上限应同时受实际 program 映射、
有效核数与最小分片工作量约束，不能只有 `min(R, 64)`。

CUDA 的 `X<=16`、`32768*X` 和以 SM 为基础的 `S` 不能直接作为 NPU 默认值。
设备属性与 compiler 保留空间也要分开：本卡报告 UB 为 216 KiB，而本地 NPU tile 配置给 A5
设置 256 KiB；这个配置常量不能单独证明一个 cooperative persistent tile 能放下，
见[tile generator 中的容量配置][N8]。

首期由 shape/能力与校准后的收益规则选择现有路径或 cooperative；cooperative 内部搜索合法的
`S/B`。现有 `cached_autotune` 不会自动比较两个不同 generated body 的完整代价，不能把加上
一个 `S` 参数视为已完成普通路径与 cooperative 的自动择优。
在特性开关默认关闭的前提下先完成接入与校准，再确定默认启用范围。

本方案不把同时开启两个开关解释为自动叠加。若后续实现显式分层，须先产生组 partial，
在组内选择 cooperative，再用独立 kernel 汇总；这是额外的图划分与性能选择问题。
CUDA 的 G=4 实验两点均未优于 split，当前不将其纳入首期实现。
NPU 还必须约束组数与组内分片映射，不能借外层分组绕过 all_vector 的合法参与集合要求。

autotune 预运行还应遵守 workspace/隐式 lock 的初始化契约：每个有效 partial 都由本次调用
覆盖；不额外引入 CUDA semaphore，也不让上一候选的状态影响下一候选。

## 4. 性能缺口的验证方法

保持相同 shape、dtype、分片算法和 compile mode，分别检查以下阶段，输出每个版本的结果校验：

| 对照 | 保持/增加的工作 | 用途 |
| --- | --- | --- |
| local partial | 每个 program 读取本片并写出 partial | 分片计算与写回的基线 |
| partial + native barrier | 相同 partial，加全体参与的屏障 | 观察完整代码变化后的同步增量，不将编译优化差异忽略 |
| barrier + owner combine | 指定分片读取同组 partial、合并、写输出 | 首期方案及小 partial 处理成本 |
| barrier + all-read combine | 每个分片均合并，指定分片写输出 | 识别终结归约重复工作；也是后续广播消费者的起点 |

`S` 按 2/4/8/16/32，`B` 按保守合法范围选点；`S=1` 单独使用无协议的普通 kernel。
每个 case 要保证真实参与者合法，区分整除/尾块、冷热缓存、Graph 开关，并在可控负载下配对测量。
如果 profiler 支持，采集实际 AIV 活跃度、各 pipe 等待、GM 事务和 UB 使用；否则先用最终指令、
metadata 与阶段实验限定结论。TA/NPUIR 提供等价原生参考实现后，才能进一步评估编译器带来的性能差距。
这些阶段对照仍是验证计划，现有 all-read 数据不能代表 owner-only 或其他下层优化的收益。

### 4.1 流水是什么，以及如何观测

这里的“流水”是 AI Core 内分工执行指令的通路及队列，不是 Inductor 的编译 pass pipeline。
V 做向量计算，MTE2 将 GM 数据搬入 UB，MTE3 将 UB 数据搬出到 GM，S 负责标量控制。
这些单元可以异步并行：例如计算 tile 0 时预取 tile 1；同一 tile 的“搬入→计算→搬出”
仍需依赖控制。`PIPE_ALL` 是选取本核所有流水的同步参数，不是另一种计算单元。
硬件分工与异步下发机制见[同步控制说明][sync-overview]。

| 需要回答的问题 | 证据与工具 | 不能单独证明什么 |
| --- | --- | --- |
| 编译器安排了哪些同步 | 后端 MLIR、LLVM IR，必要时检查设备反汇编 | IR 的文本顺序不能给出各流水实际耗时与重叠 |
| 哪条通路忙、哪条可能受限 | 上板 `PipeUtilization`，结合内存流量与核间负载 | 汇总利用率不是每条指令的执行时间线 |
| 各流水什么时候工作、空闲和重叠 | A5 上板 `PipeTimeline`，在 MindStudio Insight 中查看 | 插桩采集的时间不可直接替代未插桩 Graph + Event 样本 |
| 进一步定位到具体指令 | 支持版本中的上板 `InstrTimeline`；按需保留源码行号 | 不能把未记录通路或未显示指令的空白一律当作 barrier 等待 |
| 分析仿真器中的指令顺序与依赖 | `msprof op simulator` / `msopprof simulator` 导出 trace | 仿真时间不是这块物理卡的实测时间 |

现有结果包含后端 IR 和 Graph + Event 计时，尚未抓取该 kernel 的上板或仿真指令时间线。
要进一步定界 G2，可先抓流水重叠，再关联 MTE3 写回、跨核等待、MTE2 peer 读取；
需要同时看多个参与核，区分本核搬出慢和等待其他核两种来源。
入口与示例命令见[探针 README 的流水采集说明][pipeline-commands]。
工具的 A5 支持、`PipeTimeline`/`InstrTimeline` 参数及两者分开采集的约束见
[MindStudio 算子工具指南][profiler-guide]；Triton 仿真方法见[TA 性能分析指南][ta-profiling]。

### 4.2 全形状性能测试方案调整

**下一轮以实际 Inductor 产物作为 NPU plain 和 GPU 的测试对象；NPU cooperative 暂用调优后的 DSL。**
现有 270 行结果来自两端手写 kernel，plain 固定 `BLOCK=1024`，cooperative 只搜索 RS。
它回答固定算法下分片、同步和编译映射的影响，不能说明相对实际 Inductor 配置的收益。
统一测试入口已实现，使用方法见[多卡一键测试](../../../scripts/tests/cooperative_reductions/COMPILE_MATRIX_README.md)。
现有全形状 CSV/SVG 仍代表固定 BLOCK 的 A5/L20 实验，新入口的全量结果待采集。
小范围验证的 4090 双卡 8 个、A5 单卡 6 个主结果均通过；另确认超核 debug 错误保留耗时并排除出最佳配置。
实际命令、搜索配置及生成产物见[测试包验证记录](../../../artifacts/cooperative-compile-matrix-20261123/README.md)。

#### 测试分组与编译入口

保留 fp32、sum/amax、`X=1/7/56`、`R=4K～1M` 逐档翻倍，共 54 个操作/形状组合。
两端从相同的 PyTorch 计算定义出发：

```python
def sum_fn(a, b):
    return (a + b).sum(dim=-1)


def amax_fn(a, b):
    return (a + b).amax(dim=-1)
```

| 结果组 | 测试入口与策略 | 配置选择 |
| --- | --- | --- |
| NPU plain | `torch.compile`；关闭 cooperative、force 和上游 split reduction | NPU Inductor 的 tiling/autotune |
| NPU origin | 手写 DSL，保留原始 semaphore 协议 | 联合搜索 RS、BLOCK |
| NPU debug | 手写 DSL，保留两侧本核屏障与 `all_vector` | 独立搜索相同的 RS、BLOCK 集合 |
| GPU plain | `torch.compile`；关闭 cooperative、force 和上游 split reduction | CUDA Inductor 的 tiling/autotune |
| GPU cooperative | `torch.compile`；开启 cooperative 和 force，关闭上游 split reduction | 上游 RS 选择与局部配置调优 |

编译入口统一使用 `backend="inductor", fullgraph=True, dynamic=False`，并显式设置
`max_autotune=True`。不使用会额外改变 Graph 策略的编译 mode；Graph 计时由测试程序统一控制。
CUDA 显式关闭 `triton.cudagraphs`，NPU 同样关闭后端内部的自动 Graph 包装，避免重复捕获。
记录后端实际生成和测过的候选数量，不能仅凭设置了 `max_autotune` 就称为充分调优。
NPU 若使用额外的 tiling 搜索开关，也须记录并固定；不将编译器内部候选强行改成 DSL 的配置集合。

这里的 plain 表示关闭上述两种上游策略，**不要求编译产物必须是一行一个 program、一个 kernel，
或与旧手写 plain 使用相同 BLOCK**。NPU 已有的分核、向量化和融合照常保留，以实际产物解释性能。
NPU baseline 不通过强开尚未实现的 cooperative 来测试，也不手工改写其生成 DSL。
本轮先隔离 cooperative 与普通归约的差异；产品默认启用前，仍须与后端现有默认策略比较，
包括可用 split。该验收要求见第 3.1 节，本轮不扩展 split 的独立参数扫描。

**GPU 的强制测试与自然选择分开记录。** 当前上游选择规则为：`X<=8` 时要求 `R>=32768*X`；
`8<X<=16` 时要求 `R>=2097152`；`X>16` 不自然选择 cooperative。
本矩阵中 X=1 从 R=32K 起满足轴长条件，X=7 从 R=256K 起满足，X=56 全部不满足。
因此仅打开开关无法得到全 54 组 cooperative 性能；全覆盖使用 force，并在表头、CSV 和图例中写明。
另以 `force=False` 编译、执行并核对生成路径，保存自然命中与回退结果；不把普通路径的耗时填成
cooperative 耗时。实际 `RS=1`、屏障被消去的情况也单独标注为未分片。
规则对应 PyTorch `v2.13.0` 的提交 `cf30153c4c131c8164ee7798e5022d810682e2cb`，
见[选择函数][matrix-choices]。

GPU 仍由上游按设备 SM 数和 X 选择 RS，再调优局部归约配置；**本轮不手工修改 GPU 的 RS**。
该实现为局部配置候选填入相同 RS，并不默认比较多个 RS，见
[cooperative runtime 实现][matrix-runtime]。
因此新表写“实际选中配置”，不能继续把 GPU 结果称为“搜索后的最佳 RS”。
跨 GPU 运行时记录实际安装版本与源码，不能假定其他版本使用相同的选择函数或候选集。

#### NPU DSL 如何手动调优

本轮保持 all-read 合并、连续分片、SIMD 模式和同步协议不变，先搜索影响明确的两个参数：

| 参数 | 初始候选 | 作用 |
| --- | --- | --- |
| RS | 1、2、4、8、16、32、48、56 | 每行分片数，改变并行度、每片工作量和同步参与者 |
| BLOCK | 256、512、1024、2048、4096、8192 | 每个 program 一轮处理的元素数，改变循环次数、归约代价和 UB 需求 |

RS 与 BLOCK 联合枚举，初始每种 helper、每种操作/形状最多 48 个配置，两种 helper 共 5184 个。
不能先在 BLOCK=1024 下选 RS，再只调整这个 RS 的 BLOCK，否则可能错过更好的组合。
若最佳 BLOCK 落在搜索边界，脚本标记 `block_search_boundary=true`；资源允许时，用新输出目录
向相邻档扩展 `--blocks` 再测。RS 可通过 `--splits` 补有效核预算边界及其附近的候选。
“最佳”限定为所公布搜索空间内、检查通过并复测后的最佳，不代表全局最优。

NPU SIMD 的 `num_warps` 先保持现有 DSL 的 4，并完整记录后端实际选项；
只有确认某个额外选项影响生成代码、资源或流水安排后，才把它加入搜索。
不直接照搬 CUDA 的 warps/stages 参数组合，也不同时更换合并算法或编译模式。
后续 owner-only、persistent 或数据保留优化另做算法对照，避免与本轮 tiling 收益混合。

每个配置先编译和验证，再粗测；每组最快的三个正确候选使用完整计时设置复测。
选中配置在新进程中恢复并复测至少三轮，plain 与候选交替测量，最终报告复测中位数与波动范围，
不将探索阶段偶然最低的耗时作为最终结果。原始候选、选择依据和复测结果均保留。

同步对照保留两个问题：同一 RS/BLOCK 的 origin/debug 差异，以及两者分别调优后的最佳性能。
前者用于分析同步与编译映射，后者用于评估各自实现能达到的水平；二者都包含完整 kernel，
不能解释为裸屏障指令的耗时。

**超核配置继续保留实验结果，同时标明接入条件。** debug 的首期可接入候选仍要求
`Q*RS<=C_eff` 且后端映射满足全体同步契约，本轮 DSL 的 `XBLOCK=1`，所以 `Q=X`。
例如 56 核、X=56 时没有 RS>1 的首期单批候选；实验通过的多批结果不能自动放入这一集合。
origin 按其 semaphore 协议与实际映射单独判断，不套用 debug 的同步语义。
已完成的错误点保留性能；超时点没有有效设备耗时。RS=1 的 DSL 保留作协议开销对照，
但不冒充无屏障 plain，也不计为实际分片收益。
汇总同时给出“最快正确配置”和“满足首期接入条件的最快配置”；后者无候选时写明原因，
不删掉超核、错误或超时证据。NPU RS=64 仍沿用独立诊断范围，不自动加入本轮全量调优。

#### 统一测量范围与正确性

compile 分组测量整个编译 callable，包含该图实际生成的全部设备 kernel 和必要的初始化操作；
DSL 分组测量同一计算的完整调用，包括算法要求的 workspace/semaphore 设备初始化。
合法复用同步状态时记录其复用规则。内存分配策略与临时空间大小也要保存，不能只取最快的局部 kernel。
编译、autotune、CPU 参考、输入构造和结果回传放在计时外；编译/调优耗时单独报告。

两端使用相同种子、输入内容与 CPU 参考：先执行 fp32 加法，再用 fp64 sum 或 amax 作参考。
sum 继续采用 `atol=0.02, rtol=2e-4`，有限输入的 amax 要求精确一致。
每个配置检查首次输出、最后一次 Graph replay 输出和两组变化输入；DSL cooperative 另外执行
相同配置下的多 phase 严格整数交换，验证 partial 可见性与 workspace 重用。
检查失败但成功完成的配置可保留诊断时间，不能参与最快正确配置选择。

主结果仍使用 Graph + device Event、每图 16 次调用、每样本 10 次 replay、7 个样本中位数；
慢点沿用按 pilot 减少 batch/replays 的规则，并记录实际值。
compile 的自动调优和所有懒初始化须在捕获前完成；测试程序负责捕获，避免叠加内部 Graph replay。
这是相同地址反复运行的缓存口径，跨设备比较保留各自缓存差异，不据此推导冷数据性能。
无 Graph 的完整调用开销用于后续验收，不混入本轮主曲线。

NPU plain 与 DSL 是不同生成路径，其差异还可能包含融合、编译模式、分核与访存安排。
新结果能说明“当前实际普通路径”与“已调优 cooperative 原型”的差距，
仍不能称为 NPU Inductor 已集成 cooperative 的实测收益。
GPU 则测实际 Inductor 的两条路径，但强制矩阵的收益不等于默认选择收益。

#### 脚本改造与结果更新

保留旧固定 BLOCK 入口，新增统一采集与绘图脚本，便携包包含以下能力。

| 组件 | 已实现内容 |
| --- | --- |
| `run_matrix.sh` / `run_matrix.py` | 多卡分配完整形状组；实际 compile 与 DSL 联合搜索、独立复测、失败隔离和续跑 |
| `reduction_matrix.py` | 复用原 DSL、计时及检查；新增逐配置 DSL/IR 导出，旧固定 BLOCK 入口保留 |
| 配置与产物保存 | 复用 `BestTilingRecorder`，记录选中及编译候选、实际路径、wrapper/DSL、完整 metadata 和可导出 IR |
| `plot_compile_matrix.py` | 独立汇总 compile / DSL，导出主表、全部候选、自然选择及同 RS/BLOCK 同步对照 |
| `pack_matrix.py` 与 README | 独立便携包，不依赖源码仓；CUDA 不导入 torch_npu；同包支持 CUDA/NPU |

NPU 配置采集可复用已有
[BestTilingRecorder](../../../scripts/tools/autotune_tiling.py)，
并保存实际参数名及值，例如 `R0BLOCK_SUB`；不将 CUDA 的 `RBLOCK` 与 NPU 的字段强行视作同一配置。
旧 DSL 入口与原数据继续可复现，新增记录用 `entry`、策略开关和搜索空间区分来源。
每个候选至少保存完整配置、Grid、有效核数、编译映射信息、精度状态和计时；选中候选保存可重放的
wrapper/DSL、全部 launch 配置、完整 metadata 和可导出的 IR。
所有 cooperative 候选均补充可用 IR 导出，不能继续只有 `ir_sha256` 而没有对应文件。

新建结果 CSV，五个主分组仍各覆盖 54 组，共 270 行；无正确 cooperative 候选的行保留状态。
CSV 包含入口、是否 force、实际路径、选中配置、搜索范围、正确性、接入条件、延迟和同设备
`plain / candidate` 比值。自然选择另存记录，不将回退行归入 cooperative 曲线。
SVG 给出五组延迟、同设备收益，以及 NPU 的 RS/BLOCK 选择；GPU 图例明确“Inductor 强制 cooperative”。
同配置同步对照与超核错误耗时作为机制证据单独展示，不与正确收益混画。

新一轮验证完整后，再直接改写性能报告的主表、图和结论；旧固定 BLOCK 结果保留为机制证据。
旧矩阵不与新 compile 数据拼接，不在尚未重测时改写已有数值或给旧数据换入口标签。

[matrix-choices]:
  https://github.com/pytorch/pytorch/blob/cf30153c4c13/torch/_inductor/choices.py#L390
[matrix-runtime]:
  https://github.com/pytorch/pytorch/blob/cf30153c4c13/torch/_inductor/runtime/triton_heuristics.py#L4632

## 5. 建议交付阶段与验收

| 阶段 | 交付边界 | 验收依据 |
| --- | --- | --- |
| P0：能力契约 | A5 SIMD，同步参与者/GM/映射/锁契约；pure SIMT 明确排除 | 原生屏障、重复 phase、合法/非法 Grid、缓存与并发约束记录 |
| P1：静态最小接入 | Index、静态单 `R`、fp32 sum、非 persistent、owner-only、简单融合；no-linear 回退 | 分核、精度、协议、命中与回退均验证 |
| P2：性能缺口与选择 | `S/B` 联合选择；普通/cooperative 同图比较，拆分协议成本 | 完成 G1～G8 的归因/范围记录，冷热缓存、Graph 开关、winner 与有效核数齐全 |
| P3：融合与算子 | 连续归约/softmax、极值、类型提升、Welford 等逐项扩展 | tail、NaN/Inf、tie、空片、各 phase 空间与精度验收 |
| P4：动态 | backed dynamic R，再到有安全上界的 X 和 grouped autotune | 同一 callable 跨阈值/尾块复用；Grid、空间与回退合法，记录重编译 |
| P5：完整运行时 | C++ wrapper/AOT、Graph、多 stream、persistent、其他编译模式 | 打包重载与并发回放；能力矩阵与默认回退测试 |

验收不能只检查字符串 `cooperative_reduction` 或 kernel 数为 1。
原用例 `test_reduction_fns` 的 `(1, 1024**2)`、多个 reduction 与 dtype 是回归入口，但还包含
CUDA decorator/launch 假设。NPU 测试应检查自己的计划、partial 形状、同步作用域、store owner
和正确回退；浮点类型的基础硬件支持与 cooperative 功能分别判断。[C1]

最低测试集合应覆盖：`S=1/S>1`、`R<S*B`、非整除 `R`、输出尾块、`X` 超过准入上限、
unsupported 算子/模式回退、同一 kernel 多次调用与 workspace 重用、多个 reduction phase。
不能依赖设备超时来拒绝非法 Grid；不满足同步契约的配置应在编译或候选筛选阶段排除。

## 6. 关联文档与证据

需求价值、NPU 当前实现、上游差异、功能/性能缺口、正交矩阵与验收目标统一见[需求分析](requirements.md)。
源码链接沿用该文档的固定基线；方案尚未接入 Inductor，已有手写 kernel 测量只作为设计依据。

- [实验脚本与复现方式][P1]。
- [Index 初始化与 no-linear 回退失败记录][E7]。
- [固定配置编译矩阵][E3]、[安装后端契约片段][E1b]、[原型 adapter IR][E13]。

[N1]: https://gitcode.com/luqichao/pytorch_new/blob/961b95d9acb79551c52a77e991db9dd2613d6d86/torch_npu/_inductor/codegen/npu_combined_scheduling.py
[N3]: https://gitcode.com/luqichao/pytorch_new/blob/961b95d9acb79551c52a77e991db9dd2613d6d86/torch_npu/_inductor/codegen/triton.py#L902
[N7]: https://gitcode.com/luqichao/pytorch_new/blob/961b95d9acb79551c52a77e991db9dd2613d6d86/torch_npu/_inductor/codegen/cpp_wrapper_npu.py
[N8]: https://gitcode.com/luqichao/pytorch_new/blob/961b95d9acb79551c52a77e991db9dd2613d6d86/torch_npu/_inductor/runtime/tile_generator.py#L109
[C1]: https://github.com/limbochao/Community-Use-Case-Testing/blob/51830b03317074409a04c0ff167ee0d7ad6c897c/v2.13.0/inductor_npu/test_cooperative_reductions.py
[P1]: ../../../scripts/tests/cooperative_reductions/README.md
[E1b]: ../../../artifacts/npu-cooperative-reductions-20261118/inspect/installed-backend-contracts.txt
[E3]: ../../../artifacts/npu-cooperative-reductions-20261118/compile-fixed/compile.json
[E7]: ../../../artifacts/npu-cooperative-reductions-20261118/index-fallback.log
[E13]: ../../../artifacts/npu-cooperative-reductions-20261118/compile-fixed/cooperative_sum-simd/kernel.ttadapter
[native-mlir]: ../../../artifacts/npu-cooperative-reductions-20261120-allvector/final-native/module.hivm.opt.mlir
[native-llvm]: ../../../artifacts/npu-cooperative-reductions-20261120-allvector/final-native/kernel.ll
[native-compile]: ../../../artifacts/npu-cooperative-reductions-20261120-allvector/final-native/manifest.json
[sync-overview]: https://www.hiascend.com/doc_center/source/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0179.html
[pipe-barrier]: https://www.hiascend.com/doc_center/source/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0271.html
[cross-core-sync]: https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreSetFlag_ISASI.html
[debug-barrier]: https://triton-lang.org/main/python-api/generated/triton.language.debug_barrier.html
[installed-sync-api]: ../../../artifacts/npu-cooperative-reductions-20261118/inspect/installed-source-excerpts.txt
[sync-lowering-source]: ../../../artifacts/npu-cooperative-reductions-20261120-allvector/source/HIVMDecomposeOp.cpp
[sync-op-source]: ../../../artifacts/npu-cooperative-reductions-20261120-allvector/source/HIVMSynchronizationOps.td
[sync-trait-source]: ../../../artifacts/npu-cooperative-reductions-20261120-allvector/source/HIVMTraits.td
[upstream-x-grid]: https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/_inductor/runtime/triton_helpers.py#L824
[upstream-barrier-call]: https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/_inductor/codegen/triton.py#L5948
[upstream-workspace]: https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/_inductor/codegen/triton.py#L2716
[pipeline-commands]: ../../../scripts/tests/cooperative_reductions/README.md#流水采集与-ir-导出的区别
[profiler-guide]: https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md
[ta-profiling]: https://triton-ascend.readthedocs.io/zh-cn/latest/debug_guide/profiling.html
