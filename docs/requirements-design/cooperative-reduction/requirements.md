---
title: NPU Cooperative Reduction 需求分析（PyTorch 2.13）
---

# NPU Cooperative Reduction 需求分析（PyTorch 2.13）

状态：需求分析草案。事实依据为 PyTorch 2.13 源码及 2026-11-18 的 NPU_A5 实验。
本文说明需求价值、能力差距、目标范围与验收条件；具体实现见[方案草案](design.md)。
文中的分期优先级为评审建议，首期能力清单和性能数值目标仍需确认。

## 1. 背景与价值

### 1.1 当前问题

**输出元素少、归约轴长时，仅按输出方向安排任务，可能无法充分利用 NPU 的并行计算资源。**

下面用“将两个长向量相加，再求总和”说明这一问题。这是一个小输出、长归约轴的举例，
用于解释需求来源；2026-11-18 的 A5 实验验证了该例的生成行为。

```python
def f(a, b):
    return (a + b).sum(dim=-1)
```

`a`、`b` 均为连续 fp32 张量，shape 为 `(1, 1048576)`。计算先逐元素相加，再沿最后一维
求和，最终输出 shape 为 `(1,)`：只保留 1 个输出，但这个输出要归约 1048576 个元素。
实验使用 NPU 默认 Inductor 后端，关闭 cooperative，执行 `torch.compile(f, fullgraph=True)`。

| 观察项 | 实验结果 | 对这个例子的含义 |
| --- | --- | --- |
| 生成的 kernel | 1 个，融合 `add` 与 `sum` | 已完成算子融合，仍需看 kernel 内如何安排任务 |
| 逻辑 Grid | 1 个 program | 只有一个逻辑任务处理整个输出，没有增加归约方向的协作分片 |
| 选中配置 | `R0BLOCK=1048576`，`R0BLOCK_SUB=8192` | 同一个 program 分 128 个子块循环处理长归约轴 |
| 编译模式 | `simd_simt_template` | 这是当前普通路径的实际配置，后续性能对照应保留该基线 |
| 设备并行资源 | Inductor 与 Triton 均查询到 56 个 Vector core | 输出方向的一个 program 无法提供跨 program 的核间并行 |

生成代码中，决定循环长度的原始语句如下；省略了 load/add/归约与写回部分：

```python
    loops_r0 = (R0BLOCK + R0BLOCK_SUB - 1) // R0BLOCK_SUB
    _tmp4 = tl.full([R0BLOCK_SUB], 0, tl.float32)
    for loop_r0 in range(loops_r0):
```

**这个例子暴露的现象是：融合后的长归约仍集中在一个 program 内循环，归约轴没有提供额外的
核间并行度。** 因此需求要解决的是这种执行方式的性能局限，并保留归约前后计算的融合机会。
kernel 数量和 program 数量回答不同的问题，不能仅凭“只有一个 kernel”判断已经充分分核。
这里的并行度判断来自生成代码与配置，实验未直接采集物理核利用率。

完整信息：[生成代码][E14]、[实际选中配置与执行结果][E5]、[Vector core 查询结果][E11]。

### 1.2 需求作用

Cooperative reduction 让多个 program 在**同一个 kernel 内**共同完成一个输出的归约：
分别计算 partial，通过跨 program 数据交换和同步获得完整结果，再执行后续计算或输出。
它不改变 `sum` 等算子的数学语义，属于 Inductor 的执行优化。
实现对照：[上游分片与结果合并][U2]、[运行配置][U3]。

例如 `(1, 1048576)` 的 sum 使用 32 个分片时，每片约处理 32768 个输入，归约方向可以形成
额外并行度。收益需要覆盖 partial 存取、同步、最终合并及资源开销，不能只因归约很长就启用。

| 目标场景 | 当前关注的问题 | 预期价值 |
| --- | --- | --- |
| 小输出、长轴归约 | 输出方向并行度不足，少数 program 执行很长的局部循环 | 缩短归约关键路径，提高有效核利用率 |
| pointwise prologue → reduction → 小输出 epilogue | 归约值还需要参与融合计算 | 在同一 kernel 内完成协作和消费者计算，保留融合机会 |
| softmax、连续归约与广播 | 多个阶段需要交换完整归约结果 | 为多阶段融合提供归约结果交换能力；收益尚需真实图验证 |
| 动态长归约 | shape 改变时并行度与同步约束随之变化 | 在合法 shape 范围内延续优化，范围外保持正确回退 |

本次手写原型已表明部分长归约有优化空间；尚无接通 Inductor cooperative 后的模型级收益证据。
验收应分别证明“路径实际命中”“结果正确”“相对当前普通归约有收益”。

## 2. 特性说明与上游能力

### 2.1 概念

| 术语 | 含义 |
| --- | --- |
| `X` / `R` / `S` | 归约后保留的元素数、每个输出的归约元素数、每个输出的协作分片数 |
| program / CTA | 逻辑执行单元，不能直接等同于一个 CUDA SM 或 NPU 物理核 |
| AIV / Vector core | NPU 上执行 Vector 计算的物理单元，program 由后端映射到物理资源 |
| partial / workspace | 分片归约的中间结果，以及交换这些结果需要的临时存储 |
| GM / UB | 全局存储与核内局部缓冲；跨核交换需要明确的可见性，不能把 UB 当作跨核共享空间 |
| phase | 一轮局部计算、交换与同步；连续归约可能包含多轮 |
| combine / identity | 将 partial 合为完整结果的规则，以及空分片使用的单位元，如 sum 的 0、prod 的 1 |
| persistent reduction | 将归约工作集放入一个局部 tile 处理的策略，与循环处理多个子块的策略区别对待 |
| optimization hint | 编译选配置使用的代表值，不自动构成动态 shape 的运行时上下界 |
| compile mode | TA/NPUIR 的编译路线；本次分别检查 SIMD、template 和 pure SIMT，支持结论不能跨模式沿用 |

普通归约在每个输出组的 program 内完成局部处理；cooperative 增加同一输出的协作分片，
因此额外引入同步、结果合并和临时存储需求。逻辑 program 数与实际物理参与者之间的映射，
是判断同步正确性和性能的必要信息。
代码对照：[上游生成逻辑][U2]、[NPU launcher 契约片段][E1b]。

### 2.2 CUDA 上游如何选择

本次上游基线中，`triton.cooperative_reductions` 默认关闭，可通过环境变量
`TORCHINDUCTOR_COOPERATIVE_REDUCTIONS=1` 或 config patch 开启，见[配置定义][U0]。

`InductorChoices.should_use_cooperative_reduction()` 的[普通选择规则][U1]为：

| `X` 的 optimization hint | 要求可静态证明的 `R` 下界 |
| --- | --- |
| `X <= 8` | `32768 * X` |
| `8 < X <= 16` | `2097152` |
| `X > 16` | 不选 cooperative |

Force 开关绕过上述性能 predicate，不代表后端一定具备执行能力。
动态 shape 即使某次输入很大，也不等于编译器已经证明 `R` 的下界满足条件。

CUDA runtime 按 SM 数、`X` hint 和 `TRITON_MAX_RSPLIT=64` 选择分片数；当前给各局部归约
候选加入相同的 `RSPLIT`，并没有默认搜索多个分片数。同步使用 `x_grid_barrier`，依赖
semaphore、返回旧值的 atomic、release/acquire 和 cooperative launch 约束。
实现对照：[分片配置与 Grid][U3]、[barrier 协议][U4]。

这些是 CUDA 现有行为，用于对照。NPU 的核数、局部资源、编译模式和同步成本不同，需要
自己的准入和收益依据；CUDA 的数值阈值不直接作为 NPU 需求指标。

### 2.3 上游回归入口

指定用例 [test_reduction_fns][C1] 的输入为两份 `(1, 1024**2)` 张量，计算
`reduction_fn(x+y, dim=-1)`，覆盖 `sum`、`mean`、`prod`、`amin`、`amax`、`min`、`max`、
`var_mean`、`std`、`softmax`，dtype 为 fp16/fp32/fp64。其他用例包含布尔归约、非整除 shape、
连续归约，以及 persistent/multi-kernel 的变体。

这些用例提供能力对照和回归种子。NPU 应按硬件与编译栈能力确定支持矩阵，分别判断基础
dtype 支持与 cooperative 支持；CUDA decorator、helper 名称、屏障次数和 launch 方式不直接
作为 NPU 的验收条件。

## 3. NPU 现状与差距

### 3.1 现有实现

当前 NPU 默认 Triton 路径有两类代码生成：

- `NPUTritonScheduling / NPUIndexTritonKernel`：采用 NPU 的多轴分析、`BLOCK_SUB` 循环和 tiling。
- `NPUNoLinearTritonScheduling / NPUTritonKernel`：复用更多上游 TritonKernel 逻辑，作为可用的回退入口。

`NPUCombinedScheduling` 在两条路径之间组织尝试和回退。普通归约已有多轴分析、局部循环、
候选选择和运行基础；尚未接通的是同一输出在多个 program 之间协作的完整链路。
源码入口：[路径选择与回退][N1]、[schedule][N2]、[两类 kernel 实现][N3]。

对第 1.1 节的 `(a+b).sum(-1)`，将输入扩展到 32 行并切换开关，静态结果摘要如下。
`X` 表示输出数，所有行的归约长度均为 `R=1048576`。

| cooperative 设置 | `X=1` | `X=32` | 说明 |
| --- | --- | --- | --- |
| 关闭 | PASS，普通路径，Grid=1 | PASS，普通路径，Grid=32 | 输出数增加后可沿输出方向分核 |
| 开启，按条件选择 | FAIL，回退产物加载失败 | PASS，no-linear 普通路径，Grid=32 | 开关开启后的 PASS 不代表使用 cooperative |
| 强制选择 | FAIL | FAIL | 当前没有成功执行的 Inductor cooperative 产物 |

定向诊断进一步记录了强制 `X=1` 时的两个断点。下面按调用关系整理，省略了无关栈帧与环境路径：

```text
Index kernel 初始化
  init_cooperative_reduction() → _has_constant_mask() → max_block()
  return TRITON_MAX_BLOCK[prefix.upper()]
  KeyError: 'R'

no-linear 回退产物加载
  @triton_heuristics.cooperative_reduction(...)
  AttributeError: module 'torch_npu._inductor.runtime.triton_heuristics' has no attribute 'cooperative_reduction'
```

前一个断点是 NPU 轴处理与继承初始化的假设未对齐；后一个断点是生成的 decorator 在 NPU
runtime 中不存在。安装检查同时确认：NPU 能看到继承的上游 `x_grid_barrier`，但缺少对应的
cooperative heuristic。由此需要贯通选择、代码生成与 runtime，补一个 decorator 不能证明功能完成。
动态普通路径的复用结果单独列在第 5.3 节。

完整信息：[静态配置与结果][E5]、[Index 到 no-linear 的诊断日志][E7]、[安装接口检查][E10]。

### 3.2 与上游的差异

| 维度 | CUDA 上游 | NPU 当前证据 | 差距 |
| --- | --- | --- | --- |
| 归约并行 | 同一输出有协作分片，并在 kernel 内合并 | 受测 full reduction 的普通路径 Grid 可为 1 | 尚未形成 Inductor cooperative 执行路径 |
| 选择策略 | 已有 `X/R` predicate、分片选择及能力交互 | 继承部分上游逻辑，NPU IR 判断与逐归约选择不完全一致 | 需要 NPU 能力、合法性与收益的一致判断 |
| 代码生成 | 已有分片索引、workspace、phase flush、合并和输出归属 | Index 覆盖大量方法，继承初始化出现轴假设不匹配 | 不是补一个 decorator 即可接通 |
| runtime/autotune | 已有 cooperative heuristic、Grid 和候选计时 | 缺少对应 heuristic，tiling 另有配置格式 | 候选与运行配置尚未协同 |
| 同步 | CUDA semaphore 协议和 cooperative launch | helper 继承 CUDA 实现；TA 另有原生同步，受限原型运行通过 | 需要确认适合 NPU 的同步与参与者契约 |
| 动态 shape | 普通选择需要静态证明下界，运行还受 Grid/workspace 约束 | 普通动态归约可复用；force cooperative 仍失败 | 未验证 NPU cooperative 动态支持 |
| 资源与模式 | SM、warp、寄存器/共享内存模型 | AIV、UB、SIMD/template/pure SIMT 及后端映射 | CUDA 阈值与资源假设不能直接复用 |

NPU `num_splits()` 判断使用 `BackendFeature.REDUCE_TO_SINGLE_ELEMENT`，继承的 backend
feature 会因 cooperative 开关加入该能力；NPU IR 缺少上游“本次归约实际选中 cooperative”的
联合 predicate。因此开关可能先影响图形态，即使最终不具备执行条件。
本需求只要求准入、最终执行与回退一致，不展开其他 reduction 拆分策略。

代码对照：[上游 IR 判断][U5]、[NPU IR 判断][N5]；表中的 runtime/tiling 差异对应
[NPU 候选与运行配置][N4]、[分核与分块实现][N6]。安装源码与本地基线的差异另见附录摘要。

### 3.3 功能能力与缺口

**TA 已有原生跨核同步接口，缺口在支持边界与完整接入，不能概括为底层没有同步能力。**
已安装接口为 `sync_block_all(mode, event_id)`，scope 包含 `all_vector`、`all_cube`、`all` 和
`all_sub_vector`，event ID 为 0～15。原型调用 `ext.sync_block_all("all_vector", 0)`，在写出
partial 后等待参与者，再读取同组结果；接口本身没有按输出 row/program group 分组的参数。

固定 `num_warps=4` 的编译结果与另行取得的运行证据如下。原生 sum、可见性探针和 CUDA
helper 探针合计 9 个编译配置，其中 6 个通过、3 个失败。

| 编译模式 | 原生 sum / 可见性探针编译 | CUDA barrier 探针编译 | 已有运行证据 |
| --- | --- | --- | --- |
| SIMD | 两项均 PASS | PASS | 可见性：1/2/8 program 各 8 phase；sum 覆盖尾块、多组及 Graph |
| template | 两项均 PASS | PASS | 未运行，不能从编译通过推断运行正确 |
| pure SIMT | 两项均 FAIL，同步操作无法 legalization | FAIL，`MaxThreads out of range` | 未运行 |

Pure SIMT 的两个关键错误摘要为：

```text
原生同步：failed to legalize operation 'hivm.hir.sync_block' that was explicitly marked illegal
CUDA helper：fatal error: error in backend: MaxThreads out of range!
```

按实际型号 `Ascend950PR_9579` 复核后，错误类型一致。原生同步的 pure SIMT lowering 是当前
安装栈的明确功能缺口；CUDA helper 的线程数错误尚未进一步归因。上表中的 CUDA helper
均未发射执行，atomic 旧值、release/acquire 等运行语义仍未验证。

| 仍需确认的能力 | 当前基础与未决边界 |
| --- | --- |
| 同步参与者与分组 | 原生 API 提供全 Vector 等 scope，不能直接等同 CUDA 的按组 semaphore；是否需分组能力取决于场景 |
| GM 可见性与多 phase | 受测 SIMD 的写→同步→读通过；每轮读者完成后再覆盖空间，任意映射与控制流仍需确认 |
| atomic 旧值与内存序 | TTIR 有相关属性，adapter 表达有转换；尚未证明满足 CUDA semaphore 协议 |
| workspace、scratch/lock 与调用 ABI | compiler 已有 workspace/lock 回调，launcher 和 C++ wrapper 已有处理基础；完整生命周期仍需验收 |

第 3.1 节的失败发生在 kernel 初始化与 module load；本节的 pure SIMT 问题则由绕过 Inductor
的固定配置编译直接复现。两类证据分别说明上层接入断点与下层模式缺口。

完整信息：[同步接口与安装源码][E1a]、[固定配置编译矩阵][E3]、[实际型号复核][E12]、
[运行方法与原型][P1]、[compiler/launcher 契约片段][E1b]。

### 3.4 性能缺口

下表区分实测现象、IR 可见工作和待 profiling 项。后两者尚不构成下层性能 bug 的确定归因。

| 编号 | 性能问题 | 证据状态 | 需求上的影响 |
| --- | --- | --- | --- |
| G1 | 小工作量下 cooperative 原型退化 | `R=4096`：普通 1.013 µs、协作 4.683 µs；`R=32768` 也退化 | 按分片工作量和协议成本选择，不能只检查输出少 |
| G2 | 同步范围/pipe 粒度成本未定界 | IR 为 `ALL_VECTOR`、`PIPE_ALL`；流水影响未 profile | 明确语义和成本，再评估更细粒度优化 |
| G3 | 终结归约存在重复 peer 读取和合并 | 所有分片均读取同组 partial，只有最终 store 判断分片归属 | 识别消费者是否需要每片获得完整结果，评估重复工作的成本 |
| G4 | 小 partial 编译效率缺少基线 | adapter 有 materialize/copy/to-tensor/reduce；尚无最终指令归因 | 对齐等价算法，检查小事务、转换和合并效率 |
| G5 | 实际资源用量反馈的可解释性不足 | 成功配置 `required_ub_bits` 均为 0；compiler 将其初始化为 0 | 区分用量、容量和 unknown，支持可靠的资源判断 |
| G6 | 分片、tile、模式与 multibuffer 未联合评估 | 原型固定 SIMD、`B=1024`；普通 Inductor 配置不同 | 区分策略、算法和模式的收益与下层编译影响 |
| G7 | 隐式初始化/运行成本未定界 | 有 lock ABI，native 探针未返回显式锁布局；未采集对应设备任务 | 分别评估 kernel 和完整调用，不预设 native 必用软件锁 |

G1 的 `R=4096,B=1024,S=32` 只有 4 个非空分片，其余 28 个仍参加协议；`R=32768` 没有
这种空片仍出现退化。这同时提示分片策略和协议成本需要评估，差值不能直接记为 barrier 延迟。

G2 的 adapter 关键行如下，省略了源码位置 metadata。它显示同步覆盖全 Vector scope，相关
pipe 均为 `PIPE_ALL`；是否限制流水重叠，还需设备代码与 profiling 定界。

```text
hivm.hir.sync_block[<ALL_VECTOR>, 0 : index] tvector_pipe = <PIPE_ALL> vector_pipe = <PIPE_ALL>
```

G3 在手写原型中对应下面这段原始代码。peer load 和 `tl.sum` 位于所有分片都会执行的路径，
只有 store 使用 `shard == 0` 的 mask，因此“只有一个分片写输出”仍可能包含重复合并工作。

```python
        ext.sync_block_all("all_vector", 0)
        peers = tl.arange(0, P)
        values = tl.load(W + row * S + peers, peers < S, other=0)
        tl.store(Y + row, tl.sum(values, 0), shard == 0)
```

当每组有 `S` 个分片时，逻辑 partial 读取可达 `S²` 个元素；这不等于实际 HBM 流量。
终结归约可评估减少重复合并，广播消费者则可能确需每片得到完整结果。

G5/G7 的成功编译 metadata 摘要为：`required_ub_bits=0`、`shared_mem_dynamic_size=221184`，
且 native 探针未返回显式 `sync_block_lock_layout`。其中 221184 bytes 是 216 KiB 容量，
不能作为实际 UB 用量；`required_ub_bits=0` 也不能直接解释为该 kernel 不使用 UB。
同样，锁字段未返回不能证明原生同步一定使用或一定不使用软件锁。

G4 当前没有最终设备指令证明多余拷贝或 spill；G2/G4/G7 的最终归因仍需设备证据。
完整信息：[原型代码与运行方式][P1]、[adapter IR][E13]、[编译 metadata][E3]、
[compiler 字段与 launcher 处理][E1b]。性能数字与测量边界见第 6.1 节。

## 4. 需求范围与优先级

### 4.1 系统边界

目标是 PyTorch 2.13 NPU 默认 Inductor → Triton-Ascend（TA）→ AscendNPU-IR/BiShengIR 链路。
本文的 NPUIR 指设备编译器；另一个 `ascend_npu_ir` Inductor backend、DVM、AscendC backend
不在本轮范围内，后端接入关系见 [NPU backend loader][N0]。

当前验证平台是 NPU_A5。需求应给出明确的 SoC、compile mode、算子和 dtype 支持矩阵，
其他组合在验证前保持普通路径；不将 A5 上的受限原型外推为所有 NPU 已支持。

### 4.2 建议优先级

| 优先级 | 能力 | 范围说明 |
| --- | --- | --- |
| 基础必需 | 至少一组已声明支持的目标/模式，实现小输出长归约的 cooperative 路径 | 静态 fp32 sum 及 `(a+b).sum` 是现有证据充分的验收入口，具体首期清单待评审 |
| 基础必需 | 普通路径兼容、合法性检查、精度、autotune/cache 一致性及重复调用 | 不能因打开开关使未命中或不支持场景编译失败 |
| 扩展目标 | mean、prod、极值、布尔及带 index/复合结果的归约，更多 dtype | 按 combine、dtype 和硬件能力逐项验收；不将所有算子视为相同实现 |
| 扩展目标 | reduction/pointwise/reduction、softmax 等融合 | 结果交换与多 phase 语义需要独立验收 |
| 扩展目标 | 动态 `R`、动态 `X`、symbolic grouped autotune | 纳入总体目标，首期未支持时必须安全回退 |
| 扩展目标 | persistent、更多编译模式、C++/AOT、并发执行 | 分别给出支持状态与验收记录 |

Index/no-linear 的实现入口、具体同步原语、结果合并方式和参数布局属于方案选择，不在本文中
作为预先确定的需求约束。上表也不表示扩展目标已经承诺在首个版本全部交付。

### 4.3 不包含项

- Split reduction 的策略研究、性能优化和方案设计；只保留 cooperative 所需的 IR 一致性检查。
- 修改用户侧算子数学定义或新增 ATen 接口。
- 未验证的全 SoC、全 dtype、全模式默认启用，以及以共享卡数据直接确定生产阈值。

## 5. 功能与特性正交要求

本节的“正交”关注 cooperative 与其他特性组合时的语义、准入、缓存和回退是否成立；
资源占用与性能仍需按组合验证。

### 5.1 基础功能要求

| 编号 | 要求 | 可观察的结果 |
| --- | --- | --- |
| FR1 | 在声明支持的归约中，多个 program 对同一输出协作，保持单 kernel 内完成 | 生成代码与实际配置能确认分片、结果交换、同步和最终输出，而不只检查 kernel 数 |
| FR2 | 保持 ATen 语义，正确处理分片、尾块、identity、索引、dtype promotion 和特殊值 | 与同图 eager 结果对比；空输入和退化形状遵循对应算子语义 |
| FR3 | 区分后端能力、执行合法性和性能选择 | 不支持/不合法时走普通路径；force 不得绕过正确性与资源限制 |
| FR4 | IR、schedule、codegen 与最终运行策略一致 | 开关改变不应留下无法执行的半成品方案，失败尝试不污染后续普通路径 |
| FR5 | 候选预运行、winner、cache hit 使用一致的策略和资源需求 | 重复编译/调用与缓存恢复后，Grid、参数、workspace 及结果仍一致 |
| FR6 | 同步参与者与实际映射合法，数据交换满足必要的内存可见性和前进性 | 不依靠运行超时发现非法配置；无法确认合法性时不发射该配置 |
| FR7 | 临时存储与同步状态满足每次调用的生命周期 | 连续调用、workspace 重用和已声明支持的 Graph/stream 组合中结果正确 |
| FR8 | 特性可控，关闭与未命中时保持当前普通归约能力 | 受支持形状能验证命中，不受支持形状能验证回退；既有普通路径无功能回归 |

这些要求关注执行结果和跨层一致性。第 3.1 节的初始化/加载断点说明 FR3/FR4 尚未满足，
第 3.3 节的同步与调用契约则是 FR6/FR7 的依赖。实施时可复用现有产物、配置记录和验证工具，
不要求新增一套诊断系统。

### 5.2 正交能力矩阵

“验收要求”是目标行为；下表不代表当前 NPU 已实现这些组合。

| 组合 | 验收要求 | 当前证据/交付定位 |
| --- | --- | --- |
| 静态 shape、简单 prologue/epilogue | 支持范围内正确分片、完整合并且没有重复输出冲突 | 基础入口；Inductor cooperative 当前未接通 |
| 非整除 `R`、输出尾块、空分片 | 输入覆盖无遗漏/重复，identity 与最终结果正确 | 原型部分通过，集成路径仍需验收 |
| 动态 `R` | 真实值改变后仍满足输入覆盖、资源和同步约束；范围外有正确行为 | 普通归约复用通过；cooperative 未验证 |
| 动态 `X`、unbacked SymInt | 输出数、参与者和临时空间均有合法运行时依据 | 扩展项；未知大小不能仅凭样本 hint 准入 |
| 多逻辑归约轴和非连续 stride | 按原索引覆盖正确的元素集合；不能为复现方便改成连续输入后宣称兼容 | 分别确定支持范围，未准入时回退 |
| fp16/fp32/bf16、整数/布尔及 fp64 | 明确硬件与算子支持清单、累加类型和误差标准 | 当前原型只验证 fp32；其他 dtype 不自动视为已支持 |
| min/max index、argmin/argmax | 保持全局索引、tie-breaking、NaN 和返回结构 | 扩展项，不能只验证局部 value |
| var/std/var_mean | 复合归约、空片权重和精度符合算子语义 | 扩展项，不能把多个 partial 简单等权平均 |
| 多 phase、softmax、广播消费者 | 每轮结果可见，旧读者完成后才能重用空间 | 扩展项，需要超出单 sum 的专门用例 |
| persistent、max_autotune、symbolic group、multi_kernel | 每种实际候选及 group 都满足能力、资源与正确性要求 | 按组合交付；现有 autotuner 不自动补齐完整策略 |
| SIMD/template/pure SIMT | 支持矩阵区分编译通过和运行通过 | SIMD 有受限运行证据；template 仅有编译证据；pure SIMT native sync 失败 |
| NDDMA、multibuffer 等访存/流水优化 | 不破坏同步和相关 GM 读写依赖 | 需要对齐实际选中配置验证 |
| deterministic | 遵守已承诺的配置选择与可复现规则 | 浮点树变化的误差与复现要求分别评审 |
| NPU Graph、多个 stream | capture/replay、状态初始化和存储生命周期正确 | 原型 Graph 通过；完整 Inductor/并发未验证 |
| C++ wrapper、AOTInductor | 打包重载后仍使用合法参数、资源和执行策略 | 不能以 Python wrapper 成功代替验收 |
| scan、sort、combo/foreach、mix-order | 未接入 cooperative 的组合继续正确走已有路径 | 不因开关开启而破坏；不要求首期扩展到这些组合 |

代码检查入口：[上游 schedule 约束][U6]、[NPU schedule][N2]、[tile 与局部资源判断][N8]。
软件当前支持状态以本次实验和安装源码为准，不以“语义上可设计”代替已完成验证。

### 5.3 动态 shape 的独立要求

动态支持至少区分三个问题：能否证明选择条件成立、真实 shape 下执行是否合法、原配置是否
仍有收益。`dynamic=True` 本身不能回答其中任一个问题。

例如 `X` 从 1 变为 128，原来的分片数不能在未经确认时继续扩展为同一同步参与者集合。
验收需记录真实 Grid、临时空间需求和所选配置，并检查同一 callable 跨小值、长轴和尾块的
复用/重编译行为。

现有实验中，`dynamic=True`、`X=1`，关闭 cooperative 和按条件开启 cooperative 两组测试
均执行普通归约：Grid=1，`R0BLOCK_SUB=8192`。每组在同一个 compiled callable 上取得如下结果：

| 调用顺序 | 真实 `R` | 结果 | 新增 kernel 数 |
| --- | --- | --- | --- |
| 首次调用 | 1048576 | PASS，普通归约 | 1 |
| 复用到较小输入 | 32768 | PASS，普通归约 | 0 |
| 复用到非整除输入 | 1048593 | PASS，普通归约 | 0 |

强制 cooperative 的动态组仍失败。由此只能确认普通动态归约的复用，不能据此宣称 cooperative
动态支持已完成。完整信息：[动态开关、实际配置与复用结果][E6]。

## 6. 性能要求

### 6.1 已有性能证据及边界

以下是手写 fp32 `sum(a)` 原型，固定 SIMD、`B=1024`，没有搜索最优 tiling。
Graph 内捕获 32 次调用，预热 3 次 replay，再计时 10 次 replay；5 组样本取中位数。
所有结果通过原型的 `atol=0.02, rtol=0.0002` 检查。

| `X × R` | 分片数 `S` | 普通原型（µs） | 原生 cooperative 原型（µs） |
| --- | --- | --- | --- |
| `1 × 4096` | 32 | 1.013 | 4.683 |
| `1 × 32768` | 32 | 3.533 | 4.881 |
| `1 × 1048576` | 32 | 90.667 | 7.531 |
| `1 × 16777216` | 32 | 2623.762 | 124.551 |
| `8 × 1048576` | 4 | 93.649 | 27.878 |
| `32 × 1048576` | 1 | 255.201 | 258.221 |
| `1 × 1048593` | 32 | 90.832 | 7.539 |

数据支持“部分长归约值得接入”“小工作量不能照搬 CUDA 阈值”这两个判断。
它不是 Inductor 集成后的收益，不能直接与两输入 `(a+b).sum` 相比；也没有证明真实 softmax
或模型收益。原型误差容限仅描述本次实验，不直接作为所有算子/dtype 的产品精度标准。

实验使用经授权共享的物理卡 2。跨轮 `X=32` 普通原型中位数曾约为 96～255 µs，且多数输入
不超过本卡 128 MiB L2，Graph 重放属于热数据。当前没有独占卡、冷数据、最优 tiling、完整
调用开销或模型级收益的充分证据，不据此发布加速承诺或生产阈值。
完整信息：[计时记录与五组样本][E4]、[设备映射确认][E8]。

### 6.2 性能验收原则

| 要求 | 判定依据 |
| --- | --- |
| 与当前普通归约对齐基线 | 同一计算图、输入、layout、dtype、设备和运行条件；保留普通路径实际选中配置，不人为削弱基线 |
| 目标负载有可复现收益 | 完整图执行时间改善超出测量噪声，不能只报告分片内计算的加速 |
| 不适用负载不产生协议退化 | 小 `R`、输出并行度充分、无实际协作收益等形态应保持普通路径 |
| 区分 device 与完整调用开销 | 同时关注 kernel、临时空间/初始化，以及 Graph 开关对结果的影响 |
| 编译/调优与内存成本可控 | 记录候选数量、首编译/调优时间、临时空间及缓存恢复情况 |
| 下层性能问题可归因 | 固定算法、分片、tile 与编译模式做阶段对照，结合设备证据区分策略、codegen、TA 与 NPUIR 的影响 |

G1～G7 应在验收记录中有定位结论、影响范围或明确的待确认项。对于尚无设备证据的项目，
不能以一个总耗时差值断言是某条同步指令或 NPUIR pass 的问题。

### 6.3 需在默认启用前量化的指标

| 指标 | 当前状态 |
| --- | --- |
| 目标 shape/融合图集合、最小有效加速幅度 | 待同图校准与评审；当前只有原型选点证据 |
| 未命中场景、端到端模型允许的退化范围 | 待量化，需要普通路径与特性开关对照 |
| 不同 SoC/mode/dtype 的准入阈值或收益区间 | 待建立，不能统一复制 CUDA 的 `X/R` 常量 |
| 首编译/调优时间和 workspace 的成本预算 | 待候选方案确定后测量 |
| 各归约算子/dtype 的误差与复现标准 | 待按算子语义和硬件能力确定 |

当前需求明确了比较方式和必须量化的项目，没有将未测量的数值填作已达到的目标。

## 7. 影响范围、周边依赖与风险

### 7.1 Inductor 影响范围

| 阶段 | 需要满足的职责 |
| --- | --- |
| IR / capability / choices | 能力声明、逐归约准入和保留的图形态一致 |
| schedule / fusion | 支持范围内保持合法融合与 phase；范围外得到完整普通路径 |
| tiling / candidate generation | 共同考虑输出并行度、归约分片、局部资源和实际映射 |
| codegen / helper | 生成正确的分片计算、结果交换、同步与输出 |
| autotune / cache | 候选预运行、最终 winner、group 和缓存恢复保持一致 |
| wrapper / launcher | 处理临时空间、同步状态、stream 和已声明支持的调用方式 |

因此改动范围是跨阶段的编译策略与运行能力，不限于 codegen。现有 workspace、候选编译计时
和 wrapper 基础可复用，具体文件修改与实现工作量放在方案中评估。
可复用入口：[NPU 候选调优框架][N4]、[C++ wrapper 工作空间处理][N7]。

### 7.2 TA / NPUIR 依赖

| 依赖 | 期望下层提供或确认的能力 | 责任边界 | 性质 |
| --- | --- | --- | --- |
| D1：架构/模式及参与者 | 明确支持组合，说明 program/Grid 到物理参与者的映射和非法配置边界 | TA options/launcher 与 NPUIR 映射 | 功能依赖，未明确时不能发射相应配置 |
| D2：同步与可见性 | 所选协议的 GM 可见性、执行前进性、event/状态复用及编译重排约束 | TA 扩展/IR 依赖与 NPUIR 同步 lowering | 功能依赖，受限运行成功仍需明确契约 |
| D3：数据交换与控制流 | 支持归约结果交换及消费者所需控制流；优化不破坏同步顺序 | TA lowering 与 NPUIR 优化 | 功能与性能共同依赖，不预先指定合并方式 |
| D4：资源和调用契约 | 区分用量、容量与 unknown；明确 scratch/lock 和调用要求 | NPUIR metadata → TA → NPU runtime | ABI 影响功能，资源反馈影响性能选择 |
| D5：同步/小 partial 性能 | 定界同步、等待、小事务、合并和初始化成本 | NPUIR 主导归因，TA 提供固定 DSL/IR 对照 | 对应 G2/G4/G5/G7，按证据决定优化 |
| D6：模式与流水优化组合 | 明确各 mode 与 multibuffer 等优化的正确性和性能边界 | TA 选项与 NPUIR pass pipeline | 模式扩展依赖；pure SIMT 当前有缺口 |

这些依赖包含已有能力的契约确认、产品接入和可能的性能优化，并不表示全部都要新增下层 API。
例如 A5 原生同步已存在，是否需要更细的同步或新的结果交换能力，取决于最终支持场景和性能证据。
若方案选择 CUDA 风格 semaphore，则还需单独证明对应 atomic/内存序契约，不能默认复用成功。

### 7.3 主要风险与限制

- 版本追溯：安装包 metadata、`version.py` 和部分源码不完全对应，需使用源码 hash 和实际编译器路径。
- 正确性：开关与能力不一致、错误参与者集合或多 phase 空间重用，可能导致错误结果或执行无法推进。
- 数值：分片与 tile 改变浮点计算树，复合归约、索引和特殊值需要独立验证。
- 性能：原型固定配置、共享卡和热缓存不能代表集成效果；粗粒度同步、重复工作和资源反馈需继续定界。
- 兼容性：单模式或 Python wrapper 的成功不能直接推广到其他模式、动态、AOT 或并发。

风险应通过支持矩阵、验收证据和明确回退来收敛，不以“全部用例能编译”代替功能与性能结论。

## 8. 验收标准与未决问题

### 8.1 验收集合

| 类别 | 必须覆盖的内容 | 通过条件 |
| --- | --- | --- |
| 路径命中 | 声明支持的静态长归约和融合入口 | 实际生成并运行 cooperative；有真实协作证据 |
| 语义正确性 | 不同 `X/R`、尾块、空片、特殊值、输出结构和支持 dtype | 与对应 eager 图比较，符合约定精度及索引规则 |
| 准入/回退 | 开关关闭、未命中、不支持的算子/模式/shape、候选失败 | 现有普通能力保持，无残留 cooperative 状态；基础硬件不支持项单列 |
| 重复执行与状态 | 多次调用、临时空间重用、候选预运行和 cache hit | 结果与参数一致，不依赖上次调用留下的状态 |
| 正交组合 | 每个声明支持的第 5.2 节组合 | 有独立正确性和路径记录，未支持项记录回退 |
| 动态扩展 | 同一 callable 跨阈值、尾块和实际输出数变化 | 执行合法、结果正确，并记录复用/重编译与配置变化 |
| 性能 | 同图普通/cooperative 对照及第 6 节指标 | 可复现收益、退化边界和完整成本记录齐全 |
| 下层能力 | 目标架构/模式、映射、同步和资源契约 | 固定配置证据与集成证据一致，待确认项不被宣称为已支持 |

指定 CUDA 用例作为回归入口；NPU 的路径验证使用自己的实际产物与同步语义。
非法配置应在选择或编译阶段拒绝，不通过故意发射可能无法推进的 kernel 来验收。

### 8.2 待确认事项

| 问题 | 当前依据 | 关闭条件 |
| --- | --- | --- |
| 首期支持哪些 SoC、mode、算子与 dtype | A5、fp32 静态 sum 证据最完整，但产品能力范围未单独确认 | 评审形成明确支持矩阵和交付优先级 |
| 动态与多 phase 在哪一阶段交付 | 普通动态复用通过，cooperative 动态和 softmax 集成未验证 | 确定阶段边界，补充对应集成用例 |
| 原生同步的物理参与者和 GM 保证 | 受限 SIMD 原型通过，完整映射/运行契约仍需明确 | TA/NPUIR 确认支持组合并提供可复现证据 |
| G2/G4/G7 的真实性能占比 | 当前仅有中间 IR 和总时间 | 固定算法的阶段实验、最终设备代码或 profiling 定界 |
| 资源用量字段是否可直接用于选择 | 当前 `required_ub_bits=0`，与容量字段含义不同 | 明确单位、有效性与 unknown 处理 |
| 默认启用的收益阈值、精度和成本预算 | 原型数据不足以确定生产数值 | 完成第 6.3 节的同图校准与评审 |

### 8.3 验收记录要求

记录源码/安装版本、目标设备、输入 shape/stride/dtype、开关和 compile mode、实际候选 winner、
运行 Grid、正确性与性能方法，以及生成代码/IR 的证据链接。每项结论区分“已验证”“待确认”
和“需求目标”。复现入口见[探针说明][P1]，现有阶段提交见[版本映射][E9]。

## 附录：版本与证据

当前记录来自 2026-11-18，NPU_A5 Docker 内 `pta_213_case`，工作目录
`<workspace>/PTA_213`。经授权共享物理卡 2，CANN API 确认可见逻辑设备 0 对应物理 2，
见[设备映射记录][E8]。本版复用既有实验，没有新增设备测试。

| 项目 | 分析/实测基线 |
| --- | --- |
| 设备 | Ascend950PR_9579，28 Cube / 56 Vector core，128 MiB L2，设备报告 UB 216 KiB |
| PyTorch 安装版本 | `2.13.0+cpu`，`cf30153c4c131c8164ee7798e5022d810682e2cb` |
| torch_npu 安装信息 | metadata 为 `2.13.0+git76443ac`；version.py 为 `2.13.0+gitefc05dd` |
| Triton / Triton-Ascend | `3.6.0` / `3.6.0+dev20261112182052` |
| 实际设备编译器 | Triton 包内 BiShengIR 2.0.0，AscendNPU-IR `395c9c5646f9`，2026-11-12 |
| CANN | `cann-9.1.0`；PATH 上的 BiShengIR 1.2.0 不是本次实际编译器 |
| 本地 CUDA 源码 | `887a5ee68a445193f44f2ed9efd1d87e4d7a4242` |
| 本地 NPU 源码 | `961b95d9acb79551c52a77e991db9dd2613d6d86` |
| 用例源码 | `51830b03317074409a04c0ff167ee0d7ad6c897c` |

这里记录版本差异，是因为包信息不能单独代表实际执行内容：torch_npu 的 metadata 和
`version.py` 标识不同，部分 scheduling 文件的安装内容与本地源码 hash 不同；Triton 实际调用
包内 BiShengIR 2.0.0，PATH 上的 1.2.0 未参与本次编译。因此运行结论采用安装源码片段和
实际生成代码。CUDA 链接采用各文件最近修改的公开提交，其内容与本次本地分析基线相同。
完整信息：[安装版本与文件 SHA256][E1]、[实际 compiler 路径与版本][E2]、[源码 hash 对照][E1c]。

关键证据入口：

- [上游用例][C1]、[cooperative 选择规则][U1]、[代码生成][U2]、[runtime 候选][U3]。
- [静态 Inductor 结果][E5]、[动态复用结果][E6]、[Index 到 no-linear 的失败记录][E7]。
- [固定配置编译矩阵][E3]、[实际型号 pure SIMT 复核][E12]、[原型计时与样本][E4]。
- [安装源码片段][E1a]、[后端调用契约片段][E1b]、[原型 adapter IR][E13]。

[U0]: https://github.com/pytorch/pytorch/blob/e3966c93e0ae877c1150f9fceaab6055109ce1c8/torch/_inductor/config.py#L1997
[U1]: https://github.com/pytorch/pytorch/blob/521f9d784733cb2f4571804b0f6e4687fb7a91be/torch/_inductor/choices.py#L390
[U2]: https://github.com/pytorch/pytorch/blob/1b62beb3a54792e6d6ee0c7ac4f1af46a097e231/torch/_inductor/codegen/triton.py#L3229
[U3]: https://github.com/pytorch/pytorch/blob/521f9d784733cb2f4571804b0f6e4687fb7a91be/torch/_inductor/runtime/triton_heuristics.py#L4632
[U4]: https://github.com/pytorch/pytorch/blob/ac6173be20bb4214950e667e2bb7a75ea47c293c/torch/_inductor/runtime/triton_helpers.py#L824
[U5]: https://github.com/pytorch/pytorch/blob/98b772f8ab86176b8e0b572d88f4e09510c1b587/torch/_inductor/ir.py#L1387
[U6]: https://github.com/pytorch/pytorch/blob/d18f5dcab258c3cc62901e3bf8996ad7bc86abc9/torch/_inductor/codegen/simd.py#L787
[N0]: https://gitcode.com/luqichao/pytorch_new/blob/961b95d9acb79551c52a77e991db9dd2613d6d86/torch_npu/_inductor/__init__.py
[N1]: https://gitcode.com/luqichao/pytorch_new/blob/961b95d9acb79551c52a77e991db9dd2613d6d86/torch_npu/_inductor/codegen/npu_combined_scheduling.py
[N2]: https://gitcode.com/luqichao/pytorch_new/blob/961b95d9acb79551c52a77e991db9dd2613d6d86/torch_npu/_inductor/codegen/scheduling.py#L173
[N3]: https://gitcode.com/luqichao/pytorch_new/blob/961b95d9acb79551c52a77e991db9dd2613d6d86/torch_npu/_inductor/codegen/triton.py#L902
[N4]: https://gitcode.com/luqichao/pytorch_new/blob/961b95d9acb79551c52a77e991db9dd2613d6d86/torch_npu/_inductor/runtime/triton_heuristics.py
[N5]: https://gitcode.com/luqichao/pytorch_new/blob/961b95d9acb79551c52a77e991db9dd2613d6d86/torch_npu/_inductor/ir.py#L80
[N6]: https://gitcode.com/luqichao/pytorch_new/blob/961b95d9acb79551c52a77e991db9dd2613d6d86/torch_npu/_inductor/codegen/split_tiling.py#L100
[N7]: https://gitcode.com/luqichao/pytorch_new/blob/961b95d9acb79551c52a77e991db9dd2613d6d86/torch_npu/_inductor/codegen/cpp_wrapper_npu.py
[N8]: https://gitcode.com/luqichao/pytorch_new/blob/961b95d9acb79551c52a77e991db9dd2613d6d86/torch_npu/_inductor/runtime/tile_generator.py#L109
[C1]: https://github.com/limbochao/Community-Use-Case-Testing/blob/51830b03317074409a04c0ff167ee0d7ad6c897c/v2.13.0/inductor_npu/test_cooperative_reductions.py
[P1]: ../../../scripts/tests/cooperative_reductions/README.md
[E1]: ../../../artifacts/npu-cooperative-reductions-20261118/inspect/stack.json
[E1a]: ../../../artifacts/npu-cooperative-reductions-20261118/inspect/installed-source-excerpts.txt
[E1b]: ../../../artifacts/npu-cooperative-reductions-20261118/inspect/installed-backend-contracts.txt
[E1c]: ../../../artifacts/npu-cooperative-reductions-20261118/inspect/source-comparison.json
[E2]: ../../../artifacts/npu-cooperative-reductions-20261118/inspect/actual-npuir-compiler.json
[E3]: ../../../artifacts/npu-cooperative-reductions-20261118/compile-fixed/compile.json
[E4]: ../../../artifacts/npu-cooperative-reductions-20261118/bench-warm/bench.json
[E5]: ../../../artifacts/npu-cooperative-reductions-20261118/inductor/inductor.json
[E6]: ../../../artifacts/npu-cooperative-reductions-20261118/dynamic-reuse/inductor.json
[E7]: ../../../artifacts/npu-cooperative-reductions-20261118/index-fallback.log
[E8]: ../../../artifacts/npu-cooperative-reductions-20261118/device-map.json
[E9]: ../../../artifacts/npu-cooperative-reductions-20261118/validation-provenance.json
[E10]: ../../../artifacts/npu-cooperative-reductions-20261118/inspect/runtime-symbols.json
[E11]: ../../../artifacts/npu-cooperative-reductions-20261118/inspect/effective-cores.json
[E12]: ../../../artifacts/npu-cooperative-reductions-20261118/compile-simt-exact/compile.json
[E13]: ../../../artifacts/npu-cooperative-reductions-20261118/compile-fixed/cooperative_sum-simd/kernel.ttadapter
[E14]: ../../../artifacts/npu-cooperative-reductions-20261118/inductor/baseline-1x1048576-dynamic0/output_code_0.py
