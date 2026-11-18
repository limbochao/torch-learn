---
title: NPU Cooperative Reductions 需求调研（PyTorch 2.13）
---

# NPU Cooperative Reductions 需求调研（PyTorch 2.13）

## 1. 结论与范围

**NPU 适配 cooperative reduction 需要贯通 IR 策略、调度、分核/分块、代码生成、autotune 和运行时，
不能只增加一个 Triton decorator。** 当前 NPU 已有跨核同步基础；A5 的原生屏障能支持一个受限的
cooperative reduction 原型。完整接入 Inductor 的主要工作仍在编译器集成和组合约束。

本次得到的三个关键结果：

1. A5 上 `sync_block_all("all_vector", ...)` 的跨 program 可见性、重复同步、尾块、多组归约和
   NPU Graph 原型实验通过。不能把底层缺口概括为“没有跨核同步”。
2. 同一安装环境的纯 SIMT 编译不能合法化 `hivm.hir.sync_block`；SIMD 和 SIMD/SIMT template
   编译通过。因此必须按架构、编译模式开放能力，不能仅按 `device == npu` 打开。
3. 固定配置下，cooperative 并未稳定快于正确分核的两阶段归约。优化目标应是减少 launch、
   中间张量和融合链的额外访存，而不是把“跨核并行”的全部收益归给 cooperative。

上述结果的原始证据、配置与脚本见[实验记录](#6-实测与证据)。本文给出需求范围与建议，
没有实现或启用产品特性，也没有据共享卡实验确定上线阈值。

范围是默认 NPU Inductor → Triton-Ascend → AscendNPU-IR/BiShengIR 路径。
下文的 NPUIR 指 Triton 后面的设备编译器，不是 `torch_npu._inductor.ascend_npu_ir` 的
另一条 MLIR Inductor backend；后者及 DVM、AscendC backend 不在本需求内。[N0]

## 2. 概念和 CUDA 对照

### 2.1 三种路径

记 `X` 为归约后保留的元素数，`R` 为每个输出需要归约的元素数，`S` 为每个输出的分片数。
**program/CTA 是逻辑工作单元，不等于物理 AIV 核或 CUDA SM。** 一个物理核可能按编译器的
映射执行多个 program。`XBLOCK` 等 tiling 还会改变实际输出方向的 program 数。

| 路径 | 工作方式 | 适用的资源形态与代价 |
| --- | --- | --- |
| 普通单 program 归约 | 每个输出组由一个 program 在局部完成，可能循环多块 | `R` 小，或输出方向已有足够并行度；没有跨核合并开销 |
| Split reduction | 第一阶段分片，写 partial；后续 kernel 合并 | `X` 小而 `R` 大，跨核并行不足；多次 launch，但可以执行任意多轮 program |
| Cooperative reduction | 同一 kernel 内分片，写 partial，同步，读取 peer partial 合并 | 同样需要小 `X`、大 `R` 的并行性；还要求同步参与者能够安全共同推进 |

例如 `(1, 1048576)` 的 sum，`S=32` 时各 program 负责约 32768 个输入：

```text
split:        32 个 partial → kernel 边界 → final sum
cooperative: 32 个 partial → kernel 内屏障 → peer sum → final store
```

两种方案通常都需要 partial workspace。Cooperative 不等于“不使用 GM”，也不等于使用 atomic
直接累加最终结果。CUDA 实现让各分片读取同组 partial，分别得到完整归约结果，便于继续融合
归约后的逐元素计算，再由指定分片写标量输出。[U2]

### 2.2 CUDA 的选择规则不能直接搬到 NPU

本次上游基线中，`split_reductions` 默认开启，`triton.cooperative_reductions` 默认关闭；
后者可用 `TORCHINDUCTOR_COOPERATIVE_REDUCTIONS=1` 或 config patch 开启。两个配置可以
同时为 True，但对于同一归约，选中 cooperative 后会抑制相应的 IR split。[U0][U5]

上游 `InductorChoices.should_use_cooperative_reduction()` 的普通开关逻辑是：[U1]

| `X` 的 optimization hint | 要求可静态证明的 `R` 下界 |
| --- | --- |
| `X <= 8` | `32768 * X` |
| `8 < X <= 16` | `2097152` |
| `X > 16` | 不选 cooperative |

`force_cooperative_reductions=True` 会绕过该性能选择逻辑；它不是能力检查。
普通开关仍要求 `statically_known_geq(R, threshold)`，不能把一个较大的 shape 样本当作动态
`R` 的已知下界。

CUDA runtime 再按 SM 数、`X` hint 和 `TRITON_MAX_RSPLIT=64` 选择 `S`，为每个配置加入
`RSPLIT`。它明确限制同步 CTA 的数量，并使用 cooperative launch；其 `x_grid_barrier`
依赖返回旧值的 uint32 atomic、release/acquire 及局部 barrier。[U3][U4]
当前 runtime 给所有候选加入同一个 `S`，主要搜索局部 reduction 配置，并没有默认搜索多个
`S`；因此适配 autotune 也不能仅理解为补一个相同名字的 decorator。

这套设计同时涉及 **IR 不做多 kernel 拆分、schedule 的 phase 边界、kernel 内 workspace、
Grid、候选配置和 launch**。因此即使 Triton 已能编译一个 barrier，也没有自动完成该特性。

### 2.3 CUDA split reduction 何时进入

Split 的入口在 IR 阶段 `Reduction.num_splits()`，不是 cooperative codegen 内的另一种
barrier。普通归约要求 `split_reductions=True`、本次未选择 reduce-to-single/cooperative，
并排除 arg 类归约；`dot` 另行直接返回不拆分。随后根据访存 stride 判断 inner/outer，调用
`reduction_split_factor()`；最终 split 数大于 1 才产生分层归约。[U1][U5]

以常见连续末轴的 inner reduction 为例，当前基线有以下规则：

- 输出数 `X >= 2 * num_sm` 时不拆分；输出方向已提供足够并行度。
- `R <= 8192` 时不拆分；设备 `major >= 10` 时此边界提高到 `524288`。
- 通过上述检查后，根据 `X*R`、SM 数、每 SM 线程数和分片整除性计算分片数；并非一律固定 32。

Outer reduction 使用不同的分块模型；IR 还有小 `R`、大 `X`、未知无 hint 的 unbacked symbol、
已有 producer reduction 可复用等分支。因此这两个阈值只是 inner 路径的明确检查，不能作为
所有 layout 的充分触发条件。相比之下，cooperative 的普通选择限制到更少的输出数，并额外
要求 kernel 内同步协议合法；两者的收益区间可以重叠，最终采用不同的执行计划。

## 3. NPU Inductor 的现状与修改范围

### 3.1 当前有两条代码生成路径需要一起处理

`NPUCombinedScheduling` 先尝试 `NPUTritonScheduling / NPUIndexTritonKernel`；失败后可回退到
`NPUNoLinearTritonScheduling / NPUTritonKernel`。前者使用 NPU 自己的多轴、`BLOCK_SUB` 和循环
代码生成；后者复用更多上游 TritonKernel 逻辑。[N1][N2]

两条路径均未完成 cooperative 接入：

- Index 路径 `_get_heuristic()` 明确只返回 pointwise、reduction、persistent_reduction；
  它重写了 reduction、store、body 和 signature 生成，不能只复用上游初始化代码。[N3]
- 本次强制静态 sum 的 Index 尝试，已在继承的 `init_cooperative_reduction()` →
  `_has_constant_mask()` → `TRITON_MAX_BLOCK[prefix.upper()]` 处出现 `KeyError: 'R'`；
  NPU 轴体系与上游假设不匹配。这是首个已复现断点，修复它不代表后续协议已接通。[E7]
- No-linear 路径可能生成上游 `@triton_heuristics.cooperative_reduction`，但 imports 已替换为
  NPU runtime，后者没有这个 decorator。本次静态实验实际在此抛出 `AttributeError`。
- NPU helpers 通过 `import *` 继承了上游 `x_grid_barrier`，但这仍是 CUDA 风格 semaphore 协议，
  没有变成 NPU 原生同步实现。即使补上 decorator，workspace、peer combine、合法 Grid、
  store owner 和同步协议仍需贯通。[N4][N9]

安装环境的符号检查也确认了上述继承关系，以及 cooperative heuristic 缺失，见 [E10]。

**回退必须是一个明确的普通 reduction 方案。** 不应让 Index 失败后进入另一条同样不支持
cooperative 的路径，再以 decorator 缺失结束编译。

### 3.2 IR 决策存在开关的连带影响

NPU `num_splits()` 被 patch 到公共 `Reduction.num_splits`。其判断使用
`BackendFeature.REDUCE_TO_SINGLE_ELEMENT`；继承的上游 `get_backend_features()` 在 cooperative
开关开启时就加入这个 capability，而 NPU 的 IR 判断没有上游“本次归约实际选中 cooperative”
这一层 predicate。[N5][U5]

因此，**开启配置可能先抑制 split，即使该 shape、算子或 codegen 路径最终不能执行 cooperative**。
能力声明和逐归约选择应共用可实现的准入逻辑，而不是将一个全局 flag 当作每个 reduction 的
最终执行计划。

另一个 NPU 特点是：`X=1`、归约组恰有一个动态归约轴时，当前 `num_splits()` 返回 1；
`SplitTiling.select_split_axis()` 对动态 full reduction 保持 `grid=1`，避免没有跨核合并时覆盖
标量输出。这是后续动态 cooperative 值得优化的入口，也是必须保留的正确性回退。[N5][N6]

静态 split 策略取归约轴 hint 中大于 1 的最小值，与 CUDA 的 SM/阈值策略不同。
`split_reductions=True` 本身不能证明生成了有效的多核两阶段归约，必须检查最终 kernel、Grid
和中间张量。本次 `(a+b).sum(-1)` 的 baseline 与显式 unsplit 都只生成一个 kernel。

### 3.3 组件修改清单

“必须改”指实现 A5 SIMD、静态、单归约 phase 的最小范围；“检查/扩展”按验收范围决定。
文件名均相对 `pytorch_new/torch_npu/_inductor/`，上游复用点另行注明。

| 组件 | 需要完成的工作 | 范围判断 |
| --- | --- | --- |
| `choices.py`、配置与 backend capability | 增加 NPU 准入和收益判定；架构/模式/算子不支持时回退；IR 与 codegen 共用判定 | 必须改，中等 |
| `ir.py::num_splits` | 仅在确实采用 cooperative 时抑制多层拆分；保留正常 split 与动态单核回退 | 必须改，中等，影响方案选择 |
| `codegen/npu_combined_scheduling.py` | 处理 Index/no-linear 回退与已选方案的一致性，恢复失败尝试的状态 | 必须改，中等 |
| `codegen/scheduling.py` | kernel choices、fusion 准入、reduction/pointwise phase 边界、unsupported 组合屏蔽 | 必须改，中到大 |
| `codegen/split_tiling.py` | 为分片安排 program 维度；区分输出分核与归约分片；保证参与者和循环次数合法 | 必须改，大 |
| `codegen/triton.py` | shard 区间、partial workspace、peer combine、全局屏障、指定 store owner、尾块；同时处理两种 kernel 路径 | 必须改，大，主工作量 |
| `runtime/tile_generator.py` | 联合选择 `S`、输出 tile 和 `BLOCK_SUB`，按可用核数/局部资源筛选 | 必须改，中到大 |
| `runtime/triton_heuristics.py` | cooperative 候选、Grid、编译参数、实际 winner、cache 恢复；对性能比较明确候选层级 | 必须改，大 |
| 同步/combine helper，可落在 `runtime/triton_helpers.py` | 选择原生屏障，处理空分片 identity 与 peer 合并；也可由 codegen 直接发出已有 extension 调用 | 协议必须接入，helper 文件按方案改 |
| `codegen/wrapper.py` | 复用 `WorkspaceArg` 分配与释放，核对 stream 生命周期、参数和 runtime Grid | 必须核对，已有基础 |
| `codegen/cpp_wrapper_npu.py` | 同样传递计划、Grid、partial workspace 与编译器隐式 lock ABI | 完整 AOT/C++ 范围需要适配与验证 |
| 测试与 cache/config 接入 | 准入/回退、精度、动态、图捕获、候选一致性；配置进入图缓存与候选缓存键 | 必须改，贯穿各层 |

上游 `WorkspaceArg`、reduction combine helper、schedule marker、autotuner 编译/计时框架可以
复用。现有 NPU wrapper 已有 workspace 和编译器 lock 的处理，不应笼统认定需要重写整个
wrapper。[N7] 同样，正常 `torch.sum` 等 decomposition/ATen lowering 并非首期必须修改项；
只有既有 lowering 已将目标融合链提前拆开时，再按具体图评估。

**工作量是跨多个组件的编译策略与运行时功能，超过局部 codegen patch。** 单阶段 static sum
的受限版本与“上游测试全量通过、动态/多 phase/AOT 全兼容”是不同交付范围；现有证据不足以
给出可靠的行数或人天估算。

## 4. 特性正交性与准入边界

“可组合”表示语义上可设计实现；不表示当前 NPU Inductor 已支持。以下是建议的接入契约。

| 特性 | 上游/现有实现的约束 | NPU 接入要求与建议 |
| --- | --- | --- |
| 普通 pointwise prologue/epilogue | CUDA 支持融合，所有分片可获得最终归约值 | 首期支持简单 prologue 和标量 epilogue；只允许指定分片写唯一输出 |
| reduction → pointwise → reduction | CUDA 利用 phase flush、屏障和 workspace 周期 | 需要独立 phase/liveness 适配；workspace 重用前还要保证旧读者完成，不能只插一次 barrier |
| Softmax / chained softmax | 包含 max、sum、广播，可能有多个同步阶段 | 是有价值的目标场景；首期不以单 sum 原型代替其验收 |
| persistent reduction | CUDA cooperative 可以与 persistent 组合 | 需按 `R/S`、活跃输入、UB/寄存器重新判断；上游阈值乘数不能直接复用 |
| split reduction | 是同一归约的另一种执行方案 | 选 cooperative 时不再做同一层 split；失败时保留完整替代方案 |
| multi_kernel | CUDA 可在部分同一 schedule 的候选间选择 | NPU `create_kernel_choices()` 没有接完整上游候选流程；不能认为已自动比较完整 split pipeline |
| max_autotune / symbolic grouped autotune | NPU tile 与 group 已有自己的 schema | 候选与缓存键必须携带 `S`、同步方式、模式和合法 Grid；不同 group 不能生成不可推进的参与者集合 |
| 动态 `R` | 上游普通 predicate 要求静态证明下界；force 绕过性能判断 | 可设计非 persistent 循环版；尾块、空分片、变小 `R` 及 fallback 都需验证 |
| 动态 `X` | CUDA `S` 按 hint 选，runtime Grid 仍随 `X` 变化 | 必须约束实际参与 program 数；不可把样本 `X` 或样本 workspace 大小作为运行时上界 |
| unbacked SymInt | 不能将未知 shape 等同有 hint 的 backed symbol | 首期回退；如扩展，需要上界、workspace 与 launch 可合法计算的完整证明 |
| 多个逻辑归约轴 | 可先合并成一个 `R`；CUDA cooperative 不支持任意 tiled R 维 | 可合并且索引正确的场景再准入；nested/tiled reduction 首期排除 |
| Welford / var / std | CUDA cooperative 要合并完整 mean、M2、weight | NPU 已有自己的 variance 路径；不能平均 partial mean 或丢弃空片 count，需要专门适配 |
| argmin/argmax、min/max 返回 index | 多分片涉及全局 index、tie-breaking 和 NaN | 首期回退；不能只合并局部 value/index 而不修正分片偏移 |
| any/all、prod、极值 | 各自有 identity、mask、dtype 语义 | sum 后逐类扩展；尾块必须用正确 identity，NaN/Inf 与整型行为单独看护 |
| sort / scan | CUDA 为 sort、scan 关闭 cooperative；NPU scan 也显式关闭 | 保持排除，避免把 scan carry 误当普通归约合并 |
| foreach / combo kernel | 上游明确 `override_cooperative_reduction=False` | 保持排除；不同子 kernel 的 dispatch 分支不保证共同到达全局屏障 |
| mix-order reduction | 上游相关测试依赖 split 并关闭 cooperative | 首期排除，需独立设计访存/phase 顺序后再开放 |
| deterministic | NPU 会绕过 reduction 计时，按规则选择 tile | `S` 改变运算树；需要稳定候选顺序、选择规则和精度契约，不承诺跨配置 bitwise 一致 |
| NPU Graph、多个 stream | 原型 graph replay 通过，完整 Inductor 尚未接入 | workspace/隐式锁应按 invocation/stream 隔离；capture/replay 不得依赖未捕获的初始化 |
| C++ wrapper / AOTInductor | launch ABI、Grid 与 workspace 路径不同 | 不能以 Python wrapper 成功代替验证；需打包后、重载后和动态参数回放验收 |
| SIMD / template SIMT / pure SIMT | 本次前两者编译通过，pure SIMT 失败 | 首期限制到已完成运行验收的 SIMD；后续按模式单独开放 |
| NDDMA / multibuffer / VF 优化 | NPU 编译存在访存和异步 pipeline 优化 | 跨核同步必须具有正确的副作用/内存依赖，禁止将相关 GM 读写跨 barrier 移动；逐组合验证 |

代码依据：CUDA phase flush、Welford、feature overrides、combo 与 runtime 候选见 [U2][U3][U6]；
NPU scan 排除、tile 资源约束、deterministic 选择、grouped/C++ 路径见 [N2][N4][N8]。
上表关于后续接入的内容是设计要求，不是对当前支持状态的推断。

### 动态 shape 的关键区别

`dynamic=True` 不等于不能走 cooperative；它也不等于一定会选中 cooperative。
至少要分别检查：

1. **选择是否成立**：能否证明 `R` 足够大、输出方向并行不足，或有合法的 runtime dispatch。
2. **执行是否安全**：实际 `X` 改变后，Grid、workspace、参与者和 barrier phase 是否仍有效。
3. **配置是否适配**：原来的 `S/BLOCK_SUB` 对新的 `R` 是否仍有收益，是否需要 group 或回退。

例如 `X` 从 1 变成 128，而 `S` 仍为 32，就不能继续把 `128*32` 个逻辑 program 当成可同时
同步的一组。可以保持固定物理 worker 数并让各 worker 循环处理多个输出，但这是另一套明确的
调度与同步协议，不是简单保留当前 Grid 公式。

## 5. Triton-Ascend / NPUIR 能力评估

### 5.1 已有能力与明确缺口

| 能力 | 本次证据 | 判断与归属 |
| --- | --- | --- |
| 原生全 Vector 核同步 API | 安装包 `extension.core.sync_block_all`；TTIR/adapter 中有 `ALL_VECTOR` | 已有，不需要从零增加 Python API |
| SIMD 模式 GM 写 → 同步 → peer 读 | 1/2/8 program、8 phase 检查及 sum 原型运行正确 | A5 受测范围可用；仍需更大覆盖证明完整协议 |
| SIMD/SIMT template 的原生同步编译 | 固定 target/constexpr 编译成功 | 有编译基础，运行未由该结果证明 |
| 纯 SIMT 的原生同步 lowering | `hivm.hir.sync_block` legalization 失败 | 本安装栈的明确缺口；Triton/NPUIR 联合定位，或先禁止组合 |
| CUDA semaphore helper 编译 | SIMD、template 成功；pure SIMT 在 bisheng 报 `MaxThreads out of range` | pure SIMT 失败原因未定，不能直接归为同步原语缺失 |
| atomic 返回旧值、acquire/release 的完整语义 | TTIR 保留语义；adapter 变为 copy 旧值 + atomic store，未显式保留相同属性 | 尚未证实满足 CUDA 协议，需 compiler owner 确认及专门 litmus |
| 任意输出组的独立 barrier | 原生 API scope 是 all_vector 等，未提供按 row/program group 的参数 | 不能直接替代 CUDA 的每个 y/z 组 semaphore；可先设计全 Grid 一致 phase |
| cooperative launch/驻留能力声明 | `launch_cooperative_grid` 出现在 options；Ascend driver 未见同名消费 | 字段存在不等于 CUDA 的 cooperative launch 保证；需明确后端契约 |
| program → 物理 block 映射 | driver 将三维 Grid 展平，auto-map 时可限制实际 block 数 | 必须纳入合法性检查，不能仅检查逻辑 `S <= AIV` |
| 编译器隐式 lock/workspace ABI | driver、C++ wrapper 已有 lock 数、初始化和分配流程 | 已有基础；原生同步、graph、C++ 和并发仍需一致验收 |

安装包 API 声明 `all_cube`、`all_vector`、`all`、`all_sub_vector`，event ID 为 0～15，
并描述 GM RAW/WAR/WAW 同步；这些 scope 不是“任意 program 子组”。本次只运行
`all_vector`。见[已安装源码片段][E1a]和[后端 launch/编译接口片段][E1b]。

公开上游 issue 的维护者也区分了 Ascend 950 与早期 910 的支持状态。该 issue 已关闭，
不能把 910B2 的历史超时直接作为 A5 不支持的证据。[T1]

### 5.2 推荐的底层合作边界

首期优先 **NPU 原生 barrier + 全 Grid 统一 phase**，不把复现 CUDA semaphore 自旋协议作为
前置条件。它避免依赖尚未证实的 atomic 旧值和 release/acquire 契约，但需要明确：

- 哪些 SoC、compile mode 和 kernel 类型允许该同步；
- barrier 的物理参与者、Grid 映射、auto-map/sub-block 绑定和超额 program 的规则；
- GM 写/read 的重排与缓存可见性保证，多个 event 的重用规则；
- 全部参与者每个 phase 都到达 barrier，包括无效输出/空分片，不能放到不同步的条件分支；
- 多 stream、graph replay、隐式锁初始化和 C++ ABI 的一致性。

其中纯 SIMT lowering 失败是**已复现的功能缺口**；能力查询、分组同步、驻留和内存序契约属于
**待确认/需产品化的接口要求**。不能仅凭 Python 层错误把全部工作归到 NPUIR，也不能把一个
编译成功的 kernel 等同于正确的并发协议。
对于受限 A5 SIMD 首期方案，已有原生同步可以作为起点，目前没有证据要求先新增底层同步指令；
扩大到 pure SIMT 则需要解决已复现的 lowering 缺口。

## 6. 实测与证据

### 6.1 版本、环境与复现方法

日期：2026-11-18。Docker 内 `/home/l30023782/PTA_213`，conda `pta_213_case`。
经用户授权共享物理卡 2，通过 `ASCEND_RT_VISIBLE_DEVICES=2` 使用逻辑卡 0。
`aclrtGetPhyDevIdByUserDevId(0)` 返回物理 2，见[设备映射][E8]。
没有修改已安装的包，没有操作其他进程。

| 项目 | 实际值 |
| --- | --- |
| 设备 | Ascend950PR_9579，28 Cube / 56 Vector core，128 MiB L2，设备报告 UB 216 KiB |
| PyTorch | `2.13.0+cpu`，`torch.version.git_version=cf30153c4c131c8164ee7798e5022d810682e2cb` |
| torch_npu package metadata | `2.13.0+git76443ac` |
| torch_npu version.py | `2.13.0+gitefc05dd`，`efc05dd90b5fe31717d5f13e51fb3e49035bea30` |
| Triton / Triton-Ascend | `3.6.0` / `3.6.0+dev20261112182052` |
| 实际设备编译器 | Triton 包内 BiShengIR `2.0.0`，AscendNPU-IR `395c9c5646f9`，2026-11-12 |
| CANN / bisheng | `cann-9.1.0`；pure SIMT 诊断中的 bisheng 版本为 2026-07-30 |

本次未设置 `NPU_DEVICE_LIMIT`；Inductor 与 Triton 查询的有效 Vector core 数均为 56，见 [E11]。

`pip metadata`、`version.py` 与部分已安装 Python 源码不完全对应；因此同时保留
[源码 SHA256 与版本信息][E1]以及[本地源码对应检查][E1c]，不能只用 `pip show` 的版本号追溯。
`PATH` 上还存在 BiShengIR 1.2.0，实际 Triton 使用的是包内 2.0.0，见
[实际 compiler 路径][E2]。本地 Triton-Ascend checkout 较旧，仅用于追踪结构，不用于宣称
安装版本的实现状态。

源码分析基线：本地 CUDA PyTorch `887a5ee68a4`，NPU `pytorch_new`
`961b95d9acb79551c52a77e991db9dd2613d6d86`，用例仓
`51830b03317074409a04c0ff167ee0d7ad6c897c`。
实验相关 PyTorch choices/triton、NPU ir/triton/split_tiling/runtime 与安装文件的对应关系
见证据目录；scheduling 有差异，分析以安装片段和实际 generated code 为准。

运行脚本及参数说明见 [probe README][P1]。本次编译与首轮静态/原型实验使用脚本提交
`a2e29924870987a2f88c0594baca8e4e13d59a7f`；预热后原型计时使用 `1b18319`，
最终动态复用、固定 4 warps 编译和回退诊断使用 `ebfeedb`。
[阶段提交映射][E9]和各目录 `command.json` 保留完整参数。

### 6.2 Inductor 行为

计算为 fp32 `(a+b).sum(-1)`，`R=1048576`，`fullgraph=True`。
`enabled` 是普通 cooperative 开关；`forced` 同时设置 force；`unsplit` 关闭 split 和 cooperative。

| Shape/模式 | baseline | unsplit | enabled | forced |
| --- | --- | --- | --- | --- |
| 静态 `X=1` | PASS，1 kernel | PASS，1 kernel | FAIL，decorator 缺失 | FAIL，decorator 缺失 |
| 静态 `X=32` | PASS，1 kernel | PASS，1 kernel | PASS，普通 reduction | FAIL，decorator 缺失 |
| 动态 `R`、`X=1` | PASS，普通 reduction | 未测 | PASS，普通 reduction | FAIL，decorator 缺失 |

静态记录见 [inductor.json][E5]，动态记录见 [dynamic-reuse/inductor.json][E6]。
强制路径的两个断点是：Index 初始化的 `KeyError: 'R'`，随后 no-linear module 加载阶段的
`AttributeError: ... triton_heuristics ... cooperative_reduction`。后者尚未进入该 cooperative
kernel 的设备编译，不能拿它证明 NPUIR 不支持归约或同步。

实际 winner 也说明当前的性能基线：静态 `X=1` 选择 `R0BLOCK=1048576`、
`R0BLOCK_SUB=8192`，Grid 为 1，在同一 program 中循环；`X=32` baseline 选择输出方向
`X0BLOCK=1`、归约子块 8192。普通开关下 `X=32` 的 generated code 已变为 no-linear
`Grid1D` 路径，依然不是 cooperative。不能只根据执行成功或 kernel 数判断路径。

最终动态复用实验从 `R=1048576` 首次编译，再直接连续调用同一 callable，输入 `R=32768`
和 `R=1048593`；baseline 与 enabled 均正确，`new_kernels=0`，原代码保留 runtime `R`。
这验证了**当前普通归约的动态复用**，没有验证 NPU cooperative 的动态支持。
复用阶段没有调用会重置 Dynamo 的 `run_and_get_code()`；源码采集通过无 reset 的 hook 完成。

### 6.3 固定配置编译与原生同步

| 探针 | SIMD | SIMD/SIMT template | pure SIMT | launch |
| --- | --- | --- | --- | --- |
| native cooperative sum | PASS | PASS | FAIL：sync_block legalization | 编译探针均未 launch |
| native 8-phase visibility | PASS | PASS | FAIL：sync_block legalization | 编译探针均未 launch |
| 上游 `x_grid_barrier` | PASS | PASS | FAIL：`MaxThreads out of range` | 未 launch semaphore 自旋协议 |

上表使用显式 `GPUTarget("npu", "Ascend950", 0)`，固定 `num_warps=4`。
初次使用后端默认 32 warps 的记录另存 `compile/`；显式改为 4 后，semaphore 探针仍在 bisheng
报同一线程数错误，其来源未进一步定位，不据此提出“需要新增 atomic/barrier 指令”的结论。
原生屏障的失败则明确指向 pure SIMT 的 `sync_block` legalization。
另用实际设备型号 `Ascend950PR_9579`、4 warps 复核 pure SIMT，三项结果及错误类型一致，
见 [compile-simt-exact/compile.json][E12]。

原始 [compile-fixed/compile.json][E3] 保留 signature、constexpr、模式、编译 metadata 和完整 traceback；
对应目录保留未改格式的 TTIR、adapter 和 pre-failure IR。

单独的设备运行只使用原生 SIMD `all_vector`：1、2、8 program 的每阶段 peer 读取与期望一致，
覆盖 8 phase；sum 原型覆盖非整除尾块和多输出组，使用 NPU Graph 重复回放，均通过精度检查。
这证明受测协议可作为接入起点，不证明任意 Grid、任意 phase/control-flow 或其他架构都支持。

### 6.4 性能方法与结果

表中是手写 `sum(a)` 原型，**不是已接通的 Inductor cooperative kernel**，也不能直接与上面
两输入 `a+b` 的耗时比较。三种策略固定 SIMD、fp32、`B=1024`，没有搜索最优 tiling；
两阶段方案和 cooperative 使用相同分片区间。`S` 取不超过 `min(64, 56/X)` 的 2 次幂。
“单 program”列是每个输出一 program，并非全表都只使用一个物理核。

计时在 NPU Graph 中捕获 32 次计算，先预热 3 次 replay，再用 device event 计时 10 次 replay，
折算单次耗时；5 组样本取中位数。三种方案各自 warm up，所有结果通过 `atol=0.02`、
`rtol=0.0002` 的 fp32 检查。原始样本与配置见 [bench-warm/bench.json][E4]。

| `X × R` | `S` | 单 program/输出（µs） | 两阶段 split（µs） | 原生 cooperative（µs） |
| --- | --- | --- | --- | --- |
| `1 × 4096` | 32 | 1.013 | 4.544 | 4.683 |
| `1 × 32768` | 32 | 3.533 | 4.478 | 4.881 |
| `1 × 1048576` | 32 | 90.667 | 7.176 | 7.531 |
| `1 × 16777216` | 32 | 2623.762 | 124.459 | 124.551 |
| `8 × 1048576` | 4 | 93.649 | 27.971 | 27.878 |
| `32 × 1048576` | 1 | 255.201 | 259.276 | 258.221 |
| `1 × 1048593` | 32 | 90.832 | 7.381 | 7.539 |

本次观测支持以下判断：

- CUDA 在 `X=1、R=32768` 已达到普通 cooperative 阈值，但该 NPU 原型比单 program 慢。
  相同阈值不适合作为 NPU 的直接默认配置。
- `X=1、R=1048576` 跨核分片有明显收益，但 split 已达到约 7.18 µs；cooperative 约 7.53 µs，
  本轮没有证明省一次 launch 能抵消原生同步和 peer 合并的代价。
- `X=8` 两种跨核方案的差异很小；`X=32、S=1` 根本没有 cooperative 分片收益，应该绕过协议。
  不能把很小的正负差异当作稳定加速。

边界：共享卡有其他任务；跨轮 `X=32` 的单 program 中位数曾约 96～255 µs，说明负载或缓存
状态会改变观测。多数输入不超过本卡 128 MiB L2，重复 graph replay 也是热数据，不能据此推算
HBM 峰值带宽。当前未测独占卡、冷数据、最优 tiling、CANN 算子性能或真实 softmax 融合收益，
所以这些数据用于排除错误假设和指导后续选点，**不用于发布性能承诺或生产阈值**。

## 7. 性能模型与 NPU 触发条件建议

### 7.1 比较三种方案，而不只看 CUDA 阈值

以下是决策模型，不是已测得的硬件公式。记 `Q` 为实际输出方向 program 数，`C` 为该模式下
可用于该 kernel 的物理并行单元数，`L` 为一次 launch 的相关开销，`D` 为原输入/输出流量。

```text
T_single ≈ L + local_reduce(R, B, live_values) with Q-way parallelism
T_split  ≈ 2L + parallel_reduce(R/S, B) + partial_write/read + final_reduce(S)
T_coop   ≈ L  + parallel_reduce(R/S, B) + partial_write/read
               + barrier + peer_reduce(S) + fusion/live-value effects
```

`T_coop < T_single` 可能主要来自分核；需求是否值得上线还要看 `T_coop < T_split`。
如果 NPU Graph 降低了额外 launch 的相关开销，而全核 barrier 的开销较大，单 sum 的 cooperative
不一定占优。连续归约融合减少中间张量/重读输入时，又可能改变比较结果，需按完整融合图测量。

还应区分：

- `R` 很小：即使 `Q` 小，也可能是同步开销主导；
- `Q >= C`：输出方向已有并行度，再分 `R` 的价值变小；
- `S=1`：没有 cooperative 并行收益，应省掉同步和 peer 协议；
- `R/S` 太小：过多分片导致空片或小工作量 program；
- UB/寄存器不够：persistent 的资源风险与 `R`、dtype、融合输入和活跃值相关；
- 全局屏障：不同输出组互相等待，耗时由最慢参与者决定，跨组不均匀会增加尾部代价。

### 7.2 建议的准入顺序

```text
1. capability: SoC、compile mode、原生同步契约已验证
2. legality: 单归约形态、支持的 combine、统一 phase、合法 Grid/映射和 workspace
3. parallelism: 当前 Q 不足，存在 S > 1 且每片有足够工作量
4. resources: B、S、persistent/looping 的 UB 与寄存器资源合法
5. profitability: 预测或实测 cooperative 优于单核和 split 完整方案
6. 否则回退；force 只绕过第 5 项，不能绕过前四项
```

首期可以静态 shape 和 SIMD 为边界，用保守的配置集比较 `S` 与 `BLOCK_SUB`，逐步建立
按 `X/R/dtype/layout/fusion/graph mode` 区分的收益表。`S` 上限应同时受实际 program 映射、
有效核数与最小分片工作量约束，不能只有 `min(R, 64)`。

CUDA 的 `X<=16`、`32768*X` 和以 SM 为基础的 `S` 不能直接作为 NPU 默认值。
设备属性与 compiler 保留空间也要分开：本卡报告 UB 为 216 KiB，而本地 NPU tile 配置给 A5
设置 256 KiB；这个配置常量不能单独证明一个 cooperative persistent tile 能放下。[N8]

注意 **tiling autotune 与方案 autotune 的层级不同**：一旦 IR 已生成两阶段图或已抑制拆分，
在单个 kernel 的 `cached_autotune` 中挑 `B/S`，不会自动比较另一个完整 pipeline 的总耗时。
首期可采用明确的策略准入和原路径回退；如要三种方案实测择优，需单独设计图/方案层候选。

本次建议先建立正确分核的 split baseline，再判断 cooperative 的独立收益。
仅比当前单核路径快，不足以支持默认启用 cooperative。
autotune 的预运行还应遵守 workspace/lock 的初始化契约：原生协议要求每次覆盖全部有效 partial，
semaphore 协议如被采用则必须处理初始状态及重复调用，不能让上一候选的状态影响下一候选。

## 8. 建议交付阶段与验收

| 阶段 | 交付边界 | 验收依据 |
| --- | --- | --- |
| P0：能力契约 | A5 SIMD，同步参与者/GM/映射/锁契约；pure SIMT 明确排除 | 原生屏障、重复 phase、合法/非法 Grid、缓存与并发约束记录 |
| P1：静态最小接入 | 静态单 `R`、fp32 sum、非 persistent、简单融合；完整回退 | 结果正确，实际生成 shard/peer combine/屏障，命中与未命中图均验证 |
| P2：收益选择 | `S/B` 联合选择，单核/split/cooperative 比较，ATen 作参考 | 独占卡、冷热缓存、Graph 开关、实际 selected config，建立阈值区间 |
| P3：融合与算子 | repeated reduction/softmax、min/max、dtype promotion、Welford 等逐项扩展 | tail、NaN/Inf、tie、空片、各 phase workspace 与精度验收 |
| P4：动态 | backed dynamic R，再到有安全上界的 X 和 grouped autotune | 同一 compiled callable 跨阈值/尾块复用，合法 Grid、workspace 与回退，记录 recompile |
| P5：完整运行时 | C++ wrapper/AOT、Graph、多 stream、persistent、其他编译模式 | 打包重载与并发回放；能力矩阵与默认回退测试 |

验收不能只检查字符串 `cooperative_reduction` 或 kernel 数为 1。
原用例 `test_reduction_fns` 的 `(1, 1024**2)`、多个 reduction 与 dtype 是回归入口，但还包含
CUDA decorator/launch 假设。NPU 测试应检查自己的计划、partial 形状、同步作用域、store owner
和正确回退；浮点类型的基础硬件支持与 cooperative 功能分别判断。[C1]

最低测试集合应覆盖：`S=1/S>1`、`R<S*B`、非整除 `R`、输出尾块、`X` 超过准入上限、
unsupported 算子/模式回退、同一 kernel 多次调用与 workspace 重用、多个 reduction phase。
不能依赖设备超时来拒绝非法 Grid；不满足同步契约的配置应在编译或候选筛选阶段排除。

## 9. 源码与实验索引

代码链接固定到本次分析基线；CUDA 链接采用各文件最近修改的公开提交，文件内容与本地基线相同。
安装版本差异以证据快照为准。

- [U0] CUDA 配置默认值；[U1] `choices.py`：cooperative/persistent 与 split 启发式。
- [U2] CUDA `TritonKernel`：cooperative 初始化、peer combine、phase barrier、store owner。
- [U3] CUDA `triton_heuristics.py`：RSPLIT 候选与 Grid。
- [U4] CUDA `x_grid_barrier`：semaphore 协议。
- [U5] CUDA `Reduction.num_splits`：capability 与实际选择的联合判断。
- [U6] CUDA SIMD schedule、combo 等约束的主入口。
- [N0] NPU backend loader；[N1] combined scheduling 与回退。
- [N2] NPU scheduling；[N3] 两类 NPU Triton kernel。
- [N4] NPU runtime heuristics；[N5] IR split patch；[N6] split tiling。
- [N7] NPU C++ wrapper；[N8] tile generator 与局部资源约束。
- [N9] NPU helpers 对上游 helper 的继承与自身覆盖。
- [C1] 用户指定的 2.13 cooperative 用例。
- [P1] 实验脚本；[E1] 安装栈和源码快照；[E2] 实际 NPUIR 编译器；[E3] 编译记录。

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
[N9]: https://gitcode.com/luqichao/pytorch_new/blob/961b95d9acb79551c52a77e991db9dd2613d6d86/torch_npu/_inductor/runtime/triton_helpers.py#L5
[C1]: https://github.com/limbochao/Community-Use-Case-Testing/blob/51830b03317074409a04c0ff167ee0d7ad6c897c/v2.13.0/inductor_npu/test_cooperative_reductions.py
[P1]: ../../scripts/tests/cooperative_reductions/README.md
[E1]: ../../artifacts/npu-cooperative-reductions-20261118/inspect/stack.json
[E1a]: ../../artifacts/npu-cooperative-reductions-20261118/inspect/installed-source-excerpts.txt
[E1b]: ../../artifacts/npu-cooperative-reductions-20261118/inspect/installed-backend-contracts.txt
[E1c]: ../../artifacts/npu-cooperative-reductions-20261118/inspect/source-comparison.json
[E2]: ../../artifacts/npu-cooperative-reductions-20261118/inspect/actual-npuir-compiler.json
[E3]: ../../artifacts/npu-cooperative-reductions-20261118/compile-fixed/compile.json
[E4]: ../../artifacts/npu-cooperative-reductions-20261118/bench-warm/bench.json
[E5]: ../../artifacts/npu-cooperative-reductions-20261118/inductor/inductor.json
[E6]: ../../artifacts/npu-cooperative-reductions-20261118/dynamic-reuse/inductor.json
[E7]: ../../artifacts/npu-cooperative-reductions-20261118/index-fallback.log
[E8]: ../../artifacts/npu-cooperative-reductions-20261118/device-map.json
[E9]: ../../artifacts/npu-cooperative-reductions-20261118/validation-provenance.json
[E10]: ../../artifacts/npu-cooperative-reductions-20261118/inspect/runtime-symbols.json
[E11]: ../../artifacts/npu-cooperative-reductions-20261118/inspect/effective-cores.json
[E12]: ../../artifacts/npu-cooperative-reductions-20261118/compile-simt-exact/compile.json
[T1]: https://github.com/triton-lang/triton-ascend/issues/1738
