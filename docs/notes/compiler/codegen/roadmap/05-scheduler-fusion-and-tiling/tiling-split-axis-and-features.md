---
title: Tiling、Split Axis 与 Kernel Features
---

# Tiling、Split Axis 与 Kernel Features

本文直接回答轴选择的实现问题。这里的 `split axis` 特指 Ascend NPU backend 中决定
`tl.program_id(dim)` 的轴；它不是泛指数学上的 `split`。CUDA 上游没有同名决策器，不能把两者混为一谈。

## 源码入口

- `<workspace>/pytorch_new/torch_npu/_inductor/codegen/split_tiling.py`
  - `SplitTiling.__init__`
  - `select_split_axis()`
  - `select_tiling_axis()`
  - `select_no_loop_axis()`
  - `_build_grouped_meta()`
- `<workspace>/pytorch_new/torch_npu/_inductor/codegen/triton.py`
  - `IterationRangesEntryNPUIndex`
  - `NPUIndexTritonKernel._finalize_kernel_codegen_dims()`
- `<workspace>/pytorch_new/torch_npu/_inductor/codegen/scheduling.py`
  - `NPUTritonScheduling.candidate_tilings()`
- `<workspace>/pytorch/torch/_inductor/codegen/simd.py`
  - `SIMDScheduling.candidate_tilings()`
  - `codegen_node_schedule()`

## 实现展开：先区分三个轴属性

`IterationRangesEntryNPUIndex` 给每个 range-tree entry 增加了三个相互独立的标记：

| 属性 | 决定什么 | generated Triton 中的效果 |
|---|---|---|
| `is_split_axis` | kernel grid 的一个维度 | `tl.program_id(split_order) * BLOCK` |
| `is_tiling_axis` | 一个 program 内用 block 表示的维度 | 生成 `tl.arange(0, BLOCK_SUB)` 和 mask |
| `is_no_loop_axis` | 整条轴一次驻留，不再分块循环 | 直接使用 `base_axis`，不生成 `loop_axis` 偏移 |

因此一个轴可以同时是 split axis 和 tiling axis。前者决定 program 间的切分，后者决定 program 内的向量块；
`is_no_loop_axis` 只是 tiling axis 的一种优化形态。

调用顺序不是猜测出来的。`NPUIndexTritonKernel._finalize_kernel_codegen_dims()` 明确执行：

```text
SplitTiling(kernel)
  -> select_split_tiling_axis()
       -> select_split_axis()
       -> select_tiling_axis()
       -> apply_grouped_rewrite_if_needed()
  -> select_no_loop_axis()
  -> _select_vectorized_welford_axis()
```

`apply_grouped_rewrite_if_needed()` 可能降级已经选中的 split axis，所以阅读最终结果时应看 rewrite 之后的
`kernel.split_axis`，不能只看 `select_split_axis()` 的中间结果。

## 输入轴是怎样排序的

`SplitTiling.__init__()` 从 `kernel.range_tree_nodes.values()` 取得轴，并用 `key()` 逆序排序。规则是：

1. 名称以 `w`、`v`、`t` 开头的轴，key 前加 `zz`，因此被排到更高优先级。
2. 表达式是 `ModularIndexing` 的轴，在名称前缀后插入 `0`，使它低于对应的 floor-division 轴。
3. 其余轴直接按名称排序。
4. 排序后写入 `axis.sorted_order`；后续所有“从高维开始选择”均指这个顺序。

这不是按轴长度排序。`find_longest_dimension()` 虽然存在，但当前 `select_split_axis()` 没有调用它。
所以“最大的轴一定先被 split”不是当前实现规则。

`find_lowest_dimension()` 再从 load/store 的索引表达式分析 `kernel.low_dims`。split 选择会优先避开这些低维轴，
而 tiling 选择会优先纳入它们。这正是“高维负责 program 并行、低维负责 program 内连续访问”的代码化表达。

## NPU split axis 的完整算法

### 停止条件

`select_split_axis()` 每轮先检查 `meet_stop_condition()`。停止条件按以下顺序判断：

1. 开启 `enable_symbolic_shape_group_autotune`，kernel 含 reduction，且当前 split axes 中已有动态轴：立即停止。
2. 尝试对当前 split axes 长度乘积调用 `sizevars.size_hint()`。
3. 如果无法取得 hint：总轴数不超过 2 时选满 1 个 split axis，否则选满 2 个。
4. 如果乘积不小于 `num_vector_core`，并满足 72% 利用率公式：停止。
5. split axis 已达到 3 个：停止。

第 4 条的源码公式为：

```text
val >= num_vector_core
and
val // ceil_div(val, num_vector_core) > num_vector_core * 0.72
```

令 `rows_per_core = ceil_div(val, num_vector_core)`，那么
`val // rows_per_core` 可理解为在这种均分下实际能被填充的 core 数估计。实现要求它超过总 core 数的 72%。
因此“split 轴乘积超过 core 数”只是必要条件，不是完整条件。

最多 3 个 split axes 直接对应 Triton 三维 launch grid。达到 3 个后，即使利用率公式还未满足也会停止。

### 选择顺序

主循环可等价写成下面的伪代码：

```python
split_axis = []
while not meet_stop_condition():
    axis = first(sorted_axis, non_reduction and not_low_dim and not_selected)
    if axis is None:
        axis = first(sorted_axis, non_reduction and not_selected)
    if axis is not None:
        axis.is_split_axis = True
        split_axis.append(axis)
        continue
    if safety_counter > 10:
        break
```

这给出几个可以直接回答的问题：

- reduction axis 在主循环中永远不会被选为 split axis。
- 非 reduction 的高维轴先选；没有可选高维轴后，才允许选择 `low_dims` 中的轴。
- 是否出现多个 split axes，取决于已选轴长度乘积能否满足停止条件，而不是固定按 rank 决定。
- 三个是硬上限；十次循环只是没有候选时的安全保护，并不是允许十个 split axes。

### 空结果 fallback

主循环可能一个轴也选不到，例如只有 reduction axes。此时实现默认选择 `sorted_axis[0]`，但有一个重要例外：

```text
所有轴都是 reduction axis
且至少一个轴是动态长度
=> 保持 split_axis 为空
```

源码注释给出的硬件原因是 A5 没有跨 core reduction combine。若把动态 reduction axis 作为 grid split axis，
多个 program 会写同一个标量输出。于是动态 full reduction 必须保持 `grid == 1`，在 program 内循环运行时长度并
通过尾部 mask 处理。

静态 full reduction 不满足这个例外，会走默认 fallback。这一点说明动态和静态场景不只是 hint 不同，语义安全路径也不同。

## 什么情况下会有多个 split axes

根据主循环，多个 split axes 只会在以下链条成立时出现：

1. 第一个候选是非 reduction 高维轴。
2. 它的长度乘积不能达到 core 利用率停止条件，或 hint 无法解析且总轴数大于 2。
3. 仍存在未选择的非 reduction 轴。
4. 第二、第三个候选继续按“非低维优先、低维兜底”的顺序加入。

示例只是对算法的推演，不代表固定硬件参数。设 `num_vector_core = 20`：

| `sorted_axis` | `low_dims` | 推演结果 |
|---|---|---|
| `x2=64, x1=128, x0=16` | `{x0}` | `x2` 已足以满足条件，通常只有一个 split axis |
| `x2=4, x1=8, x0=128` | `{x0}` | 先选 `x2`，不足再选 `x1`；低维 `x0` 暂不参与 |
| `x2=2, x1=2, x0=128` | `{x0}` | 高维乘积仍不足，最后可能把 `x0` 作为第三个 split axis |
| `r0=S0` | `{r0}` 或空 | 动态 full reduction 保持空 split，grid 为 1 |

真正推演时要代入源码里的 72% 公式，不能只比较乘积是否大于 `num_vector_core`。

## static shape 与 dynamic shape 的差异

### 普通路径

静态轴长度是 `sympy.Integer`，`total_split_numels()` 可以直接相乘。动态轴会先用
`sizevars.var_to_val` 替换示例值，再由 `size_hint()` 尝试求值。求值失败时采用“轴数不超过 2 选 1 个，
否则选 2 个”的结构性 fallback。

这意味着未开启 grouped autotune 时，动态 shape 的一次示例 hint 仍可能促使算法加入第二个静态 split axis。
这个选择会固化在生成 kernel 的轴结构中，并不是每个运行时 shape 重新选轴。

### grouped symbolic autotune 路径

开启 `enable_symbolic_shape_group_autotune` 后，reduction kernel 一旦已经选到动态 split axis，停止条件立即成立。
源码注释说明：运行时 grid 已由动态 numel 提供，再为了示例 hint 补一个静态轴属于推测，而且 split axis 之后不能变成
no-loop axis，会永久破坏内层整轴驻留。

随后 `_build_grouped_meta()` 还会执行第二轮整理：

1. `_classify_split_axes()` 以 `axis.length` 是否为 `sympy.Integer` 分成动态和静态 split axes。
2. `_select_primary_group_axis()` 按 `sorted_axis` 顺序选第一个动态 split axis 为 primary。
3. 其余动态 split axes 通过 `_downgrade_split_axes()` 取消 split 标记。
4. pointwise 且仅有一个动态 split axis 时，若截至该动态轴的 split 乘积已经足够填 core，位于它之后的静态
   split axes 也会被降级。
5. 如果没有动态 split axis，reduction 尝试选择唯一动态 reduction tiling axis；pointwise 尝试选择唯一动态、
   已 tiled、未 split 的 transpose/broadcast axis。

所以 grouped 模式下“多个动态 split axes”不会直接全部保留。它将一个动态轴作为 runtime grouping 主轴，其他动态轴
记录为 `secondary_runtime_symbolic_axes`，并从 grid split 集合移除。

## tiling axis 的真实规则

`select_tiling_axis()` 不是按候选 block size 搜索；它只决定哪些逻辑轴进入 program 内的 tiled 空间。

当前调用 `select_tiling(low_dim=True, reduction=True)` 会逆序扫描 `sorted_axis`，并：

- 把所有 `low_dims` 加入 tiling axes；
- 把所有 reduction axes 加入 tiling axes；
- 同一轴只加入一次。

对于非连续的多维 reduction，`meet_stop_condition()` 要求所有 reduction axes 都已是 tiling axis。源码注释称
Triton-Ascend 当前最多支持二维，但当前函数本身没有用 `len(tiling_axis) == 2` 截断；阅读时应以实际 append 条件为准，
不要把注释误写成已实现的限制。

函数中保留了“选择最长轴”的分支，但当前两次有效路径中要么 `low_dim`、要么 `reduction` 为真时直接 `continue`；
后续以两者均为 false 调用时循环体也不会进入最长轴分支。因此当前可观察规则就是 low dims 加 reduction axes。

## no-loop axis 的 4 KiB 规则

`select_no_loop_axis()` 只从“已是 tiling、不是 reduction、不是 split”的轴中选择：

1. `low_dims` 先按静态长度升序排列；动态长度排在静态长度之后。
2. dtype 优先取第一个 split axis 的 dtype，否则用 `float32`。
3. 累计 `axis_length * dtype_bytes` 不超过 4 KiB 时，将轴标记为 `is_no_loop_axis`。
4. 一旦遇到动态长度或累计空间超过 4 KiB，立即停止，后续轴不再尝试。
5. persistent reduction 会先把 reduction axes 长度乘入累计 numel。

因此 dynamic axis 不使用示例 hint 做 no-loop 优化。这里选择保守地保留循环，与 split 选择对 dynamic hint 的处理不同。

## 这些标记怎样改变 generated Triton

`IterationRangesEntryNPUIndex` 直接消费三个标记：

```python
# split axis: program 间偏移
axis_offset = tl.program_id(split_order) * AXISBLOCK

# tiling axis: program 内向量索引
base_axis = tl.arange(0, AXISBLOCK_SUB)

# 普通 split + tiling 轴
axis = axis_offset + loop_axis * AXISBLOCK_SUB + base_axis

# no-loop 轴
axis = base_axis
```

mask 的上界也不同。split axis 使用 `min(BLOCK + offset, numel)`，普通 tiling axis 使用整个 `numel`。
因此轴选择错误会直接表现为错误的 `program_id` 维度、错误 offset、错误 loop 层次或尾块 mask；autotune 无法修复这些
语义错误。

## CUDA 与 NPU 到底哪里不同

### 上游 CUDA/SIMD 路径

上游 `SIMDScheduling.candidate_tilings()` 从节点的 pointwise/reduction ranges 和 `ReadWrites` 出发：

1. 默认候选把目标 ranges 乘积折叠成一个维度，名称为 `none`，分数为 0。
2. 对每个 `MemoryDep` 调用 `sizevars.stride_hints(dep.index, range_vars)`。
3. 找到第一个 stride 为 1 的位置，并在其后切分 ranges；尾部全是 broadcast stride 0 时放弃该切分。
4. 分数是所有非广播维度的元素数；写依赖乘 2，两个 group 是 good size 时分别再乘 2。
5. 分数达到完整计算域大小才加入候选。
6. 若 pointwise 和 reduction 都不超过一维，或 ranges 含 free unbacked symbols，直接不生成额外候选。

上游 `create_tiling()` 最多给 pointwise 使用 `z/y/x`，给 reduction 使用 `r0_/r1_`。最终
`get_tiling_and_scores()` 选择候选，再由 `SIMDKernelFeatures`、kernel heuristics 和 autotune 决定 block/warp 配置。

### NPU 路径

NPU 的 `candidate_tilings()` 基本保留 stride-scoring 结构，但默认 `none` 候选不折叠 ranges，而是把原 ranges
传给 `create_partial_tiling()`；同时没有上游对 `free_unbacked_symbols(...)` 的早退。NPU 的 `create_tiling()` 会展平
pointwise groups，并从 `w/v/t/z/y/x` 分配前缀；reduction ranges 则先合并为一个 `r` 维度。

更关键的是，候选 tiling 之后 NPU 还运行本文前述的 `SplitTiling`，显式决定 grid split axes、program 内 tiling axes
和 no-loop axes。所以下表中的术语不能一一对应：

| 问题 | CUDA 上游 | Ascend NPU |
|---|---|---|
| 如何产生多维 tiling 候选 | 依赖索引 stride 和候选评分 | 同源评分，但默认保留未折叠 ranges |
| 是否有 `SplitTiling.select_split_axis()` | 没有 | 有 |
| grid 轴如何表达 | Triton/SIMD tiling 和 kernel heuristics | `is_split_axis` + `split_order` |
| 动态 full reduction 特例 | 本文件涉及的上游函数无该分支 | A5 路径显式保持 grid 为 1 |
| symbolic grouped rewrite | 上游函数无 NPU 这套 metadata | 可降级 split axes，并构造运行时 group features |

因此正确表述是“两套 backend 在共享 SIMD scheduling 框架上的决策层次不同”，而不是“CUDA 选择 x 轴、NPU 选择 y 轴”
这种脱离具体实现的概括。

## 从一次轴选择反查问题

按下面顺序收集证据，能区分 scheduler、轴分类与 runtime config 三类问题：

1. 记录 `range_tree_nodes` 的名称、长度、prefix 和表达式类型。
2. 记录 `sorted_axis`、`low_dims`、`num_vector_core` 和 `size_hint`。
3. 逐轮代入 `meet_stop_condition()`，说明每个 split axis 为什么加入或停止。
4. 记录 grouped rewrite 前后的 `split_axis`，确认是否发生 downgrade。
5. 对照 generated kernel 中的 `tl.program_id`、`BLOCK_SUB`、`loop_axis` 和 mask。
6. 轴结构正确后，才检查 runtime autotuner 选择的 block、warp 和 launch grid。

## 验收问题

- 为什么大轴不一定先被选中？因为排序 key 基于轴名/表达式类型，不按 length。
- 何时有多个 split axes？当前乘积未满足停止条件，且还有非 reduction 候选时。
- 最多几个？三个，对应三维 grid。
- low dim 会不会 split？会，但只在非低维候选用尽后作为兜底。
- dynamic full reduction 为什么不 split？A5 没有该路径所需的跨 core combine，必须保持 grid 为 1。
- CUDA 是否执行相同算法？否；上游通过 stride-based candidate tiling 和 kernel heuristics 决策。

[返回本目标](./) | [下一篇：backend kernel codegen](../06-backend-codegen-and-runtime/triton-kernel-codegen.md)
