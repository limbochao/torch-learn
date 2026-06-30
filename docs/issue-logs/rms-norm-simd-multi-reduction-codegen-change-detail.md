---
title: RMSNorm SIMD 多 reduction 轴 codegen 修改详解
---

# RMSNorm SIMD 多 reduction 轴 codegen 修改详解

本文按 `bugfix/rms_norm_codegen_29` 分支提交 `d0f2ab254 fix` 说明代码改动。代码依据是该提交的 diff，下面按 hunk 展开；每个代码块都对应“改动前逻辑、改动后逻辑、为什么这么改”。

- `torch_npu/_inductor/codegen/triton.py`
- `torch_npu/_inductor/codegen/kernel_analysis.py`
- `torch_npu/_inductor/codegen/tile_generator.py`

这次问题的触发场景是 RMSNorm weight grad。该 kernel 有一个保留轴 `x0` 和三个 reduction 轴 `r3/r2/r1`。正确语义是对 `r3/r2/r1` 完整求和后只写一次 `out[x0]`。错误版本会把累加 buffer、最终 `tl.sum` 和 `tl.store` 放到内层 reduction tile 中，同时 flatten 前没有把 `x0` 放到保留轴位置，导致结果和 eager 不一致。

## `write_pointwise` 的 store 控制

2.9 提交在 `NPUIndexTritonKernel.codegen_body()` 内修改了局部函数 `write_pointwise`。

改动前：

```python
def write_pointwise():
    self._emit_coordinate_transforms()
    self.body.splice(self.indexing_code)
    self.body.splice(self.loads)
    self.body.splice(self.compute)
    self.body.splice(self.stores)
```

改动后：

```python
def write_pointwise(allow_stores=None):
    if allow_stores is None:
        allow_stores = self.numof_reduction_axis() <= 1
    self._emit_coordinate_transforms()
    self.body.splice(self.indexing_code)
    self.body.splice(self.loads)
    self.body.splice(self.compute)
    if allow_stores:
        self.body.splice(self.stores)
```

改动前，`write_pointwise()` 每次都会把 `self.stores` 写入当前 body。这个逻辑对普通 pointwise 或单 reduction 场景可以工作，但在多 reduction 轴场景下会把 store 提前插入到当前遍历到的 loop 层级里。RMSNorm 的错误 DSL 中，`tl.store` 出现在 `loop_r2` 内部，就是这个问题的一种表现。

改动后，默认只有 `numof_reduction_axis() <= 1` 才在 `write_pointwise()` 中写入 `self.stores`。多 reduction 轴场景下，store 默认延后，由后面的 post-loop 逻辑在完整 reduction loop 结束后统一写入。这里保留 `allow_stores` 参数，是为了后续 fused node 已经不依赖 reduction 轴时仍能显式允许 store。

## 提前计算当前 axis 是否参与 load/store

2.9 提交在 `codegen_range()` 中提前计算了：

```python
have_load_store = self.find_axis_in_load_store(range_val)
if not have_load_store:
    indexing_code = None
```

改动前，`have_load_store` 只在部分分支内临时计算。改动后，它在进入 tiling axis 分支前统一可用，后续的 `is_first_reduction_tiling`、`need_axis_loop` 和 skip 判断都能基于同一份判断结果。

这个改动本身不是 RMSNorm 精度错误的核心，但它让后续逻辑可以判断某个 reduction axis 是否真的参与当前 load/store。对于 fused kernel 中后续节点不再依赖 reduction 轴的情况，这个信息会用于跳过不必要的 reduction loop，避免生成引用不到的 axis 或错误作用域。

## 识别第一个 reduction tiling 轴

新增逻辑：

```python
is_first_reduction_tiling = (
    self.numof_reduction_axis() > 1
    and range_val.is_tiling_axis
    and range_val.prefix == "r"
    and not any(ax.prefix == "r" for ax in self.sorted_axis[:index])
)
```

改动前，codegen 只知道当前 axis 是不是 tiling axis、是不是最后一个 tiling axis，不知道它是不是多个 reduction 轴中的第一个 reduction tiling 轴。因此 `_tmp13 = tl.full(...)` 这类 prefix 初始化会跟着普通 tiling 逻辑被放到错误层级。

改动后，codegen 可以识别“即将进入完整 reduction loop 树”的位置。RMSNorm 中这相当于识别进入 `r3/r2/r1` reduction loop 的第一层。这样 `_tmp13` 可以放在第一个 reduction loop 外侧，覆盖完整的 `r3/r2/r1` 遍历，而不是在 `r2` tile 内反复清零。

## 识别需要外提 post-loop 的 reduction

新增逻辑：

```python
use_outer_reduction_post_loop = (
    self.numof_reduction_axis() > 1
    and range_val.prefix == "r"
    and bool(self.prefix._lines)
)
```

这里的 `self.prefix._lines` 表示已经生成了 reduction accumulator 初始化，例如：

```python
_tmp13 = tl.full(...)
```

改动前，只要进入某个 tiling axis 的 last-tiling 分支，就会在该 axis 后面 splice：

```python
self.body.splice(self.post_loop_combine)
self.body.splice(self.post_loop_store)
```

这会让 `tl.sum(_tmp13, 1)` 和 `tl.store(...)` 出现在内层 reduction tile 后面。RMSNorm 的错误 DSL 中，`tl.sum` 和 `tl.store` 位于 `loop_r2` 内部，只写入 partial reduction 结果。

改动后，当当前是多 reduction 轴且已经有 prefix accumulator 时，内层 reduction axis 不立即写 post-loop：

```python
if use_outer_reduction_post_loop:
    pass
else:
    ...
```

真正的 post-loop combine/store 会在第一个 reduction tiling 轴的 loop 结束后写出。这样最终 `tl.sum` 和 `tl.store` 等到完整 `r3/r2/r1` 都遍历完再执行。

## last tiling 分支中的 prefix 和 post-loop 处理

改动前，last tiling 分支只要需要 axis loop，就直接写 prefix：

```python
self.body.splice(self.prefix)
self.body.writeline(f"for loop_{range_val.name} in range(loops_{range_val.name}):")
```

改动后：

```python
if self.numof_reduction_axis() <= 1:
    self.body.splice(self.prefix)
self.body.writeline(f"for loop_{range_val.name} in range(loops_{range_val.name}):")
```

原因是多 reduction 轴场景下，prefix 不能放在最后一个 reduction tiling 轴附近，否则 accumulator 生命周期太短。prefix 要放到第一个 reduction tiling 轴外侧。

同一分支中，post-loop 也从无条件写出改成条件写出：

```python
if self.numof_reduction_axis() <= 1 or range_val.prefix != "r":
    self.body.splice(self.post_loop_combine)
self.body.splice(self.post_loop_store)
...
if self.numof_reduction_axis() > 1 and range_val.prefix == "r":
    self.body.splice(self.stores)
    self.stores.clear()
```

这里分三层含义：

第一，单 reduction 轴或非 reduction axis 仍保持原逻辑，直接写 `post_loop_combine`。

第二，多 reduction 轴的 reduction axis 不在内层写 `post_loop_combine`，避免 partial tile 上提前 `tl.sum`。

第三，`self.stores` 在多 reduction 轴的 reduction 分支中显式 splice，并立即 clear。这个补充用于处理部分 store 没有进入 `post_loop_store` 的情况，避免出现生成了计算但漏掉最终 store 的 DSL。之前 `var_mean((0,2))` 类场景就暴露过类似风险。

## 非 last tiling 分支中的 prefix 外提

改动前，非 last tiling 分支只负责写当前 loop：

```python
if not range_val.is_no_loop_axis:
    do_indent = True
    self.body.writeline(f"for loop_{range_val.name} in range(loops_{range_val.name}):")
loop_body(...)
```

改动后：

```python
if not range_val.is_no_loop_axis:
    do_indent = True
    if is_first_reduction_tiling or self.numof_reduction_axis() <= 1:
        self.body.splice(self.prefix)
    self.body.writeline(f"for loop_{range_val.name} in range(loops_{range_val.name}):")
loop_body(...)
```

这使得多 reduction 轴场景的 prefix 在“第一个 reduction tiling 轴”之前写出。RMSNorm 修复后的 DSL 中，`_tmp13` 位于：

```python
for loop_x0 in range(loops_x0):
    ...
    _tmp13 = tl.full(...)
    for loop_r3 in range(loops_r3):
        for loop_r2 in range(loops_r2):
            for loop_r1 in range(loops_r1):
                ...
```

这正是这个改动的目标：对同一个 `x0` tile，accumulator 覆盖全部 reduction tile。

## 第一个 reduction tiling 轴结束后写 post-loop

新增逻辑：

```python
if is_first_reduction_tiling and use_outer_reduction_post_loop:
    self.body.splice(self.post_loop_combine)
    for store_line in self.post_loop_store._lines:
        self.body.writeline(store_line)
    for store_line in self.stores._lines:
        self.body.writeline(store_line)
    for store_line in self._deferred_reduction_stores:
        self.body.writeline(store_line)
    self._deferred_reduction_stores.clear()
    self.stores.clear()
    self.post_loop_combine.clear()
    self.post_loop_store.clear()
```

改动前，post-loop combine/store 由 last tiling 分支处理。对于多 reduction 轴，这会落到内层 reduction axis 后，作用域不够外。

改动后，当第一个 reduction tiling axis 的完整 loop_body 结束后，说明它内部嵌套的所有 reduction axis 都已经遍历完成，此时再写：

```python
tmp13 = tl.sum(_tmp13, 1)
tl.store(...)
```

这样生成的 `tl.sum` 和 `tl.store` 与 `_tmp13` 的生命周期对齐。这里同时处理 `post_loop_store`、`stores`、`_deferred_reduction_stores`，是为了覆盖不同 store 入口，避免 fused/multi-output 场景漏写。

## 后续 fused node 跳过无关 reduction 轴

2.9 提交修改了 `first_node == False` 时的逻辑。

改动前：

```python
last_axis_order = self.tiling_axis[-1].sorted_order
if self.persistent_reduction and self.numof_reduction_axis() > 1:
    last_axis_order = last_axis_order - self.numof_reduction_axis() + 1
...
codegen_range(last_axis_order)
```

改动后：

```python
last_axis_order = self.tiling_axis[-1].sorted_order
skip_reduction_axes = False
if self.numof_reduction_axis() > 1:
    last_axis_order = last_axis_order - self.numof_reduction_axis() + 1
    skip_reduction_axes = not any(
        self.find_axis_in_load_store(axis)
        for axis in self.sorted_axis[last_axis_order:]
        if axis.prefix == "r"
    )
...
if skip_reduction_axes:
    write_pointwise(allow_stores=True)
else:
    codegen_range(last_axis_order)
```

改动前，只有 persistent reduction 且多 reduction 轴时才调整 `last_axis_order`。对于当前 SIMD multi-reduction 场景，后续 fused node 可能继续从 reduction axis 层级开始生成，导致引用到不再需要的 `r*` loop，或把 store 放到错误作用域。

改动后，只要是多 reduction 轴，就把后续 node 的起点调整到第一个 reduction axis 前。如果后续 node 的 load/store 不再引用 reduction axis，则直接调用 `write_pointwise(allow_stores=True)`。这让 reduction 结果后接 pointwise/store 的融合形态可以在 reduction 外层生成，避免重复进入 reduction loop。

这里 `allow_stores=True` 是必要的，因为前面把多 reduction 场景下 `write_pointwise()` 默认 store 关掉了。对于这种已经确定不需要 reduction axis 的后续 node，需要显式允许 store。

## reduction 后 reshape 逻辑

2.9 提交在 `reduction_resize()` 中新增 contiguous multi-reduction 特判。

改动前，逻辑会用 `dense_size_list()`，把 `dim` 对应位置置为 `1`，然后 reshape：

```python
dense_list = self.dense_size_list()
dense_list[dim] = "1"
...
return f"{value}.reshape({expand_str})"
```

这个逻辑适合单 reduction 轴。对于 RMSNorm 这类多 reduction 轴 flatten 后的二维 view，`tl.sum(_tmp13, 1)` 的结果是一维 `X0BLOCK_SUB`，需要 reshape 回原始广播布局：

```python
[1, 1, 1, X0BLOCK_SUB]
```

改动后：

```python
if self.numof_reduction_axis() > 1 and self.is_contiguous_reduction():
    if not self.golden_var_list:
        self.select_golden_varlist()

    dense_list = self.reduce_analysis.dense_size_list()
    for i, axis in enumerate(reversed(self.golden_var_list)):
        if axis.name[0] == "r":
            dense_list[i] = "1"

    expand_str = ", ".join(dense_list)
    return f"{value}.reshape({expand_str})"
```

这里不再只把一个 `dim` 置为 `1`，而是把所有 reduction 轴都置为 `1`，保留非 reduction 轴的 sub-block 形状。RMSNorm 中就是保留 `x0`，把 `r3/r2/r1` 都 reshape 成 1，生成：

```python
tmp13 = tl.sum(_tmp13, 1).reshape(1, 1, 1, X0BLOCK_SUB)
```

## reduction value flatten 前调整轴顺序

2.9 提交修改了 `reduction()` 中对 `dense_size_str` 的 reshape 逻辑。

改动前：

```python
value = self._map_tuple_or_scalar(
    lambda v: self.cse.generate(
        self.compute,
        f"tl.reshape({v}, {dense_size_str})",
        dtype=v.dtype,
    ),
    value,
)
```

这个逻辑直接 reshape，不改变 value 当前维度顺序。RMSNorm 中 `tmp11` 的真实布局来自下标方向：

```python
r3 = base_r3[:, None, None, None]
r1 = base_r1[None, :, None, None]
r2 = base_r2[None, None, :, None]
x0 = base_x0[None, None, None, :]
```

所以 value 的真实顺序是：

```text
[r3, r1, r2, x0]
```

而 reduction 后希望得到：

```text
[x0, r3/r1/r2 flatten]
```

直接 reshape 会把第 0 维 `r3` 当作 `x0` 维，导致后续 `tl.sum(_tmp13, 1)` 沿错误 lane 求和。

改动后：

```python
if self.numof_reduction_axis() > 1 and self.is_contiguous_reduction():
    if not self.golden_var_list:
        self.select_golden_varlist()
    value_order = list(reversed(self.golden_var_list))
    target_order = [x for x in value_order if x.name[0] != "r"] + [
        x for x in value_order if x.name[0] == "r"
    ]
    permute_order = [value_order.index(x) for x in target_order]
    value = self._map_tuple_or_scalar(
        lambda v: self.cse.generate(
            self.compute,
            f"tl.reshape({v}.permute({permute_order}), {dense_size_str})",
            dtype=v.dtype,
        ),
        value,
    )
```

这段逻辑先取当前 value 的维度顺序 `value_order`，再构造目标顺序：非 reduction 轴在前，reduction 轴在后。RMSNorm 中：

```text
value_order  = [r3, r1, r2, x0]
target_order = [x0, r3, r1, r2]
permute      = [3, 0, 1, 2]
```

因此修复后生成：

```python
tmp12 = tl.reshape(tmp11.permute([3, 0, 1, 2]), [X0BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB * R1BLOCK_SUB])
```

这保证 `tmp12` 的第 0 维是保留轴 `x0`，第 1 维才是所有 reduction lane 的 flatten 结果。

## reduction dim 计算

2.9 提交在 `ReductionAnalysis.analyze_reduction_dim()` 中新增：

```python
if self.numof_reduction_axis() > 1 and self.contiguous_reduction:
    dim = 0
    for x in reversed(reduction_layout_var_list):
        if x.name[0] != 'r':
            dim += 1
    return dim
```

改动前，逻辑是从 `reduction_layout_var_list` 的反向列表中找到第一个 reduction 轴，并返回它的位置：

```python
dim = -1
for i, x in enumerate(reversed(reduction_layout_var_list)):
    if x.name[0] == 'r':
        dim = i
        break
return dim
```

这个逻辑适合直接在原始 dense layout 上 reduction。修复后 contiguous multi-reduction 会被重排成：

```text
[non-reduction axes, flattened reduction axes]
```

因此 `tl.sum` 的 dim 应该等于前面非 reduction 轴的数量。RMSNorm 只有一个非 reduction 轴 `x0`，所以 dim 是 1，对应：

```python
tl.sum(_tmp13, 1)
```

如果后续存在两个保留轴，例如 `[x0, x1, r...]`，这里会返回 2，表示沿第三个维度开始的 flatten reduction lane 求和。

## SIMD / SIMT template 多 reduction 轴 tiling config 收窄

2.9 提交在 `TileGenerator.descend_split_tiling()` 最后新增：

```python
if (
    self.dual_reduction
    and self.npu_kernel_type in (NPUKernelType.SIMD, NPUKernelType.SIMT_TEMPLATE)
    and len(self.configs) > 0
):
    def reduction_remainders(conf):
        remain = 0
        for axis, name in enumerate(self.axis_name):
            if not name.startswith("r"):
                continue
            block_name = self.sub_block_name.get(axis, "")
            block = conf.kwargs.get(block_name, None)
            if block:
                remain += self.numels[axis] % block
        return remain

    ranked_configs = sorted(
        (reduction_remainders(conf), idx, conf)
        for idx, conf in enumerate(self.configs)
    )
    if ranked_configs and ranked_configs[0][0] == 0:
        self.configs = [
            conf for remain, _, conf in ranked_configs if remain == 0
        ]
```

这里的 `dual_reduction` 在当前代码中表示 `numof_reduction_axis() > 1`，不是只有两个 reduction 轴。该逻辑作用于 SIMD 和 SIMT template 多 reduction 轴 kernel。

改动前，所有候选 config 都会进入后续 autotune/选择。实际在 A2 上验证过：去掉这段过滤后，RMSNorm case 会生成结构上已经正确的 DSL，但如果选到 reduction 轴不能整除 `*_BLOCK_SUB` 的 tiling，`CHECK=1` 仍会失败。

失败原因是当前 DSL 仍使用：

```python
_tmp13 = tl.where(mask, _tmp13 + tmp12, _tmp13)
tmp13 = tl.sum(_tmp13, 1)
```

非整除 reduction tile 会产生尾块无效 lane。mask 为 false 时，lane 保留旧 accumulator；而当前多层 reduction loop 与 flatten lane 的组合还没有做到对任意 tiling 都稳定。因此这个过滤用于优先选择所有 reduction 轴都能整除 sub-block 的 config，避免尾块无效 lane。

这不是最根本的 DSL 正确性修复。更理想的长期方案是让 mask=false 的当前贡献使用 reduction neutral value，或者像 GPU Inductor 一样把多 reduction 轴规整成稳定的单一 reduction lane。当前提交选择 config 收窄，是为了在不扩大 reduction type 影响面的情况下保证已知 RMSNorm、`var_mean`、`batch_norm` 场景通过。

## 修改影响汇总

这次提交主要影响 `numof_reduction_axis() > 1` 的 NPU SIMD codegen。单 reduction 轴路径基本维持原逻辑。

对 contiguous multi-reduction，提交改变三件事：reduction value flatten 前的轴顺序、`tl.sum` 的 reduction dim、reduction 后 reshape 形状。这些改动直接修复 RMSNorm 中保留轴 `x0` 和 reduction 轴混在一起的问题。

对 multi-reduction 的 codegen body，提交改变 accumulator、post-loop combine 和 store 的写出作用域。它避免 `_tmp13` 在中间 tile 清零，也避免 partial reduction 在内层 tile 提前 store。

对 SIMD / SIMT template multi-reduction 的 tiling，提交会在存在整除 config 时丢弃非整除 config。这会收窄候选配置，可能影响性能搜索空间，但用于规避当前非整除 tail lane 下 mask/accumulator 语义不稳定的问题。后续如果补齐 DSL 对任意 tiling 的正确性，可以重新评估是否移除这段过滤。
