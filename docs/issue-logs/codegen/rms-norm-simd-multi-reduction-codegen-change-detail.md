---
title: RMSNorm SIMD 多 reduction 轴 codegen 修改详解
published: false
---

# RMSNorm SIMD 多 reduction 轴 codegen 修改详解

关联 PR：[Ascend/pytorch#39499](https://gitcode.com/Ascend/pytorch/pull/39499)。

这个 PR 的作用是修复 NPU Inductor 在连续多 reduction 轴场景下的 codegen 语义。多 reduction 轴被 flatten 后，value、mask、最终 reduction dim 和 store 位置必须按同一套轴结构生成；否则生成的 DSL 会在局部 tile 上提前写回，或者让 value 与 mask 的逻辑坐标不一致。

## store 位置

修改位置在 `NPUIndexTritonKernel.codegen_body()` 内部的 `write_pointwise()` 和 `codegen_range()`。

`write_pointwise()` 增加 `allow_stores` 参数：

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

改动前，pointwise 路径每次都会把 `self.stores` 写入当前 body。多 reduction 轴场景下，如果 pointwise store 被放在 reduction loop 还没有完全结束的位置，就会提前写回不完整结果。

改动后，单 reduction 轴及以下默认保持原行为；多 reduction 轴默认延后 store。只有调用方明确知道当前已经不需要等待 reduction loop 时，才传入 `allow_stores=True`。

`codegen_range()` 里新增了两个局部判断：

```python
is_first_reduction_tiling = (
    self.numof_reduction_axis() > 1
    and range_val.is_tiling_axis
    and range_val.prefix == "r"
    and not any(ax.prefix == "r" for ax in self.sorted_axis[:index])
)
use_outer_reduction_post_loop = (
    self.numof_reduction_axis() > 1
    and range_val.prefix == "r"
    and bool(self.prefix._lines)
)
```

`is_first_reduction_tiling` 用来识别完整 reduction 轴组的外层入口。以 RMSNorm weight grad 为例，保留轴是 `x0`，reduction 轴是 `r3/r2/r1`，进入第一个 `r` tiling 轴时就进入了完整 reduction loop。

`use_outer_reduction_post_loop` 表示当前 kernel 已经有 reduction accumulator 初始化逻辑，需要把 post-loop combine 和 store 放到完整 reduction loop 外层统一处理。

非 last tiling 分支现在会在第一个 reduction tiling loop 前写出 prefix：

```python
if not range_val.is_no_loop_axis:
    do_indent = True
    if is_first_reduction_tiling or self.numof_reduction_axis() <= 1:
        self.body.splice(self.prefix)
    self.body.writeline(f"for loop_{range_val.name} in range(loops_{range_val.name}):")
```

当当前轴是第一层 reduction tiling 且存在外层 post-loop 需求时，完整 `loop_body` 结束后统一写出 combine 和 store：

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

last tiling 分支对应增加保护：

```python
if use_outer_reduction_post_loop:
    pass
else:
    if self.numof_reduction_axis() <= 1 or range_val.prefix != "r":
        self.body.splice(self.post_loop_combine)
    self.body.splice(self.post_loop_store)
```

这几处改动共同保证：多 reduction 轴的 accumulator 在完整 reduction 范围外初始化，所有 reduction loop 更新同一个 accumulator，完整 reduction 结束后再 combine 和 store。

## 后续 fused node 的 loop 层级

修改位置在 `codegen_body()` 中 `first_node == False` 的分支。

改动前，后续 fused node 按 `last_axis_order` 继续生成。如果前一个节点是多 reduction 轴，后续 pointwise 节点可能被挂到 reduction loop 内层。

改动后，多 reduction 轴统一把生成起点调整到 reduction 轴组之前，并检查后续 node 是否还引用 reduction 轴：

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
for _ in range(last_axis_order):
    self.body.do_indent()
if skip_reduction_axes:
    write_pointwise(allow_stores=True)
else:
    codegen_range(last_axis_order)
for _ in range(last_axis_order):
    self.body.do_unindent()
```

如果后续 node 不再引用 reduction 轴，说明它只依赖 reduction 后的结果，可以直接按 pointwise 写出并允许 store。如果仍然引用 reduction 轴，则继续走原有 `codegen_range()`。

## reduction 结果 reshape

修改位置在 `NPUIndexTritonKernel.reduction_resize()`。

单 reduction 轴时，只需要把一个 reduction dim 置为 `1`。多 reduction 轴连续 flatten 后，所有 reduction 轴都已经被 reduce 掉，结果 reshape 时需要把每个 `r` 轴都变成广播维度 `1`。

当前逻辑是：

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

RMSNorm weight grad 的输出只保留 `head_dim` 对应的 `x0`。因此 reshape 后的结构是 reduction 轴全部为 `1`，只保留 `X0BLOCK_SUB`。

## value 和 mask 的轴顺序

修改位置在 `NPUIndexTritonKernel.reduction()`。

多 reduction 轴连续 flatten 时，目标 dense shape 是“非 reduction 轴在前，flatten 后的 reduction 轴在后”。因此在进入 `tl.reshape` 前，需要判断当前 value 的轴顺序是否已经等于目标顺序。

当前逻辑先初始化：

```python
dense_size_str = self.dense_size_str()
permute_order = None
need_permute = False
```

然后只在 contiguous multi-reduction 场景下计算是否需要 permute：

```python
if self.numof_reduction_axis() > 1 and self.is_contiguous_reduction():
    value_order = list(reversed(self.golden_var_list))
    target_order = [x for x in value_order if x.name[0] != "r"] + [
        x for x in value_order if x.name[0] == "r"
    ]
    permute_order = [value_order.index(x) for x in target_order]
    current_order = list(range(len(value_order)))
    need_permute = permute_order != current_order
```

这里 `value_order` 表示当前 value 的轴顺序，`target_order` 表示 reshape 前期望的轴顺序。`current_order = [0, 1, ..., n-1]` 表示不需要重排的 identity 顺序。只有 `permute_order != current_order` 时，才生成 `permute`。

value 侧统一走一处 reshape 逻辑：

```python
if need_permute:
    value = self._map_tuple_or_scalar(
        lambda v: self.cse.generate(
            self.compute,
            f"tl.reshape({v}.permute({permute_order}), {dense_size_str})",
            dtype=v.dtype,
        ),
        value,
    )
else:
    value = self._map_tuple_or_scalar(
        lambda v: self.cse.generate(
            self.compute,
            f"tl.reshape({v}, {dense_size_str})",
            dtype=v.dtype,
        ),
        value,
    )
```

mask 侧复用同一个 `need_permute` 和 `permute_order`：

```python
cond_expr = f"({' & '.join(masks)})"
if need_permute:
    cond_expr = f"{cond_expr}.permute({permute_order})"
cond = f"{cond_expr}.reshape({dense_size_str})"
```

这样 value 和 mask 在 flatten 后仍然对应同一个逻辑坐标。

## reduction dim

修改位置在 `ReductionAnalysis.analyze_reduction_dim()`。

普通路径会在 layout 中查找 reduction 轴位置。contiguous multi-reduction 场景下，value 已经 reshape 成：

```text
[non-reduction axes, flattened reduction axes]
```

因此 reduction dim 应该是非 reduction 轴数量：

```python
if self.numof_reduction_axis() > 1 and self.contiguous_reduction:
    if not self.kernel.golden_var_list:
        self.kernel.select_golden_varlist()
    return sum(1 for x in self.kernel.golden_var_list if x.name[0] != 'r')
```

RMSNorm weight grad 只有一个保留轴 `x0`，所以最终 reduction dim 是 `1`。如果以后出现两个保留轴，目标结构就是 `[x0, x1, flat_r]`，reduction dim 对应 `2`。
