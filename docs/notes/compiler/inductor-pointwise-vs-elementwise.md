---
title: Inductor Pointwise 与 Elementwise 范围区别
---

# Inductor Pointwise 与 Elementwise 范围区别

## 背景

调研动态 shape 下 Triton kernel 性能时，容易把 `@triton_heuristics.pointwise`
直接理解为狭义 elementwise kernel。这个理解不够准确。

在 Inductor 中，`pointwise` 更接近一种 loop/codegen 分类：每个输出元素都可以根据
当前 output index 独立算出结果，不需要跨输出元素聚合或全局同步。它可以包含
broadcast、view/reindex、mask、条件选择、常量生成、dtype cast、部分数据搬运和
函数式 scatter 更新。

## 结论

- 狭义 `elementwise`：输出位置 `out[i]` 只依赖输入同逻辑位置 `in[i]` 的标量计算。
  典型例子是 `relu(x)`、`x + y`，且不考虑 broadcast 和复杂 index 映射。
- Inductor/Triton `pointwise`：输出位置可以独立计算即可。输入 loader 可以对
  output index 做 reindex、broadcast、mask、offset、modular indexing 或条件选择。
- 因此 `pointwise` 覆盖范围明显大于狭义 `elementwise`。性能调研中不建议只用
  `pointwise` 作为唯一分类，最好继续细分子类。

可以把 `pointwise` 的核心语义写成：

```text
for output_index in output_shape:
    out[output_index] = f(
        loads_with_reindex_or_mask(output_index),
        output_index,
        constants,
    )
```

它不包括需要跨输出元素聚合或协作的算子，例如 reduction、scan、sort、matmul、
convolution/template kernel，以及必须走 extern/fallback 的路径。

## 代码依据

### IR 与 codegen 分类

- `torch._inductor.ir.Pointwise` 的 `get_reduction_type()` 返回 `None`，说明它不是
  reduction IR。
  - `<workspace>/pytorch/torch/_inductor/ir.py`
  - `Pointwise`
- NPU Triton codegen 中，`_get_heuristic()` 对非 reduction kernel 返回 `"pointwise"`。
  - `<workspace>/pytorch_new/torch_npu/_inductor/codegen/triton.py`
  - `NPUIndexTritonKernel._get_heuristic`
- `make_pointwise()` 最终构造 `Pointwise.create(...)`，并以 `ranges` 作为输出迭代空间。
  - `<workspace>/pytorch_new/torch_npu/_inductor/lowering_fx.py`
  - `make_pointwise`

### Broadcast 进入 pointwise

- `register_pointwise(..., broadcast=True, ...)` 默认会允许 broadcast。
  - `<workspace>/pytorch_new/torch_npu/_inductor/lowering_fx.py`
  - `register_pointwise`
- lowering 包装器在 `broadcast=True` 时先调用 `broadcast_tensors(...)`。
  - `<workspace>/pytorch_new/torch_npu/_inductor/lowering_fx.py`
  - `transform_args`
- `broadcast_tensors` 对需要扩展的输入调用 `expand`。
  - `<workspace>/pytorch_new/torch_npu/_inductor/lowering_fx.py`
  - `broadcast_tensors`
- `ExpandView.make_reindexer()` 会把被 broadcast 的维度 index 置 0。
  - `<workspace>/pytorch/torch/_inductor/ir.py`
  - `ExpandView.make_reindexer`

### View/Reindex 被内联

- `BaseView.make_loader()` 通过 `reindex(idx)` 调内部 loader。
  - `<workspace>/pytorch/torch/_inductor/ir.py`
  - `BaseView.make_loader`
- 因此 `view`、`reshape`、`permute`、`slice`、`select`、`expand` 这类 view 本身
  通常不一定生成独立 pointwise kernel，但会在后续 pointwise consumer 中被内联到
  地址计算。

## Elementwise Aten 算子枚举

这里的 elementwise 指“每个输出元素由对应位置的输入标量独立计算得到”的 aten
算子。PyTorch 语义通常允许 broadcast；如果输入 shape 完全相同，它们就是狭义
elementwise。若发生 broadcast，则仍会走 pointwise lowering，但已经属于前文说的
`broadcast_pointwise` 子类。

下面主表基于当前 `pytorch_new/torch_npu/_inductor/lowering_fx.py` 中明确通过
`register_pointwise`、`register_pointwise_numeric`、`make_pointwise` 或相关
`register_lowering(..., broadcast=True)` 注册的 aten 算子整理。

### 算术类

| Aten 算子 | 输入数 | 解释 |
| --- | --- | --- |
| `aten.add` | binary | 对应元素相加，支持 `alpha` 参数；bool 输入时可映射为逻辑或。 |
| `aten.sub` | binary | 对应元素相减，支持 `alpha` 参数。 |
| `aten.mul` | binary | 对应元素相乘；两个 bool 输入时映射为 `logical_and`。 |
| `aten.div` | binary | 除法入口，可根据 `rounding_mode` 分发到 true/floor/trunc div。 |
| `aten.div.Tensor` | binary | Tensor/Tensor true division，整数输入会先提升到浮点语义。 |
| `aten.true_divide` | binary | true division，语义等价于普通 `/` 浮点除法。 |
| `aten.pow` | binary | 幂运算，计算 `a ** b`；小整数指数可能展开成乘法链。 |
| `aten.remainder` | binary | 余数运算，结果符号遵循 PyTorch remainder 语义。 |
| `aten.neg` | unary | 取负，计算 `-x`；部分 NPU patch 中 int32/int64 会转成乘 `-1`。 |
| `aten.abs` | unary | 绝对值。 |
| `aten.square` | unary | 平方，计算 `x * x`。 |
| `aten.reciprocal` | unary | 倒数，计算 `1 / x`。 |

### 常用数学函数

| Aten 算子 | 输入数 | 解释 |
| --- | --- | --- |
| `aten.rsqrt` | unary | 平方根倒数，计算 `1 / sqrt(x)`。 |
| `aten.sqrt` | unary | 平方根。 |
| `aten.exp` | unary | 自然指数，计算 `e ** x`。 |
| `aten.exp2` | unary | 以 2 为底的指数，计算 `2 ** x`。 |
| `aten.expm1` | unary | 计算 `exp(x) - 1`，小 `x` 时数值更稳定。 |
| `aten.log` | unary | 自然对数。 |
| `aten.log1p` | unary | 计算 `log(1 + x)`，小 `x` 时数值更稳定。 |
| `aten.log2` | unary | 以 2 为底的对数。 |
| `aten.log10` | unary | 以 10 为底的对数。 |
| `aten.sin` | unary | 正弦函数。 |
| `aten.cos` | unary | 余弦函数。 |
| `aten.tan` | unary | 正切函数。 |
| `aten.asin` | unary | 反正弦函数。 |
| `aten.acos` | unary | 反余弦函数。 |
| `aten.atan` | unary | 反正切函数。 |
| `aten.atan2` | binary | 根据 `(y, x)` 计算象限正确的反正切角度。 |
| `aten.sinh` | unary | 双曲正弦函数。 |
| `aten.cosh` | unary | 双曲余弦函数。 |
| `aten.tanh` | unary | 双曲正切函数。 |
| `aten.asinh` | unary | 反双曲正弦函数。 |
| `aten.acosh` | unary | 反双曲余弦函数。 |
| `aten.atanh` | unary | 反双曲正切函数。 |
| `aten.sigmoid` | unary | Sigmoid，计算 `1 / (1 + exp(-x))`。 |
| `aten.lgamma` | unary | Gamma 函数绝对值的自然对数。 |
| `aten.erf` | unary | Gauss error function。 |
| `aten.special_erf` | unary | `aten.erf` 的 special namespace 入口，lowering 复用 `erf`。 |
| `aten.erfc` | unary | complementary error function，计算 `1 - erf(x)`。 |
| `aten.erfinv` | unary | error function 的反函数。 |

### 数值辅助函数

| Aten 算子 | 输入数 | 解释 |
| --- | --- | --- |
| `aten.ceil` | unary | 向上取整到不小于输入的最小整数值。 |
| `aten.sign` | unary | 返回符号：负数为 `-1`，零为 `0`，正数为 `1`。 |
| `aten.signbit` | unary | 返回 bool，表示输入符号位是否为负。 |
| `aten.copysign` | binary | 返回幅值来自第一个输入、符号来自第二个输入的结果。 |
| `aten.hypot` | binary | 计算 `sqrt(x * x + y * y)`，通常比手写公式更稳定。 |
| `aten.nextafter` | binary | 返回从第一个输入朝第二个输入方向的下一个可表示浮点数。 |
| `aten.maximum` | binary | 对应元素取较大值，通常遵循 NaN 传播语义。 |
| `aten.minimum` | binary | 对应元素取较小值，通常遵循 NaN 传播语义。 |
| `aten.clamp_min` | binary/scalar | 对应元素与下界取最大值，等价于 `maximum(x, min)`。 |
| `aten.clamp_max` | binary/scalar | 对应元素与上界取最小值，等价于 `minimum(x, max)`。 |

### 比较类

比较类算子的输出 dtype 是 `torch.bool`。

| Aten 算子 | 输入数 | 解释 |
| --- | --- | --- |
| `aten.lt` | binary | 小于，计算 `a < b`。 |
| `aten.le` | binary | 小于等于，计算 `a <= b`。 |
| `aten.gt` | binary | 大于，计算 `a > b`。 |
| `aten.ge` | binary | 大于等于，计算 `a >= b`。 |
| `aten.eq` | binary | 等于，计算 `a == b`。 |
| `aten.ne` | binary | 不等于，计算 `a != b`。 |

### 逻辑类

逻辑类会把输入转换成 bool 语义，输出 dtype 是 `torch.bool`。

| Aten 算子 | 输入数 | 解释 |
| --- | --- | --- |
| `aten.logical_and` | binary | 逻辑与。 |
| `aten.logical_or` | binary | 逻辑或。 |
| `aten.logical_xor` | binary | 逻辑异或。 |
| `aten.logical_not` | unary | 逻辑非。 |

### 位运算类

位运算主要面向整数/bool dtype。NPU lowering 还把 Python magic 方法形式
`aten.__and__`、`aten.__or__` 等注册到对应 bitwise lowering。

| Aten 算子 | 输入数 | 解释 |
| --- | --- | --- |
| `aten.bitwise_and` | binary | 按位与。 |
| `aten.bitwise_or` | binary | 按位或。 |
| `aten.bitwise_xor` | binary | 按位异或。 |
| `aten.bitwise_not` | unary | 按位取反；bool 输入时可映射为逻辑非。 |
| `aten.bitwise_left_shift` | binary | 按位左移。 |
| `aten.bitwise_right_shift` | binary | 按位右移。 |
| `aten.__and__` | binary | Python `&` 的 aten 入口，lower 到 `bitwise_and`。 |
| `aten.__or__` | binary | Python `|` 的 aten 入口，lower 到 `bitwise_or`。 |
| `aten.__xor__` | binary | Python `^` 的 aten 入口，lower 到 `bitwise_xor`。 |
| `aten.__lshift__` | binary | Python `<<` 的 aten 入口，lower 到 `bitwise_left_shift`。 |
| `aten.__rshift__` | binary | Python `>>` 的 aten 入口，lower 到 `bitwise_right_shift`。 |

### In-place Elementwise 别名

这些 in-place aten 最终复用对应 out-of-place elementwise lowering，然后通过
`mutate_to` / `copy_` 写回目标 buffer。性能调研时通常应把它们归到对应 elementwise
算子，再额外标记 mutation。

| Aten 算子 | 对应 out-of-place | 解释 |
| --- | --- | --- |
| `aten.add_` | `aten.add` | 原地加。 |
| `aten.sub_` | `aten.sub` | 原地减。 |
| `aten.mul_` | `aten.mul` | 原地乘。 |
| `aten.div_.Tensor` | `aten.div.Tensor` | 原地 true division。 |
| `aten.div_.Tensor_mode` | `aten.div` | 带 rounding mode 的原地除法。 |
| `aten.relu_` | `aten.relu` | 原地 ReLU。 |
| `aten.sigmoid_` | `aten.sigmoid` | 原地 sigmoid。 |
| `aten.logical_and_` | `aten.logical_and` | 原地逻辑与。 |
| `aten.logical_or_` | `aten.logical_or` | 原地逻辑或。 |
| `aten.logical_xor_` | `aten.logical_xor` | 原地逻辑异或。 |
| `aten.logical_not_` | `aten.logical_not` | 原地逻辑非。 |
| `aten.bitwise_and_` | `aten.bitwise_and` | 原地按位与。 |
| `aten.bitwise_or_` | `aten.bitwise_or` | 原地按位或。 |
| `aten.bitwise_xor_` | `aten.bitwise_xor` | 原地按位异或。 |
| `aten.bitwise_not_` | `aten.bitwise_not` | 原地按位取反。 |
| `aten.bitwise_left_shift_` | `aten.bitwise_left_shift` | 原地按位左移。 |
| `aten.bitwise_right_shift_` | `aten.bitwise_right_shift` | 原地按位右移。 |
| `aten.__iand__` | `aten.__and__` | Python `&=` 的 aten 入口。 |
| `aten.__ior__` | `aten.__or__` | Python `|=` 的 aten 入口。 |
| `aten.__ixor__` | `aten.__xor__` | Python `^=` 的 aten 入口。 |
| `aten.__ilshift__` | `aten.__lshift__` | Python `<<=` 的 aten 入口。 |
| `aten.__irshift__` | `aten.__rshift__` | Python `>>=` 的 aten 入口。 |

### 社区版额外注册与 override 机制

社区版 `torch/_inductor/lowering.py` 还有两个容易注意的扩展：

- `aten.fmod` / `prims.fmod`：浮点时使用 floating fmod，整数时使用 mod 语义。
- `pointwise_overrides_data`：在 `torch/_inductor/codegen/common.py` 中维护一批特殊
  elementwise 函数，并在 lowering 中遍历注册到 `aten` 和 `prims` namespace。当前
  可见条目包括 `special_airy_ai`、Bessel 系列、`digamma`、`special_erfcx`、
  `fma`、`igamma` 等。是否真正走 Triton 还取决于该条目是否有 triton 实现；没有
  triton 实现的条目可能走 fallback。

## Pointwise 但不属于狭义 Elementwise 的范围

### 1. 带 broadcasting 的逐元素算子

例子：

```python
a = torch.randn(B, 1)
b = torch.randn(B, N)
c = a + b
```

`a[i, 0]` 会被多个 `c[i, j]` 复用，不是同位置 elementwise。但每个输出
`c[i, j]` 都能独立计算，因此属于 Inductor pointwise。

常见算子包括：

- `add`
- `sub`
- `mul`
- `div`
- `pow`
- `maximum`
- `minimum`
- comparison：`lt`、`le`、`gt`、`ge`、`eq`、`ne`
- logical：`logical_and`、`logical_or`、`logical_xor`、`logical_not`

### 2. dtype cast / dtype conversion

例子：

```python
y = x.to(torch.float16)
y = x.float()
```

这不是数学 elementwise 算术，但每个输出元素独立转换。NPU lowering 中 `to_dtype`
调用 `make_pointwise(_to_dtype, ...)`。

### 3. clone / copy 类逐元素搬运

例子：

```python
y = x.clone()
y = torch.empty_like(x).copy_(x)
```

这类没有实质数学计算，只是按 output index 读写。`clone` 直接用 `Pointwise.create`
包装 `x.make_loader()`。

如果 `copy` 的 source 和 destination shape 不一致，还可能先 `expand` 再 `clone`。

### 4. 常量填充

例子：

```python
y = torch.full((B, N), 3.0)
y = torch.zeros((B, N))
y = torch.ones((B, N))
```

它不依赖输入，不是 input-output elementwise，但每个输出元素都能独立生成常量。
`_full` 的 `inner_fn(index)` 返回 `ops.constant(...)` 或 `ops.index_expr(...)`，然后
构造 `Pointwise.create(...)`。

### 5. iota / arange 类 index 生成

例子：

```python
y = torch.arange(n)
```

或者 prims 层的 `iota`。输出值来自 output index，而不是输入同位置元素。
`prims.iota` 的 lowering 中使用：

```python
ops.index_expr(step * index[0] + start, dtype=dtype)
```

然后构造 `Pointwise.create(...)`。

### 6. where / 条件选择

例子：

```python
y = torch.where(mask, a, b)
```

每个输出元素会读取 `mask[i]`，再选择 `a[i]` 或 `b[i]`。这不是单一算术 elementwise，
但每个 output index 独立，因此属于 pointwise。这个场景还经常叠加 broadcast。

### 7. cat / concat 的 pointwise 形式

例子：

```python
y = torch.cat([a, b], dim=1)
```

`cat` 的语义是拼接，不是 elementwise。但如果 lower 成 pointwise，每个输出 index
可以根据所在区间选择从哪个 input load。

NPU lowering 中存在 `cat` 直接构造 `Pointwise.create(...)` 的路径。社区配置里也有
`max_pointwise_cat_inputs` 和 `force_pointwise_cat`，说明 cat 可以作为 pointwise op
with masked loads 生成。

### 8. repeat / tile 类重复读取

例子：

```python
y = x.repeat(...)
```

多个输出元素可能映射到同一个输入元素，不是同位置 elementwise。lowering 中会对
index 做 `ModularIndexing` 或将 size 为 1 的维度置 0，再从原输入 load，最后构造
`Pointwise.create(...)`。

### 9. padding

例子：

```python
y = torch.constant_pad_nd(x, padding, 0)
```

输出中间区域来自 offset 后的 input，边界区域来自填充值。这不是 elementwise，但每个
输出位置可以独立判断 mask 并 load 或填常量，所以可以是 pointwise。

### 10. select_scatter / slice_scatter 的函数式更新

例子：

```python
y = torch.select_scatter(x, src, dim, index)
y = torch.slice_scatter(x, src, dim, start, end, step)
```

它们语义上是函数式更新，不是 elementwise。Inductor 可以把它表达为：每个输出位置
判断是否在更新区域；如果在区域内取 `src`，否则取 `x`。这种形式仍然是 pointwise。

源码中 `select_scatter` 和 `slice_scatter` 都构造了条件选择逻辑，然后
`Pointwise.create(...)`。

### 11. masked select / masked fill 风格的局部选择

只要语义能写成“每个输出 index 根据 mask 选择输入或常量”，就属于 pointwise 范围。
具体是否走 Triton pointwise 要看对应 lowering、mask 表达、fallback 限制。

`slice_scatter` 和 `constant_pad_nd` 中使用的 `ops.masked` / `ops.where` 是这类形态的
代码例子。

### 12. view/reindex 融入 pointwise consumer

例子：

```python
y = torch.relu(x.permute(...))
z = x.expand(...) + b
w = x.slice(...).to(...)
```

`view`、`reshape`、`permute`、`slice`、`select`、`expand` 自己通常是 view IR，不一定
单独生成 kernel。但当它们被 pointwise consumer 使用时，view 的 reindex 会被内联到
pointwise kernel 的 load 地址计算。

这类不能简单说“view 算子属于 pointwise”，更准确的说法是“view/reindex 语义可以成为
pointwise kernel 的一部分”。

### 13. Scatter / 特殊写入

`Scatter` 在 IR 中继承自 `Pointwise`，但 store index 不是普通输出 index，而是由
`output_indexer(vars)` 决定。这明显不是狭义 elementwise。

实际是否生成 Triton pointwise kernel，还要看具体 scatter lowering、冲突写、
atomic、fallback 等限制。性能调研时建议把 scatter/masked store 单独标记。

## 建议的性能调研分类

如果调研动态 shape 下 `@triton_heuristics.pointwise` 的性能，不建议只使用一个
`pointwise` 大类。建议在 generated kernel 或 scheduler metadata 上继续细分：

| 子类 | 识别线索 | 典型风险 |
| --- | --- | --- |
| `pure_elementwise` | 主要是同 shape input 的标量算术 | 通常最接近带宽上限 |
| `broadcast_pointwise` | 存在 `ExpandView` 或 broadcast size | stride 0、复用、动态 shape guard |
| `view_reindex_pointwise` | load index 中有 permute/slice/reshape reindex | 非连续访问、地址计算复杂 |
| `copy_clone_cast` | clone/copy/to_dtype | 主要受带宽和 dtype conversion 影响 |
| `constant_fill_iota` | full/zeros/ones/iota/arange | 写带宽、index expr 复杂度 |
| `where_masked_pad` | `ops.where`、`ops.masked`、padding mask | 分支/mask、边界区域比例 |
| `cat_like` | 多 input 区间选择、masked loads | 多分支、多 input、动态分段 |
| `functional_scatter` | select_scatter/slice_scatter/scatter-like | mask、src index、潜在写入语义差异 |

同时记录 NPU 侧 `npu_kernel_type`：

- `simd`
- `simt_only`
- `simt_template`
- `simd_simt_mix`

这个字段不是 pointwise/reduction 分类，而是后端编译模式。动态 shape 性能分析时，
最好同时记录 `kernel_category=pointwise` 和 `npu_kernel_type`。

## 后续问题

- 需要实际扫描 generated code，确认当前模型/用例中的 pointwise kernel 分布。
- 需要为每类子形态收集 latency、shape、dtype、load/store 数、`npu_kernel_type` 和
  runtime block 信息，避免把性能瓶颈都混在 `pointwise` 一个标签里。
- 如果要自动分类，可以先基于 generated source 的装饰器、kernel name、`tl.load`
  地址表达式、`ops.where`/mask 源码痕迹、`inductor_meta` 字段做启发式标注。
