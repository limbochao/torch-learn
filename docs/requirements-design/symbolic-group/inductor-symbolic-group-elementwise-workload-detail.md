---
title: a3782c3f8 Elementwise Workload 逐行实现说明
---

# a3782c3f8 Elementwise Workload 逐行实现说明

## 1. 文档范围

本文逐段说明 `pytorch_new` 提交 `a3782c3f8ff37d2cbdaf18f4234348b977679f25`
相对父提交 `3bbc52affa960e55474ffefa48d07b52c64c6147` 的全部代码改动。

提交标题：

```text
perf: classify elementwise symbolic workloads
```

改动统计：

| 文件 | 改动目的 | 行数 |
| --- | --- | ---: |
| `torch_npu/_inductor/codegen/split_tiling.py` | 判定 pointwise kernel 是否属于 elementwise workload | +169 / -3 |
| `torch_npu/_inductor/runtime/symbolic_grouping.py` | 在 grouped metadata payload 中保存 workload | +10 |
| `torch_npu/_inductor/codegen/triton.py` | 将 workload 写入最终 generated kernel metadata | +3 |
| `test/_inductor/test_inductor_dynamic_shapes.py` | NPU 动态 shape 集成测试 | +102 |

本文中的代码行号对应提交 `a3782c3f8`。其中“逐行”按有独立语义的 Python 语句说明；
仅承担续行、闭合括号或缩进作用的行会和所属语句合并解释。

相关设计背景见
[`inductor-symbolic-group-elementwise-workload.md`](inductor-symbolic-group-elementwise-workload.md)。

## 2. 改动前后的调用链

提交没有新增一个 Triton template。它只在既有 `pointwise` template 内增加
`group_workload` 子分类：

```text
SchedulerNode.read_writes / LoopBody.op_counts
                |
                v
SplitTiling._classify_group_workload()
                |
                v
GroupedKernelMeta(workload=...)
                |
                v
NPUIndexTritonKernel.create_inductor_meta()
                |
                v
generated Python kernel 的 inductor_meta['group_workload']
```

这里的分类不改变 DSL 的数学结果，也不改变 `group_template`。它只提供更细的性能策略标签：

```text
group_template = pointwise
group_workload = elementwise | None
```

`None` 表示普通 pointwise 或无法保守证明为 elementwise 的 pointwise。

## 3. `split_tiling.py`

文件：`torch_npu/_inductor/codegen/split_tiling.py`

### 3.1 导入和常量（第 6、19-20 行）

```python
from torch._inductor.dependencies import MemoryDep

_ELEMENTWISE_UNSUPPORTED_OPS = (
    "masked", "scan", "sort", "rand", "randn", "load_seed"
)
_NEUTRAL_CONSTANT_OPS = frozenset(("constant", "store", "output"))
```

| 行 | 说明 |
| --- | --- |
| 6 | 导入 `MemoryDep`，后续只接受能描述普通 buffer 地址访问的 read/write dependency。`StarDep`、`WeakDep`、`IndexExprDep` 等不是该类型，因此保守拒绝。 |
| 19 | 定义不准入的 `LoopBody` primitive 名称。`masked` 是语义 masked 子块，不是尾块 mask；`scan`、`sort` 有跨 lane 语义；`rand`、`randn`、`load_seed` 有 RNG 状态语义。 |
| 20 | 定义 neutral constant node 唯一允许出现的 primitive 集合。 |

`constant` 生成常量，`store` 写其临时 buffer，`output` 是 LoopBody tracing 的返回。
出现 add、load、index_expr 等其他 primitive 时，它就不是“纯常量节点”。

这两个集合并不以 ATen op 名称分类。它们检查的是 lowering 后 `LoopBody` 实际调用的
`V.ops.*` primitive。

### 3.2 `group_features` 的 workload 分发（第 345-386 行）

修改前入口为：

```python
def _build_group_features(self, primary_axis):
```

修改后为：

```python
def _build_group_features(self, workload, primary_axis):
```

`primary_axis` 是已有参数，本提交不改变它的含义；新增 `workload` 用于 pointwise 子分类。

| 行 | 代码行为 | 作用 |
| --- | --- | --- |
| 345 | 新增 `workload` 参数 | 让前面的分类结果可参与 feature 构造。 |
| 346-369 | 原有 reduction/persistent-reduction 分支 | 完整保留。它仍按动态 outer/reduction axis 选择 `outer`、`reduction` feature，不读取 workload。 |
| 370 | 判断 `workload == "elementwise"` | 只有已被保守证明为 elementwise 的 pointwise kernel 进入新分支。 |
| 371 | 返回单元素 tuple | `GroupedKernelMeta.group_features` 的类型是 tuple，即使只有一个 feature 也保持统一结构。 |
| 372-377 | 构造 `GroupFeatureSpec` | 生成新的可观测 feature 名称 `elementwise_numel`。 |
| 373 | `name="elementwise_numel"` | 区分原来的 `pointwise` feature；runtime 可据此识别该 workload 的 bucket 输入。 |
| 374 | `source="outer_product"` | feature 值等于参与轴长度的乘积。pointwise 没有 reduction product。 |
| 375 | `axis_names=self.all_axis_names()` | 使用 kernel 的全部保留迭代轴，而不是仅 split axis。普通 tiling axis 同样决定单 program 内循环量。 |
| 376 | bucket 为 `(num_vector_core * 4096,)` | 与原 pointwise 使用同一阈值，避免在没有性能数据时擅自调整性能策略；本次先分出标签和扩展入口。 |
| 379-386 | 原 pointwise fallback | `workload is None` 时保持 `name="pointwise"`、相同 source 和 bucket，保证未准入场景行为不变。 |

### 3.3 访问签名：忽略变量名、固定 offset，保留迭代映射（第 388-430 行）

elementwise 的工程定义不是“算子名字逐元素”，而是 read/write 是否采用同一直接迭代映射。
访问签名的类型是：

```text
(normalized_index_without_offset, normalized_iteration_sizes)
```

#### 3.3.1 `_alpha_rename_access_vars`（第 388-394 行）

```python
@staticmethod
def _alpha_rename_access_vars(dep):
    replacements = {
        var: sympy.Symbol(
            f"elementwise_dim_{idx}", integer=True, nonnegative=True
        )
        for idx, var in enumerate(dep.var_names)
    }
    return replacements
```

| 行 | 说明 |
| --- | --- |
| 388 | 静态方法，不依赖 `SplitTiling` 的其他状态。 |
| 389 | 入参是已经 `normalize()` 的 `MemoryDep`。 |
| 390-393 | 按 `var_names` 的位置创建替换表。例如 `(c0, c2)` 统一改成 `(elementwise_dim_0, elementwise_dim_1)`。 |
| 391 | `integer=True, nonnegative=True` 保持 loop index 的基本约束，使 SymPy 化简语义与原迭代变量一致。 |
| 394 | 返回替换表，不修改 `dep`。`MemoryDep` 是 frozen dataclass，后续 `sympy_subs` 也返回新表达式。 |

统一改名解决“同一种映射恰好使用不同临时符号”的假不等。例如：

```text
c0 * N + c1
d4 * N + d9
```

在按位置改名后都变成：

```text
elementwise_dim_0 * N + elementwise_dim_1
```

#### 3.3.2 `_make_access_signature`（第 396-409 行）

```python
def _make_access_signature(self, dep):
    if dep.is_indirect() or dep.mode is not None:
        return None
    normalized = dep.normalize()
    replacements = self._alpha_rename_access_vars(normalized)
    index = sympy_subs(
        normalized.index - normalized.get_offset(),
        replacements,
    )
    sizes = tuple(sympy_subs(size, replacements) for size in normalized.size)
    return (
        V.graph.sizevars.simplify(index),
        tuple(V.graph.sizevars.simplify(size) for size in sizes),
    )
```

| 行 | 说明 |
| --- | --- |
| 397 | `dep.is_indirect()` 为真时，索引依赖从内存读取的值，如 gather/index_select；地址不再是纯迭代变量函数，直接拒绝。 |
| 397 | `dep.mode is not None` 表示特殊 store mode，例如 atomic add；它不是普通的一一写入，也拒绝。 |
| 398 | 被拒绝时返回 `None`，调用者不把它当成可比较签名。分类失败只回退普通 pointwise，不抛异常。 |
| 399 | 调用 `MemoryDep.normalize()`。它为该 dependency 生成规范化副本，合并能安全合并的迭代轴；不会改变 kernel 的 split/tiling axis。 |
| 400 | 获取按位置统一变量名的替换表。 |
| 401-404 | 从 index 中减去 `get_offset()` 后再替换变量名。固定 offset 只改变起始 storage 地址，不改变 lane 到相对地址的映射。 |
| 405 | 对每个迭代轴 size 使用同一套变量替换。index 相同但迭代域不同仍不能视为同一访问。 |
| 406-409 | 通过 `V.graph.sizevars.simplify()` 化简 index 和 size；最终返回二元组。 |

示例：对 `x[4:]` 的访问，原 index 可为 `4 + i`。去 offset 后是 `i`，因此与输出的
`i` 保持同一映射。对 `x[::2]`，index 为 `2*i`，去 offset 后仍是 `2*i`，与输出 `i`
不同，因而拒绝。

#### 3.3.3 `_same_access_signature` 与 `_all_access_signatures_equal`（第 411-430 行）

```python
def _same_access_signature(left, right):
    left_index, left_sizes = left
    right_index, right_sizes = right
    if len(left_sizes) != len(right_sizes):
        return False
    sizevars = V.graph.sizevars
    return sizevars.statically_known_equals(left_index, right_index) and all(
        sizevars.statically_known_equals(left_size, right_size)
        for left_size, right_size in zip(left_sizes, right_sizes)
    )
```

| 行 | 说明 |
| --- | --- |
| 413-414 | 解构两个签名，使 index 与 axis size 分别比较。 |
| 415-416 | 迭代轴数量不同直接返回 `False`。broadcast 常在此或后续 size/index 比较被排除。 |
| 417 | 缓存 `sizevars`，后续的等价判断统一使用当前图的 ShapeEnv。 |
| 418 | index 必须可静态证明相等。该 API 不增加 guard；无法证明时保守返回 `False`。 |
| 419-421 | `zip` 逐轴比较 size，`all()` 要求每个 size 都能静态证明相等。 |
| 423-425 | 空签名集合没有参考签名，返回 `False`。 |
| 426-430 | 以第一个签名为 reference，所有后续签名必须与其等价。 |

相同 unbacked symbol 或化简后结构相同的表达式仍可返回 `True`；例如 `u == u`、
`2*u + 1 == u + u + 1`。不同 unbacked symbol 且 ShapeEnv 没有等式约束时返回 `False`。

### 3.4 节点级语义筛选（第 432-478 行）

#### 3.4.1 primitive 与常量节点判断（第 432-439 行）

```python
def _has_unsupported_elementwise_semantics(node):
    return any(node._body.has_op(op) for op in _ELEMENTWISE_UNSUPPORTED_OPS)

def _is_constant_only_body(node):
    op_names = set(node._body.op_counts)
    return bool(node._body.op_counts.get("constant")) and op_names <= _NEUTRAL_CONSTANT_OPS
```

| 行 | 说明 |
| --- | --- |
| 433-434 | 逐个查询 `LoopBody.op_counts` 是否出现不支持 primitive；只要出现一个就拒绝。 |
| 437 | 读取所有已记录 primitive 名称。 |
| 438 | 要求至少有一次 `constant`，防止空 body 被误当作常量生成。 |
| 439 | 要求 primitive 名称集合是 `{constant, store, output}` 的子集。`<=` 允许其中某些 tracing 细节不存在。 |

`op_counts` 在 `LoopBodyBlock` tracing 时由 `CountOps` 记录，因此它是 SchedulerNode 的
lowering 后语义，而不是 FX/ATen 图中的原始 op 名。

#### 3.4.2 `_classify_elementwise_node`（第 441-478 行）

这个函数返回三种结果：

```text
("direct", signature)          有直接 tensor read/write 的节点
("neutral_constant", signature) 纯常量临时节点
None                            不准入
```

| 行 | 说明 |
| --- | --- |
| 442-443 | 有副作用、alias 或 mutation 时拒绝。此类节点不能只按地址等价理解。 |
| 444-445 | 出现不支持 primitive 时拒绝。 |
| 447-448 | 从 SchedulerNode 的 `read_writes` 提取逻辑 read/write dependency。它们代表依赖关系，不要求与 DSL 中每一次 `tl.load/tl.store` 一一对应。 |
| 449-450 | 没有 write 或出现 `index_exprs` 时拒绝。前者没有输出映射可比较；后者常见于 iota/arange 等纯索引表达式。 |
| 451-452 | 所有 write 必须是 `MemoryDep`。 |
| 454 | 为每个 write 构造签名。 |
| 455-456 | 任意 write 为 indirect/atomic 时其签名为 `None`，拒绝。 |
| 457-458 | 多个 write 的地址映射必须相同。 |
| 459 | 保存第一个 write 签名，作为该 node 的 reference。 |
| 461 | read 为空时进入常量生成分支。 |
| 462-465 | 只有纯常量 body 可返回 `neutral_constant`；它仍要在 kernel 级别被额外验证。 |
| 466 | 非常量且无 read 的节点拒绝，例如独立 iota、RNG 或未知生成逻辑。 |
| 468-469 | 有 read 时，所有 read 也必须是 `MemoryDep`。 |
| 470 | 为所有 read 构造签名。 |
| 471-472 | 任意 read 无法构造直接签名时拒绝。 |
| 473-477 | 每个 read 都必须与 write reference 相同。broadcast、transpose、stride slice 会在这里失败。 |
| 478 | 通过后返回 `direct` 和 reference。 |

对于 broadcast `x[M,N] + bias[N]`，概念签名是：

```text
x read / out write: index = d0 * N + d1, sizes = (M, N)
bias read:           index = d1,          sizes = (N)
```

因此 bias read 无法与 write reference 相等，函数返回 `None`。

### 3.5 融合 kernel 级检查（第 480-535 行）

Scheduler fusion 后一个 kernel 可以有多个 SchedulerNode；仅每个 node 自身 direct 还不够，
还需要验证 node 之间可共同视为同一 workload。

#### 3.5.1 neutral constant 的消费检查（第 480-495 行）

```python
consumed_names = {
    dep.name
    for node, _, _ in classifications
    for dep in node.read_writes.reads
    if isinstance(dep, MemoryDep)
}
return all(
    dep.name in consumed_names
    for node, kind, _ in classifications
    if kind == "neutral_constant"
    for dep in node.read_writes.writes
)
```

| 行 | 说明 |
| --- | --- |
| 484-489 | 收集整个融合 kernel 内所有 `MemoryDep` read 的 buffer name，即“被其他节点消费”的逻辑 buffer 集合。 |
| 490-495 | 对每个 neutral constant node 的每个 write，要求其 buffer name 出现在该集合。 |

这避免把水平融合进同一 kernel、但并未被后续计算使用的独立 `full` 输出误认成临时常量。

#### 3.5.2 外部 direct read 检查（第 497-511 行）

```python
produced_names = {
    dep.name
    for node, _, _ in classifications
    for dep in node.read_writes.writes
    if isinstance(dep, MemoryDep)
}
return any(
    dep.name not in produced_names
    for node, kind, _ in classifications
    if kind == "direct"
    for dep in node.read_writes.reads
    if isinstance(dep, MemoryDep)
)
```

| 行 | 说明 |
| --- | --- |
| 499-504 | 收集 kernel 内产生的全部 buffer name。 |
| 505-511 | 查找任一个 direct node read，其 name 不属于 produced set；这说明该 read 来自 kernel 外部输入。 |

因此：

```text
x + full_like(x, 2)  -> 有外部 x read，可准入
full_like(x, 2)      -> 没有 direct read，不准入
full -> relu(full)   -> 全部 read 都来自 kernel 内，不准入
```

#### 3.5.3 `_classify_group_workload`（第 513-535 行）

| 行 | 说明 |
| --- | --- |
| 514-515 | 非 pointwise template 直接返回 `None`。reduction 的 feature 行为完全保留。 |
| 516 | 从 `SIMDKernelFeatures.scheduler_nodes()` 获取实际 SchedulerNode；该接口会过滤 Enable/DisableReduction marker。 |
| 517-518 | 空 schedule 不可分类。 |
| 520 | 创建收集 `(node, kind, reference)` 的列表。 |
| 521-526 | 逐 node 分类；任意一个 node 返回 `None`，整个 kernel 回退普通 pointwise。 |
| 528 | 取出全部 node reference signature。 |
| 529-530 | 所有 node 的 write 映射也必须一致。 |
| 531-532 | 纯常量 node 的输出必须被消费。 |
| 533-534 | kernel 至少需要一个外部 direct read。 |
| 535 | 全部条件通过后，唯一返回值为字符串 `"elementwise"`。 |

### 3.6 将分类接入 grouped metadata（第 537-578 行）

```python
dynamic_split_axes, static_split_axes = self._classify_split_axes()
template = self._grouped_template_name()
workload = self._classify_group_workload(template)
```

| 行 | 说明 |
| --- | --- |
| 538 | 保留原有动态/静态 split axis 分析。 |
| 539 | 只计算一次 template，避免后续 return 中重复调用。 |
| 540 | 在 SplitTiling 阶段完成 workload 分类。若不是 pointwise 或无法准入，值为 `None`。 |
| 541-565 | 原有 primary axis、secondary axis 和 reduction-tiling fallback 逻辑保留不变。 |
| 566 | 将 workload 传给 `_build_group_features`，使 elementwise 走新 feature 名称。 |
| 567-568 | 原有保护：没有可用 feature 时不创建 grouped metadata。 |
| 569-578 | 构造 `GroupedKernelMeta`。 |
| 570 | `enabled=True` 的语义不变：只表示 kernel 进入 grouped path。 |
| 571 | template 仍为 `pointwise/reduction/persistent_reduction`。 |
| 572 | 新增 `workload=workload`，保存 elementwise 分类结果。 |
| 573-578 | 保留原有 primary axis、静态 split、secondary symbolic axis、feature 和 runtime block 参数。 |

重要边界：`_build_grouped_meta()` 本身只会在 group 开关打开、template 被允许后调用；如果
没有动态 split axis 或动态 reduction tiling axis，原有逻辑会在此函数中返回 `None`。本提交没有
扩大静态 shape 或 group 关闭时的行为范围。

## 4. `symbolic_grouping.py`：metadata schema 与 payload 兼容

文件：`torch_npu/_inductor/runtime/symbolic_grouping.py`

### 4.1 `GroupedKernelMeta` 字段（第 50-59 行）

```python
class GroupedKernelMeta:
    enabled: bool
    template: str
    workload: str | None
    primary_group_axis: str | None
```

| 行 | 说明 |
| --- | --- |
| 50 | dataclass 继续保持 `frozen=True, slots=True`，metadata 创建后不可原地改写。 |
| 51 | grouped metadata 的类型定义。 |
| 52-53 | 原字段：是否启用及大类 template。 |
| 54 | 新字段。当前合法值仅为 `None` 或 `"elementwise"`。 |
| 55-59 | 原有 axis、feature、runtime block 字段不变。 |

### 4.2 `to_payload`（第 61-81 行）

| 行 | 说明 |
| --- | --- |
| 61 | 该方法将 dataclass 转换为可保存/传递的 `dict[str, object]`。 |
| 62-64 | 输出原有 `enabled`、`template`。 |
| 65 | 新增 `"workload": self.workload`。`None` 也显式写入 payload，避免调用方靠缺字段猜测版本。 |
| 66-80 | 原有字段序列化逻辑不变。 |

### 4.3 `from_payload`（第 83-108 行）

| 行 | 说明 |
| --- | --- |
| 83-84 | 静态反序列化入口。 |
| 85 | 创建新 `GroupedKernelMeta`。 |
| 86-87 | 保持对 `enabled/template` 的类型校验。 |
| 88 | 新增 workload 解析和白名单校验。因为使用 `payload.get("workload")`，旧 payload 缺字段时得到 `None`，保持向后兼容。 |
| 89-108 | 原有其他字段反序列化不变。 |

### 4.4 `_require_group_workload`（第 390-394 行）

```python
def _require_group_workload(value: object) -> str | None:
    workload = _require_optional_str(value, "workload")
    if workload not in (None, "elementwise"):
        raise ValueError(f"unsupported group workload: {workload}")
    return workload
```

| 行 | 说明 |
| --- | --- |
| 390 | 定义 workload 的 schema 校验入口。 |
| 391 | 复用可选字符串校验；非字符串且非 `None` 的值先抛出 `TypeError`。 |
| 392 | 白名单只允许当前已实现的两个状态。 |
| 393 | 新增未知 workload 时显式失败，防止 runtime 静默误解 payload。 |
| 394 | 返回已验证的值。 |

## 5. `triton.py`：写入 generated kernel metadata

文件：`torch_npu/_inductor/codegen/triton.py`

### 5.1 grouped meta 存在时（第 1987-2002 行）

```python
grouped_meta = getattr(self, "grouped_autotune_meta", None)
if grouped_meta is not None:
    inductor_meta.update(
        {
            "group_enabled": bool(grouped_meta.enabled),
            "group_template": grouped_meta.template,
            "group_workload": grouped_meta.workload,
            ...
        }
    )
```

| 行 | 说明 |
| --- | --- |
| 1987 | 读取 SplitTiling 保存到 kernel 的 grouped metadata。 |
| 1988 | 非 `None` 表示该 kernel 成功建立 grouped plan。 |
| 1991 | 保留 `group_enabled`。 |
| 1992 | 保留 `group_template`。 |
| 1993 | 新增 `group_workload`。这是生成 Python kernel、debug trace 和 UT 可观察到分类结果的唯一 codegen 输出字段。 |
| 1994-2000 | 原有 grouped launch 所需字段不变。 |

### 5.2 grouped meta 不存在时（第 2003-2015 行）

| 行 | 说明 |
| --- | --- |
| 2003 | `grouped_meta is None` 的 fallback 分支。 |
| 2006-2007 | 原有 `group_enabled=False`、`group_template=None`。 |
| 2008 | 新增 `group_workload=None`，使所有 generated kernel 都有稳定字段，UT 不必区分字段缺失和分类结果为空。 |
| 2009-2013 | 清空原有 grouped 相关字段。 |
| 2015 | 返回完整 `inductor_meta`。 |

### 5.3 grouped path 后续禁用时（第 2093-2105 行）

`_disable_grouped_autotune()` 会在 grouped plan 或 benchmark 参数不支持时清理 metadata。

| 行 | 说明 |
| --- | --- |
| 2093-2098 | 原有 debug log，记录禁用原因。 |
| 2099-2100 | 清除 enabled/template。 |
| 2101 | 同步清除 `group_workload`，避免最终 fallback kernel 带着过期的 `elementwise` 标签。 |
| 2102-2105 | 清除其他 grouped runtime 字段。 |

这一行是 metadata 一致性保护：`group_enabled=False` 时 workload 必须也是 `None`。

## 6. `test_inductor_dynamic_shapes.py`：NPU 集成测试

文件：`test/_inductor/test_inductor_dynamic_shapes.py`

### 6.1 导入与测试注册（第 24、536、855-857 行）

| 行 | 说明 |
| --- | --- |
| 24 | 新增 `run_and_get_code`，它在执行 compiled function 时收集 generated Python code。测试以此检查最终 metadata，而不是 mock 私有分类函数。 |
| 532-534 | 分节标题，说明这里测试的是 symbolic grouped-autotune 的 workload 分类。 |
| 536 | 新增独立 `TestSymbolicGroupElementwise`。它不复用 `DynamicShapeTestMixin._compile()`，因为该 helper 使用 `dynamic=True`，可能把多个维度都符号化。 |
| 855 | 保留已有 `TestFusionDynamicShapes` 参数化注册。 |
| 856 | 注册新 class，确保 common-utils 参数化框架发现其中测试。 |
| 857 | 保留 nonzero 测试注册。 |

### 6.2 setUp / tearDown（第 537-549 行）

| 行 | 说明 |
| --- | --- |
| 537-538 | 调用父类 setUp，保留测试框架初始化。 |
| 539 | 局部导入 NPU Inductor config，避免模块导入时永久修改配置。 |
| 541 | 保存 config module 引用。 |
| 542 | 保存原始 group 开关，避免该测试影响同文件的其他动态 shape 测试。 |
| 543 | 强制开启 symbolic grouped autotune。 |
| 544 | 清除 Dynamo cache，避免复用前一个测试在 group 关闭状态下缓存的 graph。 |
| 546-549 | 无论断言成功或失败都恢复开关、再次 reset，然后调用父类 tearDown。 |

### 6.3 公共执行和断言 helper（第 551-576 行）

```python
def _run_and_check(self, fn, inputs, next_inputs, expected_workload):
```

| 行 | 说明 |
| --- | --- |
| 552 | 同时遍历首次输入和第二次输入。 |
| 553-554 | 仅当 tensor 的第 0 维确实变化时，对首次输入 `mark_dynamic(..., 0)`。这样测试构造一个动态轴、其余轴静态，符合当前 grouped representative 每个 feature 只支持一个 symbolic axis 的条件。 |
| 556 | eager 执行，得到精度基准。 |
| 557 | 以 Inductor backend 编译。 |
| 558 | 执行首次输入并收集 generated code。 |
| 559 | 对首次输入执行 `assert_close`。 |
| 561-563 | 用第二组 shape 做 eager/compiled 精度比较，验证动态轴运行时变化不破坏正确性。 |
| 565 | 将期望值构造成 generated code 中的文本，例如 `"'group_workload': 'elementwise'"` 或 `"'group_workload': None"`。 |
| 566-572 | 在收集到的每一段 code 中，同时要求 grouped 开启、template 为 pointwise、workload 等于期望。三个条件放在同一段 code 中，避免从不同 kernel 文本拼出假阳性。 |
| 573-576 | 若没有匹配 kernel，失败信息打印全部 code，便于定位 metadata 或 kernel 拆分问题。 |

### 6.4 基础 elementwise（第 578-590 行）

```python
def fn(x, y):
    return torch.relu(x + y) * 0.5
```

| 行 | 说明 |
| --- | --- |
| 579-580 | 同 shape 直接 read、直接 write 的正例。 |
| 582-585 | 第一组 shape 为 `(257, 1031)`；非 2 的幂可覆盖 tail mask。 |
| 586-589 | 第二组仅改变第 0 维为 263。 |
| 590 | 期望 workload 为 `"elementwise"`。 |

### 6.5 broadcast 拒绝（第 592-599 行）

```python
def fn(x, bias):
    return torch.relu(x + bias) * 0.5
```

| 行 | 说明 |
| --- | --- |
| 593-594 | `x` 为 `[M, 1031]`，`bias` 为 `[1031]`，add lowering 产生 broadcast read。 |
| 596 | bias 保持静态一维，确保只让 `x` 的第 0 维动态。 |
| 597-598 | 两次调用只改变 `M`。 |
| 599 | 期望 `None`。group path 仍可开启，只有 elementwise 子分类回退。 |

### 6.6 被消费常量临时值（第 601-608 行）

```python
generated = torch.full_like(x, 2.0)
return torch.relu(x + generated)
```

| 行 | 说明 |
| --- | --- |
| 602-604 | 构造 full 临时值并被 add 消费，覆盖 `neutral_constant` + 外部 direct read 的正例。 |
| 606-607 | 两组单输入动态 shape。 |
| 608 | 期望 `elementwise`。 |

### 6.7 独立数据生成（第 610-616 行）

```python
return torch.full_like(x, 2.0)
```

| 行 | 说明 |
| --- | --- |
| 611-612 | 只生成 full 输出，不读取 `x` 的数据。`x` 只提供 shape 信息。 |
| 614-615 | 使用与正例相同的动态 shape 结构，避免把差异归因于 shape。 |
| 616 | 期望 `None`，覆盖“必须有外部 direct read”的 kernel 级保护。 |

### 6.8 strided reindex（第 618-630 行）

```python
return torch.relu(x[:, ::2] + y)
```

| 行 | 说明 |
| --- | --- |
| 619-620 | `x[:, ::2]` 使 x 的内层 read index 含 `2*x1`，而 y/read 和 output/store 使用 `x1`。 |
| 622-625 | x 的末维为 2062，slice 后正好为 1031，能与 y 相加。 |
| 626-629 | 第二组仅改变动态 batch 维。 |
| 630 | 期望 `None`，覆盖 stride reindex 的映射不一致。 |

## 7. 实际 NPU 验证证据

验证环境：

```text
host/container : NPU_A2 / l30023782_210
base source    : 3bbc52affa960e55474ffefa48d07b52c64c6147
installed path : /usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu
```

由于容器安装包来自 base source commit，验证时仅覆盖安装包中本提交改动的三个 Python 文件：

```text
torch_npu/_inductor/codegen/split_tiling.py
torch_npu/_inductor/codegen/triton.py
torch_npu/_inductor/runtime/symbolic_grouping.py
```

执行命令：

```bash
cd /home/l30023782/Ascend/PTA_210
source pytorch_new/env.sh
TORCHINDUCTOR_CACHE_DIR=/home/l30023782/Ascend/PTA_210/
symbolic_elementwise_validation/ut_cache_final \
python pytorch_new/test/_inductor/test_inductor_dynamic_shapes.py \
  TestSymbolicGroupElementwise -v
```

结果：

```text
Ran 5 tests in 76.169s
OK
```

验证通过的 workload 结果：

| 测试 | `group_enabled` | `group_template` | `group_workload` | 精度 |
| --- | --- | --- | --- | --- |
| 基础 add/relu/mul | `True` | `pointwise` | `elementwise` | 两组动态 shape `assert_close` 通过 |
| broadcast add | `True` | `pointwise` | `None` | 两组动态 shape `assert_close` 通过 |
| 消费 `full_like` | `True` | `pointwise` | `elementwise` | 两组动态 shape `assert_close` 通过 |
| standalone `full_like` | `True` | `pointwise` | `None` | 两组动态 shape `assert_close` 通过 |
| `x[:, ::2] + y` | `True` | `pointwise` | `None` | 两组动态 shape `assert_close` 通过 |

## 8. 保持不变的行为与限制

1. `group_template` 没有新增 `elementwise`，仍是 `pointwise`。
2. reduction 和 persistent reduction 的 feature 构造逻辑不读取 workload。
3. group 未开启、template 不允许、没有可用动态 group axis 时，不生成 grouped metadata；
   generated metadata 中的 `group_workload` 为 `None`。
4. workload 目前只有 `None` 和 `"elementwise"` 两个合法值；后续要新增 workload，必须同时扩展
   `_require_group_workload()`、feature 构造、runtime 策略和测试。
5. 分类是保守准入。无法静态证明访问签名相同、出现未知 dependency 或特殊语义时会 false negative，
   但不会改变 kernel 正确性，只会继续使用原 pointwise group 策略。
6. `rand/randn/load_seed` 已在生产分类逻辑中拒绝。没有将 RNG 纳入当前 NPU 集成精度测试，原因是
   eager 与 compiled 路径的随机流不保证逐元素一致，即使重置 seed 也不能使用 `assert_close` 作为
   稳定断言。

## 9. 最终结论

`a3782c3f8` 将 elementwise 作为 pointwise grouped autotune 的 workload 标签，而不是新的 kernel
template。分类基于 lowering 后的 `MemoryDep` read/write 映射和 `LoopBody` 语义，结果经
`GroupedKernelMeta`、`inductor_meta` 传到 generated kernel。五个 NPU 动态 shape 集成测试同时
证明了数值正确性和 workload metadata 符合预期。
