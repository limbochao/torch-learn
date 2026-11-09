---
title: Dynamic IR、SizeVars 与 Indexing
---

# Dynamic IR、SizeVars 与 Indexing

## 生成链

```mermaid
flowchart LR
    SymShape["symbolic shape"] --> SizeVars["SizeVarAllocator"]
    SizeVars --> Simplify["simplify/guard/check"]
    Simplify --> Index["symbolic index formula"]
    Index --> Emitter["kernel load/store expression"]
```

sizevars 负责表达式化简、等价判断、stride/index 变量和必要的 runtime checks。动态 index 不能只看静态
hint；需要检查它如何被化简、如何生成 guard，以及目标 emitter 是否保留正确的符号依赖。

## 源码入口

- `<workspace>/pytorch/torch/_inductor/sizevars.py`：`SizeVarAllocator`、simplify、range/check、
  stride/index helper。
- `<workspace>/pytorch/torch/_inductor/graph.py`：symbolic size/layout 进入 GraphLowering。
- `<workspace>/pytorch/torch/_inductor/codegen/wrapper.py`：runtime size expression。

## 练习

比较静态和动态版本的同一 index formula，标出常量折叠、runtime scalar、guard 和 mask 的差异。验证
non-contiguous dynamic 输入，不要只测试 contiguous。

## 实现展开：SizeVarAllocator 的职责

`torch/_inductor/sizevars.py:62` 的 `SizeVarAllocator` 是 Inductor 侧的符号表达式服务对象。它不负责
创建 Dynamo 的原始 symbol，而是负责在 lowering/codegen 阶段：

- `simplify(expr)`：使用 replacements、range 和 SymPy 规则化简表达式。
- `evaluate_expr(expr)`：在可求值时得到运行时/编译时结果。
- `guard_int(expr)`/`guard_int_seq()`：当 backend 需要 Python int 时建立必要的 runtime guard。
- `stride_order(index, vars)`（约 961）：根据 index 对各维访问的影响排序。
- `simplify_loops()`：合并或重排等价 loop/index 结构。

典型动态 index 的数据流是：

```text
SymInt s0, s1
  -> lowering creates index = i * s1 + j
  -> SizeVarAllocator.simplify(index)
  -> static terms folded, symbolic terms preserved
  -> wrapper emits s1 runtime argument
  -> Triton/C++ emitter emits pointer + i * s1 + j
```

`guard_int()` 是一个边界信号：如果某个 backend 选择必须依赖具体 int，而当前 expression 无法安全
静态求值，系统可能生成 runtime assertion、specialization 或走另一条 kernel path。看到一个动态值被
`int()` 化时，要查它是合法 guard 还是意外丢失了符号约束。

## 与 layout/index 的关系

动态 shape 和 dynamic stride 是两个维度。`s0` 可能只出现在 size/grid，stride 仍可能是静态连续布局；
transpose 或 arbitrary `as_strided` 则会让 pointer expression 同时依赖动态 stride。生成 kernel 时必须
逐个检查 size、stride、storage offset 是否被保留，不能因为 shape dynamic 就假设 layout dynamic，或反过来。

## `SizeVarAllocator` 的判定顺序

动态 index 进入 backend 前通常经过三种操作：

```text
原始 SymPy index -> simplify(expr) -> stride_order(index, vars)
                    -> evaluate_expr/guard_int（backend 需要具体值时）
```

`stride_order()` 影响 loop/range 排列，却不改变 tensor 语义 index；`guard_int()` 可能产生约束或触发
specialization，不能无证据地替换成 Python `int`。因此动态 shape 下“轴顺序变化”与“轴长度变成常量”是两个独立事件。

设 contiguous `(s0, 128)` 的二维输出：

```text
静态/半动态: offset = i * 128 + j，grid/mask 使用 s0
动态 stride: offset = i * stride0 + j * stride1
```

只有后一种需要把 stride scalar 一并放入 ABI；检查 generated pointer 时应分别标注 size、stride、storage offset 的
symbolic 来源。
