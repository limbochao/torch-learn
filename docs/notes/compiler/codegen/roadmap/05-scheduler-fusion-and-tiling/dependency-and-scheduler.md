---
title: Dependency、SchedulerNode 与拓扑顺序
---

# Dependency、SchedulerNode 与拓扑顺序

## 数据流

```mermaid
flowchart LR
    Buffer["IR buffer read/write"] --> Dep["Read/Write dependency"]
    Dep --> Node["SchedulerNode"]
    Node --> Order["topological order"]
    Order --> Group["candidate kernel group"]
```

dependency 不仅表示先后，还可能携带读写区域、mutation、alias、device 和 layout 约束。SchedulerNode
是 scheduler 视角的计算单元，可能来自单个 IR node，也可能代表 fused group。

## 源码入口

- `<workspace>/pytorch/torch/_inductor/dependencies.py`：read/write dependency。
- `<workspace>/pytorch/torch/_inductor/scheduler.py`：`Scheduler`、`SchedulerNode`、
  `FusedSchedulerNode`。

## 练习

为 `add -> relu -> sum` 画 dependency graph，标出每个 buffer 的 writer、reader 和可能的 realization。
再比较 transpose 输入导致的 dependency/index 变化。

## 需要回答的问题

- 一个 node 没有显式 data edge 时，为什么仍可能不能重排？
- mutation 和 alias 如何限制 scheduler 的自由度？
- fused node 的 users 如何合并原始节点的 users？

## 实现展开：ReadWrites 如何变成 SchedulerNode

`dependencies.py` 将 loop body 中的 `ops.load`、`ops.store`、`ops.index_expr` 收集成 `ReadWrites`。
其中：

- `MemoryDep(name, index, size)` 表示对某个 buffer 的具体索引访问。
- `StarDep(name)` 表示无法精确到单一 index 的整 buffer/广义依赖。
- `WeakDep(name)` 表示 mutation 等弱依赖，需要额外规则处理。

`dependencies.extract_read_writes()` 先从 `LoopBody` 收集这些 dep，再由 `SchedulerNode._compute_attrs()`
保存到 `self.read_writes`。同时，它调用 IR node 的 `simplify_and_reorder()`，得到 `_sizes` 和规范化
后的 loop body，并通过 backend 的 `group_fn` 形成 `self.group`。

初始化 `Scheduler`（`scheduler.py:2757`）时，代码会：

```text
operations
  -> create_scheduler_node()
  -> prune_deps()
  -> compute_dependencies()
  -> topological_sort_schedule()
  -> dead_node_elimination()
  -> compute_ancestors()
```

mutation 还会维护 `mutation_real_name` 和 `mutation_renames`，把同一逻辑 buffer 的不同版本映射回依赖
图中的可比较名称，避免错误地形成 cycle。分析一个 scheduler 顺序时，除了看 data edge，还要查看这两张
mutation map。

## `Scheduler` 的实际建图顺序

阅读 `Scheduler.__init__()` 时，重点不是“它最后有一个 DAG”，而是每一步改变了什么集合：

```text
IR nodes
  -> create_scheduler_node
  -> prune_deps
  -> compute_dependencies
  -> topological_sort_schedule
  -> dead_node_elimination
  -> compute_ancestors
```

`create_scheduler_node()` 对可调度的 `ir.ComputedBuffer` 创建 `SchedulerNode`，对已经融合或不可拆的对象保留
相应 node 类型。`prune_deps()` 删除同一节点内部可证明无用的依赖；这一步之后不能再把原始 IR 的所有 load
都当成 scheduler edge。`compute_dependencies()` 才把跨节点的 `MemoryDep`/`WeakDep` 变成 predecessor、successor
和 mutation 关系。

拓扑排序的输入不是纯 `read -> write` 图：mutation rename 会把同一 buffer 的版本链显式加入排序约束，避免
“写 A、读 A、再写 A”被错误重排。`compute_ancestors()` 随后为每个 node 缓存可达祖先，fusion 检查会复用该集合，
而不是每次重新遍历图。

## 为什么 `MemoryDep` 比 buffer 名字更重要

`MemoryDep(name, index, size)` 的 `index` 是 SymPy 表达式。两个节点都读 `buf0` 并不说明它们访问同一数据：

```text
producer: store(buf0, x)
consumer: load(buf0, x)       -> 可能消除中间 buffer
other:    load(buf0, x + 1)   -> 仍需邻域依赖，不能按同一元素融合
```

`StarDep` 没有精确索引，scheduler 必须采用保守的整 buffer 约束；`WeakDep` 则主要保证 mutation 顺序，不能
像普通 `MemoryDep` 那样用于消除一次 load。调试 fusion 时应打印 dep 类型、`index`、`size`，而不是只打印 buffer 名。

## 一个可复现的检查表

对任意两个 scheduler nodes，按源码中的数据结构逐项记录：

1. `node.read_writes.reads/writes` 中的 dep 类型和索引。
2. `node.ancestors` 是否包含对方。
3. `mutation_real_name` 是否把不同名字归并为同一逻辑 buffer。
4. `group` 中的 `(numel, rnumel)` 是否一致或可通过 backend `group_fn` 归一。
5. topological schedule 中两者的相对位置，以及 dead-node elimination 是否删除了中间结果。

这五项能把“依赖不存在”“依赖被 prune”“mutation 约束”“domain 不兼容”区分开；只看最终 kernel 数量无法定位这些
分支。
