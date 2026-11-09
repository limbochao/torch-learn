---
title: 目标 2：PyTorch 执行语义与 FX
---

# 目标 2：PyTorch 执行语义与 FX

## 目标

理解 eager 程序进入编译器前的语义载体。完成后，应当能读懂 FX graph，区分 Python API、ATen
operator、Tensor metadata 和 device dispatch，不会把 FX code generation 与 Inductor kernel codegen
混为一谈。

## 核心概念

### Python API、ATen 与 Dispatcher

用户写的是 `torch.add`、`Tensor.relu()` 或 module，编译链最终更关注规范化的 operator 和 tensor
metadata。Dispatcher 根据 operator、dispatch key 和 device/backend 选择具体实现。

需要理解的边界：

- operator schema 描述调用签名和 alias/mutation 信息。
- meta/fake 实现可以只推导输出 metadata，不执行真实计算。
- eager kernel 的选择与 Inductor 是否为该 op 生成 fused kernel 是两件事。
- 不支持 lowering 的 op 可能成为 extern kernel、fallback 或 graph break，具体取决于所在阶段。

### FX Graph

FX graph 是由 node 构成的有向数据流图。重点 node 类型包括：

- `placeholder`：graph 输入。
- `get_attr`：module 参数或常量属性。
- `call_function`：调用函数/operator。
- `call_method`：调用对象方法。
- `call_module`：调用子 module。
- `output`：graph 输出。

FX 自身也有 `Graph.python_code()` 等“codegen”能力，用于把 graph 重新生成为 Python；这不是
Inductor 把 tensor loop 生成 Triton/C++ kernel 的 backend codegen。

## 代码入口

- `<workspace>/pytorch/torch/fx/graph.py`：`Graph`、`Node` 和 FX Python code generation。
- `<workspace>/pytorch/torch/fx/graph_module.py`：`GraphModule`。
- `<workspace>/pytorch/torch/_subclasses/fake_tensor.py`：`FakeTensorMode`。
- `<workspace>/pytorch/torch/_ops.py`：operator overload 对象和调用入口。
- `<workspace>/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h`：Dispatcher 核心接口。

源码阅读问题：

1. FX node 的 `target`、`args`、`kwargs` 分别表示什么？
2. `node.meta` 中哪些 tensor metadata 会被后续编译阶段消费？
3. FakeTensor 如何在不分配真实设备数据时传播 shape、stride 和 dtype？
4. inplace/view 操作包含哪些 alias 或 mutation 语义？

## 源码范围

本章覆盖“用户 Python/Tensor API 如何形成编译器可消费的图”，止于 FX graph、FakeTensor metadata
和 operator dispatch 语义，不进入 Dynamo 的 frame 捕获和 Inductor lowering。

- **纳入范围**：`torch/fx/graph.py`、`graph_module.py`、`torch/_ops.py`、FakeTensor 实现，以及
  `aten/src/ATen/core/dispatch/Dispatcher.h` 中与 operator、metadata、alias 和 dispatch 直接相关的
  接口。
- **暂不深入**：`torch/_dynamo/` 的 bytecode transformation、AOTAutograd、Inductor IR、Triton/C++
  emitter；它们分别属于目标 3 至 6。
- **建议顺序**：先看 operator/dispatch，再看 FX node/container，随后看 FakeTensor metadata，最后看
  view、alias 和 mutation。

## 章节文档地图

- [Overview 与 Architecture](overview-and-architecture.md)
- [Dispatcher、ATen Operator 与 Meta 实现](dispatcher-aten-and-meta.md)
- [FX Graph、Node 与 Python Codegen](fx-graph-and-python-codegen.md)
- [FakeTensor 与 Tensor Metadata](fake-tensor-and-metadata.md)
- [View、Alias 与 Mutation](view-alias-and-mutation.md)

## 练习

### 练习 1：手工 symbolic trace

对纯 module 使用 `torch.fx.symbolic_trace`，打印：

```python
traced.graph.print_tabular()
print(traced.code)
```

逐个标记 node 的 opcode、target、输入和用户。修改一个 node 后执行 `lint()` 和 `recompile()`，观察
Graph 与 GraphModule 代码之间的关系。

### 练习 2：查看 ATen graph

使用适合当前 PyTorch 版本的 export/Dynamo graph 获取接口，观察高级 module 如何分解成 ATen op。
对 `relu`、view、transpose 和 inplace op 做对比，记录 node metadata。

### 练习 3：shape/stride 传播

为 contiguous 和 transpose 输入捕获 graph，比较相同 operator 的 metadata。确认 graph 中的逻辑计算
相同，不代表 layout 和后续 codegen 相同。

## 产出物

- 一张 `Python API -> operator -> FX node -> backend` 的分层图。
- pointwise 用例的 FX graph 注释版。
- contiguous/transpose 两种输入的 metadata 对照表。

## 验收清单

- [ ] 能解释六类常见 FX node 及其数据依赖。
- [ ] 能从 FX graph 找到 graph 输入、常量、算子和输出。
- [ ] 能区分 FX Python codegen 与 Inductor backend codegen。
- [ ] 能解释 FakeTensor 为什么是编译期 shape/layout 分析的基础。
- [ ] 能指出 view、mutation 和普通 pointwise operator 的语义差异。

## 常见误区

- FX graph 是编译输入之一，不是最终 kernel IR。
- `traced.code` 可运行不代表它已经经过 Inductor 优化。
- 算子名称相同不代表 layout、dtype、device 和 alias 条件相同。
- eager kernel 没有被直接调用，不代表该 operator 的语义消失；语义可能已经被 lowering 到 fused loop。

[上一目标](../01-compiler-foundations/) | [返回总路线](../) |
[下一目标](../03-dynamo-and-aotautograd/)
