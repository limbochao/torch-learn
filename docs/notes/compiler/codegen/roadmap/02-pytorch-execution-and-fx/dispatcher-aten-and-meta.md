---
title: Dispatcher、ATen Operator 与 Meta 实现
---

# Dispatcher、ATen Operator 与 Meta 实现

## 核心链路

```mermaid
flowchart LR
    Call["torch.add / Tensor.relu"] --> Overload["operator overload"]
    Overload --> Schema["operator schema"]
    Schema --> Key["dispatch key + device"]
    Key --> Kernel["eager kernel or meta kernel"]
```

Dispatcher 选择 eager 实现，meta/fake 实现则主要计算输出 metadata。Inductor 是否有 lowering 是另一条
链路，不能从 eager kernel 存在推断 lowering 一定存在。

## 源码入口

- `<workspace>/pytorch/torch/_ops.py`：operator overload 对象。
- `<workspace>/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h`：Dispatcher 接口。
- `<workspace>/pytorch/torch/_subclasses/fake_tensor.py`：fake dispatch 和 metadata propagation。
- `<workspace>/pytorch/torch/_inductor/lowering.py`：backend lowering registry，见目标 4。

## 练习问题

- 一个 operator schema 如何表达默认参数、alias 或 mutation？
- meta kernel 需要返回哪些信息才能让后续编译继续？
- 为什么 eager dispatch 成功仍可能在 Inductor lowering 阶段失败？

## 实现展开：从 overload 到 dispatch kernel

`torch.add` 这类 Python API 最终会落到 `torch._ops.OpOverload`/`OpOverloadPacket`。调用时，Dispatcher
根据 operator schema、输入 tensor 的 dispatch key 和 device 选择实现。简化后的数据流是：

```text
Python call
  -> OpOverload.__call__
  -> Dispatcher::call / typed_call
  -> dispatch key set
  -> CPU/CUDA/NPU/Meta 等 kernel
```

源码阅读入口：

- `torch/_ops.py`：Python 侧 overload 对象和调用转发。
- `aten/src/ATen/core/dispatch/Dispatcher.h`：C++ Dispatcher 注册/查找/调用接口。
- `torch/_subclasses/fake_tensor.py`：FakeTensor mode 下把真实 dispatch 替换为 metadata-aware dispatch。
- `torch/_inductor/lowering.py:527`：`register_lowering`，这是另一条 backend lowering 注册表。

关键区别是：Dispatcher 解决“eager 或 meta 谁来实现这个 operator”，lowering registry 解决“Inductor 如何
把这个 operator 转成 IR”。例如 `aten.add` 可以有 CPU eager kernel、Meta kernel 和 NPU kernel，同时
Inductor 还可能把它 lowering 成 `Pointwise`，这四者不是同一个实现。

## 最小追踪

对 `x + y` 记录四个 target：Python 表达式、FX target、Dispatcher operator、Inductor lowering key。
如果 FX target 是 overload packet 而 lowering 查找的是具体 overload，继续沿 `get_overloads()` 和
`register_lowering()` 展开，避免把 registry miss 误判成 dispatcher miss。

## Dispatcher lookup 的实际输入

一个 `OpOverload` 调用进入 C++ Dispatcher 后，lookup key 不是 Python 函数名，而是 operator schema 对应的
`DispatchKeySet`。Tensor 参数的 device、layout、autograd state 和 active dispatch modes 共同参与 key 选择；没有
匹配 kernel 时，Dispatcher 按 fallback/alias key 规则继续查找或抛出错误。

因此同一个 `aten.add.Tensor` 可以分别命中 CPU、CUDA、PrivateUse1/NPU、Autograd 或 Meta 实现。`Meta` kernel 只计算
size/stride/dtype 等 metadata；它不执行真实 device load/store。FakeTensorMode 会在 dispatch 前后包住这个过程，保存
symbolic size 和 alias 信息。

## overload、decomposition 与 lowering 的边界

```text
Python torch.add
  -> OpOverloadPacket
  -> 具体 OpOverload（schema/default overload）
  -> Dispatcher 按 DispatchKeySet lookup
  -> Meta/Fake 或 device eager kernel
  -> Inductor lowering registry 以同一个具体 overload 查找
```

decomposition 可以在 AOT/Inductor 前端把一个 overload 改写为其他 ATen operators；这发生在 lowering lookup 之前或
相关预处理阶段。若具体 overload 没有 lowering，GraphLowering 还可能创建 fallback operation。因此“Meta kernel 成功”只
证明 metadata dispatch 成功，“lowering 成功”还需要独立证据。

## 最小证据表

记录一个算子时保存：Python callable、具体 overload/schema、输入 dispatch keys、命中的 kernel、fake/meta 输出 metadata、
Inductor lowering 函数和最终 IR class。缺少具体 overload 时，容易把 packet 级别的名字与 registry key 混淆。
