---
title: 目标 6：Backend Codegen 与 Runtime
---

# 目标 6：Backend Codegen 与 Runtime

## 目标

理解已调度计算如何变成 Triton/C++/template/extern kernel、Python/C++ wrapper 和可执行模块。完成后，
应当能逐行解释一个小型 generated kernel，并从 wrapper 跟到 launcher、cache 和 stream。

## 两类 Codegen 产物

### Kernel codegen

负责生成真正执行计算的目标代码，例如：

- Triton kernel；
- CPU C++ loop/vector kernel；
- GEMM/template kernel；
- 外部库调用描述；
- NPU MLIR/DSL 等后端表示。

### Wrapper codegen

负责组织 kernel 周围的运行逻辑，例如：

- 输入 shape/stride assertion；
- symbolic size 读取和表达式计算；
- buffer allocation/reuse/free；
- device guard 和 stream 获取；
- grid/launch 参数；
- kernel 调用顺序和输出返回。

只检查 kernel 源码不足以覆盖完整 codegen 正确性。错误参数、stream 或 buffer lifetime 也会造成 runtime
失败和精度问题。

## 代码入口

- `<workspace>/pytorch/torch/_inductor/graph.py`
  - `GraphLowering.codegen`
  - `GraphLowering.compile_to_module`
- `<workspace>/pytorch/torch/_inductor/codegen/triton.py`：Triton kernel emitter。
- `<workspace>/pytorch/torch/_inductor/codegen/cpp.py`：CPU C++ loop emitter。
- `<workspace>/pytorch/torch/_inductor/codegen/wrapper.py`：Python wrapper codegen。
- `<workspace>/pytorch/torch/_inductor/codegen/cpp_wrapper_cpu.py`：CPU C++ wrapper。
- `<workspace>/pytorch/torch/_inductor/codecache.py`：source/binary/module cache。
- `<workspace>/pytorch/torch/_inductor/runtime/triton_heuristics.py`
  - `CachingAutotuner`
  - compile result / launcher 构造
  - precompile 和 run

## 机械阅读 Generated Kernel

按固定顺序阅读，不要从复杂表达式中间开始：

1. 读取函数签名和 metadata，映射每个 pointer/scalar 参数。
2. 确认 grid 和 `program_id` 对应哪一层 block。
3. 写出 block offset 与 lane offset。
4. 把 address expression 还原成 tensor shape/stride 公式。
5. 区分语义 mask、tail mask 和 padding mask。
6. 跟踪 load 后 value 的 shape/layout 变化。
7. 对 reduction 标出 accumulator 初始化、combine、final reduce。
8. 确认 store 地址、mask 和所在 loop scope。

如果无法把某行 generated code 映射回 tensor 语义，先回到 IR/indexing 和 node schedule，不要猜测 DSL
内建函数的意图。

## 练习

### 练习 1：CPU 全链路

在本地 CPU 上编译 pointwise 用例，设置独立 cache 目录，收集 output wrapper 和生成的 C++。标出输入
检查、buffer allocation、kernel 定义和调用。

### 练习 2：Triton pointwise

在可用 trace 或设备环境中，逐行注释 `add + relu` kernel。手算一个 program instance 访问的元素范围，
并解释尾块 mask。

### 练习 3：Triton reduction

逐行标记 preserved/reduction axes、load mask、accumulator、`tl.sum` 和 store。改变 reduction size，比较
persistent/non-persistent 或 block 配置，但只陈述当前环境中有证据的选择。

### 练习 4：cache 与 recompilation

冷 cache 执行、同进程重复执行、重新进程执行，记录 source cache、binary cache 和 autotune cache 的
命中情况。区分 Dynamo graph cache 与 Inductor code cache。

### 练习 5：launcher

从 wrapper 的 kernel call 跟踪到 runtime launcher，记录 grid、stream、num warps/stages 或对应 backend
参数。确认哪些参数在编译时确定，哪些在调用时由 symbolic shape 计算。

## 源码范围

本章从 scheduler 已决定 kernel group 开始，覆盖 kernel emitter、wrapper、code cache、compile result、
autotune 和 launcher，止于设备 runtime 被调用的边界。

- **纳入范围**：`codegen/triton.py`、`cpp.py`、`wrapper.py`、`cpp_wrapper_cpu.py`、template 相关
  emitter、`codecache.py` 和 `runtime/triton_heuristics.py`。
- **暂不深入**：Dynamo/AOTAutograd、Inductor lowering/scheduler 的决策过程、dynamic shape 的符号推理、
  NPU 专用实现；这些在目标 2 至 5、7、8 展开。
- **建议顺序**：先读 Triton/C++ kernel emitter，再读 wrapper，再读 cache/compile result，最后读
  autotune、grid 和 stream launch。

## 章节文档地图

- [Overview 与 Architecture](overview-and-architecture.md)
- [Triton Kernel Codegen](triton-kernel-codegen.md)
- [C++、Template 与 Extern Codegen](cpp-template-and-extern-codegen.md)
- [Wrapper Codegen 与 Kernel ABI](wrapper-codegen-and-abi.md)
- [Code Cache、Compile Result 与 Autotune](cache-compile-and-autotune.md)

## 建议调试开关

具体开关随版本变化，应先查当前版本配置。常用入口包括：

```bash
TORCH_COMPILE_DEBUG=1 \
TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-codegen-study \
python repro.py
```

也可以使用 PyTorch logging/config 接口输出 graph、guards、recompiles、scheduler 或 generated code。
环境变量名称和可用 artifact 以当前源码及运行日志为准。

## 产出物

- 一份带逐行语义注释的 pointwise generated kernel。
- 一份 reduction kernel 的 axis/load/reduce/store 图。
- wrapper 中 buffer、stream、grid 和 launch 参数映射表。
- 冷编译与 cache hit 的调用路径对照。

## 验收清单

- [ ] 能区分 kernel codegen 与 wrapper codegen。
- [ ] 能由 `tl.load` 或 C++ pointer expression 还原 shape/stride 地址。
- [ ] 能确认 mask 保护的具体访问范围。
- [ ] 能从 wrapper 跟到 cache、launcher 和 device stream。
- [ ] 能区分 compile latency、autotune latency 和 steady-state runtime。

## 常见误区

- generated Python wrapper 不是 eager Python 运算，它主要协调已编译 kernel。
- 能生成源码不等于 kernel 编译成功，kernel 编译成功也不等于 runtime 或结果正确。
- cache hit 会显著改变耗时；性能比较必须说明 cache 和 warmup 状态。
- 只截取报错行会丢失 kernel signature、metadata 和 loop scope，通常不足以判断根因。

[上一目标](../05-scheduler-fusion-and-tiling/) | [返回总路线](../) |
[下一目标](../07-dynamic-shapes/)
