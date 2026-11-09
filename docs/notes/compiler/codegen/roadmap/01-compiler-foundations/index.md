---
title: 目标 1：编译器与张量程序基础
---

# 目标 1：编译器与张量程序基础

## 目标

建立阅读 Inductor 的最小编译器模型。完成后，应当能区分语义、IR、调度和目标代码，并能根据
shape/stride 手工推导简单张量访问地址。

## 前置知识

- Python 基础和基本的 PyTorch tensor 操作。
- 能阅读简单 C++ 和 Python 类。
- 不要求先学完整的 LLVM 或 MLIR。

## 必须掌握的概念

### 编译阶段

- frontend：接收源程序并保留可编译语义。
- IR：用适合分析和变换的数据结构表示程序。
- lowering：从高层语义转换到更接近执行方式的表示。
- optimization：在保持语义的前提下改变程序结构。
- scheduling：确定执行顺序、融合、循环和分块策略。
- code generation：把已确定的计算结构写成目标语言或机器可编译代码。
- runtime：加载产物、准备参数并启动执行。

同一个系统可以有多层 IR，也可以把 scheduling 和 codegen 紧密结合。不要把所有转换都叫
“codegen”，否则无法定位问题属于哪一阶段。

### 张量内存模型

对二维 tensor，元素地址可抽象为：

```text
offset(i, j) = storage_offset + i * stride[0] + j * stride[1]
```

`view`、`transpose` 和 broadcast 往往改变 shape/stride 或索引映射，不一定产生数据复制。
generated kernel 的 `tl.load(base + expression)` 正是在实现这类地址公式。

### 循环与并行模型

- iteration domain：输出元素对应的迭代空间。
- reduction domain：多个输入元素汇聚为较少输出元素的空间。
- tiling：把大迭代空间拆成 block/tile。
- vectorization：让一次指令处理多个 lane。
- fusion：让多个逻辑算子共享一个循环或 kernel，减少中间读写和 launch。
- mask：约束尾块、条件分支或不规则访问的有效 lane。

## 源码观察入口

这一阶段不要求通读实现，只需把概念映射到下面的代码对象：

- `<workspace>/pytorch/torch/_inductor/ir.py`：Inductor IR 和 layout 对象。
- `<workspace>/pytorch/torch/_inductor/scheduler.py`：dependency、fusion 和调度对象。
- `<workspace>/pytorch/torch/_inductor/codegen/common.py`：backend codegen 公共抽象。
- `<workspace>/pytorch/torch/_inductor/codegen/triton.py`：Triton kernel emitter。
- `<workspace>/pytorch/torch/_inductor/codegen/cpp.py`：CPU C++ loop codegen。

## 源码范围

本章只建立阅读 codegen 所需的编译器和张量程序模型，不追踪完整的 PyTorch 编译调用链。

- **纳入范围**：`torch/_inductor/ir.py` 中的表示、`scheduler.py` 中的调度概念、
  `codegen/common.py` 的公共接口，以及 `codegen/triton.py`、`codegen/cpp.py` 中能够直接体现
  loop、index、mask 和 store 的局部实现。
- **暂不深入**：Dynamo bytecode 捕获、AOTAutograd、具体 lowering 注册、设备 runtime、NPU 专用
  autotune 和 MLIR；这些内容分别在后续章节展开。
- **建议顺序**：先读张量 layout/index，再读 iteration/reduction domain，最后读 tiling、mask 和
  fusion 对生成循环的影响。

## 章节文档地图

- [Overview 与 Architecture](overview-and-architecture.md)
- [编译阶段与 IR 边界](compiler-pipeline-and-ir.md)
- [Tensor Layout、Stride 与 Index](tensor-layout-and-index.md)
- [Loop Domain、Tiling 与 Mask](loop-domain-tiling-and-mask.md)
- [Fusion 与并行执行模型](fusion-and-parallel-model.md)

## 练习

### 练习 1：手算 stride

创建 `(3, 4)` tensor，分别执行 transpose、slice 和 expand，记录每一步的 shape、stride 和
`storage_offset`。任选三个逻辑坐标，手算底层 offset，再与实际元素对应。

### 练习 2：写朴素循环

不用张量算子，用伪代码分别表达：

```text
y = relu(x + 1)
z = sum(x, dim=-1)
```

标注 iteration domain、reduction domain、load、compute 和 store。再尝试把两个连续 pointwise 操作
融合到一个循环。

### 练习 3：手工 tiling

将长度 1000 的循环按 block size 256 分块，写出 block offset、lane offset 和边界 mask。理解为什么
最后一个 block 不能无条件访问全部 256 个元素。

## 产出物

- 一页术语表，必须包含 IR、lowering、scheduler、tiling、fusion、layout、stride 和 mask。
- 一张连续与 transpose tensor 的地址映射表。
- pointwise 和 reduction 的朴素循环及 tiled 循环伪代码。

## 验收清单

- [ ] 能说明 lowering、scheduling、codegen、runtime 的边界。
- [ ] 能根据 shape/stride 写出二维 tensor 的线性地址公式。
- [ ] 能区分 iteration axis 与 reduction axis。
- [ ] 能解释 tail mask 为什么存在。
- [ ] 能说明 fusion 的收益以及寄存器压力、并行度下降等潜在代价。

## 常见误区

- contiguous 是具体 stride 关系，不等于“二维数据物理存成二维”。底层存储仍是线性空间。
- reshape 不保证零拷贝；能否成为 view 取决于原 layout。
- fusion 并非越多越好。资源占用、重计算和调度约束都可能让拆分更优。
- tiling 不只是性能参数。错误的 axis、mask 或 store 作用域会直接改变语义。

[返回总路线](../)
