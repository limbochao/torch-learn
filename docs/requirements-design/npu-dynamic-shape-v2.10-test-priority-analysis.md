---
title: CUDA 社区 v2.10.0 用例的 NPU 动态 Shape 看护优先级分析
---

# CUDA 社区 v2.10.0 用例的 NPU 动态 Shape 看护优先级分析

## 结论

本次共分析 105 个用例，其中 56 个建议作为 NPU 动态 shape 重要用例看护，49 个不建议进入核心看护集。

“重要”表示该用例覆盖 NPU `torch.compile` 动态 shape 的核心正确性、符号传播、数据依赖 shape、
前反向、常见模型算子或编译复用链路。它不表示社区原用例可以不经适配直接在 NPU 上运行。

“不重要”仅表示不建议进入 NPU 动态 shape **核心**看护集，不代表用例本身没有价值。CUDA/Triton 专属、
CPU/meta 专属、静态 shape、纯性能、纯未特化浮点参数和过窄的内部回归用例，应由对应专项测试看护。

## 判定依据

1. 分析源码为 CUDA 社区 `v2.10.0`，提交 `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`：
   `test/inductor/test_padding.py`、`test/inductor/test_unbacked_symints.py` 和
   `test/inductor/test_torchinductor_dynamic_shapes.py`。
2. NPU 现有动态 shape 测试明确把跨 shape 编译复用、动态 tiling、间接访存、view/reshape/cat、
   `nonzero` 数据依赖输出和 unbacked SymInt 运行时约束列为核心场景，见
   `pytorch_new/test/_inductor/test_inductor_dynamic_shapes.py:1-18,63-85,795-803`。
3. `pytorch_new/torch_npu/_inductor/__init__.py:286` 和
   `pytorch_new/torch_npu/_inductor/dvm/mlir_fusion.py:455` 显式关闭 `comprehensive_padding`。
   因此 `test_padding.py` 中即使名称带 dynamic，其目标仍是 CUDA/Triton padding layout 策略，
   不作为当前 NPU 动态 shape 核心用例。
4. 直接调用社区 Triton kernel、检查 Triton 源码字符串、依赖 CUDA SM80/FlashAttention、仅在 CPU/meta
   运行的用例，不具备 NPU 核心看护价值；其中可迁移的算子语义应改写为 NPU 后端断言后再进入专项集。

## 逐项分析

完整表格单独存放在
[npu-dynamic-shape-v2.10-test-priority-analysis.csv](npu-dynamic-shape-v2.10-test-priority-analysis.csv)，
包含“文件名、用例、用例分类（NPU 视角）、用例功能、是否为重要用例”五列。

## 使用建议

建议先把 56 个“是”用例按以下顺序迁移或对齐到 NPU：

1. `nonzero`/`item`/`tolist` 产生的 unbacked SymInt 与 tensor factory、view、split、cat 的组合。
2. 动态 shape 的 MM/BMM/Linear/LayerNorm/reduction/einsum/softmax 等常见模型算子。
3. 动态 stride、前反向保存、运行时 `_check`、编译次数和跨 shape 复用。
4. combo kernel/autotune/sort 等依赖 NPU 后端能力的专项路径。

迁移验收不应只运行一个 example shape。至少准备两个不同输入 shape；对于 `nonzero` 等数据依赖输出，
还应在相同输入 shape 下改变有效元素数量，并检查 eager/compiled 结果、输出 size/stride 和编译次数。
