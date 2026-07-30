---
title: CUDA 社区 v2.10.0 用例的 NPU 动态 Shape 看护优先级分析
---

# CUDA 社区 v2.10.0 用例的 NPU 动态 Shape 看护优先级分析

## 结论

本次共分析 105 个用例，其中 57 个建议作为 NPU 动态 shape 重要用例看护，48 个不建议进入核心看护集。

“重要”表示该用例覆盖 NPU `torch.compile` 动态 shape 的核心正确性、符号传播、数据依赖 shape、
前反向、常见模型算子或编译复用链路。它不表示社区原用例可以不经适配直接在 NPU 上运行。
用例属于算子、扩展或 codegen 专项，不作为降低重要性的理由；只要目标能力适用于 NPU 且覆盖上述核心风险，
就应进入核心看护集。

“不重要”仅表示不建议进入 NPU 动态 shape **核心**看护集，不代表用例本身没有价值。NPU 明确关闭的能力、
目标算子仅走 NPU fallback 且未额外覆盖核心符号传播、CUDA 专属、CPU/meta 专属、静态 shape、纯性能、
纯未特化浮点参数和过窄的内部回归用例，不进入核心看护集。

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
4. NPU 后端使用 `triton-ascend`。用户 Triton kernel、dynamic reduction 和 persistent reduction 等适用于
   NPU 的 codegen 核心能力，应适配 CUDA 专属断言后进入核心看护集；CUDA FlashAttention 和仅在 CPU/meta
   运行的路径仍不适用。
5. `pytorch_new/torch_npu/_inductor/lowering_fallback_list.py` 中的算子不会进入 NPU Inductor lowering 和融合。
   当用例主要验证此类算子本身时，不作为 NPU codegen 核心用例；`nonzero` 等用于产生数据依赖 shape、并继续
   验证 unbacked SymInt 传播的场景除外，因为该传播链路本身是 NPU 动态 shape 的核心能力。

## 逐项分析

完整表格单独存放在
[npu-dynamic-shape-v2.10-test-priority-analysis.csv](npu-dynamic-shape-v2.10-test-priority-analysis.csv)，
包含“文件名、用例、用例分类（NPU 视角）、用例功能、是否为重要用例”五列。

## 使用建议

建议先把 57 个“是”用例按以下顺序迁移或对齐到 NPU：

1. `nonzero`/`item`/`tolist` 产生的 unbacked SymInt 与 tensor factory、view、split、cat 的组合。
2. 动态 shape 的 MM/BMM/Linear/LayerNorm/reduction/einsum/softmax 等常见模型算子。
3. 动态 stride、前反向保存、运行时 `_check`、编译次数和跨 shape 复用。
4. combo kernel/autotune/sort 等依赖 NPU 后端能力的专项路径。

迁移验收不应只运行一个 example shape。至少准备两个不同输入 shape；对于 `nonzero` 等数据依赖输出，
还应在相同输入 shape 下改变有效元素数量，并检查 eager/compiled 结果、输出 size/stride 和编译次数。
