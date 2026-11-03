---
title: "CICD 性能分析：OPTForCausalLM"
---

# OPTForCausalLM：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=2；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 92.748133 | 69.839784 | 24.70% |
| GPU | 54.743527 | 64.170959 | -17.22% |

达到 GPU **整网**提升比时，NPU 整步目标为 108.720374 ms；当前相对该目标的差额为 -38.880590 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/OPTForCausalLM_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/OPTForCausalLM_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/OPTForCausalLM_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/OPTForCausalLM_Infer/GPU_wrapper_calls.txt)。

### 1. log_softmax→NLLLoss 的完整 log probability 写回

**优化入口：NPU Inductor nll_loss_forward decomposition/lowering、producer-consumer fusion。** 外部 NLLLoss 消费 log_softmax 完整矩阵；GPU loss 融合 kernel 可只提取 target logit 和行归约统计。检查 NLLLoss 唯一消费者边界，避免最后一个完整词表循环及中间矩阵写回。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__log_softmax_clone_matmul_sli_13` | 1.0 | 1.272026 | 1.82% |
| `triton_poi_fused_clone_slice_14` | 1.0 | 0.002442 | 0.00% |
| `aclnnInplaceOne_OnesLikeAiCore_OnesLike` | 1.0 | 0.003351 | 0.00% |
| `aclnnNLLLoss_NLLLossAiCore_NLLLoss` | 1.0 | 0.013173 | 0.02% |

本项 kernel/区间合计 **1.290992 ms，占整步 1.85%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 2.379939 | 1.290992 | ≤45.76% |
| GPU（Eager 下界 / compiled 上界） | 2.447325 | 1.130406 | ≥53.81% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **8.06 个百分点**。NPU 为最终 projection 后至 NLLLoss 的逐步精确窗口，包含 shift/clone/copy；GPU Eager 只计 log_softmax+NLLLoss 下界，compiled 包含 prepare 及全部 loss kernel。预算是保守下界。

NPU Eager 对账项：`aclnnInplaceOne_OnesLikeAiCore_OnesLike`, `aclnnInplaceCopy_SliceAiCore_Slice`, `aclnnLogSoftmax_LogSoftmaxAiCore_LogSoftmaxV2`, `aclnnNLLLoss_NLLLossAiCore_NLLLoss`。

GPU Eager 对账项：`_log_softmax`, `nll_loss_forward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__unsafe_view_prepare_softmax_online_view_8` | 1.0 | 1.124281 | 1.75% |
| `triton_red_fused__log_softmax__unsafe_view_clone_constant_pad_nd_nll_loss_forward_slice_view_9` | 1.0 | 0.006125 | 0.01% |

**贡献粗估：0.191712 ms/step，相当于当前 NPU Inductor 整步耗时的 0.27%，使整网提升比增加 0.207 个百分点，单独达到该预算后整网提升比约为 24.91%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused__log_softmax_clone_matmul_sli_13`：NPU_g01_output_code.py:1334；输入 `f32[4096, 50272][50272, 1]npu:0`；size_hints=`{'x2': 2, 'x3': 2047, 'r1': 50272}`；split_axis=`[0, 1]`。
- `triton_poi_fused_clone_slice_14`：NPU_g01_output_code.py:1418；输入 `i64[2, 2048][2048, 1]npu:0`；size_hints=`{'y0': 2, 'x1': 2047}`；split_axis=`[0, 1]`。
- `aclnnInplaceOne_OnesLikeAiCore_OnesLike`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused__softmax__to_copy_add_arange__4` | 6.589630 / 12.0 | 9.44% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_poi_fused_addmm_relu_8` | 0.481705 / 12.0 | 0.69% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_add_addmm_embedding_mul_sub_v_11` | 0.307379 / 10.0 | 0.44% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_per_fused__softmax__to_copy_add_arange__4`：NPU_g01_output_code.py:394；输入 `f32[24, 2048, 2048][4194304, 2048, 1]npu:0`；size_hints=`{'y0': 24, 'x1': 2048, 'r2': 2048}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_red_fused__scaled_dot_product_efficient_attention_add_addmm_all_arange_bitwise_not_clone_eq_expand_full_gt_masked_fill_mul_ones_transpose_triu_unsqueeze_view_1`, `triton_per_fused__to_copy_add_addmm_cumsum_embedding_mul_native_layer_norm_ones_sub_view_6`, `triton_poi_fused__scaled_dot_product_efficient_attention_add_addmm_all_arange_bitwise_not_clone_eq_expand_full_gt_masked_fill_mul_ones_transpose_triu_unsqueeze_view_2`。
- `triton_poi_fused_addmm_relu_8`：NPU_g01_output_code.py:718；输入 `f32[3072][1]npu:0`；size_hints=`{'y0': 4096, 'x1': 3072}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_relu_4`, `triton_red_fused__scaled_dot_product_efficient_attention_add_addmm_all_arange_bitwise_not_clone_eq_expand_full_gt_masked_fill_mul_ones_transpose_triu_unsqueeze_view_1`, `triton_per_fused__to_copy_add_addmm_cumsum_embedding_mul_native_layer_norm_ones_sub_view_6`。
- `triton_poi_fused_add_addmm_embedding_mul_sub_v_11`：NPU_g01_output_code.py:1020；输入 `f32[2, 2048, 768][1572864, 768, 1]npu:0`；size_hints=`{'y0': 4096, 'x1': 768}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_relu_4`, `triton_red_fused__scaled_dot_product_efficient_attention_add_addmm_all_arange_bitwise_not_clone_eq_expand_full_gt_masked_fill_mul_ones_transpose_triu_unsqueeze_view_1`, `triton_per_fused__to_copy_add_addmm_cumsum_embedding_mul_native_layer_norm_ones_sub_view_6`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=2；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 258.887296 | 301.438821 | -16.44% |
| GPU | 163.739390 | 206.727004 | -26.25% |

达到 GPU **整网**提升比时，NPU 整步目标为 326.854735 ms；当前相对该目标的差额为 -25.415914 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/OPTForCausalLM_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/OPTForCausalLM_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/OPTForCausalLM_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/OPTForCausalLM_Train/GPU_wrapper_calls.txt)。

### 1. ReLU backward 与 bias 梯度归约

**优化入口：NPU Inductor scheduler 融合决策、codegen/reduction tiling。** 具体检查 threshold_backward 输出写回与沿 batch/edge 维 sum 的联合调度；GPU 将 pointwise 与 reduction 分拆或采用不同归约 kernel。重点是归约 kernel 的并行度和多阶段策略。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused_sum_threshold_backward_1` | 12.0 | 18.747805 | 6.22% |

本项 kernel/区间合计 **18.747805 ms，占整步 6.22%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 1.701854 | 18.747805 | ≤-1001.61% |
| GPU（Eager 下界 / compiled 上界） | 3.343242 | 2.502012 | ≥25.16% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **1026.77 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnReduceSum_ReduceSumOpAiCore_ReduceSum`, `aclnnThresholdBackward_ReluGrad_ReluGrad`。

GPU Eager 对账项：`threshold_backward`, `sum`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_threshold_backward_3` | 12.0 | 1.516183 | 0.73% |
| `triton_red_fused_sum_1` | 60.0 | 0.310223 | 0.15% |
| `triton_per_fused_sum_2` | 108.0 | 0.264696 | 0.13% |
| `sum` | 12.0 | 0.198203 | 0.10% |
| `triton_red_fused_sum_4` | 12.0 | 0.183557 | 0.09% |
| `triton_per_fused_sum_5` | 12.0 | 0.029150 | 0.01% |

**贡献粗估：17.474172 ms/step，相当于当前 NPU Inductor 整步耗时的 5.80%，使整网提升比增加 6.750 个百分点，单独达到该预算后整网提升比约为 -9.69%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_per_fused_sum_threshold_backward_1`：NPU_g04_output_code.py:132；输入 `f32[4096, 3072][3072, 1]npu:0`；size_hints=`{'x0': 3072, 'r1': 4096}`；split_axis=`[0]`。

### 2. prims.fma metadata 重复执行第一组 Mul/Add

**优化入口：PyTorch Inductor inductor_prims.make_prim.meta 与 prims._prim_impl；NPU prims.fma fallback 暴露真实执行。** softmax backward 后连续 Mul/Add/Mul/Add 中第一对来自 metadata 执行真实 impl_aten，随后真实实现再次执行。第一对结果不参与最终输出；GPU 生成 kernel 内执行有效公式，没有 metadata 设备调用。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnMul_MulAiCore_Mul` | 12.0 | 6.876388 | 2.28% |
| `aclnnAdd_AddAiCore_Add` | 12.0 | 10.101049 | 3.35% |

本项 kernel/区间合计 **16.977438 ms，占整步 5.63%**。

仅计算这些相邻窗口中的第一对，不使用整网 Add/Mul 汇总。预算为已证实重复调用的当前设备耗时；修复后整网时间仍须实测。另一个有效 Mul/Add 对不是可全额删除的工作。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `_efficient_attention_backward` | 12.0 | 73.902843 | 35.75% |

**贡献粗估：16.977438 ms/step，相当于当前 NPU Inductor 整步耗时的 5.63%，使整网提升比增加 6.558 个百分点，单独达到该预算后整网提升比约为 -9.88%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnMul_MulAiCore_Mul`：外部调用；见 wrapper 调用边界。
- `aclnnAdd_AddAiCore_Add`：外部调用；见 wrapper 调用边界。

历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused__softmax_backward_data_7` | 13.801833 / 12.0 | 4.58% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_poi_fused_div_masked_fill_threshold_bac_8` | 8.007771 / 12.0 | 2.66% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_per_fused__softmax_add_lift_fresh_maxim_4` | 7.793883 / 12.0 | 2.59% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |

- `triton_per_fused__softmax_backward_data_7`：NPU_g04_output_code.py:616；输入 `f32[24, 2048, 2048][4194304, 2048, 1]npu:0`；size_hints=`{'y0': 24, 'x1': 2048, 'r2': 2048}`；split_axis=`[0, 1]`。GPU 检查入口：需对照外部 ACLNN/CUDA 调用和 FX 图。
- `triton_poi_fused_div_masked_fill_threshold_bac_8`：NPU_g04_output_code.py:690；输入 `b8[2, 12, 2048, 2048][50331648, 4194304, 2048, 1]npu:0`；size_hints=`{'t0': 2, 'z1': 12, 'y2': 2048, 'x3': 2048}`；split_axis=`[0, 1, 2]`。GPU 检查入口：`triton_poi_fused_threshold_backward_3`, `triton_red_fused_add_all_arange_bitwise_not_clone_eq_expand_full_gt_masked_fill_mul_triu_unsqueeze_view_0`。
- `triton_per_fused__softmax_add_lift_fresh_maxim_4`：NPU_g05_output_code.py:385；输入 `f32[24, 2048, 2048][4194304, 2048, 1]npu:0`；size_hints=`{'y3': 2, 'y4': 12, 'x1': 2048, 'r2': 2048}`；split_axis=`[0, 1, 2]`。GPU 检查入口：`triton_red_fused__log_softmax__log_softmax_backward_data_add_arange_clone_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_view_0`, `triton_per_fused_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_6`, `triton_poi_fused_add_addmm_native_dropout_view_4`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
