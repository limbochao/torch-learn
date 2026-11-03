---
title: "CICD 性能分析：XGLMForCausalLM"
---

# XGLMForCausalLM：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=8；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 56.914148 | 57.083685 | -0.30% |
| GPU | 50.077310 | 49.242039 | 1.67% |

达到 GPU **整网**提升比时，NPU 整步目标为 55.964841 ms；当前相对该目标的差额为 1.118845 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/XGLMForCausalLM_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/XGLMForCausalLM_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/XGLMForCausalLM_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/XGLMForCausalLM_Infer/GPU_wrapper_calls.txt)。

### 1. log_softmax→NLLLoss 的完整 log probability 写回

**优化入口：NPU Inductor nll_loss_forward decomposition/lowering、producer-consumer fusion。** 外部 NLLLoss 消费 log_softmax 完整矩阵；GPU loss 融合 kernel 可只提取 target logit 和行归约统计。检查 NLLLoss 唯一消费者边界，避免最后一个完整词表循环及中间矩阵写回。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__log_softmax_10` | 1.0 | 1.821455 | 3.19% |
| `triton_poi_fused_copy_expand_new_zeros_scalar__11` | 1.0 | 0.003000 | 0.01% |
| `aclnnInplaceOne_OnesLikeAiCore_OnesLike` | 1.0 | 0.003272 | 0.01% |
| `aclnnNLLLoss_NLLLossAiCore_NLLLoss` | 1.0 | 0.010923 | 0.02% |

本项 kernel/区间合计 **1.838650 ms，占整步 3.22%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 1.653745 | 1.838650 | ≤-11.18% |
| GPU（Eager 下界 / compiled 上界） | 5.686962 | 1.660908 | ≥70.79% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **81.98 个百分点**。NPU 为最终 projection 后至 NLLLoss 的逐步精确窗口，包含 shift/clone/copy；GPU Eager 只计 log_softmax+NLLLoss 下界，compiled 包含 prepare 及全部 loss kernel。预算是保守下界。

NPU Eager 对账项：`aclnnInplaceFillScalar_FillAiCore_Fill`, `aclnnInplaceZero_ZerosLikeAiCore_ZerosLike`, `aclnnInplaceCopy_SliceAiCore_Slice`, `aclnnInplaceCopy_ViewCopyAiCore_ViewCopy`, `aclnnInplaceCopy_BroadcastToAiCore_BroadcastTo`, `aclnnLogSoftmax_LogSoftmaxAiCore_LogSoftmaxV2`, `aclnnInplaceOne_OnesLikeAiCore_OnesLike`, `aclnnNLLLoss_NLLLossAiCore_NLLLoss`。

GPU Eager 对账项：`_log_softmax`, `nll_loss_forward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__unsafe_view_prepare_softmax_online_view_10` | 1.0 | 1.491619 | 3.03% |
| `triton_per_fused__softmax_add_arange_expand_full_lift_fresh_lt_masked_fill_maximum_prepare_softmax_online_unsqueeze_view_3` | 24.0 | 0.165698 | 0.34% |
| `triton_per_fused__log_softmax__unsafe_view_clone_constant_pad_nd_nll_loss_forward_slice_view_11` | 1.0 | 0.003591 | 0.01% |

**贡献粗估：1.355665 ms/step，相当于当前 NPU Inductor 整步耗时的 2.37%，使整网提升比增加 2.382 个百分点，单独达到该预算后整网提升比约为 2.08%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused__log_softmax_10`：NPU_g01_output_code.py:948；输入 `f32[1024, 256008][256008, 1]npu:0`；size_hints=`{'x0': 1024, 'r1': 256008}`；split_axis=`[0]`。
- `triton_poi_fused_copy_expand_new_zeros_scalar__11`：NPU_g01_output_code.py:1038；输入 `i64[8, 128][128, 1]npu:0`；size_hints=`{'y0': 8, 'x1': 128}`；split_axis=`[0, 1]`。
- `aclnnInplaceOne_OnesLikeAiCore_OnesLike`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `aclnnGeluV2_GeluV2_GeluV2` | 0.645939 / 24.0 | 1.13% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_per_fused__softmax_add_arange_expand_fu_3` | 0.395061 / 24.0 | 0.69% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_poi_fused_addmm_6` | 0.330770 / 24.0 | 0.58% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `aclnnGeluV2_GeluV2_GeluV2`：外部调用；见 wrapper 调用边界。GPU 检查入口：需对照外部 ACLNN/CUDA 调用和 FX 图。
- `triton_per_fused__softmax_add_arange_expand_fu_3`：NPU_g01_output_code.py:346；输入 `f32[128, 128, 128][16384, 128, 1]npu:0`；size_hints=`{'y0': 128, 'x1': 128, 'r2': 128}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_per_fused_add_addmm_native_layer_norm_view_8`, `triton_per_fused__softmax_add_arange_expand_full_lift_fresh_lt_masked_fill_maximum_prepare_softmax_online_unsqueeze_view_3`, `triton_per_fused_add_addmm_native_layer_norm_view_7`。
- `triton_poi_fused_addmm_6`：NPU_g01_output_code.py:602；输入 `f32[4096][1]npu:0`；size_hints=`{'y0': 1024, 'x1': 4096}`；split_axis=`[0]`。GPU 检查入口：`triton_per_fused_add_addmm_native_layer_norm_view_8`, `triton_poi_fused_addmm_gelu_view_6`, `triton_poi_fused_addmm_clone_transpose_view_2`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=8；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 183.448519 | 193.356245 | -5.40% |
| GPU | 151.951925 | 150.024565 | 1.27% |

达到 GPU **整网**提升比时，NPU 整步目标为 181.121655 ms；当前相对该目标的差额为 12.234590 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/XGLMForCausalLM_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/XGLMForCausalLM_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/XGLMForCausalLM_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/XGLMForCausalLM_Train/GPU_wrapper_calls.txt)。

### 1. prims.fma metadata 重复执行第一组 Mul/Add

**优化入口：PyTorch Inductor inductor_prims.make_prim.meta 与 prims._prim_impl；NPU prims.fma fallback 暴露真实执行。** softmax backward 后连续 Mul/Add/Mul/Add 中第一对来自 metadata 执行真实 impl_aten，随后真实实现再次执行。第一对结果不参与最终输出；GPU 生成 kernel 内执行有效公式，没有 metadata 设备调用。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnMul_MulAiCore_Mul` | 24.0 | 0.210090 | 0.11% |
| `aclnnAdd_AddAiCore_Add` | 24.0 | 0.165798 | 0.09% |

本项 kernel/区间合计 **0.375888 ms，占整步 0.19%**。

仅计算这些相邻窗口中的第一对，不使用整网 Add/Mul 汇总。预算为已证实重复调用的当前设备耗时；修复后整网时间仍须实测。另一个有效 Mul/Add 对不是可全额删除的工作。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused__softmax__softmax_backward_data_add_lift_fresh_maximum_native_dropout_backward_view_8` | 24.0 | 0.583081 | 0.39% |
| `triton_poi_fused__softmax__softmax_backward_data_add_div_eq_lift_fresh_lt_masked_fill_maximum_view_where_9` | 24.0 | 0.285164 | 0.19% |

**贡献粗估：0.375888 ms/step，相当于当前 NPU Inductor 整步耗时的 0.19%，使整网提升比增加 0.205 个百分点，单独达到该预算后整网提升比约为 -5.20%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnMul_MulAiCore_Mul`：外部调用；见 wrapper 调用边界。
- `aclnnAdd_AddAiCore_Add`：外部调用；见 wrapper 调用边界。

历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad` | 2.448884 / 1.0 | 1.27% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_add_view_0` | 2.151486 / 1.0 | 1.11% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_add_copy__0` | 2.142647 / 1.0 | 1.11% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad`：外部调用；见 wrapper 调用边界。GPU 检查入口：需对照外部 ACLNN/CUDA 调用和 FX 图。
- `triton_poi_fused_add_view_0`：NPU_g05_output_code.py:58；输入 `f32[8, 128, 256008][32769024, 256008, 1]npu:0`；size_hints=`{'z0': 8, 'y1': 128, 'x2': 256008}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_red_fused__log_softmax__log_softmax_backward_data_add_arange_clone_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_view_0`, `triton_per_fused_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_4`, `triton_per_fused__softmax__softmax_backward_data_add_lift_fresh_maximum_native_dropout_backward_view_8`。
- `triton_poi_fused_add_copy__0`：NPU_g07_output_code.py:58；输入 `f32[256008, 1024][1024, 1]npu:0`；size_hints=`{'y0': 256008, 'x1': 1024}`；split_axis=`[0]`。GPU 检查入口：`triton_red_fused__log_softmax__log_softmax_backward_data_add_arange_clone_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_view_0`, `triton_per_fused_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_4`, `triton_per_fused__softmax__softmax_backward_data_add_lift_fresh_maximum_native_dropout_backward_view_8`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
