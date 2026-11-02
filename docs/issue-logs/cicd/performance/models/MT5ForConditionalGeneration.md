---
title: "CICD 性能分析：MT5ForConditionalGeneration"
---

# MT5ForConditionalGeneration：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=4；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 11.674852 | 10.388546 | 11.02% |
| GPU | 11.111035 | 8.337969 | 24.96% |

达到 GPU **整网**提升比时，NPU 整步目标为 8.761071 ms；当前相对该目标的差额为 1.627475 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/MT5ForConditionalGeneration_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/MT5ForConditionalGeneration_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/MT5ForConditionalGeneration_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/MT5ForConditionalGeneration_Infer/GPU_wrapper_calls.txt)。

### 1. log_softmax→NLLLoss 的完整 log probability 写回

**优化入口：NPU Inductor nll_loss_forward decomposition/lowering、producer-consumer fusion。** 外部 NLLLoss 消费 log_softmax 完整矩阵；GPU loss 融合 kernel 可只提取 target logit 和行归约统计。检查 NLLLoss 唯一消费者边界，避免最后一个完整词表循环及中间矩阵写回。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__log_softmax_16` | 1.0 | 1.093196 | 10.52% |
| `aclnnInplaceOne_OnesLikeAiCore_OnesLike` | 1.0 | 0.003322 | 0.03% |
| `aclnnNLLLoss_NLLLossAiCore_NLLLoss` | 1.0 | 0.009517 | 0.09% |

本项 kernel/区间合计 **1.106035 ms，占整步 10.65%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.845627 | 1.106035 | ≤-30.79% |
| GPU（Eager 下界 / compiled 上界） | 2.590313 | 0.752078 | ≥70.97% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **101.76 个百分点**。NPU 为最终 projection 后至 NLLLoss 的逐步精确窗口，包含 shift/clone/copy；GPU Eager 只计 log_softmax+NLLLoss 下界，compiled 包含 prepare 及全部 loss kernel。预算是保守下界。

NPU Eager 对账项：`aclnnInplaceOne_OnesLikeAiCore_OnesLike`, `aclnnLogSoftmax_LogSoftmaxAiCore_LogSoftmaxV2`, `aclnnNLLLoss_NLLLossAiCore_NLLLoss`。

GPU Eager 对账项：`_log_softmax`, `nll_loss_forward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__unsafe_view_prepare_softmax_online_view_18` | 1.0 | 0.669718 | 8.03% |
| `triton_per_fused__softmax__to_copy_abs_add_arange_div_embedding_full_like_gt_log_lt_minimum_mul_permute_prepare_softmax_online_slice_sub_unsqueeze_view_where_6` | 8.0 | 0.032187 | 0.39% |
| `triton_per_fused__softmax_add_prepare_softmax_online_view_17` | 7.0 | 0.025271 | 0.30% |
| `triton_per_fused__softmax_add_prepare_softmax_online_16` | 8.0 | 0.018247 | 0.22% |
| `triton_per_fused__softmax__to_copy_add_arange_div_embedding_expand_full_full_like_gt_log_lt_minimum_mul_neg_permute_prepare_softmax_online_slice_sub_triu_unsqueeze_view_where_zeros_like_3` | 1.0 | 0.004400 | 0.05% |
| `triton_per_fused__log_softmax__unsafe_view_nll_loss_forward_view_19` | 1.0 | 0.002256 | 0.03% |

**贡献粗估：0.860513 ms/step，相当于当前 NPU Inductor 整步耗时的 8.28%，使整网提升比增加 7.371 个百分点，单独达到该预算后整网提升比约为 18.39%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused__log_softmax_16`：NPU_g01_output_code.py:1693；输入 `f32[512, 250112][250112, 1]npu:0`；size_hints=`{'x0': 512, 'r1': 250112}`；split_axis=`[0]`。
- `aclnnInplaceOne_OnesLikeAiCore_OnesLike`：外部调用；见 wrapper 调用边界。
- `aclnnNLLLoss_NLLLossAiCore_NLLLoss`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused__safe_softmax__to_copy_add_ar_3` | 0.101655 / 8.0 | 0.98% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_poi_fused_add_matmul_mul_pow_tanh_8` | 0.088111 / 16.0 | 0.85% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_per_fused__safe_softmax__to_copy_abs_ad_6` | 0.086154 / 8.0 | 0.83% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |

- `triton_per_fused__safe_softmax__to_copy_add_ar_3`：NPU_g01_output_code.py:346；输入 `f32[24, 128, 128][16384, 128, 1]npu:0`；size_hints=`{'z0': 4, 'y1': 6, 'x2': 128, 'r3': 128}`；split_axis=`[0, 1, 2]`。GPU 检查入口：`triton_poi_fused__unsafe_view_add_mul_pow_tanh_8`, `triton_per_fused__softmax__to_copy_abs_add_arange_div_embedding_full_like_gt_log_lt_minimum_mul_permute_prepare_softmax_online_slice_sub_unsqueeze_view_where_6`, `triton_per_fused__softmax_add_prepare_softmax_online_view_17`。
- `triton_poi_fused_add_matmul_mul_pow_tanh_8`：NPU_g01_output_code.py:929；输入 `f32[512, 1024][1024, 1]npu:0`；size_hints=`{'z0': 4, 'y1': 128, 'x2': 1024}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused__unsafe_view_add_mul_pow_tanh_8`, `triton_per_fused__softmax__to_copy_abs_add_arange_div_embedding_full_like_gt_log_lt_minimum_mul_permute_prepare_softmax_online_slice_sub_unsqueeze_view_where_6`, `triton_per_fused__softmax_add_prepare_softmax_online_view_17`。
- `triton_per_fused__safe_softmax__to_copy_abs_ad_6`：NPU_g01_output_code.py:708；输入 `f32[24, 128, 128][16384, 128, 1]npu:0`；size_hints=`{'z0': 4, 'y1': 6, 'x2': 128, 'r3': 128}`；split_axis=`[0, 1, 2]`。GPU 检查入口：`triton_poi_fused__unsafe_view_add_mul_pow_tanh_8`, `triton_per_fused__softmax__to_copy_abs_add_arange_div_embedding_full_like_gt_log_lt_minimum_mul_permute_prepare_softmax_online_slice_sub_unsqueeze_view_where_6`, `triton_per_fused__softmax_add_prepare_softmax_online_view_17`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=4；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 43.414721 | 41.073613 | 5.39% |
| GPU | 72.035832 | 40.323586 | 44.02% |

达到 GPU **整网**提升比时，NPU 整步目标为 24.302312 ms；当前相对该目标的差额为 16.771301 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/MT5ForConditionalGeneration_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/MT5ForConditionalGeneration_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/MT5ForConditionalGeneration_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/MT5ForConditionalGeneration_Train/GPU_wrapper_calls.txt)。

### 1. NLLLossBackward→LogSoftmaxBackward 的完整中间矩阵

**优化入口：NPU Inductor nll_loss_backward / _log_softmax_backward_data decomposition 与 lowering。** 两条独立外部调用之间生成完整 dlogP；无类别权重时每个 token 的 dlogP 仅 target 类别非零。GPU 将 loss 导数融合生成，目标是直接计算 dlogits，避免完整中间矩阵；最终 dlogits 仍须生成。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnNLLLossBackward_NLLLossGradAiCore_NLLLossGrad` | 1.0 | 0.385892 | 0.94% |
| `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad` | 1.0 | 1.231228 | 3.00% |

本项 kernel/区间合计 **1.617120 ms，占整步 3.94%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 1.582735 | 1.617120 | ≤-2.17% |
| GPU（Eager 下界 / compiled 上界） | 3.023625 | 2.441591 | ≥19.25% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **21.42 个百分点**。GPU compiled 上界故意包含全部前向与反向 loss/prepare kernel。需要完整 GPU wrapper 进一步切开前后向后才能给更紧预算。

NPU Eager 对账项：`aclnnNLLLossBackward_NLLLossGradAiCore_NLLLossGrad`, `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad`。

GPU Eager 对账项：`_log_softmax_backward_data`, `nll_loss_backward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__log_softmax__log_softmax_backward_data_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_triu_view_0` | 1.0 | 1.640926 | 4.07% |
| `triton_red_fused__log_softmax_prepare_softmax_online_15` | 1.0 | 0.666440 | 1.65% |
| `triton_per_fused__softmax_add_native_dropout_prepare_softmax_online_view_14` | 7.0 | 0.043306 | 0.11% |
| `triton_per_fused__softmax_add_embedding_native_dropout_permute_prepare_softmax_online_slice_unsqueeze_view_9` | 7.0 | 0.038872 | 0.10% |
| `triton_per_fused__softmax_add_native_dropout_prepare_softmax_online_13` | 8.0 | 0.037057 | 0.09% |
| `triton_per_fused__softmax_add_arange_embedding_expand_full_gt_mul_native_dropout_permute_prepare_softmax_online_slice_triu_unsqueeze_view_12` | 1.0 | 0.006394 | 0.02% |
| `triton_per_fused__softmax_add_embedding_native_dropout_permute_prepare_softmax_online_slice_unsqueeze_view_4` | 1.0 | 0.006170 | 0.02% |
| `triton_per_fused__log_softmax_nll_loss_forward_triu_view_16` | 1.0 | 0.002426 | 0.01% |

**贡献粗估：0.339055 ms/step，相当于当前 NPU Inductor 整步耗时的 0.83%，使整网提升比增加 0.781 个百分点，单独达到该预算后整网提升比约为 6.17%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnNLLLossBackward_NLLLossGradAiCore_NLLLossGrad`：外部调用；见 wrapper 调用边界。
- `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad`：外部调用；见 wrapper 调用边界。

### 2. prims.fma metadata 重复执行第一组 Mul/Add

**优化入口：PyTorch Inductor inductor_prims.make_prim.meta 与 prims._prim_impl；NPU prims.fma fallback 暴露真实执行。** softmax backward 后连续 Mul/Add/Mul/Add 中第一对来自 metadata 执行真实 impl_aten，随后真实实现再次执行。第一对结果不参与最终输出；GPU 生成 kernel 内执行有效公式，没有 metadata 设备调用。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnMul_MulAiCore_Mul` | 24.0 | 0.112292 | 0.27% |
| `aclnnAdd_AddAiCore_Add` | 24.0 | 0.082377 | 0.20% |

本项 kernel/区间合计 **0.194670 ms，占整步 0.47%**。

仅计算这些相邻窗口中的第一对，不使用整网 Add/Mul 汇总。预算为已证实重复调用的当前设备耗时；修复后整网时间仍须实测。另一个有效 Mul/Add 对不是可全额删除的工作。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused__softmax_backward_data_add_embedding_dense_backward_native_dropout_backward_permute_slice_backward_squeeze_sum_view_18` | 1.0 | 0.088697 | 0.22% |
| `triton_per_fused__softmax_backward_data_native_dropout_backward_view_13` | 16.0 | 0.085036 | 0.21% |
| `triton_poi_fused__softmax_backward_data_add_embedding_dense_backward_native_dropout_backward_permute_slice_backward_squeeze_sum_view_22` | 1.0 | 0.045860 | 0.11% |
| `triton_per_fused__softmax__softmax_backward_data_add_native_dropout_backward_view_8` | 8.0 | 0.044676 | 0.11% |

**贡献粗估：0.194670 ms/step，相当于当前 NPU Inductor 整步耗时的 0.47%，使整网提升比增加 0.448 个百分点，单独达到该预算后整网提升比约为 5.84%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnMul_MulAiCore_Mul`：外部调用；见 wrapper 调用边界。
- `aclnnAdd_AddAiCore_Add`：外部调用；见 wrapper 调用边界。

历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_add_copy__0` | 2.118131 / 2.0 | 5.16% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_add_21` | 1.100618 / 1.0 | 2.68% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_red_fused__log_softmax_14` | 1.073017 / 1.0 | 2.61% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |

- `triton_poi_fused_add_copy__0`：NPU_g03_output_code.py:58；输入 `f32[250112, 512][512, 1]npu:0`；size_hints=`{'y0': 250112, 'x1': 512}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_23`, `triton_poi_fused__unsafe_view_add_mul_native_dropout_backward_pow_tanh_tanh_backward_view_4`, `triton_per_fused__unsafe_view_add_mean_mul_native_dropout_pow_rsqrt_8`。
- `triton_poi_fused_add_21`：NPU_g01_output_code.py:2231；输入 `f32[250112, 512][512, 1]npu:0`；size_hints=`{'y0': 250112, 'x1': 512}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_23`, `triton_poi_fused__unsafe_view_add_mul_native_dropout_backward_pow_tanh_tanh_backward_view_4`, `triton_per_fused__unsafe_view_add_mean_mul_native_dropout_pow_rsqrt_8`。
- `triton_red_fused__log_softmax_14`：NPU_g02_output_code.py:1641；输入 `f32[512, 250112][250112, 1]npu:0`；size_hints=`{'x0': 512, 'r1': 250112}`；split_axis=`[0]`。GPU 检查入口：`triton_red_fused__log_softmax__log_softmax_backward_data_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_triu_view_0`, `triton_red_fused__log_softmax_prepare_softmax_online_15`, `triton_per_fused__log_softmax_nll_loss_forward_triu_view_16`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
