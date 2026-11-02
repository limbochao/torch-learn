---
title: "CICD 性能分析：GPT2ForSequenceClassification"
---

# GPT2ForSequenceClassification：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=4；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 55.379151 | 41.809755 | 24.50% |
| GPU | 45.468808 | 34.914074 | 23.21% |

达到 GPU **整网**提升比时，NPU 整步目标为 42.523916 ms；当前相对该目标的差额为 -0.714160 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/GPT2ForSequenceClassification_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/GPT2ForSequenceClassification_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/GPT2ForSequenceClassification_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/GPT2ForSequenceClassification_Infer/GPU_wrapper_calls.txt)。

### 1. log_softmax→NLLLoss 的完整 log probability 写回

**优化入口：NPU Inductor nll_loss_forward decomposition/lowering、producer-consumer fusion。** 外部 NLLLoss 消费 log_softmax 完整矩阵；GPU loss 融合 kernel 可只提取 target logit 和行归约统计。检查 NLLLoss 唯一消费者边界，避免最后一个完整词表循环及中间矩阵写回。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused__to_copy_argmax_eq_8` | 1.0 | 0.016257 | 0.04% |
| `triton_poi_fused_arange_index_matmul_sub_9` | 1.0 | 0.002259 | 0.01% |
| `triton_poi_fused__log_softmax_10` | 1.0 | 0.002760 | 0.01% |
| `aclnnInplaceOne_OnesLikeAiCore_OnesLike` | 1.0 | 0.002214 | 0.01% |
| `aclnnNLLLoss_NLLLossAiCore_NLLLoss` | 1.0 | 0.006068 | 0.01% |

本项 kernel/区间合计 **0.029558 ms，占整步 0.07%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.024077 | 0.029558 | ≤-22.77% |
| GPU（Eager 下界 / compiled 上界） | 0.002896 | 0.001568 | ≥45.85% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **68.62 个百分点**。NPU 为最终 projection 后至 NLLLoss 的逐步精确窗口，包含 shift/clone/copy；GPU Eager 只计 log_softmax+NLLLoss 下界，compiled 包含 prepare 及全部 loss kernel。预算是保守下界。

NPU Eager 对账项：`aclnnIndex_IndexAiCore_Index`, `aclnnEqScalar_EqualAiCore_Equal`, `aclnnArange_ArangeAiCore_Range`, `aclnnInplaceCopy_CastAiCore_Cast`, `aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2`, `aclnnSubs_SubAiCore_Sub`, `aclnnLogSoftmax_LogSoftmaxAiCore_LogSoftmaxV2`, `aclnnInplaceOne_OnesLikeAiCore_OnesLike`, `aclnnNLLLoss_NLLLossAiCore_NLLLoss`。

GPU Eager 对账项：`nll_loss_forward`, `_log_softmax`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused__log_softmax_nll_loss_forward_prepare_softmax_online_13` | 1.0 | 0.001568 | 0.00% |

**贡献粗估：0.016521 ms/step，相当于当前 NPU Inductor 整步耗时的 0.04%，使整网提升比增加 0.030 个百分点，单独达到该预算后整网提升比约为 24.53%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_per_fused__to_copy_argmax_eq_8`：NPU_g01_output_code.py:761；输入 `i64[4, 1024][1024, 1]npu:0`；size_hints=`{'x0': 4, 'r1': 1024}`；split_axis=`[0]`。
- `triton_poi_fused_arange_index_matmul_sub_9`：NPU_g01_output_code.py:827；输入 `i64[4][1]npu:0`；size_hints=`{'y0': 4, 'x1': 2}`；split_axis=`[0, 1]`。
- `triton_poi_fused__log_softmax_10`：NPU_g01_output_code.py:892；输入 `f32[4, 2][2, 1]npu:0`；size_hints=`{'y0': 4, 'x1': 2}`；split_axis=`[0, 1]`。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_add_addmm_mul_pow_tanh_view_4` | 0.649131 / 12.0 | 1.55% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_per_fused_add_addmm_native_layer_norm_v_6` | 0.398791 / 11.0 | 0.95% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_per_fused_add_addmm_native_layer_norm_v_5` | 0.296294 / 11.0 | 0.71% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |

- `triton_poi_fused_add_addmm_mul_pow_tanh_view_4`：NPU_g01_output_code.py:405；输入 `f32[3072][1]npu:0`；size_hints=`{'y0': 4096, 'x1': 3072}`；split_axis=`[0]`。GPU 检查入口：`triton_per_fused_add_addmm_native_layer_norm_view_8`, `triton_poi_fused_add_addmm_mul_pow_tanh_view_6`, `triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_4`。
- `triton_per_fused_add_addmm_native_layer_norm_v_6`：NPU_g01_output_code.py:583；输入 `f32[768][1]npu:0`；size_hints=`{'y0': 4, 'x1': 1024, 'r2': 768}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_per_fused_add_addmm_native_layer_norm_view_8`, `triton_poi_fused_add_addmm_mul_pow_tanh_view_6`, `triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_4`。
- `triton_per_fused_add_addmm_native_layer_norm_v_5`：NPU_g01_output_code.py:484；输入 `f32[4, 1024, 768][786432, 768, 1]npu:0`；size_hints=`{'y0': 4, 'x1': 1024, 'r2': 768}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_per_fused_add_addmm_native_layer_norm_view_8`, `triton_poi_fused_add_addmm_mul_pow_tanh_view_6`, `triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_4`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=4；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 167.969723 | 191.515054 | -14.02% |
| GPU | 165.215314 | 129.168405 | 21.82% |

达到 GPU **整网**提升比时，NPU 整步目标为 131.321854 ms；当前相对该目标的差额为 60.193200 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/GPT2ForSequenceClassification_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/GPT2ForSequenceClassification_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/GPT2ForSequenceClassification_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/GPT2ForSequenceClassification_Train/GPU_wrapper_calls.txt)。

### 1. prims.fma metadata 重复执行第一组 Mul/Add

**优化入口：PyTorch Inductor inductor_prims.make_prim.meta 与 prims._prim_impl；NPU prims.fma fallback 暴露真实执行。** softmax backward 后连续 Mul/Add/Mul/Add 中第一对来自 metadata 执行真实 impl_aten，随后真实实现再次执行。第一对结果不参与最终输出；GPU 生成 kernel 内执行有效公式，没有 metadata 设备调用。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnMul_MulAiCore_Mul` | 12.0 | 3.548282 | 1.85% |
| `aclnnAdd_AddAiCore_Add` | 12.0 | 5.139258 | 2.68% |

本项 kernel/区间合计 **8.687540 ms，占整步 4.54%**。

仅计算这些相邻窗口中的第一对，不使用整网 Add/Mul 汇总。预算为已证实重复调用的当前设备耗时；修复后整网时间仍须实测。另一个有效 Mul/Add 对不是可全额删除的工作。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `_efficient_attention_backward` | 12.0 | 39.629716 | 30.68% |

**贡献粗估：8.687540 ms/step，相当于当前 NPU Inductor 整步耗时的 4.54%，使整网提升比增加 5.172 个百分点，单独达到该预算后整网提升比约为 -8.85%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnMul_MulAiCore_Mul`：外部调用；见 wrapper 调用边界。
- `aclnnAdd_AddAiCore_Add`：外部调用；见 wrapper 调用边界。

历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。

### 2. 独立 sum 生成归约

**优化入口：NPU Inductor codegen/reduction tiling、runtime autotune。** 选中 kernel 仅有 sum 和 view/transpose。检查输入 stride、归约维大小、split_axis、persistent/分阶段选择，与 GPU 归约拆分对照。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused_sum_view_14` | 12.0 | 7.322639 | 3.82% |

本项 kernel/区间合计 **7.322639 ms，占整步 3.82%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.633947 | 7.322639 | ≤-1055.09% |
| GPU（Eager 下界 / compiled 上界） | 1.006724 | 0.705599 | ≥29.91% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **1085.00 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnReduceSum_ReduceSumOpAiCore_ReduceSum`。

GPU Eager 对账项：`sum`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_add_mul_pow_sum_tanh_tanh_backward_view_8` | 12.0 | 0.183061 | 0.14% |
| `sum` | 12.0 | 0.182709 | 0.14% |
| `triton_red_fused_cat_sum_transpose_view_12` | 12.0 | 0.140806 | 0.11% |
| `triton_red_fused_native_dropout_backward_sum_view_6` | 24.0 | 0.123470 | 0.10% |
| `triton_per_fused_add_mul_pow_sum_tanh_tanh_backward_view_9` | 12.0 | 0.028509 | 0.02% |
| `triton_per_fused_cat_sum_transpose_view_13` | 12.0 | 0.028366 | 0.02% |
| `triton_poi_fused_embedding_dense_backward_sum_16` | 1.0 | 0.016704 | 0.01% |
| `triton_poi_fused_embedding_dense_backward_sum_2` | 1.0 | 0.001975 | 0.00% |

**贡献粗估：6.878314 ms/step，相当于当前 NPU Inductor 整步耗时的 3.59%，使整网提升比增加 4.095 个百分点，单独达到该预算后整网提升比约为 -9.92%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_per_fused_sum_view_14`：NPU_g01_output_code.py:1274；输入 `f32[4, 1024, 2304][2359296, 2304, 1]npu:0`；size_hints=`{'x0': 2304, 'r1': 4096}`；split_axis=`[0]`。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused__softmax__to_copy_full_native_10` | 9.809914 / 11.0 | 5.12% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_per_fused_add_mul_pow_sum_tanh_tanh_bac_6` | 9.787516 / 12.0 | 5.11% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_per_fused__softmax_backward_data_matmul_9` | 6.519964 / 11.0 | 3.40% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |

- `triton_per_fused__softmax__to_copy_full_native_10`：NPU_g02_output_code.py:942；输入 `b8[1, 1, 1024, 1024][1048576, 1048576, 1024, 1]npu:0`；size_hints=`{'z0': 4, 'y1': 12, 'x2': 1024, 'r3': 1024}`；split_axis=`[0, 1, 2]`。GPU 检查入口：`triton_per_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_view_9`, `triton_per_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_view_10`, `triton_per_fused_add_arange_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_unsqueeze_5`。
- `triton_per_fused_add_mul_pow_sum_tanh_tanh_bac_6`：NPU_g01_output_code.py:613；输入 `f32[4, 1024, 3072][3145728, 3072, 1]npu:0`；size_hints=`{'x0': 3072, 'r1': 4096}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_mul_pow_tanh_tanh_backward_view_7`, `triton_poi_fused_add_mul_pow_tanh_view_8`, `triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_view_10`。
- `triton_per_fused__softmax_backward_data_matmul_9`：NPU_g01_output_code.py:873；输入 `f32[48, 1024, 1024][1048576, 1024, 1]npu:0`；size_hints=`{'z0': 4, 'y1': 12, 'x2': 1024, 'r3': 1024}`；split_axis=`[0, 1, 2]`。GPU 检查入口：`triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_view_10`, `triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_view_15`, `triton_red_fused_native_dropout_backward_sum_view_6`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
