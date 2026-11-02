---
title: "CICD 性能分析：DistillGPT2"
---

# DistillGPT2：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=16；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 79.519549 | 65.780203 | 17.28% |
| GPU | 69.530430 | 57.469323 | 17.35% |

达到 GPU **整网**提升比时，NPU 整步目标为 65.725678 ms；当前相对该目标的差额为 0.054525 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/DistillGPT2_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/DistillGPT2_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/DistillGPT2_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/DistillGPT2_Infer/GPU_wrapper_calls.txt)。

### 1. log_softmax→NLLLoss 的完整 log probability 写回

**优化入口：NPU Inductor nll_loss_forward decomposition/lowering、producer-consumer fusion。** 外部 NLLLoss 消费 log_softmax 完整矩阵；GPU loss 融合 kernel 可只提取 target logit 和行归约统计。检查 NLLLoss 唯一消费者边界，避免最后一个完整词表循环及中间矩阵写回。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__log_softmax_clone_matmul_sli_8` | 1.0 | 2.718616 | 4.13% |
| `triton_poi_fused_clone_slice_9` | 1.0 | 0.002852 | 0.00% |
| `aclnnInplaceOne_OnesLikeAiCore_OnesLike` | 1.0 | 0.003209 | 0.00% |
| `aclnnNLLLoss_NLLLossAiCore_NLLLoss` | 1.0 | 0.016825 | 0.03% |

本项 kernel/区间合计 **2.741501 ms，占整步 4.17%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 4.626172 | 2.741501 | ≤40.74% |
| GPU（Eager 下界 / compiled 上界） | 5.057800 | 2.497971 | ≥50.61% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **9.87 个百分点**。NPU 为最终 projection 后至 NLLLoss 的逐步精确窗口，包含 shift/clone/copy；GPU Eager 只计 log_softmax+NLLLoss 下界，compiled 包含 prepare 及全部 loss kernel。预算是保守下界。

NPU Eager 对账项：`aclnnInplaceCopy_SliceAiCore_Slice`, `aclnnLogSoftmax_LogSoftmaxAiCore_LogSoftmaxV2`, `aclnnInplaceOne_OnesLikeAiCore_OnesLike`, `aclnnNLLLoss_NLLLossAiCore_NLLLoss`。

GPU Eager 对账项：`_log_softmax`, `nll_loss_forward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_prepare_softmax_online_view_0` | 1.0 | 2.486931 | 4.33% |
| `triton_red_fused__log_softmax_clone_constant_pad_nd_nll_loss_forward_slice_view_1` | 1.0 | 0.011040 | 0.02% |

**贡献粗估：0.456705 ms/step，相当于当前 NPU Inductor 整步耗时的 0.69%，使整网提升比增加 0.574 个百分点，单独达到该预算后整网提升比约为 17.85%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused__log_softmax_clone_matmul_sli_8`：NPU_g01_output_code.py:775；输入 `f32[8192, 50257][50257, 1]npu:0`；size_hints=`{'x2': 16, 'x3': 511, 'r1': 50257}`；split_axis=`[0, 1]`。
- `triton_poi_fused_clone_slice_9`：NPU_g01_output_code.py:859；输入 `i64[16, 512][512, 1]npu:0`；size_hints=`{'y0': 16, 'x1': 511}`；split_axis=`[0, 1]`。
- `aclnnInplaceOne_OnesLikeAiCore_OnesLike`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_add_addmm_mul_pow_tanh_view_4` | 0.625997 / 6.0 | 0.95% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_per_fused_add_addmm_native_layer_norm_v_6` | 0.305027 / 5.0 | 0.46% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_per_fused_add_addmm_native_layer_norm_v_5` | 0.261551 / 5.0 | 0.40% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |

- `triton_poi_fused_add_addmm_mul_pow_tanh_view_4`：NPU_g01_output_code.py:408；输入 `f32[3072][1]npu:0`；size_hints=`{'y0': 8192, 'x1': 3072}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_addmm_mul_pow_tanh_view_6`, `triton_per_fused_add_addmm_native_layer_norm_view_8`, `triton_per_fused_add_addmm_native_layer_norm_view_7`。
- `triton_per_fused_add_addmm_native_layer_norm_v_6`：NPU_g01_output_code.py:586；输入 `f32[768][1]npu:0`；size_hints=`{'y0': 16, 'x1': 512, 'r2': 768}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_add_addmm_mul_pow_tanh_view_6`, `triton_per_fused_add_addmm_native_layer_norm_view_8`, `triton_per_fused_add_addmm_native_layer_norm_view_7`。
- `triton_per_fused_add_addmm_native_layer_norm_v_5`：NPU_g01_output_code.py:487；输入 `f32[16, 512, 768][393216, 768, 1]npu:0`；size_hints=`{'y0': 16, 'x1': 512, 'r2': 768}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_add_addmm_mul_pow_tanh_view_6`, `triton_per_fused_add_addmm_native_layer_norm_view_8`, `triton_per_fused_add_addmm_native_layer_norm_view_7`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=16；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 237.305787 | 227.562242 | 4.11% |
| GPU | 212.564318 | 165.649050 | 22.07% |

达到 GPU **整网**提升比时，NPU 整步目标为 184.929806 ms；当前相对该目标的差额为 42.632436 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/DistillGPT2_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/DistillGPT2_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/DistillGPT2_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/DistillGPT2_Train/GPU_wrapper_calls.txt)。

### 1. prims.fma metadata 重复执行第一组 Mul/Add

**优化入口：PyTorch Inductor inductor_prims.make_prim.meta 与 prims._prim_impl；NPU prims.fma fallback 暴露真实执行。** softmax backward 后连续 Mul/Add/Mul/Add 中第一对来自 metadata 执行真实 impl_aten，随后真实实现再次执行。第一对结果不参与最终输出；GPU 生成 kernel 内执行有效公式，没有 metadata 设备调用。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnMul_MulAiCore_Mul` | 6.0 | 1.732885 | 0.76% |
| `aclnnAdd_AddAiCore_Add` | 6.0 | 2.572496 | 1.13% |

本项 kernel/区间合计 **4.305381 ms，占整步 1.89%**。

仅计算这些相邻窗口中的第一对，不使用整网 Add/Mul 汇总。预算为已证实重复调用的当前设备耗时；修复后整网时间仍须实测。另一个有效 Mul/Add 对不是可全额删除的工作。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `_efficient_attention_backward` | 6.0 | 20.440757 | 12.34% |

**贡献粗估：4.305381 ms/step，相当于当前 NPU Inductor 整步耗时的 1.89%，使整网提升比增加 1.814 个百分点，单独达到该预算后整网提升比约为 5.92%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnMul_MulAiCore_Mul`：外部调用；见 wrapper 调用边界。
- `aclnnAdd_AddAiCore_Add`：外部调用；见 wrapper 调用边界。

历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused__softmax__to_copy_full_native_14` | 4.766360 / 5.0 | 2.09% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad` | 3.481986 / 1.0 | 1.53% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_add_0` | 3.416719 / 1.0 | 1.50% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_per_fused__softmax__to_copy_full_native_14`：NPU_g02_output_code.py:1288；输入 `b8[1, 1, 1024, 1024][1048576, 1048576, 1024, 1]npu:0`；size_hints=`{'z0': 16, 'y1': 12, 'x2': 512, 'r3': 512}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_red_fused__log_softmax__log_softmax_backward_data_arange_clone_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_view_0`, `triton_per_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_view_10`, `triton_per_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_view_9`。
- `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad`：外部调用；见 wrapper 调用边界。GPU 检查入口：需对照外部 ACLNN/CUDA 调用和 FX 图。
- `triton_poi_fused_add_0`：NPU_g01_output_code.py:56；输入 `f32[16, 512, 50257][25731584, 50257, 1]npu:0`；size_hints=`{'z0': 16, 'y1': 512, 'x2': 50257}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_add_mul_pow_tanh_tanh_backward_view_5`, `triton_poi_fused_add_mul_pow_tanh_view_8`, `triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_view_7`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
