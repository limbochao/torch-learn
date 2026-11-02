---
title: "CICD 性能分析：PLBartForCausalLM"
---

# PLBartForCausalLM：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=8；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 76.695707 | 74.351114 | 3.06% |
| GPU | 64.598353 | 68.262971 | -5.67% |

达到 GPU **整网**提升比时，NPU 整步目标为 81.046599 ms；当前相对该目标的差额为 -6.695485 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/PLBartForCausalLM_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/PLBartForCausalLM_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/PLBartForCausalLM_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/PLBartForCausalLM_Infer/GPU_wrapper_calls.txt)。

### 1. log_softmax→NLLLoss 的完整 log probability 写回

**优化入口：NPU Inductor nll_loss_forward decomposition/lowering、producer-consumer fusion。** 外部 NLLLoss 消费 log_softmax 完整矩阵；GPU loss 融合 kernel 可只提取 target logit 和行归约统计。检查 NLLLoss 唯一消费者边界，避免最后一个完整词表循环及中间矩阵写回。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__log_softmax_7` | 1.0 | 2.601429 | 3.50% |
| `aclnnInplaceOne_OnesLikeAiCore_OnesLike` | 1.0 | 0.003326 | 0.00% |
| `aclnnNLLLoss_NLLLossAiCore_NLLLoss` | 1.0 | 0.016162 | 0.02% |

本项 kernel/区间合计 **2.620917 ms，占整步 3.53%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 2.303117 | 2.620917 | ≤-13.80% |
| GPU（Eager 下界 / compiled 上界） | 5.056589 | 2.483353 | ≥50.89% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **64.69 个百分点**。NPU 为最终 projection 后至 NLLLoss 的逐步精确窗口，包含 shift/clone/copy；GPU Eager 只计 log_softmax+NLLLoss 下界，compiled 包含 prepare 及全部 loss kernel。预算是保守下界。

NPU Eager 对账项：`aclnnLogSoftmax_LogSoftmaxAiCore_LogSoftmaxV2`, `aclnnInplaceOne_OnesLikeAiCore_OnesLike`, `aclnnNLLLoss_NLLLossAiCore_NLLLoss`。

GPU Eager 对账项：`_log_softmax`, `nll_loss_forward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__unsafe_view_prepare_softmax_online_view_4` | 1.0 | 2.471884 | 3.62% |
| `triton_red_fused__log_softmax__unsafe_view_nll_loss_forward_view_5` | 1.0 | 0.011469 | 0.02% |

**贡献粗估：1.489828 ms/step，相当于当前 NPU Inductor 整步耗时的 2.00%，使整网提升比增加 1.943 个百分点，单独达到该预算后整网提升比约为 5.00%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused__log_softmax_7`：NPU_g01_output_code.py:641；输入 `f32[8192, 50005][50005, 1]npu:0`；size_hints=`{'x0': 8192, 'r1': 50005}`；split_axis=`[0]`。
- `aclnnInplaceOne_OnesLikeAiCore_OnesLike`：外部调用；见 wrapper 调用边界。
- `aclnnNLLLoss_NLLLossAiCore_NLLLoss`：外部调用；见 wrapper 调用边界。

### 2. GELU 前向外部调用边界

**优化入口：NPU Inductor decomposition/lowering、scheduler 的 bias/GELU 融合。** NPU 保留 ACLNN GELU；GPU 将 GELU 与相邻 bias/pointwise 合入生成 kernel。优化具体外部调用边界，而非仅调整 ACLNN GELU。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnGeluV2_GeluV2_GeluV2` | 6.0 | 0.995565 | 1.34% |

本项 kernel/区间合计 **0.995565 ms，占整步 1.34%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.904851 | 0.995565 | ≤-10.03% |
| GPU（Eager 下界 / compiled 上界） | 1.640935 | 1.598083 | ≥2.61% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **12.64 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnGeluV2_GeluV2_GeluV2`。

GPU Eager 对账项：`gelu`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_addmm_gelu_view_3` | 6.0 | 1.598083 | 2.34% |

**贡献粗估：0.114345 ms/step，相当于当前 NPU Inductor 整步耗时的 0.15%，使整网提升比增加 0.149 个百分点，单独达到该预算后整网提升比约为 3.21%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnGeluV2_GeluV2_GeluV2`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused__safe_softmax_add_arange_expa_3` | 5.172832 / 6.0 | 6.96% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_per_fused_add_addmm_native_layer_norm_v_5` | 0.591273 / 12.0 | 0.80% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_addmm_6` | 0.439463 / 6.0 | 0.59% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_per_fused__safe_softmax_add_arange_expa_3`：NPU_g01_output_code.py:340；输入 `f32[96, 1024, 1024][1048576, 1024, 1]npu:0`；size_hints=`{'z0': 8, 'y1': 12, 'x2': 1024, 'r3': 1024}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_per_fused_add_addmm_native_layer_norm_view_2`, `triton_poi_fused__scaled_dot_product_efficient_attention_arange_constant_pad_nd_expand_full_gt_mul_slice_transpose_triu_unsqueeze_view_1`, `triton_red_fused_add_arange_embedding_mul_native_layer_norm_unsqueeze_0`。
- `triton_per_fused_add_addmm_native_layer_norm_v_5`：NPU_g01_output_code.py:502；输入 `f32[8, 1024, 768][786432, 768, 1]npu:0`；size_hints=`{'y0': 8, 'x1': 1024, 'r2': 768}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_3`, `triton_per_fused_add_addmm_native_layer_norm_view_2`, `triton_red_fused_add_arange_embedding_mul_native_layer_norm_unsqueeze_0`。
- `triton_poi_fused_addmm_6`：NPU_g01_output_code.py:581；输入 `f32[3072][1]npu:0`；size_hints=`{'y0': 8192, 'x1': 3072}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_3`, `triton_per_fused_add_addmm_native_layer_norm_view_2`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=8；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 228.987131 | 249.080303 | -8.77% |
| GPU | 177.909427 | 197.429583 | -10.97% |

达到 GPU **整网**提升比时，NPU 整步目标为 254.111513 ms；当前相对该目标的差额为 -5.031210 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/PLBartForCausalLM_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/PLBartForCausalLM_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/PLBartForCausalLM_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/PLBartForCausalLM_Train/GPU_wrapper_calls.txt)。

GPU profile 中有以下 kernel 未在本批 wrapper 找到同名定义，需要补充对应编译图；本场景不依赖这些缺失 DSL 做额外的算子族收益归因：

- `triton_red_fused__log_softmax__unsafe_view_mm_prepare_softmax_online_view_1`
- `triton_poi_fused_mm_t_0`
- `triton_red_fused__log_softmax__unsafe_view_mm_nll_loss_forward_view_2`

### 1. prims.fma metadata 重复执行第一组 Mul/Add

**优化入口：PyTorch Inductor inductor_prims.make_prim.meta 与 prims._prim_impl；NPU prims.fma fallback 暴露真实执行。** softmax backward 后连续 Mul/Add/Mul/Add 中第一对来自 metadata 执行真实 impl_aten，随后真实实现再次执行。第一对结果不参与最终输出；GPU 生成 kernel 内执行有效公式，没有 metadata 设备调用。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnMul_MulAiCore_Mul` | 6.0 | 3.429213 | 1.38% |
| `aclnnAdd_AddAiCore_Add` | 6.0 | 5.045709 | 2.03% |

本项 kernel/区间合计 **8.474922 ms，占整步 3.40%**。

仅计算这些相邻窗口中的第一对，不使用整网 Add/Mul 汇总。预算为已证实重复调用的当前设备耗时；修复后整网时间仍须实测。另一个有效 Mul/Add 对不是可全额删除的工作。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `_efficient_attention_backward` | 6.0 | 40.500024 | 20.51% |

**贡献粗估：8.474922 ms/step，相当于当前 NPU Inductor 整步耗时的 3.40%，使整网提升比增加 3.701 个百分点，单独达到该预算后整网提升比约为 -5.07%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnMul_MulAiCore_Mul`：外部调用；见 wrapper 调用边界。
- `aclnnAdd_AddAiCore_Add`：外部调用；见 wrapper 调用边界。

历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused__softmax_add_native_dropout_v_2` | 10.110772 / 6.0 | 4.06% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_per_fused__softmax__softmax_backward_da_8` | 7.221750 / 6.0 | 2.90% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad` | 3.473825 / 1.0 | 1.39% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_per_fused__softmax_add_native_dropout_v_2`：NPU_g05_output_code.py:230；输入 `f32[96, 1024, 1024][1048576, 1024, 1]npu:0`；size_hints=`{'y0': 96, 'x1': 1024, 'r2': 1024}`；split_axis=`[0]`。GPU 检查入口：`triton_red_fused__log_softmax__log_softmax_backward_data_add_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_view_0`, `triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_view_6`, `triton_poi_fused_add_view_8`。
- `triton_per_fused__softmax__softmax_backward_da_8`：NPU_g04_output_code.py:803；输入 `f32[96, 1024, 1024][1048576, 1024, 1]npu:0`；size_hints=`{'y0': 96, 'x1': 1024, 'r2': 1024}`；split_axis=`[0]`。GPU 检查入口：`triton_red_fused__log_softmax__log_softmax_backward_data_add_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_view_0`, `triton_per_fused_native_dropout_backward_native_layer_norm_backward_1`, `triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_view_6`。
- `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad`：外部调用；见 wrapper 调用边界。GPU 检查入口：需对照外部 ACLNN/CUDA 调用和 FX 图。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
