---
title: "CICD 性能分析：TrOCRForCausalLM"
---

# TrOCRForCausalLM：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=32；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 161.007659 | 164.462927 | -2.15% |
| GPU | 156.349703 | 147.882565 | 5.42% |

达到 GPU **整网**提升比时，NPU 整步目标为 152.288269 ms；当前相对该目标的差额为 12.174658 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/TrOCRForCausalLM_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/TrOCRForCausalLM_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/TrOCRForCausalLM_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/TrOCRForCausalLM_Infer/GPU_wrapper_calls.txt)。

### 1. log_softmax→NLLLoss 的完整 log probability 写回

**优化入口：NPU Inductor nll_loss_forward decomposition/lowering、producer-consumer fusion。** 外部 NLLLoss 消费 log_softmax 完整矩阵；GPU loss 融合 kernel 可只提取 target logit 和行归约统计。检查 NLLLoss 唯一消费者边界，避免最后一个完整词表循环及中间矩阵写回。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__log_softmax_7` | 1.0 | 2.654778 | 1.61% |
| `aclnnInplaceOne_OnesLikeAiCore_OnesLike` | 1.0 | 0.003167 | 0.00% |
| `aclnnNLLLoss_NLLLossAiCore_NLLLoss` | 1.0 | 0.016541 | 0.01% |

本项 kernel/区间合计 **2.674486 ms，占整步 1.63%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 2.332432 | 2.674486 | ≤-14.67% |
| GPU（Eager 下界 / compiled 上界） | 5.064695 | 2.316967 | ≥54.25% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **68.92 个百分点**。NPU 为最终 projection 后至 NLLLoss 的逐步精确窗口，包含 shift/clone/copy；GPU Eager 只计 log_softmax+NLLLoss 下界，compiled 包含 prepare 及全部 loss kernel。预算是保守下界。

NPU Eager 对账项：`aclnnLogSoftmax_LogSoftmaxAiCore_LogSoftmaxV2`, `aclnnInplaceOne_OnesLikeAiCore_OnesLike`, `aclnnNLLLoss_NLLLossAiCore_NLLLoss`。

GPU Eager 对账项：`_log_softmax`, `nll_loss_forward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__unsafe_view_mm_prepare_softmax_online_view_7` | 1.0 | 2.304372 | 1.56% |
| `triton_red_fused__log_softmax__unsafe_view_mm_nll_loss_forward_view_8` | 1.0 | 0.012595 | 0.01% |

**贡献粗估：1.607458 ms/step，相当于当前 NPU Inductor 整步耗时的 0.98%，使整网提升比增加 0.998 个百分点，单独达到该预算后整网提升比约为 -1.15%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused__log_softmax_7`：NPU_g01_output_code.py:637；输入 `f32[8192, 50265][50265, 1]npu:0`；size_hints=`{'x0': 8192, 'r1': 50265}`；split_axis=`[0]`。
- `aclnnInplaceOne_OnesLikeAiCore_OnesLike`：外部调用；见 wrapper 调用边界。
- `aclnnNLLLoss_NLLLossAiCore_NLLLoss`：外部调用；见 wrapper 调用边界。

### 2. GELU 前向外部调用边界

**优化入口：NPU Inductor decomposition/lowering、scheduler 的 bias/GELU 融合。** NPU 保留 ACLNN GELU；GPU 将 GELU 与相邻 bias/pointwise 合入生成 kernel。优化具体外部调用边界，而非仅调整 ACLNN GELU。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnGeluV2_GeluV2_GeluV2` | 12.0 | 2.692127 | 1.64% |

本项 kernel/区间合计 **2.692127 ms，占整步 1.64%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 2.359302 | 2.692127 | ≤-14.11% |
| GPU（Eager 下界 / compiled 上界） | 4.552189 | 4.372792 | ≥3.94% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **18.05 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnGeluV2_GeluV2_GeluV2`。

GPU Eager 对账项：`gelu`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_addmm_gelu_view_5` | 12.0 | 4.372792 | 2.96% |

**贡献粗估：0.425803 ms/step，相当于当前 NPU Inductor 整步耗时的 0.26%，使整网提升比增加 0.264 个百分点，单独达到该预算后整网提升比约为 -1.88%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnGeluV2_GeluV2_GeluV2`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused__safe_softmax_add_arange_expa_3` | 7.347876 / 12.0 | 4.47% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_per_fused_add_addmm_native_layer_norm_v_5` | 1.482069 / 24.0 | 0.90% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_addmm_6` | 1.345467 / 12.0 | 0.82% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_per_fused__safe_softmax_add_arange_expa_3`：NPU_g01_output_code.py:336；输入 `f32[512, 256, 256][65536, 256, 1]npu:0`；size_hints=`{'z0': 32, 'y1': 16, 'x2': 256, 'r3': 256}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused__scaled_dot_product_efficient_attention_add_addmm_arange_clone_expand_full_lt_masked_fill_mul_transpose_unsqueeze_view_2`, `triton_per_fused_add_addmm_native_layer_norm_view_4`, `triton_poi_fused__scaled_dot_product_efficient_attention_add_addmm_arange_clone_expand_full_lt_masked_fill_mul_transpose_unsqueeze_view_1`。
- `triton_per_fused_add_addmm_native_layer_norm_v_5`：NPU_g01_output_code.py:498；输入 `f32[32, 256, 1024][262144, 1024, 1]npu:0`；size_hints=`{'y0': 32, 'x1': 256, 'r2': 1024}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_5`, `triton_poi_fused__scaled_dot_product_efficient_attention_add_addmm_arange_clone_expand_full_lt_masked_fill_mul_transpose_unsqueeze_view_2`, `triton_per_fused_add_addmm_native_layer_norm_view_4`。
- `triton_poi_fused_addmm_6`：NPU_g01_output_code.py:577；输入 `f32[4096][1]npu:0`；size_hints=`{'y0': 8192, 'x1': 4096}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_5`, `triton_poi_fused__scaled_dot_product_efficient_attention_add_addmm_arange_clone_expand_full_lt_masked_fill_mul_transpose_unsqueeze_view_2`, `triton_per_fused_add_addmm_native_layer_norm_view_4`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=32；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 485.386276 | 503.080015 | -3.65% |
| GPU | 416.993162 | 400.896566 | 3.86% |

达到 GPU **整网**提升比时，NPU 整步目标为 466.649597 ms；当前相对该目标的差额为 36.430418 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/TrOCRForCausalLM_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/TrOCRForCausalLM_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/TrOCRForCausalLM_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/TrOCRForCausalLM_Train/GPU_wrapper_calls.txt)。

GPU profile 中有以下 kernel 未在本批 wrapper 找到同名定义，需要补充对应编译图；本场景不依赖这些缺失 DSL 做额外的算子族收益归因：

- `triton_red_fused__log_softmax__unsafe_view_mm_prepare_softmax_online_view_1`
- `triton_poi_fused_mm_t_0`
- `triton_red_fused__log_softmax__unsafe_view_mm_nll_loss_forward_view_2`

### 1. prims.fma metadata 重复执行第一组 Mul/Add

**优化入口：PyTorch Inductor inductor_prims.make_prim.meta 与 prims._prim_impl；NPU prims.fma fallback 暴露真实执行。** softmax backward 后连续 Mul/Add/Mul/Add 中第一对来自 metadata 执行真实 impl_aten，随后真实实现再次执行。第一对结果不参与最终输出；GPU 生成 kernel 内执行有效公式，没有 metadata 设备调用。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnMul_MulAiCore_Mul` | 12.0 | 2.436722 | 0.48% |
| `aclnnAdd_AddAiCore_Add` | 12.0 | 3.492735 | 0.69% |

本项 kernel/区间合计 **5.929457 ms，占整步 1.18%**。

仅计算这些相邻窗口中的第一对，不使用整网 Add/Mul 汇总。预算为已证实重复调用的当前设备耗时；修复后整网时间仍须实测。另一个有效 Mul/Add 对不是可全额删除的工作。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused__softmax_backward_data_10` | 12.0 | 7.062039 | 1.76% |

**贡献粗估：5.929457 ms/step，相当于当前 NPU Inductor 整步耗时的 1.18%，使整网提升比增加 1.222 个百分点，单独达到该预算后整网提升比约为 -2.42%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnMul_MulAiCore_Mul`：外部调用；见 wrapper 调用边界。
- `aclnnAdd_AddAiCore_Add`：外部调用；见 wrapper 调用边界。

历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused__softmax_backward_data_9` | 4.304802 / 12.0 | 0.86% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_poi_fused_gelu_backward_view_3` | 3.759252 / 12.0 | 0.75% | NPU Inductor activation backward tiling/融合；核对 GPU 对应激活及相邻 bias 边界 |
| `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad` | 3.502793 / 1.0 | 0.70% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_per_fused__softmax_backward_data_9`：NPU_g04_output_code.py:853；输入 `f32[512, 256, 256][65536, 256, 1]npu:0`；size_hints=`{'y0': 512, 'x1': 256, 'r2': 256}`；split_axis=`[0]`。GPU 检查入口：`triton_per_fused__softmax_backward_data_10`。
- `triton_poi_fused_gelu_backward_view_3`：NPU_g04_output_code.py:342；输入 `f32[8192, 4096][4096, 1]npu:0`；size_hints=`{'z0': 32, 'y1': 256, 'x2': 4096}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_gelu_gelu_backward_view_4`, `triton_red_fused_gelu_gelu_backward_sum_view_5`, `triton_per_fused_gelu_gelu_backward_sum_view_7`。
- `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad`：外部调用；见 wrapper 调用边界。GPU 检查入口：需对照外部 ACLNN/CUDA 调用和 FX 图。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
