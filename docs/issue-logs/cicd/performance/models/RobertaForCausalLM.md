---
title: "CICD 性能分析：RobertaForCausalLM"
---

# RobertaForCausalLM：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=16；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 115.139050 | 104.205410 | 9.50% |
| GPU | 86.114546 | 84.867685 | 1.45% |

达到 GPU **整网**提升比时，NPU 整步目标为 113.471940 ms；当前相对该目标的差额为 -9.266530 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/RobertaForCausalLM_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/RobertaForCausalLM_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/RobertaForCausalLM_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/RobertaForCausalLM_Infer/GPU_wrapper_calls.txt)。

### 1. log_softmax→NLLLoss 的完整 log probability 写回

**优化入口：NPU Inductor nll_loss_forward decomposition/lowering、producer-consumer fusion。** 外部 NLLLoss 消费 log_softmax 完整矩阵；GPU loss 融合 kernel 可只提取 target logit 和行归约统计。检查 NLLLoss 唯一消费者边界，避免最后一个完整词表循环及中间矩阵写回。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__log_softmax_clone_slice_view_7` | 1.0 | 2.687460 | 2.58% |
| `triton_poi_fused_clone_slice_8` | 1.0 | 0.002959 | 0.00% |
| `aclnnInplaceOne_OnesLikeAiCore_OnesLike` | 1.0 | 0.003928 | 0.00% |
| `aclnnNLLLoss_NLLLossAiCore_NLLLoss` | 1.0 | 0.015666 | 0.02% |

本项 kernel/区间合计 **2.710013 ms，占整步 2.60%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 4.617044 | 2.710013 | ≤41.30% |
| GPU（Eager 下界 / compiled 上界） | 5.116787 | 2.451650 | ≥52.09% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **10.78 个百分点**。NPU 为最终 projection 后至 NLLLoss 的逐步精确窗口，包含 shift/clone/copy；GPU Eager 只计 log_softmax+NLLLoss 下界，compiled 包含 prepare 及全部 loss kernel。预算是保守下界。

NPU Eager 对账项：`aclnnInplaceOne_OnesLikeAiCore_OnesLike`, `aclnnInplaceCopy_SliceAiCore_Slice`, `aclnnLogSoftmax_LogSoftmaxAiCore_LogSoftmaxV2`, `aclnnNLLLoss_NLLLossAiCore_NLLLoss`。

GPU Eager 对账项：`_log_softmax`, `nll_loss_forward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_prepare_softmax_online_view_6` | 1.0 | 2.441294 | 2.88% |
| `triton_red_fused__log_softmax_clone_constant_pad_nd_nll_loss_forward_slice_view_7` | 1.0 | 0.010355 | 0.01% |

**贡献粗估：0.497810 ms/step，相当于当前 NPU Inductor 整步耗时的 0.48%，使整网提升比增加 0.432 个百分点，单独达到该预算后整网提升比约为 9.93%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused__log_softmax_clone_slice_view_7`：NPU_g01_output_code.py:585；输入 `f32[8192, 50265][50265, 1]npu:0`；size_hints=`{'x2': 16, 'x3': 511, 'r1': 50265}`；split_axis=`[0, 1]`。
- `triton_poi_fused_clone_slice_8`：NPU_g01_output_code.py:669；输入 `i64[16, 512][512, 1]npu:0`；size_hints=`{'y0': 16, 'x1': 511}`；split_axis=`[0, 1]`。
- `aclnnInplaceOne_OnesLikeAiCore_OnesLike`：外部调用；见 wrapper 调用边界。

### 2. GELU 前向外部调用边界

**优化入口：NPU Inductor decomposition/lowering、scheduler 的 bias/GELU 融合。** NPU 保留 ACLNN GELU；GPU 将 GELU 与相邻 bias/pointwise 合入生成 kernel。优化具体外部调用边界，而非仅调整 ACLNN GELU。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnGeluV2_GeluV2_GeluV2` | 13.0 | 2.028551 | 1.95% |

本项 kernel/区间合计 **2.028551 ms，占整步 1.95%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 1.850604 | 2.028551 | ≤-9.62% |
| GPU（Eager 下界 / compiled 上界） | 3.337077 | 3.237339 | ≥2.99% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **12.60 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnGeluV2_GeluV2_GeluV2`。

GPU Eager 对账项：`gelu`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_addmm_gelu_view_4` | 12.0 | 3.207783 | 3.78% |
| `triton_per_fused_addmm_gelu_native_layer_norm_view_5` | 1.0 | 0.029556 | 0.03% |

**贡献粗估：0.233258 ms/step，相当于当前 NPU Inductor 整步耗时的 0.22%，使整网提升比增加 0.203 个百分点，单独达到该预算后整网提升比约为 9.70%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnGeluV2_GeluV2_GeluV2`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused_add_addmm_native_layer_norm_v_3` | 1.272460 / 24.0 | 1.22% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_addmm_4` | 0.878446 / 12.0 | 0.84% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_clone_permute_2` | 0.278940 / 12.0 | 0.27% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_per_fused_add_addmm_native_layer_norm_v_3`：NPU_g01_output_code.py:304；输入 `f32[768][1]npu:0`；size_hints=`{'y0': 16, 'x1': 512, 'r2': 768}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_4`, `triton_per_fused_add_addmm_native_layer_norm_view_3`, `triton_per_fused_addmm_gelu_native_layer_norm_view_5`。
- `triton_poi_fused_addmm_4`：NPU_g01_output_code.py:383；输入 `f32[3072][1]npu:0`；size_hints=`{'y0': 8192, 'x1': 3072}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_4`, `triton_per_fused_add_addmm_native_layer_norm_view_3`, `triton_per_fused_addmm_gelu_native_layer_norm_view_5`。
- `triton_poi_fused_clone_permute_2`：NPU_g01_output_code.py:231；输入 `f32[16, 12, 512, 64][393216, 32768, 64, 1]npu:0`；size_hints=`{'t0': 16, 'z1': 512, 'y2': 12, 'x3': 64}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_red_fused__log_softmax_clone_constant_pad_nd_nll_loss_forward_slice_view_7`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=16；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 357.134498 | 413.713966 | -15.84% |
| GPU | 281.458430 | 275.092940 | 2.26% |

达到 GPU **整网**提升比时，NPU 整步目标为 349.057511 ms；当前相对该目标的差额为 64.656455 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/RobertaForCausalLM_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/RobertaForCausalLM_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/RobertaForCausalLM_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/RobertaForCausalLM_Train/GPU_wrapper_calls.txt)。

### 1. prims.fma metadata 重复执行第一组 Mul/Add

**优化入口：PyTorch Inductor inductor_prims.make_prim.meta 与 prims._prim_impl；NPU prims.fma fallback 暴露真实执行。** softmax backward 后连续 Mul/Add/Mul/Add 中第一对来自 metadata 执行真实 impl_aten，随后真实实现再次执行。第一对结果不参与最终输出；GPU 生成 kernel 内执行有效公式，没有 metadata 设备调用。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnMul_MulAiCore_Mul` | 12.0 | 3.551462 | 0.86% |
| `aclnnAdd_AddAiCore_Add` | 12.0 | 5.186535 | 1.25% |

本项 kernel/区间合计 **8.737997 ms，占整步 2.11%**。

仅计算这些相邻窗口中的第一对，不使用整网 Add/Mul 汇总。预算为已证实重复调用的当前设备耗时；修复后整网时间仍须实测。另一个有效 Mul/Add 对不是可全额删除的工作。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `_efficient_attention_backward` | 12.0 | 39.791210 | 14.46% |

**贡献粗估：8.737997 ms/step，相当于当前 NPU Inductor 整步耗时的 2.11%，使整网提升比增加 2.447 个百分点，单独达到该预算后整网提升比约为 -13.40%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnMul_MulAiCore_Mul`：外部调用；见 wrapper 调用边界。
- `aclnnAdd_AddAiCore_Add`：外部调用；见 wrapper 调用边界。

历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。

### 2. NLLLossBackward→LogSoftmaxBackward 的完整中间矩阵

**优化入口：NPU Inductor nll_loss_backward / _log_softmax_backward_data decomposition 与 lowering。** 两条独立外部调用之间生成完整 dlogP；无类别权重时每个 token 的 dlogP 仅 target 类别非零。GPU 将 loss 导数融合生成，目标是直接计算 dlogits，避免完整中间矩阵；最终 dlogits 仍须生成。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnNLLLossBackward_NLLLossGradAiCore_NLLLossGrad` | 1.0 | 1.256527 | 0.30% |
| `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad` | 1.0 | 3.514964 | 0.85% |

本项 kernel/区间合计 **4.771491 ms，占整步 1.15%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 4.659711 | 4.771491 | ≤-2.40% |
| GPU（Eager 下界 / compiled 上界） | 8.178233 | 7.740530 | ≥5.35% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **7.75 个百分点**。GPU compiled 上界故意包含全部前向与反向 loss/prepare kernel。需要完整 GPU wrapper 进一步切开前后向后才能给更紧预算。

NPU Eager 对账项：`aclnnNLLLossBackward_NLLLossGradAiCore_NLLLossGrad`, `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad`。

GPU Eager 对账项：`_log_softmax_backward_data`, `nll_loss_backward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__log_softmax__log_softmax_backward_data_arange_clone_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_view_0` | 1.0 | 5.473942 | 1.99% |
| `triton_red_fused__log_softmax_prepare_softmax_online_8` | 1.0 | 2.256802 | 0.82% |
| `triton_red_fused__log_softmax_clone_nll_loss_forward_slice_view_9` | 1.0 | 0.009786 | 0.00% |

**贡献粗估：0.361170 ms/step，相当于当前 NPU Inductor 整步耗时的 0.09%，使整网提升比增加 0.101 个百分点，单独达到该预算后整网提升比约为 -15.74%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnNLLLossBackward_NLLLossGradAiCore_NLLLossGrad`：外部调用；见 wrapper 调用边界。
- `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused__softmax_all_amax_div_isfinit_8` | 14.325747 / 11.0 | 3.46% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_per_fused__softmax__softmax_backward_da_11` | 7.105822 / 12.0 | 1.72% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_poi_fused_add_0` | 3.501054 / 1.0 | 0.85% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_per_fused__softmax_all_amax_div_isfinit_8`：NPU_g02_output_code.py:796；输入 `f32[192, 512, 512][262144, 512, 1]npu:0`；size_hints=`{'z0': 16, 'y1': 12, 'x2': 512, 'r3': 512}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_per_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_view_5`, `triton_per_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_view_3`, `triton_per_fused_add_embedding_expand_native_dropout_native_layer_norm_native_layer_norm_backward_1`。
- `triton_per_fused__softmax__softmax_backward_da_11`：NPU_g01_output_code.py:1075；输入 `b8[16, 12, 512, 1][6144, 512, 1, 1]npu:0`；size_hints=`{'z0': 16, 'y1': 12, 'x2': 512, 'r3': 512}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_view_9`, `triton_poi_fused_native_dropout_backward_native_layer_norm_backward_12`, `triton_poi_fused_add_native_dropout_backward_view_13`。
- `triton_poi_fused_add_0`：NPU_g01_output_code.py:56；输入 `f32[16, 512, 50265][25735680, 50265, 1]npu:0`；size_hints=`{'z0': 16, 'y1': 512, 'x2': 50265}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_per_fused_add_native_layer_norm_backward_view_11`, `triton_per_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_view_5`, `triton_poi_fused_add_native_layer_norm_backward_view_10`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
