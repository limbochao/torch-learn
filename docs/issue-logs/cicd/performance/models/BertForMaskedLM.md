---
title: "CICD 性能分析：BertForMaskedLM"
---

# BertForMaskedLM：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=16；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 101.377946 | 92.497133 | 8.76% |
| GPU | 81.420622 | 83.132457 | -2.10% |

达到 GPU **整网**提升比时，NPU 整步目标为 103.509375 ms；当前相对该目标的差额为 -11.012242 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/BertForMaskedLM_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/BertForMaskedLM_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/BertForMaskedLM_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/BertForMaskedLM_Infer/GPU_wrapper_calls.txt)。

### 1. log_softmax→NLLLoss 的完整 log probability 写回

**优化入口：NPU Inductor nll_loss_forward decomposition/lowering、producer-consumer fusion。** 外部 NLLLoss 消费 log_softmax 完整矩阵；GPU loss 融合 kernel 可只提取 target logit 和行归约统计。检查 NLLLoss 唯一消费者边界，避免最后一个完整词表循环及中间矩阵写回。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__log_softmax_6` | 1.0 | 1.579837 | 1.71% |
| `aclnnInplaceOne_OnesLikeAiCore_OnesLike` | 1.0 | 0.003137 | 0.00% |
| `aclnnNLLLoss_NLLLossAiCore_NLLLoss` | 1.0 | 0.015616 | 0.02% |

本项 kernel/区间合计 **1.598591 ms，占整步 1.73%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 1.415077 | 1.598591 | ≤-12.97% |
| GPU（Eager 下界 / compiled 上界） | 3.037840 | 1.322049 | ≥56.48% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **69.45 个百分点**。NPU 为最终 projection 后至 NLLLoss 的逐步精确窗口，包含 shift/clone/copy；GPU Eager 只计 log_softmax+NLLLoss 下界，compiled 包含 prepare 及全部 loss kernel。预算是保守下界。

NPU Eager 对账项：`aclnnInplaceOne_OnesLikeAiCore_OnesLike`, `aclnnLogSoftmax_LogSoftmaxAiCore_LogSoftmaxV2`, `aclnnNLLLoss_NLLLossAiCore_NLLLoss`。

GPU Eager 对账项：`_log_softmax`, `nll_loss_forward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_prepare_softmax_online_view_5` | 1.0 | 1.311681 | 1.58% |
| `triton_red_fused__log_softmax_nll_loss_forward_view_6` | 1.0 | 0.010369 | 0.01% |

**贡献粗估：0.982758 ms/step，相当于当前 NPU Inductor 整步耗时的 1.06%，使整网提升比增加 0.969 个百分点，单独达到该预算后整网提升比约为 9.73%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused__log_softmax_6`：NPU_g01_output_code.py:509；输入 `f32[8192, 30522][30522, 1]npu:0`；size_hints=`{'x0': 8192, 'r1': 30522}`；split_axis=`[0]`。
- `aclnnInplaceOne_OnesLikeAiCore_OnesLike`：外部调用；见 wrapper 调用边界。
- `aclnnNLLLoss_NLLLossAiCore_NLLLoss`：外部调用；见 wrapper 调用边界。

### 2. GELU 前向外部调用边界

**优化入口：NPU Inductor decomposition/lowering、scheduler 的 bias/GELU 融合。** NPU 保留 ACLNN GELU；GPU 将 GELU 与相邻 bias/pointwise 合入生成 kernel。优化具体外部调用边界，而非仅调整 ACLNN GELU。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnGeluV2_GeluV2_GeluV2` | 13.0 | 2.042318 | 2.21% |

本项 kernel/区间合计 **2.042318 ms，占整步 2.21%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 1.847301 | 2.042318 | ≤-10.56% |
| GPU（Eager 下界 / compiled 上界） | 3.324164 | 3.193000 | ≥3.95% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **14.50 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnGeluV2_GeluV2_GeluV2`。

GPU Eager 对账项：`gelu`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_addmm_gelu_view_3` | 12.0 | 3.166138 | 3.81% |
| `triton_per_fused_addmm_gelu_native_layer_norm_view_4` | 1.0 | 0.026862 | 0.03% |

**贡献粗估：0.267908 ms/step，相当于当前 NPU Inductor 整步耗时的 0.29%，使整网提升比增加 0.264 个百分点，单独达到该预算后整网提升比约为 9.02%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnGeluV2_GeluV2_GeluV2`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused_add_addmm_native_layer_norm_v_2` | 1.225533 / 24.0 | 1.32% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_addmm_3` | 0.871110 / 12.0 | 0.94% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_clone_permute_1` | 0.262215 / 12.0 | 0.28% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_per_fused_add_addmm_native_layer_norm_v_2`：NPU_g01_output_code.py:236；输入 `f32[768][1]npu:0`；size_hints=`{'y0': 16, 'x1': 512, 'r2': 768}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_3`, `triton_per_fused_add_addmm_native_layer_norm_view_2`, `triton_per_fused_add_embedding_expand_native_layer_norm_0`。
- `triton_poi_fused_addmm_3`：NPU_g01_output_code.py:315；输入 `f32[3072][1]npu:0`；size_hints=`{'y0': 8192, 'x1': 3072}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_3`, `triton_per_fused_add_addmm_native_layer_norm_view_2`, `triton_per_fused_addmm_gelu_native_layer_norm_view_4`。
- `triton_poi_fused_clone_permute_1`：NPU_g01_output_code.py:163；输入 `f32[16, 12, 512, 64][393216, 32768, 64, 1]npu:0`；size_hints=`{'t0': 16, 'z1': 512, 'y2': 12, 'x3': 64}`；split_axis=`[0, 1]`。GPU 检查入口：需对照外部 ACLNN/CUDA 调用和 FX 图。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=16；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 312.829786 | 348.279075 | -11.33% |
| GPU | 251.303884 | 242.039795 | 3.69% |

达到 GPU **整网**提升比时，NPU 整步目标为 301.297600 ms；当前相对该目标的差额为 46.981475 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/BertForMaskedLM_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/BertForMaskedLM_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/BertForMaskedLM_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/BertForMaskedLM_Train/GPU_wrapper_calls.txt)。

### 1. prims.fma metadata 重复执行第一组 Mul/Add

**优化入口：PyTorch Inductor inductor_prims.make_prim.meta 与 prims._prim_impl；NPU prims.fma fallback 暴露真实执行。** softmax backward 后连续 Mul/Add/Mul/Add 中第一对来自 metadata 执行真实 impl_aten，随后真实实现再次执行。第一对结果不参与最终输出；GPU 生成 kernel 内执行有效公式，没有 metadata 设备调用。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnMul_MulAiCore_Mul` | 12.0 | 3.522018 | 1.01% |
| `aclnnAdd_AddAiCore_Add` | 12.0 | 5.142259 | 1.48% |

本项 kernel/区间合计 **8.664277 ms，占整步 2.49%**。

仅计算这些相邻窗口中的第一对，不使用整网 Add/Mul 汇总。预算为已证实重复调用的当前设备耗时；修复后整网时间仍须实测。另一个有效 Mul/Add 对不是可全额删除的工作。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `_efficient_attention_backward` | 12.0 | 39.468144 | 16.31% |

**贡献粗估：8.664277 ms/step，相当于当前 NPU Inductor 整步耗时的 2.49%，使整网提升比增加 2.770 个百分点，单独达到该预算后整网提升比约为 -8.56%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnMul_MulAiCore_Mul`：外部调用；见 wrapper 调用边界。
- `aclnnAdd_AddAiCore_Add`：外部调用；见 wrapper 调用边界。

历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。

### 2. NLLLossBackward→LogSoftmaxBackward 的完整中间矩阵

**优化入口：NPU Inductor nll_loss_backward / _log_softmax_backward_data decomposition 与 lowering。** 两条独立外部调用之间生成完整 dlogP；无类别权重时每个 token 的 dlogP 仅 target 类别非零。GPU 将 loss 导数融合生成，目标是直接计算 dlogits，避免完整中间矩阵；最终 dlogits 仍须生成。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnNLLLossBackward_NLLLossGradAiCore_NLLLossGrad` | 1.0 | 0.746944 | 0.21% |
| `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad` | 1.0 | 2.145857 | 0.62% |

本项 kernel/区间合计 **2.892800 ms，占整步 0.83%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 2.892742 | 2.892800 | ≤-0.00% |
| GPU（Eager 下界 / compiled 上界） | 4.676092 | 4.378754 | ≥6.36% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **6.36 个百分点**。GPU compiled 上界故意包含全部前向与反向 loss/prepare kernel。需要完整 GPU wrapper 进一步切开前后向后才能给更紧预算。

NPU Eager 对账项：`aclnnNLLLossBackward_NLLLossGradAiCore_NLLLossGrad`, `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad`。

GPU Eager 对账项：`_log_softmax_backward_data`, `nll_loss_backward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__log_softmax__log_softmax_backward_data_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_view_0` | 1.0 | 3.030216 | 1.25% |
| `triton_red_fused__log_softmax_prepare_softmax_online_6` | 1.0 | 1.339172 | 0.55% |
| `triton_red_fused__log_softmax_nll_loss_forward_view_7` | 1.0 | 0.009367 | 0.00% |

**贡献粗估：0.183998 ms/step，相当于当前 NPU Inductor 整步耗时的 0.05%，使整网提升比增加 0.059 个百分点，单独达到该预算后整网提升比约为 -11.27%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnNLLLossBackward_NLLLossGradAiCore_NLLLossGrad`：外部调用；见 wrapper 调用边界。
- `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused__softmax_all_amax_div_isfinit_6` | 14.116388 / 11.0 | 4.05% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_per_fused__softmax__softmax_backward_da_11` | 6.776238 / 12.0 | 1.95% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_per_fused_add_addmm_native_dropout_nati_5` | 2.890409 / 24.0 | 0.83% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |

- `triton_per_fused__softmax_all_amax_div_isfinit_6`：NPU_g02_output_code.py:670；输入 `f32[192, 512, 512][262144, 512, 1]npu:0`；size_hints=`{'z0': 16, 'y1': 12, 'x2': 512, 'r3': 512}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_per_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_view_4`, `triton_per_fused_add_embedding_expand_native_dropout_native_layer_norm_native_layer_norm_backward_0`, `triton_per_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_view_2`。
- `triton_per_fused__softmax__softmax_backward_da_11`：NPU_g01_output_code.py:1079；输入 `b8[16, 12, 512, 1][6144, 512, 1, 1]npu:0`；size_hints=`{'z0': 16, 'y1': 12, 'x2': 512, 'r3': 512}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_view_9`, `triton_poi_fused_native_dropout_backward_native_layer_norm_backward_12`, `triton_poi_fused_add_native_dropout_backward_view_13`。
- `triton_per_fused_add_addmm_native_dropout_nati_5`：NPU_g02_output_code.py:546；输入 `i64[37][1]npu:0`；size_hints=`{'y0': 16, 'x1': 512, 'r2': 768}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_per_fused_add_native_layer_norm_backward_view_11`, `triton_per_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_view_4`, `triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_view_9`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
