---
title: "CICD 性能分析：LayoutLMForMaskedLM"
---

# LayoutLMForMaskedLM：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=16；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 101.881538 | 92.453109 | 9.25% |
| GPU | 103.696650 | 88.732579 | 14.43% |

达到 GPU **整网**提升比时，NPU 整步目标为 87.179399 ms；当前相对该目标的差额为 5.273710 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/LayoutLMForMaskedLM_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/LayoutLMForMaskedLM_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/LayoutLMForMaskedLM_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/LayoutLMForMaskedLM_Infer/GPU_wrapper_calls.txt)。

### 1. log_softmax→NLLLoss 的完整 log probability 写回

**优化入口：NPU Inductor nll_loss_forward decomposition/lowering、producer-consumer fusion。** 外部 NLLLoss 消费 log_softmax 完整矩阵；GPU loss 融合 kernel 可只提取 target logit 和行归约统计。检查 NLLLoss 唯一消费者边界，避免最后一个完整词表循环及中间矩阵写回。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__log_softmax_6` | 1.0 | 1.571940 | 1.70% |
| `aclnnInplaceOne_OnesLikeAiCore_OnesLike` | 1.0 | 0.003142 | 0.00% |
| `aclnnNLLLoss_NLLLossAiCore_NLLLoss` | 1.0 | 0.015517 | 0.02% |

本项 kernel/区间合计 **1.590599 ms，占整步 1.72%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 1.412645 | 1.590599 | ≤-12.60% |
| GPU（Eager 下界 / compiled 上界） | 3.041163 | 1.339640 | ≥55.95% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **68.55 个百分点**。NPU 为最终 projection 后至 NLLLoss 的逐步精确窗口，包含 shift/clone/copy；GPU Eager 只计 log_softmax+NLLLoss 下界，compiled 包含 prepare 及全部 loss kernel。预算是保守下界。

NPU Eager 对账项：`aclnnInplaceOne_OnesLikeAiCore_OnesLike`, `aclnnLogSoftmax_LogSoftmaxAiCore_LogSoftmaxV2`, `aclnnNLLLoss_NLLLossAiCore_NLLLoss`。

GPU Eager 对账项：`_log_softmax`, `nll_loss_forward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_prepare_softmax_online_view_8` | 1.0 | 1.329166 | 1.50% |
| `triton_red_fused__log_softmax_nll_loss_forward_view_9` | 1.0 | 0.010474 | 0.01% |

**贡献粗估：0.968325 ms/step，相当于当前 NPU Inductor 整步耗时的 1.05%，使整网提升比增加 0.950 个百分点，单独达到该预算后整网提升比约为 10.20%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused__log_softmax_6`：NPU_g01_output_code.py:567；输入 `f32[8192, 30522][30522, 1]npu:0`；size_hints=`{'x0': 8192, 'r1': 30522}`；split_axis=`[0]`。
- `aclnnInplaceOne_OnesLikeAiCore_OnesLike`：外部调用；见 wrapper 调用边界。
- `aclnnNLLLoss_NLLLossAiCore_NLLLoss`：外部调用；见 wrapper 调用边界。

### 2. GELU 前向外部调用边界

**优化入口：NPU Inductor decomposition/lowering、scheduler 的 bias/GELU 融合。** NPU 保留 ACLNN GELU；GPU 将 GELU 与相邻 bias/pointwise 合入生成 kernel。优化具体外部调用边界，而非仅调整 ACLNN GELU。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnGeluV2_GeluV2_GeluV2` | 13.0 | 2.017028 | 2.18% |

本项 kernel/区间合计 **2.017028 ms，占整步 2.18%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 1.853563 | 2.017028 | ≤-8.82% |
| GPU（Eager 下界 / compiled 上界） | 3.327016 | 3.219972 | ≥3.22% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **12.04 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnGeluV2_GeluV2_GeluV2`。

GPU Eager 对账项：`gelu`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_addmm_gelu_view_6` | 12.0 | 3.192054 | 3.60% |
| `triton_per_fused_addmm_gelu_native_layer_norm_view_7` | 1.0 | 0.027918 | 0.03% |

**贡献粗估：0.223102 ms/step，相当于当前 NPU Inductor 整步耗时的 0.24%，使整网提升比增加 0.219 个百分点，单独达到该预算后整网提升比约为 9.47%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnGeluV2_GeluV2_GeluV2`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused_add_addmm_native_layer_norm_v_2` | 1.228130 / 24.0 | 1.33% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_addmm_3` | 0.877469 / 12.0 | 0.95% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_clone_permute_1` | 0.245933 / 12.0 | 0.27% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_per_fused_add_addmm_native_layer_norm_v_2`：NPU_g01_output_code.py:294；输入 `f32[768][1]npu:0`；size_hints=`{'y0': 16, 'x1': 512, 'r2': 768}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_6`, `triton_poi_fused_addmm_clone_transpose_view_1`, `triton_per_fused_add_addmm_native_layer_norm_view_5`。
- `triton_poi_fused_addmm_3`：NPU_g01_output_code.py:373；输入 `f32[3072][1]npu:0`；size_hints=`{'y0': 8192, 'x1': 3072}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_6`, `triton_poi_fused_addmm_clone_transpose_view_1`, `triton_per_fused_add_addmm_native_layer_norm_view_5`。
- `triton_poi_fused_clone_permute_1`：NPU_g01_output_code.py:221；输入 `f32[16, 12, 512, 64][393216, 32768, 64, 1]npu:0`；size_hints=`{'t0': 16, 'z1': 512, 'y2': 12, 'x3': 64}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_addmm_clone_transpose_view_1`, `triton_poi_fused_addmm_clone_transpose_view_2`, `triton_poi_fused_clone_transpose_view_4`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=16；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 305.775699 | 349.305439 | -14.24% |
| GPU | 303.770742 | 249.548092 | 17.85% |

达到 GPU **整网**提升比时，NPU 整步目标为 251.195167 ms；当前相对该目标的差额为 98.110272 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/LayoutLMForMaskedLM_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/LayoutLMForMaskedLM_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/LayoutLMForMaskedLM_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/LayoutLMForMaskedLM_Train/GPU_wrapper_calls.txt)。

### 1. 首层 attention constexpr/checker 回退

**优化入口：NPU Inductor NPUIndexTritonKernel.codegen_kernel、TTIR checker constexpr 参数传递及回退 tiling。** 首层 kernel 为 25.091903 ms/step；后续同类 triton_per_fused__softmax_all_amax_div_isfinit_8 为 1.283046 ms/次。此前源码/TTIR checker 已确认 seed offset constexpr 缺失常量触发回退。GPU 同类 attention 没有这一首层回退。实际 compile_mode 不能仅凭 metadata 标签区分。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused__softmax_all_amax_div_isfinite_matmul_native_dropout_sub_5` | 1.0 | 25.091903 | 7.18% |

本项 kernel/区间合计 **25.091903 ms，占整步 7.18%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 14.655329 | 25.091903 | ≤-71.21% |
| GPU（Eager 下界 / compiled 上界） | 7.604581 | 8.501592 | ≥-11.80% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **59.42 个百分点**。保守四方筛选：NPU 将全部 12 层前向两次 BMM 之间的 Eager 工作收费给首层，Inductor 只计首层；GPU compiled 计全部 12 层前向 softmax/dropout kernel，GPU Eager 计全部 softmax，以及混合 dropout 中 12 次 attention 调用的下界。dropout 下界=max(12×min, total−(count−12)×max)，不按平均值分摊混合形状。因此这里是很保守的提升比缺口预算；恢复到 NPU 后续层实现的独立控制估算为 23.808857 ms，与本预算口径不同，不能相加。根因仍采用历史 checker/source 证据。

NPU Eager 对账项：`aclnnAdd_AddAiCore_Add`, `aclnnDivs_Muls_Muls`, `aclnnDropoutV3_DropoutV3AiCore_DropOutV3`, `aclnnMatmul_TransposeAiCore_Transpose`, `aclnnSoftmax_SoftmaxAiCore_SoftmaxV2`。

GPU Eager 对账项：`_softmax`, `native_dropout (12 calls, min/max lower bound)`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused__softmax_abs_all_amax_eq_mul_native_dropout_ne_sub_view_9` | 11.0 | 7.791527 | 3.12% |
| `triton_per_fused__softmax_abs_all_amax_eq_mul_native_dropout_ne_sub_view_5` | 1.0 | 0.710064 | 0.28% |

**贡献粗估：8.707880 ms/step，相当于当前 NPU Inductor 整步耗时的 2.49%，使整网提升比增加 2.848 个百分点，单独达到该预算后整网提升比约为 -11.39%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_per_fused__softmax_all_amax_div_isfinite_matmul_native_dropout_sub_5`：NPU_g02_output_code.py:534；输入 `f32[192, 512, 512][262144, 512, 1]npu:0`；size_hints=`{'x': 131072, 'r0_': 512}`；split_axis=`[0]`。

历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。

### 2. prims.fma metadata 重复执行第一组 Mul/Add

**优化入口：PyTorch Inductor inductor_prims.make_prim.meta 与 prims._prim_impl；NPU prims.fma fallback 暴露真实执行。** softmax backward 后连续 Mul/Add/Mul/Add 中第一对来自 metadata 执行真实 impl_aten，随后真实实现再次执行。第一对结果不参与最终输出；GPU 生成 kernel 内执行有效公式，没有 metadata 设备调用。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnMul_MulAiCore_Mul` | 12.0 | 3.511247 | 1.01% |
| `aclnnAdd_AddAiCore_Add` | 12.0 | 5.147488 | 1.47% |

本项 kernel/区间合计 **8.658735 ms，占整步 2.48%**。

仅计算这些相邻窗口中的第一对，不使用整网 Add/Mul 汇总。预算为已证实重复调用的当前设备耗时；修复后整网时间仍须实测。另一个有效 Mul/Add 对不是可全额删除的工作。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused__softmax__softmax_backward_data_mul_native_dropout_backward_sub_view_11` | 12.0 | 11.427901 | 4.58% |

**贡献粗估：8.658735 ms/step，相当于当前 NPU Inductor 整步耗时的 2.48%，使整网提升比增加 2.832 个百分点，单独达到该预算后整网提升比约为 -11.40%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnMul_MulAiCore_Mul`：外部调用；见 wrapper 调用边界。
- `aclnnAdd_AddAiCore_Add`：外部调用；见 wrapper 调用边界。

历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。

### 3. NLLLossBackward→LogSoftmaxBackward 的完整中间矩阵

**优化入口：NPU Inductor nll_loss_backward / _log_softmax_backward_data decomposition 与 lowering。** 两条独立外部调用之间生成完整 dlogP；无类别权重时每个 token 的 dlogP 仅 target 类别非零。GPU 将 loss 导数融合生成，目标是直接计算 dlogits，避免完整中间矩阵；最终 dlogits 仍须生成。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnNLLLossBackward_NLLLossGradAiCore_NLLLossGrad` | 1.0 | 0.744702 | 0.21% |
| `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad` | 1.0 | 2.150434 | 0.62% |

本项 kernel/区间合计 **2.895137 ms，占整步 0.83%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 2.849485 | 2.895137 | ≤-1.60% |
| GPU（Eager 下界 / compiled 上界） | 4.681924 | 4.372730 | ≥6.60% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **8.21 个百分点**。GPU compiled 上界故意包含全部前向与反向 loss/prepare kernel。需要完整 GPU wrapper 进一步切开前后向后才能给更紧预算。

NPU Eager 对账项：`aclnnNLLLossBackward_NLLLossGradAiCore_NLLLossGrad`, `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad`。

GPU Eager 对账项：`_log_softmax_backward_data`, `nll_loss_backward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__log_softmax__log_softmax_backward_data_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_view_0` | 1.0 | 3.026411 | 1.21% |
| `triton_red_fused__log_softmax_prepare_softmax_online_11` | 1.0 | 1.336808 | 0.54% |
| `triton_red_fused__log_softmax_nll_loss_forward_view_12` | 1.0 | 0.009511 | 0.00% |

**贡献粗估：0.233832 ms/step，相当于当前 NPU Inductor 整步耗时的 0.07%，使整网提升比增加 0.076 个百分点，单独达到该预算后整网提升比约为 -14.16%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnNLLLossBackward_NLLLossGradAiCore_NLLLossGrad`：外部调用；见 wrapper 调用边界。
- `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused__softmax_all_amax_div_isfinit_8` | 14.113511 / 11.0 | 4.04% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_per_fused__softmax__softmax_backward_da_11` | 6.657793 / 12.0 | 1.91% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_per_fused_add_addmm_native_dropout_nati_7` | 2.889306 / 24.0 | 0.83% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |

- `triton_per_fused__softmax_all_amax_div_isfinit_8`：NPU_g02_output_code.py:843；输入 `f32[192, 512, 512][262144, 512, 1]npu:0`；size_hints=`{'z0': 16, 'y1': 12, 'x2': 512, 'r3': 512}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_per_fused__softmax__softmax_backward_data_mul_native_dropout_backward_sub_view_11`, `triton_per_fused__softmax_abs_all_amax_eq_mul_native_dropout_ne_sub_view_9`, `triton_per_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_view_7`。
- `triton_per_fused__softmax__softmax_backward_da_11`：NPU_g01_output_code.py:1079；输入 `b8[16, 12, 512, 1][6144, 512, 1, 1]npu:0`；size_hints=`{'z0': 16, 'y1': 12, 'x2': 512, 'r3': 512}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_per_fused__softmax__softmax_backward_data_mul_native_dropout_backward_sub_view_11`, `triton_per_fused__softmax_abs_all_amax_eq_mul_native_dropout_ne_sub_view_9`, `triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_view_9`。
- `triton_per_fused_add_addmm_native_dropout_nati_7`：NPU_g02_output_code.py:719；输入 `i64[37][1]npu:0`；size_hints=`{'y0': 16, 'x1': 512, 'r2': 768}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_per_fused__softmax_abs_all_amax_eq_mul_native_dropout_ne_sub_view_9`, `triton_per_fused_add_native_layer_norm_backward_view_15`, `triton_per_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_view_7`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
