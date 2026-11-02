---
title: "CICD 性能分析：BartForCausalLM"
---

# BartForCausalLM：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=4；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 82.123560 | 82.792606 | -0.81% |
| GPU | 77.755871 | 79.150302 | -1.79% |

达到 GPU **整网**提升比时，NPU 整步目标为 83.596319 ms；当前相对该目标的差额为 -0.803713 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/BartForCausalLM_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/BartForCausalLM_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/BartForCausalLM_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/BartForCausalLM_Infer/GPU_wrapper_calls.txt)。

### 1. log_softmax→NLLLoss 的完整 log probability 写回

**优化入口：NPU Inductor nll_loss_forward decomposition/lowering、producer-consumer fusion。** 外部 NLLLoss 消费 log_softmax 完整矩阵；GPU loss 融合 kernel 可只提取 target logit 和行归约统计。检查 NLLLoss 唯一消费者边界，避免最后一个完整词表循环及中间矩阵写回。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__log_softmax_6` | 1.0 | 1.381065 | 1.67% |
| `aclnnInplaceOne_OnesLikeAiCore_OnesLike` | 1.0 | 0.003409 | 0.00% |
| `aclnnNLLLoss_NLLLossAiCore_NLLLoss` | 1.0 | 0.012714 | 0.02% |

本项 kernel/区间合计 **1.397188 ms，占整步 1.69%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 1.166809 | 1.397188 | ≤-19.74% |
| GPU（Eager 下界 / compiled 上界） | 2.513559 | 1.245641 | ≥50.44% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **70.19 个百分点**。NPU 为最终 projection 后至 NLLLoss 的逐步精确窗口，包含 shift/clone/copy；GPU Eager 只计 log_softmax+NLLLoss 下界，compiled 包含 prepare 及全部 loss kernel。预算是保守下界。

NPU Eager 对账项：`aclnnInplaceOne_OnesLikeAiCore_OnesLike`, `aclnnLogSoftmax_LogSoftmaxAiCore_LogSoftmaxV2`, `aclnnNLLLoss_NLLLossAiCore_NLLLoss`。

GPU Eager 对账项：`_log_softmax`, `nll_loss_forward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__unsafe_view_prepare_softmax_online_view_4` | 1.0 | 1.239075 | 1.57% |
| `triton_red_fused__log_softmax__unsafe_view_nll_loss_forward_view_5` | 1.0 | 0.006567 | 0.01% |

**贡献粗估：0.818954 ms/step，相当于当前 NPU Inductor 整步耗时的 0.99%，使整网提升比增加 0.997 个百分点，单独达到该预算后整网提升比约为 0.18%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused__log_softmax_6`：NPU_g01_output_code.py:512；输入 `f32[4096, 50265][50265, 1]npu:0`；size_hints=`{'x0': 4096, 'r1': 50265}`；split_axis=`[0]`。
- `aclnnInplaceOne_OnesLikeAiCore_OnesLike`：外部调用；见 wrapper 调用边界。
- `aclnnNLLLoss_NLLLossAiCore_NLLLoss`：外部调用；见 wrapper 调用边界。

### 2. GELU 前向外部调用边界

**优化入口：NPU Inductor decomposition/lowering、scheduler 的 bias/GELU 融合。** NPU 保留 ACLNN GELU；GPU 将 GELU 与相邻 bias/pointwise 合入生成 kernel。优化具体外部调用边界，而非仅调整 ACLNN GELU。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnGeluV2_GeluV2_GeluV2` | 12.0 | 1.181526 | 1.43% |

本项 kernel/区间合计 **1.181526 ms，占整步 1.43%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 1.190261 | 1.181526 | ≤0.73% |
| GPU（Eager 下界 / compiled 上界） | 2.066148 | 1.900537 | ≥8.02% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **7.28 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnGeluV2_GeluV2_GeluV2`。

GPU Eager 对账项：`gelu`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_addmm_gelu_view_3` | 12.0 | 1.900537 | 2.40% |

**贡献粗估：0.086670 ms/step，相当于当前 NPU Inductor 整步耗时的 0.10%，使整网提升比增加 0.106 个百分点，单独达到该预算后整网提升比约为 -0.71%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnGeluV2_GeluV2_GeluV2`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused_add_addmm_native_layer_norm_v_4` | 0.788848 / 24.0 | 0.95% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_addmm_5` | 0.592865 / 12.0 | 0.72% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_addmm_clone_transpose_view_1` | 0.433511 / 36.0 | 0.52% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_per_fused_add_addmm_native_layer_norm_v_4`：NPU_g01_output_code.py:373；输入 `f32[4, 1024, 1024][1048576, 1024, 1]npu:0`；size_hints=`{'y0': 4, 'x1': 1024, 'r2': 1024}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_3`, `triton_per_fused_add_addmm_native_layer_norm_view_2`, `triton_per_fused_add_arange_embedding_mul_native_layer_norm_unsqueeze_0`。
- `triton_poi_fused_addmm_5`：NPU_g01_output_code.py:452；输入 `f32[4096][1]npu:0`；size_hints=`{'y0': 4096, 'x1': 4096}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_3`, `triton_per_fused_add_addmm_native_layer_norm_view_2`。
- `triton_poi_fused_addmm_clone_transpose_view_1`：NPU_g01_output_code.py:159；输入 `f32[1024][1]npu:0`；size_hints=`{'t0': 4, 'z1': 16, 'y2': 1024, 'x3': 64}`；split_axis=`[0, 1, 2]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_3`, `triton_per_fused_add_addmm_native_layer_norm_view_2`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=4；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 252.342776 | 292.002617 | -15.72% |
| GPU | 210.658226 | 235.370849 | -11.73% |

达到 GPU **整网**提升比时，NPU 整步目标为 281.945475 ms；当前相对该目标的差额为 10.057142 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/BartForCausalLM_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/BartForCausalLM_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/BartForCausalLM_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/BartForCausalLM_Train/GPU_wrapper_calls.txt)。

### 1. GELU backward 与 sum 梯度归约

**优化入口：NPU Inductor scheduler、codegen/reduction tiling。** 具体检查融合 gelu_backward 后的 bias sum；GPU 相应融合/归约成本全部计入上界。优化输出维分块、归约维切分和融合取舍。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused_gelu_backward_sum_view_4` | 12.0 | 13.092238 | 4.48% |

本项 kernel/区间合计 **13.092238 ms，占整步 4.48%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 2.847750 | 13.092238 | ≤-359.74% |
| GPU（Eager 下界 / compiled 上界） | 4.468297 | 3.873427 | ≥13.31% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **373.05 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnReduceSum_ReduceSumOpAiCore_ReduceSum`, `aclnnGeluBackwardV2_GeluGradV2AiCore_GeluGradV2`。

GPU Eager 对账项：`gelu_backward`, `sum`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_gelu_gelu_backward_view_4` | 12.0 | 2.674862 | 1.14% |
| `triton_red_fused_native_dropout_backward_sum_view_3` | 60.0 | 0.641818 | 0.27% |
| `sum` | 13.0 | 0.271601 | 0.12% |
| `triton_red_fused_gelu_gelu_backward_sum_view_5` | 12.0 | 0.250359 | 0.11% |
| `triton_per_fused_gelu_gelu_backward_sum_view_6` | 12.0 | 0.034787 | 0.01% |

**贡献粗估：10.623612 ms/step，相当于当前 NPU Inductor 整步耗时的 3.64%，使整网提升比增加 4.210 个百分点，单独达到该预算后整网提升比约为 -11.51%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_per_fused_gelu_backward_sum_view_4`：NPU_g03_output_code.py:435；输入 `f32[4, 1024, 4096][4194304, 4096, 1]npu:0`；size_hints=`{'x0': 4096, 'r1': 4096}`；split_axis=`[0]`。

### 2. LayerNorm backward 归约调度

**优化入口：NPU Inductor codegen/reduction tiling 与 runtime autotune。** 检查该 kernel 的 reduction size_hints、split_axis 和 program 并行度；GPU 对应归约 kernel 列于本项。优先检查归约维切分与多阶段归约，不能据此断言 Triton_ascend/npuir 有缺陷。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused_native_layer_norm_backward_0` | 12.0 | 4.590552 | 1.57% |

本项 kernel/区间合计 **4.590552 ms，占整步 1.57%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 1.289596 | 4.590552 | ≤-255.97% |
| GPU（Eager 下界 / compiled 上界） | 1.232162 | 3.919786 | ≥-218.12% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **37.85 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnLayerNormBackward_LayerNormGradV3_LayerNormGradV3`。

GPU Eager 对账项：`native_layer_norm_backward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_7` | 12.0 | 1.208050 | 0.51% |
| `triton_red_fused_add_native_layer_norm_backward_view_8` | 12.0 | 0.823674 | 0.35% |
| `triton_per_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_view_2` | 12.0 | 0.608385 | 0.26% |
| `triton_per_fused_native_dropout_backward_native_layer_norm_backward_2` | 12.0 | 0.553570 | 0.24% |
| `triton_red_fused_native_layer_norm_backward_0` | 12.0 | 0.391793 | 0.17% |
| `triton_per_fused_native_layer_norm_backward_1` | 108.0 | 0.275899 | 0.12% |
| `native_layer_norm_backward` | 2.0 | 0.058415 | 0.02% |

**贡献粗估：0.488055 ms/step，相当于当前 NPU Inductor 整步耗时的 0.17%，使整网提升比增加 0.193 个百分点，单独达到该预算后整网提升比约为 -15.52%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_per_fused_native_layer_norm_backward_0`：NPU_g03_output_code.py:66；输入 `f32[4, 1024, 1024][1048576, 1024, 1]npu:0`；size_hints=`{'x0': 1024, 'r1': 4096}`；split_axis=`[0]`。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused_native_dropout_backward_sum_v_2` | 10.030561 / 48.0 | 3.44% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_per_fused_add_native_layer_norm_backwar_5` | 6.460890 / 12.0 | 2.21% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_per_fused_clone_sum_transpose_view_8` | 3.399104 / 12.0 | 1.16% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |

- `triton_per_fused_native_dropout_backward_sum_v_2`：NPU_g03_output_code.py:255；输入 `f32[4, 1024, 1024][1048576, 1024, 1]npu:0`；size_hints=`{'x0': 1024, 'r1': 4096}`；split_axis=`[0]`。GPU 检查入口：`triton_per_fused_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_7`, `triton_red_fused_native_dropout_backward_sum_view_3`, `triton_per_fused_native_dropout_backward_native_layer_norm_backward_2`。
- `triton_per_fused_add_native_layer_norm_backwar_5`：NPU_g03_output_code.py:504；输入 `f32[4, 1024, 1024][1048576, 1024, 1]npu:0`；size_hints=`{'x0': 1024, 'r1': 4096}`；split_axis=`[0]`。GPU 检查入口：`triton_red_fused__log_softmax__log_softmax_backward_data_add_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_view_0`, `triton_per_fused_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_7`, `triton_red_fused_add_native_layer_norm_backward_view_8`。
- `triton_per_fused_clone_sum_transpose_view_8`：NPU_g03_output_code.py:770；输入 `f32[4, 16, 1024, 64][1048576, 65536, 64, 1]npu:0`；size_hints=`{'x4': 16, 'x5': 64, 'r2': 4, 'r3': 1024}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_red_fused_native_dropout_backward_sum_view_3`, `triton_red_fused_gelu_gelu_backward_sum_view_5`, `triton_per_fused_gelu_gelu_backward_sum_view_6`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
