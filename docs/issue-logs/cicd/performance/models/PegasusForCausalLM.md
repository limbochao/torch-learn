---
title: "CICD 性能分析：PegasusForCausalLM"
---

# PegasusForCausalLM：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=32；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 78.613746 | 80.057862 | -1.84% |
| GPU | 72.795821 | 71.503588 | 1.78% |

达到 GPU **整网**提升比时，NPU 整步目标为 77.218236 ms；当前相对该目标的差额为 2.839626 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/PegasusForCausalLM_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/PegasusForCausalLM_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/PegasusForCausalLM_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/PegasusForCausalLM_Infer/GPU_wrapper_calls.txt)。

### 1. log_softmax→NLLLoss 的完整 log probability 写回

**优化入口：NPU Inductor nll_loss_forward decomposition/lowering、producer-consumer fusion。** 外部 NLLLoss 消费 log_softmax 完整矩阵；GPU loss 融合 kernel 可只提取 target logit 和行归约统计。检查 NLLLoss 唯一消费者边界，避免最后一个完整词表循环及中间矩阵写回。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__log_softmax_10` | 1.0 | 1.374713 | 1.72% |
| `aclnnInplaceOne_OnesLikeAiCore_OnesLike` | 1.0 | 0.003213 | 0.00% |
| `aclnnNLLLoss_NLLLossAiCore_NLLLoss` | 1.0 | 0.012870 | 0.02% |

本项 kernel/区间合计 **1.390795 ms，占整步 1.74%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 1.166489 | 1.390795 | ≤-19.23% |
| GPU（Eager 下界 / compiled 上界） | 2.513043 | 1.250230 | ≥50.25% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **69.48 个百分点**。NPU 为最终 projection 后至 NLLLoss 的逐步精确窗口，包含 shift/clone/copy；GPU Eager 只计 log_softmax+NLLLoss 下界，compiled 包含 prepare 及全部 loss kernel。预算是保守下界。

NPU Eager 对账项：`aclnnLogSoftmax_LogSoftmaxAiCore_LogSoftmaxV2`, `aclnnInplaceOne_OnesLikeAiCore_OnesLike`, `aclnnNLLLoss_NLLLossAiCore_NLLLoss`。

GPU Eager 对账项：`_log_softmax`, `nll_loss_forward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__unsafe_view_prepare_softmax_online_view_7` | 1.0 | 1.243404 | 1.74% |
| `triton_red_fused__log_softmax__unsafe_view_nll_loss_forward_view_8` | 1.0 | 0.006826 | 0.01% |

**贡献粗估：0.810471 ms/step，相当于当前 NPU Inductor 整步耗时的 1.01%，使整网提升比增加 1.031 个百分点，单独达到该预算后整网提升比约为 -0.81%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused__log_softmax_10`：NPU_g01_output_code.py:935；输入 `f32[4096, 50265][50265, 1]npu:0`；size_hints=`{'x0': 4096, 'r1': 50265}`；split_axis=`[0]`。
- `aclnnInplaceOne_OnesLikeAiCore_OnesLike`：外部调用；见 wrapper 调用边界。
- `aclnnNLLLoss_NLLLossAiCore_NLLLoss`：外部调用；见 wrapper 调用边界。

### 2. GELU 前向外部调用边界

**优化入口：NPU Inductor decomposition/lowering、scheduler 的 bias/GELU 融合。** NPU 保留 ACLNN GELU；GPU 将 GELU 与相邻 bias/pointwise 合入生成 kernel。优化具体外部调用边界，而非仅调整 ACLNN GELU。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnGeluV2_GeluV2_GeluV2` | 12.0 | 1.176677 | 1.47% |

本项 kernel/区间合计 **1.176677 ms，占整步 1.47%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 1.190606 | 1.176677 | ≤1.17% |
| GPU（Eager 下界 / compiled 上界） | 2.011944 | 1.916329 | ≥4.75% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **3.58 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnGeluV2_GeluV2_GeluV2`。

GPU Eager 对账项：`gelu`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_addmm_gelu_view_3` | 12.0 | 1.916329 | 2.68% |

**贡献粗估：0.042653 ms/step，相当于当前 NPU Inductor 整步耗时的 0.05%，使整网提升比增加 0.054 个百分点，单独达到该预算后整网提升比约为 -1.78%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnGeluV2_GeluV2_GeluV2`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused__safe_softmax_add_arange_expa_3` | 1.502975 / 12.0 | 1.88% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_poi_fused_addmm_6` | 0.575942 / 12.0 | 0.72% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_per_fused_add_addmm_native_layer_norm_v_8` | 0.563136 / 11.0 | 0.70% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |

- `triton_per_fused__safe_softmax_add_arange_expa_3`：NPU_g01_output_code.py:332；输入 `f32[512, 128, 128][16384, 128, 1]npu:0`；size_hints=`{'z0': 32, 'y1': 16, 'x2': 128, 'r3': 128}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_per_fused_add_addmm_native_layer_norm_view_5`, `triton_per_fused_add_addmm_native_layer_norm_view_4`, `triton_per_fused_add_addmm_native_layer_norm_view_6`。
- `triton_poi_fused_addmm_6`：NPU_g01_output_code.py:589；输入 `f32[4096][1]npu:0`；size_hints=`{'y0': 4096, 'x1': 4096}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_3`, `triton_per_fused_add_addmm_native_layer_norm_view_5`, `triton_per_fused_add_addmm_native_layer_norm_view_4`。
- `triton_per_fused_add_addmm_native_layer_norm_v_8`：NPU_g01_output_code.py:754；输入 `f32[32, 128, 1024][131072, 1024, 1]npu:0`；size_hints=`{'y0': 32, 'x1': 128, 'r2': 1024}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_3`, `triton_per_fused_add_addmm_native_layer_norm_view_5`, `triton_per_fused_add_addmm_native_layer_norm_view_4`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=32；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 239.297278 | 282.669740 | -18.12% |
| GPU | 188.104584 | 189.466602 | -0.72% |

达到 GPU **整网**提升比时，NPU 整步目标为 241.029969 ms；当前相对该目标的差额为 41.639771 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/PegasusForCausalLM_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/PegasusForCausalLM_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/PegasusForCausalLM_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/PegasusForCausalLM_Train/GPU_wrapper_calls.txt)。

### 1. GELU backward 与 sum 梯度归约

**优化入口：NPU Inductor scheduler、codegen/reduction tiling。** 具体检查融合 gelu_backward 后的 bias sum；GPU 相应融合/归约成本全部计入上界。优化输出维分块、归约维切分和融合取舍。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused_gelu_backward_sum_view_3` | 12.0 | 13.106795 | 4.64% |

本项 kernel/区间合计 **13.106795 ms，占整步 4.64%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 2.825173 | 13.106795 | ≤-363.93% |
| GPU（Eager 下界 / compiled 上界） | 4.518283 | 4.261875 | ≥5.67% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **369.60 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnReduceSum_ReduceSumOpAiCore_ReduceSum`, `aclnnGeluBackwardV2_GeluGradV2AiCore_GeluGradV2`。

GPU Eager 对账项：`gelu_backward`, `sum`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_gelu_gelu_backward_view_2` | 12.0 | 2.689942 | 1.42% |
| `triton_red_fused_native_dropout_backward_sum_view_7` | 48.0 | 0.506108 | 0.27% |
| `triton_per_fused_native_dropout_backward_sum_view_1` | 108.0 | 0.295077 | 0.16% |
| `sum` | 12.0 | 0.276978 | 0.15% |
| `triton_red_fused_gelu_gelu_backward_sum_view_3` | 12.0 | 0.262353 | 0.14% |
| `triton_red_fused_native_dropout_backward_sum_view_0` | 12.0 | 0.195570 | 0.10% |
| `triton_per_fused_gelu_gelu_backward_sum_view_4` | 12.0 | 0.035847 | 0.02% |

**贡献粗估：10.441948 ms/step，相当于当前 NPU Inductor 整步耗时的 3.69%，使整网提升比增加 4.364 个百分点，单独达到该预算后整网提升比约为 -13.76%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_per_fused_gelu_backward_sum_view_3`：NPU_g03_output_code.py:309；输入 `f32[32, 128, 4096][524288, 4096, 1]npu:0`；size_hints=`{'x0': 4096, 'r1': 4096}`；split_axis=`[0]`。

### 2. LayerNorm backward 归约调度

**优化入口：NPU Inductor codegen/reduction tiling 与 runtime autotune。** 检查该 kernel 的 reduction size_hints、split_axis 和 program 并行度；GPU 对应归约 kernel 列于本项。优先检查归约维切分与多阶段归约，不能据此断言 Triton_ascend/npuir 有缺陷。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused_native_layer_norm_backward_vi_4` | 12.0 | 4.546475 | 1.61% |

本项 kernel/区间合计 **4.546475 ms，占整步 1.61%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 1.390882 | 4.546475 | ≤-226.88% |
| GPU（Eager 下界 / compiled 上界） | 1.732816 | 3.561665 | ≥-105.54% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **121.34 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnLayerNormBackward_LayerNormGradV3_LayerNormGradV3`。

GPU Eager 对账项：`native_layer_norm_backward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_5` | 12.0 | 1.220230 | 0.64% |
| `triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8` | 12.0 | 1.040158 | 0.55% |
| `triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_9` | 12.0 | 0.947092 | 0.50% |
| `triton_red_fused_native_layer_norm_backward_view_6` | 12.0 | 0.297947 | 0.16% |
| `native_layer_norm_backward` | 2.0 | 0.056238 | 0.03% |

**贡献粗估：1.687628 ms/step，相当于当前 NPU Inductor 整步耗时的 0.60%，使整网提升比增加 0.705 个百分点，单独达到该预算后整网提升比约为 -17.42%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_per_fused_native_layer_norm_backward_vi_4`：NPU_g03_output_code.py:375；输入 `f32[4096, 1024][1024, 1]npu:0`；size_hints=`{'x0': 1024, 'r1': 4096}`；split_axis=`[0]`。

### 3. prims.fma metadata 重复执行第一组 Mul/Add

**优化入口：PyTorch Inductor inductor_prims.make_prim.meta 与 prims._prim_impl；NPU prims.fma fallback 暴露真实执行。** softmax backward 后连续 Mul/Add/Mul/Add 中第一对来自 metadata 执行真实 impl_aten，随后真实实现再次执行。第一对结果不参与最终输出；GPU 生成 kernel 内执行有效公式，没有 metadata 设备调用。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnMul_MulAiCore_Mul` | 12.0 | 0.254824 | 0.09% |
| `aclnnAdd_AddAiCore_Add` | 12.0 | 0.315787 | 0.11% |

本项 kernel/区间合计 **0.570610 ms，占整步 0.20%**。

仅计算这些相邻窗口中的第一对，不使用整网 Add/Mul 汇总。预算为已证实重复调用的当前设备耗时；修复后整网时间仍须实测。另一个有效 Mul/Add 对不是可全额删除的工作。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `_efficient_attention_backward` | 12.0 | 6.755140 | 3.57% |

**贡献粗估：0.570610 ms/step，相当于当前 NPU Inductor 整步耗时的 0.20%，使整网提升比增加 0.238 个百分点，单独达到该预算后整网提升比约为 -17.89%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnMul_MulAiCore_Mul`：外部调用；见 wrapper 调用边界。
- `aclnnAdd_AddAiCore_Add`：外部调用；见 wrapper 调用边界。

历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused_native_dropout_backward_sum_v_1` | 9.999137 / 48.0 | 3.54% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_per_fused_add_native_layer_norm_backwar_12` | 8.003458 / 12.0 | 2.83% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_per_fused_clone_sum_transpose_view_10` | 3.325722 / 12.0 | 1.18% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |

- `triton_per_fused_native_dropout_backward_sum_v_1`：NPU_g03_output_code.py:129；输入 `f32[32, 128, 1024][131072, 1024, 1]npu:0`；size_hints=`{'x0': 1024, 'r1': 4096}`；split_axis=`[0]`。GPU 检查入口：`triton_per_fused_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_5`, `triton_red_fused_native_dropout_backward_sum_view_7`, `triton_per_fused_native_dropout_backward_sum_view_1`。
- `triton_per_fused_add_native_layer_norm_backwar_12`：NPU_g03_output_code.py:1065；输入 `f32[4096, 1024][1024, 1]npu:0`；size_hints=`{'x0': 1024, 'r1': 4096}`；split_axis=`[0]`。GPU 检查入口：`triton_red_fused__log_softmax__log_softmax_backward_data_add_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_view_0`, `triton_per_fused_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_5`, `triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8`。
- `triton_per_fused_clone_sum_transpose_view_10`：NPU_g03_output_code.py:850；输入 `f32[512, 128, 64][8192, 64, 1]npu:0`；size_hints=`{'x4': 16, 'x5': 64, 'r2': 32, 'r3': 128}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_red_fused_native_dropout_backward_sum_view_7`, `triton_per_fused_native_dropout_backward_sum_view_1`, `triton_red_fused_gelu_gelu_backward_sum_view_3`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
