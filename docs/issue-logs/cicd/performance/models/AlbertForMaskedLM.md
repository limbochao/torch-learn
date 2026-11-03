---
title: "CICD 性能分析：AlbertForMaskedLM"
---

# AlbertForMaskedLM：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=4；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 488.357897 | 462.793387 | 5.23% |
| GPU | 392.682080 | 367.770261 | 6.34% |

达到 GPU **整网**提升比时，NPU 整步目标为 457.376387 ms；当前相对该目标的差额为 5.417000 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/AlbertForMaskedLM_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/AlbertForMaskedLM_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/AlbertForMaskedLM_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/AlbertForMaskedLM_Infer/GPU_wrapper_calls.txt)。

### 1. log_softmax→NLLLoss 的完整 log probability 写回

**优化入口：NPU Inductor nll_loss_forward decomposition/lowering、producer-consumer fusion。** 外部 NLLLoss 消费 log_softmax 完整矩阵；GPU loss 融合 kernel 可只提取 target logit 和行归约统计。检查 NLLLoss 唯一消费者边界，避免最后一个完整词表循环及中间矩阵写回。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__log_softmax_6` | 1.0 | 0.428867 | 0.09% |
| `aclnnInplaceOne_OnesLikeAiCore_OnesLike` | 1.0 | 0.003004 | 0.00% |
| `aclnnNLLLoss_NLLLossAiCore_NLLLoss` | 1.0 | 0.011206 | 0.00% |

本项 kernel/区间合计 **0.443077 ms，占整步 0.10%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.379549 | 0.443077 | ≤-16.74% |
| GPU（Eager 下界 / compiled 上界） | 0.741934 | 0.346048 | ≥53.36% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **70.10 个百分点**。NPU 为最终 projection 后至 NLLLoss 的逐步精确窗口，包含 shift/clone/copy；GPU Eager 只计 log_softmax+NLLLoss 下界，compiled 包含 prepare 及全部 loss kernel。预算是保守下界。

NPU Eager 对账项：`aclnnInplaceOne_OnesLikeAiCore_OnesLike`, `aclnnLogSoftmax_LogSoftmaxAiCore_LogSoftmaxV2`, `aclnnNLLLoss_NLLLossAiCore_NLLLoss`。

GPU Eager 对账项：`_log_softmax`, `nll_loss_forward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_prepare_softmax_online_view_6` | 1.0 | 0.341971 | 0.09% |
| `triton_red_fused__log_softmax_nll_loss_forward_view_7` | 1.0 | 0.004077 | 0.00% |

**贡献粗估：0.266050 ms/step，相当于当前 NPU Inductor 整步耗时的 0.06%，使整网提升比增加 0.054 个百分点，单独达到该预算后整网提升比约为 5.29%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused__log_softmax_6`：NPU_g01_output_code.py:615；输入 `f32[2048, 30000][30000, 1]npu:0`；size_hints=`{'x0': 2048, 'r1': 30000}`；split_axis=`[0]`。
- `aclnnInplaceOne_OnesLikeAiCore_OnesLike`：外部调用；见 wrapper 调用边界。
- `aclnnNLLLoss_NLLLossAiCore_NLLLoss`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_add_addmm_mul_pow_tanh_view_3` | 1.959433 / 12.0 | 0.42% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_per_fused_add_addmm_native_layer_norm_v_2` | 0.935302 / 12.0 | 0.20% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_per_fused_add_addmm_native_layer_norm_v_4` | 0.896666 / 12.0 | 0.19% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |

- `triton_poi_fused_add_addmm_mul_pow_tanh_view_3`：NPU_g01_output_code.py:336；输入 `f32[16384][1]npu:0`；size_hints=`{'y0': 2048, 'x1': 16384}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_addmm_mul_pow_tanh_view_3`, `triton_red_fused_add_addmm_native_layer_norm_view_2`, `triton_red_fused_add_addmm_native_layer_norm_view_4`。
- `triton_per_fused_add_addmm_native_layer_norm_v_2`：NPU_g01_output_code.py:239；输入 `f32[2048, 4096][4096, 1]npu:0`；size_hints=`{'y0': 4, 'x1': 512, 'r2': 4096}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_add_addmm_mul_pow_tanh_view_3`, `triton_red_fused_add_addmm_native_layer_norm_view_2`, `triton_red_fused_add_addmm_native_layer_norm_view_4`。
- `triton_per_fused_add_addmm_native_layer_norm_v_4`：NPU_g01_output_code.py:415；输入 `f32[4096][1]npu:0`；size_hints=`{'y0': 4, 'x1': 512, 'r2': 4096}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_add_addmm_mul_pow_tanh_view_3`, `triton_red_fused_add_addmm_native_layer_norm_view_2`, `triton_red_fused_add_addmm_native_layer_norm_view_4`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=4；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 1444.716219 | 1594.116173 | -10.34% |
| GPU | 1079.256015 | 990.814690 | 8.19% |

达到 GPU **整网**提升比时，NPU 整步目标为 1326.326685 ms；当前相对该目标的差额为 267.789488 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/AlbertForMaskedLM_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/AlbertForMaskedLM_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/AlbertForMaskedLM_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/AlbertForMaskedLM_Train/GPU_wrapper_calls.txt)。

### 1. 独立 sum 生成归约

**优化入口：NPU Inductor codegen/reduction tiling、runtime autotune。** 选中 kernel 仅有 sum 和 view/transpose。检查输入 stride、归约维大小、split_axis、persistent/分阶段选择，与 GPU 归约拆分对照。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused_sum_view_16` | 2.0 | 24.252617 | 1.52% |

本项 kernel/区间合计 **24.252617 ms，占整步 1.52%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 2.990131 | 24.252617 | ≤-711.09% |
| GPU（Eager 下界 / compiled 上界） | 4.639855 | 8.153063 | ≥-75.72% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **635.37 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnReduceSum_ReduceSumOpAiCore_ReduceSum`。

GPU Eager 对账项：`sum`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_add_mul_pow_sum_tanh_tanh_backward_view_20` | 1.0 | 2.235402 | 0.23% |
| `triton_red_fused_add_native_layer_norm_backward_sum_view_15` | 11.0 | 1.923243 | 0.19% |
| `triton_red_fused_sum_transpose_view_14` | 38.0 | 1.768456 | 0.18% |
| `triton_red_fused_add_native_layer_norm_backward_sum_view_12` | 12.0 | 1.419731 | 0.14% |
| `sum` | 12.0 | 0.507051 | 0.05% |
| `triton_red_fused_sum_4` | 1.0 | 0.289845 | 0.03% |
| `triton_poi_fused_embedding_dense_backward_sum_25` | 1.0 | 0.003507 | 0.00% |
| `triton_per_fused_add_sum_view_22` | 1.0 | 0.002678 | 0.00% |
| `triton_red_fused_add_mul_pow_sum_tanh_tanh_backward_view_8` | 1.0 | 0.001949 | 0.00% |
| `triton_poi_fused_embedding_dense_backward_sum_1` | 1.0 | 0.001200 | 0.00% |

**贡献粗估：18.998416 ms/step，相当于当前 NPU Inductor 整步耗时的 1.19%，使整网提升比增加 1.315 个百分点，单独达到该预算后整网提升比约为 -9.03%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_per_fused_sum_view_16`：NPU_g01_output_code.py:2344；输入 `f32[4, 512, 4096][2097152, 4096, 1]npu:0`；size_hints=`{'x0': 4096, 'r1': 2048}`；split_axis=`[0]`。

### 2. prims.fma metadata 重复执行第一组 Mul/Add

**优化入口：PyTorch Inductor inductor_prims.make_prim.meta 与 prims._prim_impl；NPU prims.fma fallback 暴露真实执行。** softmax backward 后连续 Mul/Add/Mul/Add 中第一对来自 metadata 执行真实 impl_aten，随后真实实现再次执行。第一对结果不参与最终输出；GPU 生成 kernel 内执行有效公式，没有 metadata 设备调用。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnMul_MulAiCore_Mul` | 12.0 | 4.695385 | 0.29% |
| `aclnnAdd_AddAiCore_Add` | 12.0 | 6.740735 | 0.42% |

本项 kernel/区间合计 **11.436120 ms，占整步 0.72%**。

仅计算这些相邻窗口中的第一对，不使用整网 Add/Mul 汇总。预算为已证实重复调用的当前设备耗时；修复后整网时间仍须实测。另一个有效 Mul/Add 对不是可全额删除的工作。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `_efficient_attention_backward` | 12.0 | 51.044632 | 5.15% |

**贡献粗估：11.436120 ms/step，相当于当前 NPU Inductor 整步耗时的 0.72%，使整网提升比增加 0.792 个百分点，单独达到该预算后整网提升比约为 -9.55%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnMul_MulAiCore_Mul`：外部调用；见 wrapper 调用边界。
- `aclnnAdd_AddAiCore_Add`：外部调用；见 wrapper 调用边界。

历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。

### 3. NLLLossBackward→LogSoftmaxBackward 的完整中间矩阵

**优化入口：NPU Inductor nll_loss_backward / _log_softmax_backward_data decomposition 与 lowering。** 两条独立外部调用之间生成完整 dlogP；无类别权重时每个 token 的 dlogP 仅 target 类别非零。GPU 将 loss 导数融合生成，目标是直接计算 dlogits，避免完整中间矩阵；最终 dlogits 仍须生成。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnNLLLossBackward_NLLLossGradAiCore_NLLLossGrad` | 1.0 | 0.190980 | 0.01% |
| `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad` | 1.0 | 0.566380 | 0.04% |

本项 kernel/区间合计 **0.757359 ms，占整步 0.05%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.714119 | 0.757359 | ≤-6.06% |
| GPU（Eager 下界 / compiled 上界） | 1.244755 | 1.120560 | ≥9.98% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **16.03 个百分点**。GPU compiled 上界故意包含全部前向与反向 loss/prepare kernel。需要完整 GPU wrapper 进一步切开前后向后才能给更紧预算。

NPU Eager 对账项：`aclnnNLLLossBackward_NLLLossGradAiCore_NLLLossGrad`, `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad`。

GPU Eager 对账项：`_log_softmax_backward_data`, `nll_loss_backward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__log_softmax__log_softmax_backward_data_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_view_3` | 1.0 | 0.724847 | 0.07% |
| `triton_red_fused__log_softmax_prepare_softmax_online_6` | 1.0 | 0.391176 | 0.04% |
| `triton_red_fused__log_softmax_nll_loss_forward_view_7` | 1.0 | 0.004538 | 0.00% |

**贡献粗估：0.114492 ms/step，相当于当前 NPU Inductor 整步耗时的 0.01%，使整网提升比增加 0.008 个百分点，单独达到该预算后整网提升比约为 -10.33%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnNLLLossBackward_NLLLossGradAiCore_NLLLossGrad`：外部调用；见 wrapper 调用边界。
- `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad`：外部调用；见 wrapper 调用边界。

### 额外具体检查点

额外优先核对 `_20`（90.679 ms）和 `_17`（51.366 ms）。`_20` 的名字含 matmul_backward，实际 DSL 为多个 tl.sum、permute 和累加，并没有执行裸矩阵乘；其 x4/x5=64/64、r2/r3=4/512，外层 x4 使用 Python range。`_17` 同时读取 12 个来源并逐个归约。应检查共享参数梯度累加导致的多归约融合、codegen 轴映射和寄存器压力。GPU 采用多条 sum/transpose/activation 归约。当前没有把这两个 kernel 的完整语义逐调用对齐，不能把 142.044 ms 全部当作可回收收益。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused_clone_matmul_backward_permute_20` | 90.678594 / 2.0 | 5.69% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_per_fused_add_mul_pow_sum_tanh_tanh_bac_17` | 51.365636 / 1.0 | 3.22% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_per_fused__softmax_backward_data_matmul_10` | 9.139415 / 12.0 | 0.57% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |

- `triton_per_fused_clone_matmul_backward_permute_20`：NPU_g01_output_code.py:3967；输入 `f32[256, 512, 64][32768, 64, 1]npu:0`；size_hints=`{'x4': 64, 'x5': 64, 'r2': 4, 'r3': 512}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_red_fused_add_mul_pow_sum_tanh_tanh_backward_view_20`, `triton_red_fused_add_native_layer_norm_backward_sum_view_15`, `triton_red_fused_sum_transpose_view_14`。
- `triton_per_fused_add_mul_pow_sum_tanh_tanh_bac_17`：NPU_g01_output_code.py:3073；输入 `f32[4, 512, 16384][8388608, 16384, 1]npu:0`；size_hints=`{'x0': 16384, 'r1': 2048}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_mul_pow_tanh_tanh_backward_view_11`, `triton_poi_fused_add_mul_pow_tanh_view_3`, `triton_red_fused_add_native_layer_norm_backward_view_17`。
- `triton_per_fused__softmax_backward_data_matmul_10`：NPU_g01_output_code.py:1042；输入 `f32[256, 512, 512][262144, 512, 1]npu:0`；size_hints=`{'z0': 4, 'y1': 64, 'x2': 512, 'r3': 512}`；split_axis=`[0, 1]`。GPU 检查入口：需对照外部 ACLNN/CUDA 调用和 FX 图。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
