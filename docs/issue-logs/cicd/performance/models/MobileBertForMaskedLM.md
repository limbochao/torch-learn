---
title: "CICD 性能分析：MobileBertForMaskedLM"
---

# MobileBertForMaskedLM：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

2026-11-03 更新：[Train 参数梯度归约专项分析](../mobilebert-parameter-reduction-20261103.md)已在 NPU_A5 物理卡 2、`pta_213_cicd`、`test/cicd_benchmark` 对应安装版本复测，确认 IR 分段归约策略与 tiling 候选漏搜；下方原整网统计保持原批次口径。

## Infer

Batch=128；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 71.405178 | 65.363471 | 8.46% |
| GPU | 84.108132 | 54.636697 | 35.04% |

达到 GPU **整网**提升比时，NPU 整步目标为 46.384850 ms；当前相对该目标的差额为 18.978621 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/MobileBertForMaskedLM_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/MobileBertForMaskedLM_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/MobileBertForMaskedLM_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/MobileBertForMaskedLM_Infer/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 额外具体检查点

`triton_red_fused__log_softmax_12` 的 4.669 ms 是具体候选，但 GPU 的 prepare_softmax_online 已耗时约 6.105 ms，不能看到 NPU 比自身 Eager 慢就认定 GPU 有更高收益；该局部保守筛选未通过，暂不承诺优化预算。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_red_fused__log_softmax_12` | 4.668885 / 1.0 | 7.14% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_poi_fused_addmm_6` | 2.120228 / 96.0 | 3.24% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_add_addmm_mul_view_7` | 0.933994 / 96.0 | 1.43% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_red_fused__log_softmax_12`：NPU_g01_output_code.py:877；输入 `f32[16384, 30522][30522, 1]npu:0`；size_hints=`{'x0': 16384, 'r1': 30522}`；split_axis=`[0]`。GPU 检查入口：`triton_red_fused__log_softmax__unsafe_view_add_nll_loss_forward_view_14`, `triton_per_fused__log_softmax__unsafe_view_add_nll_loss_forward_view_15`。
- `triton_poi_fused_addmm_6`：NPU_g01_output_code.py:476；输入 `f32[512][1]npu:0`；size_hints=`{'y0': 16384, 'x1': 512}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_relu_view_8`, `triton_poi_fused_add_addmm_mul_view_9`, `triton_poi_fused_add_addmm_mul_view_10`。
- `triton_poi_fused_add_addmm_mul_view_7`：NPU_g01_output_code.py:542；输入 `f32[128][1]npu:0`；size_hints=`{'y0': 16384, 'x1': 128}`；split_axis=`[0]`。GPU 检查入口：`triton_red_fused__unsafe_view_add_prepare_softmax_online_13`, `triton_poi_fused_addmm_relu_view_8`, `triton_per_fused__softmax_abs_all_amax_div_eq_mul_ne_sub_view_5`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=128；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 221.888467 | 295.193592 | -33.04% |
| GPU | 268.925023 | 215.610848 | 19.82% |

达到 GPU **整网**提升比时，NPU 整步目标为 177.899253 ms；当前相对该目标的差额为 117.294340 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/MobileBertForMaskedLM_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/MobileBertForMaskedLM_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/MobileBertForMaskedLM_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/MobileBertForMaskedLM_Train/GPU_wrapper_calls.txt)。

### 1. add/mul/sum 参数梯度归约

**优化入口：NPU Inductor `ir.py::num_splits` 的 outer reduction 分段策略，以及 runtime `TileGenerator` 候选生成。** 2026-11-03 已对最大单项 `_11` 实测：未分段的 16384-token 归约为 246.390 μs/次；候选只覆盖窄通道 tile。显式两阶段 ATen 自动 codegen 为 22.544 μs/次。详见[专项定位](../mobilebert-parameter-reduction-20261103.md)；下表仍保留七组 kernel 的原操作族粗筛口径。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_add_sum_view_0` | 1.0 | 2.585491 | 0.88% |
| `triton_red_fused_mul_sum_view_5` | 1.0 | 0.086612 | 0.03% |
| `triton_red_fused_mul_sum_view_7` | 48.0 | 9.252472 | 3.13% |
| `triton_red_fused_mul_sum_view_9` | 168.0 | 15.300343 | 5.18% |
| `triton_red_fused_add_mul_sum_view_11` | 96.0 | 25.847447 | 8.76% |
| `triton_red_fused_add_mul_sum_view_13` | 24.0 | 8.921167 | 3.02% |
| `triton_red_fused_add_mul_sum_view_20` | 24.0 | 5.468952 | 1.85% |

本项 kernel/区间合计 **67.462484 ms，占整步 22.85%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 28.757175 | 67.462484 | ≤-134.59% |
| GPU（Eager 下界 / compiled 上界） | 38.384838 | 44.291971 | ≥-15.39% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **119.20 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnAdd_AddAiCore_Add`, `aclnnMul_MulAiCore_Mul`, `aclnnReduceSum_ReduceSumOpAiCore_ReduceSum`, `aclnnAdd_AxpyAiCore_Axpy`。

GPU Eager 对账项：`sum`, `mul`, `add`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_add_mul_sum_view_19` | 23.0 | 6.103080 | 2.83% |
| `triton_poi_fused__log_softmax__log_softmax_backward_data__unsafe_view_add_arange_eq_expand_mm_nll_loss_backward_nll_loss_forward_scalar_tensor_view_2` | 1.0 | 6.071003 | 2.82% |
| `triton_red_fused__log_softmax__log_softmax_backward_data__unsafe_view_add_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_view_0` | 1.0 | 6.023934 | 2.79% |
| `triton_poi_fused_add_addmm_mul_view_11` | 24.0 | 4.312604 | 2.00% |
| `triton_red_fused_nll_loss_forward_sum_threshold_backward_view_10` | 96.0 | 3.573482 | 1.66% |
| `triton_red_fused__log_softmax__log_softmax_backward_data__unsafe_view_add_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_sum_view_1` | 1.0 | 2.686653 | 1.25% |
| `triton_red_fused__log_softmax__unsafe_view_add_prepare_softmax_online_14` | 1.0 | 2.642253 | 1.23% |
| `triton_per_fused__softmax_abs_all_amax_div_eq_mul_native_dropout_ne_sub_view_6` | 23.0 | 2.406298 | 1.12% |
| `triton_per_fused__softmax__softmax_backward_data_div_mul_native_dropout_backward_sub_view_14` | 24.0 | 2.156044 | 1.00% |
| `triton_red_fused_add_mul_sum_view_11` | 72.0 | 1.727827 | 0.80% |
| `triton_poi_fused_add_addmm_mul_view_10` | 96.0 | 1.084955 | 0.50% |
| `triton_red_fused_add_mul_sum_view_12` | 24.0 | 1.076891 | 0.50% |
| `triton_red_fused_sum_view_9` | 576.0 | 0.802718 | 0.37% |
| `triton_red_fused_add_embedding_dense_backward_mul_sum_view_21` | 1.0 | 0.693198 | 0.32% |
| `triton_poi_fused_add_mul_view_8` | 24.0 | 0.611388 | 0.28% |
| `triton_red_fused_add_mul_sum_view_18` | 24.0 | 0.563310 | 0.26% |
| `triton_red_fused_mul_sum_view_8` | 24.0 | 0.526735 | 0.24% |
| `triton_red_fused_sum_16` | 72.0 | 0.285010 | 0.13% |
| `triton_red_fused_native_layer_norm_native_layer_norm_backward_nll_loss_forward_relu_sum_threshold_backward_view_6` | 172.0 | 0.254174 | 0.12% |
| `triton_poi_fused_add_mul_view_4` | 24.0 | 0.217454 | 0.10% |
| `triton_poi_fused_add_addmm_embedding_mul_slice_view_2` | 1.0 | 0.118438 | 0.05% |
| `triton_red_fused_mul_sum_view_7` | 1.0 | 0.105073 | 0.05% |
| `triton_per_fused__softmax_abs_all_amax_div_eq_mul_native_dropout_ne_sub_view_12` | 1.0 | 0.086150 | 0.04% |
| `triton_poi_fused_add_slice_t_26` | 1.0 | 0.059313 | 0.03% |
| `triton_poi_fused_add_constant_pad_nd_embedding_dense_backward_slice_slice_backward_view_25` | 1.0 | 0.048473 | 0.02% |
| `triton_red_fused__log_softmax__unsafe_view_add_nll_loss_forward_view_15` | 1.0 | 0.012444 | 0.01% |
| `sum` | 2.0 | 0.012319 | 0.01% |
| `triton_red_fused_native_layer_norm_native_layer_norm_backward_nll_loss_forward_relu_sum_threshold_backward_view_5` | 1.0 | 0.011366 | 0.01% |
| `triton_red_fused_embedding_dense_backward_slice_sum_23` | 1.0 | 0.010930 | 0.01% |
| `triton_poi_fused_add_constant_pad_nd_embedding_dense_backward_slice_slice_backward_view_24` | 1.0 | 0.007267 | 0.00% |
| `triton_per_fused__log_softmax__unsafe_view_add_nll_loss_forward_view_16` | 1.0 | 0.001187 | 0.00% |

**贡献粗估：34.279800 ms/step，相当于当前 NPU Inductor 整步耗时的 11.61%，使整网提升比增加 15.449 个百分点，单独达到该预算后整网提升比约为 -17.59%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused_add_sum_view_0`：NPU_g01_output_code.py:60；输入 `f32[128, 128, 30522][3906816, 30522, 1]npu:0`；size_hints=`{'x0': 30522, 'r1': 16384}`；split_axis=`[0]`。
- `triton_red_fused_mul_sum_view_5`：NPU_g01_output_code.py:468；输入 `f32[16384, 512][512, 1]npu:0`；size_hints=`{'x0': 512, 'r1': 16384}`；split_axis=`[0]`。
- `triton_red_fused_mul_sum_view_7`：NPU_g01_output_code.py:594；输入 `f32[16384, 128][128, 1]npu:0`；size_hints=`{'x0': 128, 'r1': 16384}`；split_axis=`[0]`。

### 2. prims.fma metadata 重复执行第一组 Mul/Add

**优化入口：PyTorch Inductor inductor_prims.make_prim.meta 与 prims._prim_impl；NPU prims.fma fallback 暴露真实执行。** softmax backward 后连续 Mul/Add/Mul/Add 中第一对来自 metadata 执行真实 impl_aten，随后真实实现再次执行。第一对结果不参与最终输出；GPU 生成 kernel 内执行有效公式，没有 metadata 设备调用。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnMul_MulAiCore_Mul` | 24.0 | 0.515588 | 0.17% |
| `aclnnAdd_AddAiCore_Add` | 24.0 | 0.546524 | 0.19% |

本项 kernel/区间合计 **1.062112 ms，占整步 0.36%**。

仅计算这些相邻窗口中的第一对，不使用整网 Add/Mul 汇总。预算为已证实重复调用的当前设备耗时；修复后整网时间仍须实测。另一个有效 Mul/Add 对不是可全额删除的工作。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused__softmax__softmax_backward_data_div_mul_native_dropout_backward_sub_view_14` | 24.0 | 2.156044 | 1.00% |

**贡献粗估：1.062112 ms/step，相当于当前 NPU Inductor 整步耗时的 0.36%，使整网提升比增加 0.479 个百分点，单独达到该预算后整网提升比约为 -32.56%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnMul_MulAiCore_Mul`：外部调用；见 wrapper 调用边界。
- `aclnnAdd_AddAiCore_Add`：外部调用；见 wrapper 调用边界。

历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。

### 额外具体检查点

最大单项 `triton_red_fused_add_mul_sum_view_11`（25.847 ms，96 次/step）已在 2026-11-03 深挖：主要问题为 NPU Inductor `ir.py::num_splits` 未拆分长 outer reduction，以及 `TileGenerator` 遗漏宽通道 tile。原实现设备耗时 246.390 μs/次；同 DSL 固定 tile 为 103.624 μs，显式两阶段 ATen 自动 codegen 为 22.544 μs，两阶段 Triton 诊断版为 13.211 μs。源码、四方保守筛选与收益外推见[专项分析](../mobilebert-parameter-reduction-20261103.md)；没有修改框架或验证整网修复。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused__softmax_all_amax_div_isfinit_8` | 11.544797 / 23.0 | 3.91% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_red_fused__log_softmax_18` | 4.683339 / 1.0 | 1.59% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad` | 4.229120 / 1.0 | 1.43% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_per_fused__softmax_all_amax_div_isfinit_8`：NPU_g02_output_code.py:598；输入 `f32[512, 128, 128][16384, 128, 1]npu:0`；size_hints=`{'z0': 128, 'y1': 4, 'x2': 128, 'r3': 128}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_per_fused__softmax_abs_all_amax_div_eq_mul_native_dropout_ne_sub_view_6`, `triton_per_fused__softmax__softmax_backward_data_div_mul_native_dropout_backward_sub_view_14`, `triton_per_fused__softmax_abs_all_amax_div_eq_mul_native_dropout_ne_sub_view_12`。
- `triton_red_fused__log_softmax_18`：NPU_g02_output_code.py:1383；输入 `f32[16384, 30522][30522, 1]npu:0`；size_hints=`{'x0': 16384, 'r1': 30522}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused__log_softmax__log_softmax_backward_data__unsafe_view_add_arange_eq_expand_mm_nll_loss_backward_nll_loss_forward_scalar_tensor_view_2`, `triton_red_fused__log_softmax__log_softmax_backward_data__unsafe_view_add_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_view_0`, `triton_red_fused__log_softmax__log_softmax_backward_data__unsafe_view_add_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_sum_view_1`。
- `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad`：外部调用；见 wrapper 调用边界。GPU 检查入口：需对照外部 ACLNN/CUDA 调用和 FX 图。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
