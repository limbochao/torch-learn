---
title: "CICD 性能分析：torch_multimodal_clip"
---

# torch_multimodal_clip：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=32；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 26.171212 | 29.034701 | -10.94% |
| GPU | 21.072679 | 19.310255 | 8.36% |

达到 GPU **整网**提升比时，NPU 整步目标为 23.982369 ms；当前相对该目标的差额为 5.052332 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/torch_multimodal_clip_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/torch_multimodal_clip_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/torch_multimodal_clip_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/torch_multimodal_clip_Infer/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 额外具体检查点

额外具体点为 `aclnnBatchMatMul_BroadcastToAiCore_BroadcastTo`（约 2.231 ms）。NPU wrapper 的 QKV projection 为 bmm，右矩阵 (50,768,2304)、stride=(0,1,768)；GPU 把同一投影展平为 mm((1600,768),(768,2304))。NPU Eager 没有该 BroadcastTo kernel。优先检查 NPU matmul→bmm lowering/布局选择以及 CANN 对 stride=0 batch 的展开；这是插入的广播开销，若可消除，该项成本就是回收上限，需逐次 profiler 关联后确认，不能连同 BMM 全部算收益。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_addmm_mul_sigmoid_view_8` | 0.272016 / 12.0 | 0.94% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_addmm_mul_sigmoid_view_19` | 0.200588 / 12.0 | 0.69% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_per_fused_add_addmm_clone_native_layer__9` | 0.181104 / 11.0 | 0.62% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |

- `triton_poi_fused_addmm_mul_sigmoid_view_8`：NPU_g01_output_code.py:661；输入 `f32[3072][1]npu:0`；size_hints=`{'y0': 1600, 'x1': 3072}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_mul_sigmoid_view_17`, `triton_poi_fused_addmm_mul_sigmoid_view_8`, `triton_per_fused_add_addmm_clone_native_layer_norm_transpose_view_9`。
- `triton_poi_fused_addmm_mul_sigmoid_view_19`：NPU_g01_output_code.py:1587；输入 `f32[2048][1]npu:0`；size_hints=`{'y0': 2464, 'x1': 2048}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_mul_sigmoid_view_17`, `triton_poi_fused_addmm_mul_sigmoid_view_8`, `triton_per_fused_add_addmm_clone_native_layer_norm_transpose_view_9`。
- `triton_per_fused_add_addmm_clone_native_layer__9`：NPU_g01_output_code.py:743；输入 `f32[32, 50, 768][38400, 768, 1]npu:0`；size_hints=`{'y0': 32, 'x1': 50, 'r2': 768}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_addmm_mul_sigmoid_view_17`, `triton_poi_fused_addmm_mul_sigmoid_view_8`, `triton_per_fused_add_addmm_clone_native_layer_norm_transpose_view_9`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=32；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 79.708819 | 84.737356 | -6.31% |
| GPU | 67.406561 | 60.213061 | 10.67% |

达到 GPU **整网**提升比时，NPU 整步目标为 71.202446 ms；当前相对该目标的差额为 13.534910 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/torch_multimodal_clip_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/torch_multimodal_clip_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/torch_multimodal_clip_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/torch_multimodal_clip_Train/GPU_wrapper_calls.txt)。

### 1. LayerNorm backward 归约调度

**优化入口：NPU Inductor codegen/reduction tiling 与 runtime autotune。** 检查该 kernel 的 reduction size_hints、split_axis 和 program 并行度；GPU 对应归约 kernel 列于本项。优先检查归约维切分与多阶段归约，不能据此断言 Triton_ascend/npuir 有缺陷。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused_native_layer_norm_backward_3` | 1.0 | 0.013417 | 0.02% |
| `triton_per_fused_native_layer_norm_backward_4` | 1.0 | 0.152786 | 0.18% |
| `triton_per_fused_native_layer_norm_backward_vi_9` | 24.0 | 3.321622 | 3.92% |
| `triton_per_fused_native_layer_norm_backward_16` | 1.0 | 0.004475 | 0.01% |
| `triton_per_fused_native_layer_norm_backward_17` | 1.0 | 0.004759 | 0.01% |
| `triton_per_fused_native_layer_norm_backward_vi_23` | 13.0 | 0.371392 | 0.44% |

本项 kernel/区间合计 **3.868450 ms，占整步 4.57%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 1.405912 | 3.868450 | ≤-175.16% |
| GPU（Eager 下界 / compiled 上界） | 1.142528 | 2.077402 | ≥-81.82% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **93.33 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnLayerNormBackward_TransposeAiCore_Transpose`, `aclnnLayerNormBackward_LayerNormGradV3_LayerNormGradV3`, `aclnnLayerNormBackward_SliceAiCore_Slice`。

GPU Eager 对账项：`native_layer_norm_backward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_view_16` | 22.0 | 0.326541 | 0.54% |
| `triton_per_fused_add_addmm_clone_native_layer_norm_native_layer_norm_backward_transpose_view_5` | 11.0 | 0.317964 | 0.53% |
| `triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_14` | 8.0 | 0.239690 | 0.40% |
| `triton_red_fused_add_native_layer_norm_backward_transpose_view_36` | 11.0 | 0.162260 | 0.27% |
| `triton_per_fused_add_native_layer_norm_backward_view_29` | 12.0 | 0.152052 | 0.25% |
| `triton_red_fused_native_layer_norm_backward_view_12` | 23.0 | 0.150029 | 0.25% |
| `triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_transpose_view_3` | 12.0 | 0.117683 | 0.20% |
| `triton_per_fused_native_layer_norm_backward_5` | 74.0 | 0.095389 | 0.16% |
| `triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_13` | 8.0 | 0.086965 | 0.14% |
| `triton_red_fused_native_layer_norm_backward_view_30` | 12.0 | 0.066242 | 0.11% |
| `triton_red_fused_native_layer_norm_backward_transpose_view_37` | 12.0 | 0.065883 | 0.11% |
| `triton_per_fused_native_layer_norm_backward_transpose_view_35` | 12.0 | 0.036827 | 0.06% |
| `triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_16` | 1.0 | 0.034958 | 0.06% |
| `triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_permute_view_12` | 1.0 | 0.028766 | 0.05% |
| `triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_18` | 1.0 | 0.028295 | 0.05% |
| `triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_permute_view_21` | 1.0 | 0.025694 | 0.04% |
| `triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_permute_view_11` | 1.0 | 0.017581 | 0.03% |
| `triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_17` | 2.0 | 0.017572 | 0.03% |
| `triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_permute_view_10` | 1.0 | 0.017306 | 0.03% |
| `triton_red_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_38` | 1.0 | 0.016387 | 0.03% |
| `triton_per_fused_add_native_layer_norm_backward_permute_view_17` | 1.0 | 0.016269 | 0.03% |
| `triton_red_fused_native_layer_norm_backward_4` | 1.0 | 0.010845 | 0.02% |
| `triton_per_fused_native_layer_norm_backward_3` | 1.0 | 0.010301 | 0.02% |
| `triton_red_fused_add_native_layer_norm_backward_permute_view_18` | 1.0 | 0.008436 | 0.01% |
| `triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_15` | 1.0 | 0.007364 | 0.01% |
| `triton_red_fused_add_native_layer_norm_native_layer_norm_backward_39` | 1.0 | 0.006663 | 0.01% |
| `triton_poi_fused_native_layer_norm_backward_select_backward_23` | 1.0 | 0.003850 | 0.01% |
| `triton_poi_fused__unsafe_view_clone_native_layer_norm_backward_permute_6` | 1.0 | 0.003725 | 0.01% |
| `triton_per_fused_native_layer_norm_backward_21` | 1.0 | 0.002109 | 0.00% |
| `triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_select_transpose_view_24` | 1.0 | 0.002080 | 0.00% |
| `triton_per_fused_native_layer_norm_backward_22` | 1.0 | 0.001680 | 0.00% |

**贡献粗估：1.312151 ms/step，相当于当前 NPU Inductor 整步耗时的 1.55%，使整网提升比增加 1.646 个百分点，单独达到该预算后整网提升比约为 -4.66%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_per_fused_native_layer_norm_backward_3`：NPU_g01_output_code.py:309；输入 `f32[32, 77, 512][39424, 512, 1]npu:0`；size_hints=`{'y0': 32, 'x1': 77, 'r2': 512}`；split_axis=`[0, 1]`。
- `triton_per_fused_native_layer_norm_backward_4`：NPU_g01_output_code.py:397；输入 `f32[32, 77, 512][39424, 512, 1]npu:0`；size_hints=`{'x0': 512, 'r1': 2464}`；split_axis=`[0]`。
- `triton_per_fused_native_layer_norm_backward_vi_9`：NPU_g01_output_code.py:857；输入 `f32[2464, 512][512, 1]npu:0`；size_hints=`{'x0': 512, 'r2': 77, 'r1': 32}`；split_axis=`[0]`。

### 额外具体检查点

前向另有 `aclnnBatchMatMul_BroadcastToAiCore_BroadcastTo`（约 2.243 ms）；NPU 使用带 stride=0 权重的 QKV bmm，GPU 使用展平 mm。检查 matmul→bmm lowering/布局与 CANN batch 广播展开，属于额外 materialization 候选。2.243 ms 是消除该额外 kernel 的上限，尚未计入达到 GPU 局部提升比的预算。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused_matmul_backward_native_layer__29` | 0.851202 / 12.0 | 1.00% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_add_mul_sigmoid_sigmoid_backw_6` | 0.536105 / 12.0 | 0.63% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_per_fused__unsafe_view_clone_sum_10` | 0.468832 / 23.0 | 0.55% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |

- `triton_per_fused_matmul_backward_native_layer__29`：NPU_g01_output_code.py:2701；输入 `f32[1600, 768][768, 1]npu:0`；size_hints=`{'x0': 768, 'r2': 32, 'r1': 50}`；split_axis=`[0]`。GPU 检查入口：`triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_view_16`, `triton_per_fused_add_addmm_clone_native_layer_norm_native_layer_norm_backward_transpose_view_5`, `triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_14`。
- `triton_poi_fused_add_mul_sigmoid_sigmoid_backw_6`：NPU_g01_output_code.py:566；输入 `f32[2464, 2048][2048, 1]npu:0`；size_hints=`{'z0': 77, 'y1': 32, 'x2': 2048}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_add_copy__3`, `triton_poi_fused_add_mul_sigmoid_sigmoid_backward_view_26`, `triton_poi_fused_add_mul_sigmoid_sigmoid_backward_view_8`。
- `triton_per_fused__unsafe_view_clone_sum_10`：NPU_g01_output_code.py:932；输入 `f32[77, 32, 512][16384, 512, 1]npu:0`；size_hints=`{'x0': 512, 'r1': 2464}`；split_axis=`[0]`。GPU 检查入口：`triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_view_16`, `triton_per_fused_add_addmm_clone_native_layer_norm_native_layer_norm_backward_transpose_view_5`, `triton_poi_fused_addmm_clone_squeeze_transpose_unsqueeze_view_8`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
