---
title: "CICD 性能分析：basic_gnn_edgecnn"
---

# basic_gnn_edgecnn：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=1；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 1.987049 | 1.364681 | 31.32% |
| GPU | 5.353945 | 2.762886 | 48.40% |

达到 GPU **整网**提升比时，NPU 整步目标为 1.025411 ms；当前相对该目标的差额为 0.339270 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_edgecnn_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_edgecnn_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_edgecnn_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_edgecnn_Infer/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_index_select_relu_select_sub_3` | 0.280373 / 2.0 | 20.54% | NPU Inductor gather/scatter lowering 与 producer-consumer 融合 |
| `aclnnInplaceCopy_BroadcastToAiCore_BroadcastTo` | 0.147457 / 3.0 | 10.81% | NPU Inductor 布局/materialization 与 CANN 广播路径；先定位触发的外部调用 |
| `triton_poi_fused_index_select_select_sub_0` | 0.144060 / 1.0 | 10.56% | NPU Inductor gather/scatter lowering 与 producer-consumer 融合 |

- `triton_poi_fused_index_select_relu_select_sub_3`：NPU_g01_output_code.py:246；输入 `i64[2, 200000][200000, 1]npu:0`；size_hints=`{'y0': 200000, 'x1': 64}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_index_select_relu_select_sub_3`, `triton_poi_fused_addmm_relu_1`, `triton_poi_fused_index_select_select_sub_0`。
- `aclnnInplaceCopy_BroadcastToAiCore_BroadcastTo`：外部调用；见 wrapper 调用边界。GPU 检查入口：需对照外部 ACLNN/CUDA 调用和 FX 图。
- `triton_poi_fused_index_select_select_sub_0`：NPU_g01_output_code.py:64；输入 `i64[2, 200000][200000, 1]npu:0`；size_hints=`{'y0': 200000, 'x1': 64}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_index_select_relu_select_sub_3`, `triton_poi_fused_index_select_select_sub_0`, `triton_poi_fused_expand_new_zeros_scatter_reduce_select_view_2`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=1；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 7.761574 | 16.760050 | -115.94% |
| GPU | 11.795400 | 5.889982 | 50.07% |

达到 GPU **整网**提升比时，NPU 整步目标为 3.875708 ms；当前相对该目标的差额为 12.884342 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_edgecnn_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_edgecnn_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_edgecnn_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_edgecnn_Train/GPU_wrapper_calls.txt)。

### 1. ReLU backward 与 bias 梯度归约

**优化入口：NPU Inductor scheduler 融合决策、codegen/reduction tiling。** 具体检查 threshold_backward 输出写回与沿 batch/edge 维 sum 的联合调度；GPU 将 pointwise 与 reduction 分拆或采用不同归约 kernel。重点是归约 kernel 的并行度和多阶段策略。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_sum_threshold_backward_3` | 3.0 | 8.405490 | 50.15% |

本项 kernel/区间合计 **8.405490 ms，占整步 50.15%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.347957 | 8.405490 | ≤-2315.67% |
| GPU（Eager 下界 / compiled 上界） | 0.918399 | 0.649202 | ≥29.31% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **2344.98 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnReduceSum_ReduceSumOpAiCore_ReduceSum`, `aclnnThresholdBackward_ReluGrad_ReluGrad`。

GPU Eager 对账项：`threshold_backward`, `sum`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_threshold_backward_2` | 3.0 | 0.476144 | 8.08% |
| `triton_red_fused_sum_0` | 6.0 | 0.148996 | 2.53% |
| `triton_red_fused_sum_1` | 6.0 | 0.009949 | 0.17% |
| `threshold_backward` | 2.0 | 0.009635 | 0.16% |
| `sum` | 1.0 | 0.004477 | 0.08% |

**贡献粗估：8.159524 ms/step，相当于当前 NPU Inductor 整步耗时的 48.68%，使整网提升比增加 105.127 个百分点，单独达到该预算后整网提升比约为 -10.81%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused_sum_threshold_backward_3`：NPU_g01_output_code.py:243；输入 `f32[200000, 64][64, 1]npu:0`；size_hints=`{'x0': 64, 'r1': 200000}`；split_axis=`[0]`。

### 额外具体检查点

`triton_red_fused_sum_threshold_backward_3` 的输入为 f32[200000,64]；split_axis=[0] 只沿 64 个输出通道分 program，200000 条边在 program 内 loop_r1 归约，并在同一循环写回 ReLU 导数。应在 NPUIndexTritonKernel.reduction / triton_config_npu_index 检查归约维拆分，或拆开 pointwise 与 bias sum 并采用多阶段归约。当前源码没有沿 edge 维分派独立 program；实际 tile/config 仍待补。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_red_fused_add_div_expand_gather_mul_sum_6` | 3.408717 / 2.0 | 20.34% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_red_fused_div_expand_gather_mul_sum_vie_2` | 1.070179 / 1.0 | 6.39% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_add_index_add_neg_slice_5` | 0.394489 / 2.0 | 2.35% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_red_fused_add_div_expand_gather_mul_sum_6`：NPU_g01_output_code.py:481；输入 `b8[200000, 64][64, 1]npu:0`；size_hints=`{'x0': 64, 'r1': 200000}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_threshold_backward_2`, `triton_poi_fused_add_index_add_neg_new_zeros_slice_4`, `triton_poi_fused__to_copy_expand_scatter_add_view_1`。
- `triton_red_fused_div_expand_gather_mul_sum_vie_2`：NPU_g01_output_code.py:171；输入 `b8[200000, 64][64, 1]npu:0`；size_hints=`{'x0': 64, 'r1': 200000}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused__to_copy_expand_scatter_add_view_1`, `triton_red_fused_sum_0`, `triton_poi_fused_div_expand_gather_mul_view_2`。
- `triton_poi_fused_add_index_add_neg_slice_5`：NPU_g01_output_code.py:393；输入 `i64[200000][1]npu:0`；size_hints=`{'y0': 200000, 'x1': 64}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_index_add_neg_new_zeros_slice_4`, `triton_poi_fused_index_add_new_zeros_slice_3`, `triton_poi_fused_add_copy__1`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
