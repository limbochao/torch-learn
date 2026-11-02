---
title: "CICD 性能分析：dlrm"
---

# dlrm：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=2048；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 2.502826 | 2.496321 | 0.26% |
| GPU | 1.079799 | 1.064851 | 1.38% |

达到 GPU **整网**提升比时，NPU 整步目标为 2.468178 ms；当前相对该目标的差额为 0.028142 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/dlrm_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/dlrm_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/dlrm_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/dlrm_Infer/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_addmm_relu_3` | 0.020992 / 3.0 | 0.84% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_addmm_relu_0` | 0.004747 / 1.0 | 0.19% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_index_lift_fresh_2` | 0.004666 / 1.0 | 0.19% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_poi_fused_addmm_relu_3`：NPU_g01_output_code.py:252；输入 `f32[1024][1]npu:0`；size_hints=`{'y0': 2048, 'x1': 1024}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_relu_4`, `triton_poi_fused_addmm_cat_relu_2`, `triton_poi_fused_addmm_relu_0`。
- `triton_poi_fused_addmm_relu_0`：NPU_g01_output_code.py:60；输入 `f32[512][1]npu:0`；size_hints=`{'y0': 2048, 'x1': 512}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_relu_4`, `triton_poi_fused_addmm_cat_relu_2`, `triton_poi_fused_addmm_relu_0`。
- `triton_poi_fused_index_lift_fresh_2`：NPU_g01_output_code.py:187；输入 `i64[36][1]npu:0`；size_hints=`{'y0': 2048, 'x1': 36}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_cat_index_lift_fresh_3`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=1024；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 5.259739 | 5.270753 | -0.21% |
| GPU | 2.578556 | 2.554851 | 0.92% |

达到 GPU **整网**提升比时，NPU 整步目标为 5.211385 ms；当前相对该目标的差额为 0.059368 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/dlrm_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/dlrm_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/dlrm_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/dlrm_Train/GPU_wrapper_calls.txt)。

### 1. ReLU backward 与 bias 梯度归约

**优化入口：NPU Inductor scheduler 融合决策、codegen/reduction tiling。** 具体检查 threshold_backward 输出写回与沿 batch/edge 维 sum 的联合调度；GPU 将 pointwise 与 reduction 分拆或采用不同归约 kernel。重点是归约 kernel 的并行度和多阶段策略。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused_sum_threshold_backward_0` | 1.0 | 0.001422 | 0.03% |
| `triton_per_fused_sum_threshold_backward_3` | 2.0 | 0.038290 | 0.73% |
| `triton_per_fused_sum_threshold_backward_7` | 1.0 | 0.011193 | 0.21% |

本项 kernel/区间合计 **0.050905 ms，占整步 0.97%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.058926 | 0.050905 | ≤13.61% |
| GPU（Eager 下界 / compiled 上界） | 0.057297 | 0.042074 | ≥26.57% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **12.96 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnReduceSum_ReduceSumOpAiCore_ReduceSum`, `aclnnThresholdBackward_BroadcastToAiCore_BroadcastTo`, `aclnnThresholdBackward_ReluGrad_ReluGrad`。

GPU Eager 对账项：`sum`, `threshold_backward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_threshold_backward_5` | 2.0 | 0.008068 | 0.32% |
| `triton_red_fused_sum_2` | 3.0 | 0.007446 | 0.29% |
| `triton_per_fused_sum_3` | 3.0 | 0.006307 | 0.25% |
| `triton_poi_fused_mm_t_threshold_backward_1` | 1.0 | 0.003645 | 0.14% |
| `triton_poi_fused_threshold_backward_11` | 1.0 | 0.002394 | 0.09% |
| `sum` | 1.0 | 0.002333 | 0.09% |
| `triton_poi_fused_add_slice_threshold_backward_transpose_view_7` | 1.0 | 0.002042 | 0.08% |
| `triton_red_fused_sum_12` | 1.0 | 0.001872 | 0.07% |
| `triton_poi_fused_addmm_cat_index_lift_fresh_relu_threshold_backward_1` | 1.0 | 0.001642 | 0.06% |
| `triton_red_fused_sum_9` | 1.0 | 0.001485 | 0.06% |
| `triton_per_fused_sum_13` | 1.0 | 0.001290 | 0.05% |
| `triton_per_fused_sum_10` | 1.0 | 0.001210 | 0.05% |
| `triton_per_fused_sum_threshold_backward_0` | 1.0 | 0.001187 | 0.05% |
| `triton_poi_fused_addmm_relu_threshold_backward_5` | 1.0 | 0.001155 | 0.05% |

**贡献粗估：0.007635 ms/step，相当于当前 NPU Inductor 整步耗时的 0.14%，使整网提升比增加 0.145 个百分点，单独达到该预算后整网提升比约为 -0.06%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_per_fused_sum_threshold_backward_0`：NPU_g01_output_code.py:64；输入 `b8[1024, 1][1, 1]npu:0`；size_hints=`{'r0': 1024}`；split_axis=`[0]`。
- `triton_per_fused_sum_threshold_backward_3`：NPU_g01_output_code.py:256；输入 `f32[1024, 1024][1024, 1]npu:0`；size_hints=`{'x0': 1024, 'r1': 1024}`；split_axis=`[0]`。
- `triton_per_fused_sum_threshold_backward_7`：NPU_g01_output_code.py:535；输入 `f32[1024, 512][512, 1]npu:0`；size_hints=`{'x0': 512, 'r1': 1024}`；split_axis=`[0]`。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused_add_slice_sum_threshold_backw_5` | 0.021206 / 1.0 | 0.40% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_per_fused_mm_sum_t_threshold_backward_1` | 0.016995 / 1.0 | 0.32% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_addmm_relu_3` | 0.014919 / 3.0 | 0.28% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_per_fused_add_slice_sum_threshold_backw_5`：NPU_g01_output_code.py:407；输入 `b8[1024, 64][64, 1]npu:0`；size_hints=`{'x0': 64, 'r1': 1024}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_threshold_backward_5`, `triton_red_fused_sum_2`, `triton_per_fused_sum_3`。
- `triton_per_fused_mm_sum_t_threshold_backward_1`：NPU_g01_output_code.py:133；输入 `f32[1024, 1024][1024, 1]npu:0`；size_hints=`{'x0': 1024, 'r1': 1024}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_threshold_backward_5`, `triton_red_fused_sum_2`, `triton_per_fused_sum_3`。
- `triton_poi_fused_addmm_relu_3`：NPU_g02_output_code.py:257；输入 `f32[1024][1]npu:0`；size_hints=`{'y0': 1024, 'x1': 1024}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_cat_relu_2`, `triton_poi_fused_addmm_relu_4`, `triton_poi_fused_addmm_relu_0`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
