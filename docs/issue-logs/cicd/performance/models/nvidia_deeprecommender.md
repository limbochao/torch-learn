---
title: "CICD 性能分析：nvidia_deeprecommender"
---

# nvidia_deeprecommender：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=256；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 4.951618 | 4.917715 | 0.68% |
| GPU | 4.377126 | 4.938638 | -12.83% |

达到 GPU **整网**提升比时，NPU 整步目标为 5.586828 ms；当前相对该目标的差额为 -0.669114 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/nvidia_deeprecommender_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/nvidia_deeprecommender_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/nvidia_deeprecommender_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/nvidia_deeprecommender_Infer/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_addmm_elu_2` | 0.229995 / 1.0 | 4.68% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_addmm_elu_0` | 0.009234 / 4.0 | 0.19% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_addmm_elu_1` | 0.002865 / 1.0 | 0.06% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_poi_fused_addmm_elu_2`：NPU_g01_output_code.py:206；输入 `f32[197951][1]npu:0`；size_hints=`{'y0': 256, 'x1': 197951}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_elu_2`, `triton_poi_fused_addmm_elu_0`, `triton_poi_fused_addmm_elu_1`。
- `triton_poi_fused_addmm_elu_0`：NPU_g01_output_code.py:64；输入 `f32[512][1]npu:0`；size_hints=`{'y0': 256, 'x1': 512}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_elu_2`, `triton_poi_fused_addmm_elu_0`, `triton_poi_fused_addmm_elu_1`。
- `triton_poi_fused_addmm_elu_1`：NPU_g01_output_code.py:135；输入 `f32[1024][1]npu:0`；size_hints=`{'y0': 256, 'x1': 1024}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_elu_2`, `triton_poi_fused_addmm_elu_0`, `triton_poi_fused_addmm_elu_1`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=256；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 15.112002 | 14.964449 | 0.98% |
| GPU | 13.587924 | 14.069104 | -3.54% |

达到 GPU **整网**提升比时，NPU 整步目标为 15.647152 ms；当前相对该目标的差额为 -0.682704 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/nvidia_deeprecommender_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/nvidia_deeprecommender_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/nvidia_deeprecommender_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/nvidia_deeprecommender_Train/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_add_copy__0` | 0.847333 / 1.0 | 5.66% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_add_copy__6` | 0.837179 / 1.0 | 5.59% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_per_fused_elu_backward_sum_0` | 0.502336 / 1.0 | 3.36% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |

- `triton_poi_fused_add_copy__0`：NPU_g03_output_code.py:58；输入 `f32[512, 197952][197952, 1]npu:0`；size_hints=`{'y0': 512, 'x1': 197952}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_copy__5`, `triton_poi_fused_add_copy__0`, `triton_poi_fused_add_copy__2`。
- `triton_poi_fused_add_copy__6`：NPU_g03_output_code.py:393；输入 `f32[197952, 512][512, 1]npu:0`；size_hints=`{'y0': 197952, 'x1': 512}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_copy__5`, `triton_poi_fused_add_copy__0`, `triton_poi_fused_add_copy__2`。
- `triton_per_fused_elu_backward_sum_0`：NPU_g01_output_code.py:69；输入 `f32[256, 197952][197952, 1]npu:0`；size_hints=`{'x0': 197952, 'r1': 256}`；split_axis=`[0]`。GPU 检查入口：`triton_red_fused_elu_backward_sum_0`, `triton_poi_fused_elu_backward_1`, `triton_red_fused_sum_2`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
