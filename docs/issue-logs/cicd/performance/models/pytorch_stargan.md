---
title: "CICD 性能分析：pytorch_stargan"
---

# pytorch_stargan：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=16；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 3.902258 | 3.651759 | 6.42% |
| GPU | 11.302245 | 11.222801 | 0.70% |

达到 GPU **整网**提升比时，NPU 整步目标为 3.874828 ms；当前相对该目标的差额为 -0.223070 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/pytorch_stargan_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/pytorch_stargan_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/pytorch_stargan_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/pytorch_stargan_Infer/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused__native_batch_norm_legit_no_t_5` | 0.115413 / 6.0 | 3.16% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused__native_batch_norm_legit_no_t_2` | 0.109787 / 2.0 | 3.01% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused__native_batch_norm_legit_no_t_4` | 0.104517 / 7.0 | 2.86% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |

- `triton_poi_fused__native_batch_norm_legit_no_t_5`：NPU_g01_output_code.py:513；输入 `f32[1, 4096, 32, 32][4194304, 1024, 32, 1]npu:0`；size_hints=`{'z0': 16, 'y1': 256, 'x2': 1024}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused__native_batch_norm_legit_no_training_relu_repeat_view_2`, `triton_poi_fused__native_batch_norm_legit_no_training_add_repeat_view_8`, `triton_poi_fused__native_batch_norm_legit_no_training_relu_repeat_view_6`。
- `triton_poi_fused__native_batch_norm_legit_no_t_2`：NPU_g01_output_code.py:196；输入 `f32[16, 64, 128, 128][1048576, 16384, 128, 1]npu:0`；size_hints=`{'y2': 16, 'y3': 64, 'x1': 16384}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused__native_batch_norm_legit_no_training_relu_repeat_view_2`, `triton_poi_fused__native_batch_norm_legit_no_training_add_repeat_view_8`, `triton_poi_fused__native_batch_norm_legit_no_training_relu_repeat_view_6`。
- `triton_poi_fused__native_batch_norm_legit_no_t_4`：NPU_g01_output_code.py:390；输入 `f32[16, 256, 32, 32][262144, 1024, 32, 1]npu:0`；size_hints=`{'y2': 16, 'y3': 256, 'x1': 1024}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused__native_batch_norm_legit_no_training_relu_repeat_view_2`, `triton_poi_fused__native_batch_norm_legit_no_training_add_repeat_view_8`, `triton_poi_fused__native_batch_norm_legit_no_training_relu_repeat_view_6`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=16；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 18.542527 | 15.755271 | 15.03% |
| GPU | 37.616093 | 36.164914 | 3.86% |

达到 GPU **整网**提升比时，NPU 整步目标为 17.827181 ms；当前相对该目标的差额为 -2.071910 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/pytorch_stargan_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/pytorch_stargan_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/pytorch_stargan_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/pytorch_stargan_Train/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_red_fused_native_batch_norm_backward_th_1` | 0.299755 / 2.0 | 1.90% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_per_fused_native_batch_norm_backward_th_14` | 0.186622 / 6.0 | 1.18% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused__native_batch_norm_legit_no_t_3` | 0.184864 / 2.0 | 1.17% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |

- `triton_red_fused_native_batch_norm_backward_th_1`：NPU_g01_output_code.py:154；输入 `f32[16, 64, 128, 128][1048576, 16384, 128, 1]npu:0`；size_hints=`{'x0': 1024, 'r1': 16384}`；split_axis=`[0]`。GPU 检查入口：`triton_red_fused_convolution_backward_native_batch_norm_backward_threshold_backward_view_1`, `triton_red_fused_convolution_backward_native_batch_norm_backward_threshold_backward_view_4`, `triton_per_fused_convolution_backward_native_batch_norm_backward_threshold_backward_view_10`。
- `triton_per_fused_native_batch_norm_backward_th_14`：NPU_g01_output_code.py:1190；输入 `f32[16, 256, 32, 32][262144, 1024, 32, 1]npu:0`；size_hints=`{'x0': 4096, 'r1': 1024}`；split_axis=`[0]`。GPU 检查入口：`triton_red_fused_convolution_backward_native_batch_norm_backward_threshold_backward_view_1`, `triton_red_fused_convolution_backward_native_batch_norm_backward_threshold_backward_view_4`, `triton_per_fused_convolution_backward_native_batch_norm_backward_threshold_backward_view_10`。
- `triton_poi_fused__native_batch_norm_legit_no_t_3`：NPU_g02_output_code.py:245；输入 `f32[16, 64, 128, 128][1048576, 16384, 128, 1]npu:0`；size_hints=`{'y2': 16, 'y3': 64, 'x1': 16384}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_red_fused__native_batch_norm_legit_functional_relu_repeat_view_2`, `triton_per_fused__native_batch_norm_legit_functional_add_relu_repeat_view_13`, `triton_per_fused__native_batch_norm_legit_functional_relu_repeat_view_10`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
