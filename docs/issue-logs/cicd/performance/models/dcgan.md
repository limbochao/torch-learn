---
title: "CICD 性能分析：dcgan"
---

# dcgan：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=64；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 0.260175 | 0.353907 | -36.03% |
| GPU | 1.126273 | 2.212362 | -96.43% |

达到 GPU **整网**提升比时，NPU 整步目标为 0.511068 ms；当前相对该目标的差额为 -0.157161 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/dcgan_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/dcgan_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/dcgan_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/dcgan_Infer/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused__native_batch_norm_legit_no_t_1` | 0.039932 / 1.0 | 11.28% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_leaky_relu_0` | 0.011891 / 0.9 | 3.36% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_leaky_relu_2` | 0.009573 / 1.0 | 2.70% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_poi_fused__native_batch_norm_legit_no_t_1`：NPU_g01_output_code.py:140；输入 `f32[64, 128, 16, 16][32768, 256, 16, 1]npu:0`；size_hints=`{'y2': 64, 'y1': 128, 'y0': 2, 'x3': 128}`；split_axis=`[0, 1, 2]`。GPU 检查入口：`triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_4`, `triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_6`, `triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_8`。
- `triton_poi_fused_leaky_relu_0`：NPU_g01_output_code.py:57；输入 `f32[64, 64, 32, 32][65536, 1024, 32, 1]npu:0`；size_hints=`{'t0': 64, 'z1': 64, 'y2': 32, 'x3': 32}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_convolution_leaky_relu_7`, `triton_poi_fused_leaky_relu_2`, `triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_4`。
- `triton_poi_fused_leaky_relu_2`：NPU_g01_output_code.py:222；输入 `f32[64, 128, 16, 16][32768, 1, 2048, 128]npu:0`；size_hints=`{'t0': 64, 'z1': 16, 'y2': 16, 'x3': 128}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_convolution_leaky_relu_7`, `triton_poi_fused_leaky_relu_2`, `triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_4`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=64；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 1.298396 | 1.264226 | 2.63% |
| GPU | 2.679155 | 2.642355 | 1.37% |

达到 GPU **整网**提升比时，NPU 整步目标为 1.280562 ms；当前相对该目标的差额为 -0.016337 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/dcgan_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/dcgan_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/dcgan_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/dcgan_Train/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_convolution_backward_leaky_re_4` | 0.024601 / 1.0 | 1.95% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_leaky_relu_0` | 0.013794 / 1.0 | 1.09% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_add_copy__5` | 0.013613 / 1.0 | 1.08% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_poi_fused_convolution_backward_leaky_re_4`：NPU_g01_output_code.py:555；输入 `f32[64, 64, 32, 32][65536, 1024, 32, 1]npu:0`；size_hints=`{'t0': 64, 'z1': 64, 'y2': 32, 'x3': 32}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_red_fused_leaky_relu_backward_native_batch_norm_backward_3`, `triton_poi_fused_convolution_backward_leaky_relu_backward_7`, `triton_poi_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_6`。
- `triton_poi_fused_leaky_relu_0`：NPU_g02_output_code.py:57；输入 `f32[64, 64, 32, 32][65536, 1024, 32, 1]npu:0`；size_hints=`{'t0': 64, 'z1': 64, 'y2': 32, 'x3': 32}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_leaky_relu_0`, `triton_poi_fused__native_batch_norm_legit_functional_leaky_relu_3`, `triton_red_fused__native_batch_norm_legit_functional_copy__leaky_relu_4`。
- `triton_poi_fused_add_copy__5`：NPU_g03_output_code.py:364；输入 `f32[512, 256, 4, 4][4096, 16, 4, 1]npu:0`；size_hints=`{'t0': 512, 'z1': 256, 'y2': 4, 'x3': 4}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_copy__5`, `triton_red_fused__native_batch_norm_legit_functional_copy__leaky_relu_4`, `triton_poi_fused_add_copy__7`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
