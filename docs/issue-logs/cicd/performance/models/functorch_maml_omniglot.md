---
title: "CICD 性能分析：functorch_maml_omniglot"
---

# functorch_maml_omniglot：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=1；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 0.064966 | 0.049930 | 23.14% |
| GPU | 0.059630 | 0.052939 | 11.22% |

达到 GPU **整网**提升比时，NPU 整步目标为 0.057676 ms；当前相对该目标的差额为 -0.007746 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/functorch_maml_omniglot_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/functorch_maml_omniglot_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/functorch_maml_omniglot_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/functorch_maml_omniglot_Infer/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `aclnnMaxPool2dWithIndices_MaxPoolWithArgmaxV3AiCore_MaxPoolWithArgmaxV3` | 0.019766 / 3.0 | 39.59% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_per_fused__native_batch_norm_legit_relu_0` | 0.004479 / 1.0 | 8.97% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_per_fused__native_batch_norm_legit_relu_1` | 0.003567 / 1.0 | 7.14% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |

- `aclnnMaxPool2dWithIndices_MaxPoolWithArgmaxV3AiCore_MaxPoolWithArgmaxV3`：外部调用；见 wrapper 调用边界。GPU 检查入口：需对照外部 ACLNN/CUDA 调用和 FX 图。
- `triton_per_fused__native_batch_norm_legit_relu_0`：NPU_g01_output_code.py:72；输入 `f32[5, 64, 26, 26][43264, 676, 26, 1]npu:0`；size_hints=`{'x0': 64, 'r2': 5, 'r1': 676}`；split_axis=`[0]`。GPU 检查入口：`triton_red_fused__native_batch_norm_legit_convolution_relu_0`, `triton_poi_fused__native_batch_norm_legit_convolution_max_pool2d_with_indices_relu_2`, `triton_per_fused__native_batch_norm_legit_convolution_max_pool2d_with_indices_relu_7`。
- `triton_per_fused__native_batch_norm_legit_relu_1`：NPU_g01_output_code.py:161；输入 `f32[5, 64, 11, 11][7744, 121, 11, 1]npu:0`；size_hints=`{'x0': 64, 'r2': 5, 'r1': 121}`；split_axis=`[0]`。GPU 检查入口：`triton_red_fused__native_batch_norm_legit_convolution_relu_0`, `triton_poi_fused__native_batch_norm_legit_convolution_max_pool2d_with_indices_relu_2`, `triton_per_fused__native_batch_norm_legit_convolution_max_pool2d_with_indices_relu_7`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=1；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 0.212294 | 0.182855 | 13.87% |
| GPU | 0.174378 | 0.132588 | 23.97% |

达到 GPU **整网**提升比时，NPU 整步目标为 0.161418 ms；当前相对该目标的差额为 0.021437 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/functorch_maml_omniglot_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/functorch_maml_omniglot_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/functorch_maml_omniglot_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/functorch_maml_omniglot_Train/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `aclnnMaxPool2dWithIndices_MaxPoolWithArgmaxV3AiCore_MaxPoolWithArgmaxV3` | 0.020037 / 3.0 | 10.96% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `aclnnScatterAdd_ScatterElements_ScatterElementsV2` | 0.011590 / 3.0 | 6.34% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_add_copy__1` | 0.009810 / 9.0 | 5.36% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `aclnnMaxPool2dWithIndices_MaxPoolWithArgmaxV3AiCore_MaxPoolWithArgmaxV3`：外部调用；见 wrapper 调用边界。GPU 检查入口：需对照外部 ACLNN/CUDA 调用和 FX 图。
- `aclnnScatterAdd_ScatterElements_ScatterElementsV2`：外部调用；见 wrapper 调用边界。GPU 检查入口：需对照外部 ACLNN/CUDA 调用和 FX 图。
- `triton_poi_fused_add_copy__1`：NPU_g03_output_code.py:121；输入 `f32[64][1]npu:0`；size_hints=`{'x0': 64}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_copy__1`, `triton_poi_fused_add_copy__2`, `triton_poi_fused_add_copy__0`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
