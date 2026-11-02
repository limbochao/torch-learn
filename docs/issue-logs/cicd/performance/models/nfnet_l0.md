---
title: "CICD 性能分析：nfnet_l0"
---

# nfnet_l0：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=128；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 58.527466 | 46.892745 | 19.88% |
| GPU | 221.127549 | 157.224468 | 28.90% |

达到 GPU **整网**提升比时，NPU 整步目标为 41.613764 ms；当前相对该目标的差额为 5.278981 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/nfnet_l0_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/nfnet_l0_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/nfnet_l0_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/nfnet_l0_Infer/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_silu_30` | 2.222617 / 18.0 | 4.74% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_add_mul_sigmoid_41` | 1.713988 / 2.0 | 3.66% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_add_mul_sigmoid_silu_16` | 1.425677 / 1.0 | 3.04% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_poi_fused_silu_30`：NPU_g01_output_code.py:2350；输入 `f32[128, 384, 18, 18][124416, 324, 18, 1]npu:0`；size_hints=`{'t0': 128, 'z1': 384, 'y2': 18, 'x3': 18}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused__native_batch_norm_legit_add_convolution_mean_mul_relu_sigmoid_silu_view_18`, `triton_poi_fused__native_batch_norm_legit_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_view_26`, `triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_view_10`。
- `triton_poi_fused_add_mul_sigmoid_41`：NPU_g01_output_code.py:3211；输入 `f32[128, 1536, 18, 18][497664, 324, 18, 1]npu:0`；size_hints=`{'y1': 128, 'y0': 1536, 'x2': 324}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused__native_batch_norm_legit_add_convolution_mean_mul_relu_sigmoid_silu_view_18`, `triton_poi_fused__native_batch_norm_legit_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_view_26`, `triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_view_10`。
- `triton_poi_fused_add_mul_sigmoid_silu_16`：NPU_g01_output_code.py:1292；输入 `f32[128, 256, 72, 72][1327104, 5184, 72, 1]npu:0`；size_hints=`{'y0': 32768, 'x1': 5184}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused__native_batch_norm_legit_add_convolution_mean_mul_relu_sigmoid_silu_view_18`, `triton_poi_fused__native_batch_norm_legit_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_view_26`, `triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_view_10`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=128；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 140.453107 | 131.778025 | 6.18% |
| GPU | 349.804233 | 272.279674 | 22.16% |

达到 GPU **整网**提升比时，NPU 整步目标为 109.325510 ms；当前相对该目标的差额为 22.452515 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/nfnet_l0_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/nfnet_l0_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/nfnet_l0_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/nfnet_l0_Train/GPU_wrapper_calls.txt)。

### 1. avg_pool2d 生成 kernel

**优化入口：NPU Inductor avg_pool2d lowering、pointwise tiling/runtime 配置。** 该 kernel 是 avg_pool2d 本身的生成实现；先核对输出点并行映射及 tile，必要时选择 ACLNN 回退。若相同 DSL/配置仍慢，再交 Triton_ascend/npuir。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_avg_pool2d_16` | 1.0 | 0.485396 | 0.37% |
| `triton_poi_fused_avg_pool2d_26` | 1.0 | 0.368915 | 0.28% |
| `triton_poi_fused_avg_pool2d_42` | 1.0 | 18.855676 | 14.31% |

本项 kernel/区间合计 **19.709987 ms，占整步 14.96%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.752289 | 19.709987 | ≤-2520.00% |
| GPU（Eager 下界 / compiled 上界） | 1.635334 | 1.631657 | ≥0.22% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **2520.23 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnAvgPool2d_AvgPoolV2AiCore_AvgPoolV2`。

GPU Eager 对账项：`avg_pool2d`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_avg_pool2d_17` | 1.0 | 0.838084 | 0.31% |
| `triton_poi_fused_avg_pool2d_29` | 1.0 | 0.441634 | 0.16% |
| `triton_poi_fused_avg_pool2d_46` | 1.0 | 0.351939 | 0.13% |

**贡献粗估：18.959389 ms/step，相当于当前 NPU Inductor 整步耗时的 14.39%，使整网提升比增加 13.499 个百分点，单独达到该预算后整网提升比约为 19.68%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_poi_fused_avg_pool2d_16`：NPU_g02_output_code.py:1290；输入 `f32[128, 256, 56, 56][802816, 3136, 56, 1]npu:0`；size_hints=`{'y0': 917504, 'x1': 28}`；split_axis=`[0]`。
- `triton_poi_fused_avg_pool2d_26`：NPU_g02_output_code.py:2063；输入 `f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0`；size_hints=`{'y0': 917504, 'x1': 14}`；split_axis=`[0]`。
- `triton_poi_fused_avg_pool2d_42`：NPU_g02_output_code.py:3330；输入 `f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0`；size_hints=`{'y0': 1376256, 'x1': 7}`；split_axis=`[0]`。

### 额外具体检查点

`triton_poi_fused_avg_pool2d_42` 对 f32[128,1536,14,14] 做 2×2/stride=2 pooling。DSL 将输出组织成 y0=1376256、x1=7，固定 X1BLOCK_SUB=7，只沿 y0 分配 program，每个输出用四条 stride=2 load，再加和除 4；没有复杂数学运算。优先核对宽度 7 的非整齐 tile、y0 分块和实际生成的访存/向量指令。资料可确认是这个生成实现异常，尚不能区分 tile 选择与 Triton_ascend/npuir 编译质量。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `aclnnAvgPool2dBackward_AvgPoolV2Grad_AvgPoolV2Grad` | 5.148650 / 3.0 | 3.91% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_per_fused_add_mul_sigmoid_sigmoid_backw_27` | 1.479441 / 1.0 | 1.12% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_silu_31` | 1.245492 / 18.0 | 0.95% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `aclnnAvgPool2dBackward_AvgPoolV2Grad_AvgPoolV2Grad`：外部调用；见 wrapper 调用边界。GPU 检查入口：需对照外部 ACLNN/CUDA 调用和 FX 图。
- `triton_per_fused_add_mul_sigmoid_sigmoid_backw_27`：NPU_g01_output_code.py:3062；输入 `f32[128, 256, 56, 56][802816, 3136, 56, 1]npu:0`；size_hints=`{'y0': 128, 'x1': 256, 'r2': 3136}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused__native_batch_norm_legit_add_convolution_mul_sigmoid_silu_view_16`, `triton_per_fused__native_batch_norm_legit_convolution_mean_mul_view_36`, `triton_red_fused_add_avg_pool2d_backward_mul_sigmoid_sigmoid_backward_silu_backward_sum_29`。
- `triton_poi_fused_silu_31`：NPU_g02_output_code.py:2450；输入 `f32[128, 384, 14, 14][75264, 196, 14, 1]npu:0`；size_hints=`{'t0': 128, 'z1': 384, 'y2': 14, 'x3': 14}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused__native_batch_norm_legit_add_convolution_mul_sigmoid_silu_view_16`, `triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_view_5`, `triton_poi_fused_add_mul_sigmoid_silu_28`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
