---
title: "CICD 性能分析：nanogpt"
---

# nanogpt：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=1；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 1.554152 | 1.227220 | 21.04% |
| GPU | 1.677339 | 1.350179 | 19.50% |

达到 GPU **整网**提升比时，NPU 整步目标为 1.251019 ms；当前相对该目标的差额为 -0.023800 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/nanogpt_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/nanogpt_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/nanogpt_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/nanogpt_Infer/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_add_addmm_mul_pow_tanh_view_4` | 0.036596 / 12.0 | 2.98% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_per_fused_addmm_native_layer_norm_6` | 0.036515 / 11.0 | 2.98% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_per_fused_addmm_native_layer_norm_5` | 0.030710 / 11.0 | 2.50% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |

- `triton_poi_fused_add_addmm_mul_pow_tanh_view_4`：NPU_g01_output_code.py:401；输入 `f32[3072][1]npu:0`；size_hints=`{'y0': 64, 'x1': 3072}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_add_addmm_mul_pow_tanh_view_2`, `triton_per_fused_add_addmm_native_layer_norm_view_4`, `triton_per_fused_add_addmm_native_layer_norm_view_3`。
- `triton_per_fused_addmm_native_layer_norm_6`：NPU_g01_output_code.py:568；输入 `f32[1, 64, 768][49152, 768, 1]npu:0`；size_hints=`{'x0': 64, 'r1': 768}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_addmm_mul_pow_tanh_view_2`, `triton_per_fused_add_addmm_native_layer_norm_view_4`, `triton_per_fused_add_addmm_native_layer_norm_view_3`。
- `triton_per_fused_addmm_native_layer_norm_5`：NPU_g01_output_code.py:479；输入 `f32[1, 64, 768][49152, 768, 1]npu:0`；size_hints=`{'x0': 64, 'r1': 768}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_addmm_mul_pow_tanh_view_2`, `triton_per_fused_add_addmm_native_layer_norm_view_4`, `triton_per_fused_add_addmm_native_layer_norm_view_3`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=1；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 6.270929 | 5.963910 | 4.90% |
| GPU | 8.040806 | 6.748682 | 16.07% |

达到 GPU **整网**提升比时，NPU 整步目标为 5.263217 ms；当前相对该目标的差额为 0.700693 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/nanogpt_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/nanogpt_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/nanogpt_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/nanogpt_Train/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_add_matmul_backward_t_9` | 0.526532 / 1.0 | 8.83% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_add_copy__0` | 0.304951 / 1.0 | 5.11% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_add_copy__8` | 0.256608 / 12.0 | 4.30% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_poi_fused_add_matmul_backward_t_9`：NPU_g01_output_code.py:763；输入 `f32[768, 1][1, 768]npu:0`；size_hints=`{'y0': 50304, 'x1': 768}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_copy__6`, `triton_poi_fused_add_copy__0`, `triton_poi_fused_add_mm_t_view_12`。
- `triton_poi_fused_add_copy__0`：NPU_g03_output_code.py:58；输入 `f32[50304, 768][768, 1]npu:0`；size_hints=`{'y0': 50304, 'x1': 768}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_copy__6`, `triton_poi_fused_add_copy__0`, `triton_poi_fused_add_mm_t_view_12`。
- `triton_poi_fused_add_copy__8`：NPU_g03_output_code.py:502；输入 `f32[768, 3072][3072, 1]npu:0`；size_hints=`{'y0': 768, 'x1': 3072}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_copy__6`, `triton_poi_fused_add_copy__0`, `triton_poi_fused_add_mm_t_view_12`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
