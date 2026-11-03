---
title: "CICD 性能分析：llama"
---

# llama：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=32；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 3.926699 | 3.550468 | 9.58% |
| GPU | 3.705570 | 3.334278 | 10.02% |

达到 GPU **整网**提升比时，NPU 整步目标为 3.533251 ms；当前相对该目标的差额为 0.017217 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/llama_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/llama_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/llama_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/llama_Infer/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_matmul_mul_silu_4` | 0.091468 / 8.0 | 2.58% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_copy_slice_view_1` | 0.052973 / 16.0 | 1.49% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_clone_transpose_2` | 0.033332 / 8.0 | 0.94% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_poi_fused_matmul_mul_silu_4`：NPU_g01_output_code.py:365；输入 `f32[1024, 1536][1536, 1]npu:0`；size_hints=`{'z0': 32, 'y1': 32, 'x2': 1536}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused__unsafe_view_mul_silu_8`, `triton_per_fused__softmax_abs_all_amax_div_eq_mul_ne_sub_view_4`, `triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_14`。
- `triton_poi_fused_copy_slice_view_1`：NPU_g01_output_code.py:149；输入 `f32[32, 32, 8, 32, 2][16384, 512, 64, 2, 1]npu:0`；size_hints=`{'y0': 32, 'x1': 16384}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_copy_slice_view_1`, `triton_poi_fused_clone_slice_transpose_3`, `triton_poi_fused_clone_slice_transpose_5`。
- `triton_poi_fused_clone_transpose_2`：NPU_g01_output_code.py:203；输入 `f32[32, 8, 32, 64][16384, 2048, 64, 1]npu:0`；size_hints=`{'t0': 32, 'z1': 32, 'y2': 8, 'x3': 64}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_clone_slice_transpose_3`, `triton_poi_fused_clone_slice_transpose_5`, `triton_poi_fused_clone_transpose_view_2`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

本批目录没有此场景，无法给出性能结论。
