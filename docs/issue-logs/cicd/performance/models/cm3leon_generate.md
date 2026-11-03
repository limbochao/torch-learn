---
title: "CICD 性能分析：cm3leon_generate"
---

# cm3leon_generate：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=1；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 228.157075 | 241.420536 | -5.81% |
| GPU | 330.396771 | 310.871115 | 5.91% |

达到 GPU **整网**提升比时，NPU 整步目标为 214.673539 ms；当前相对该目标的差额为 26.746996 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/cm3leon_generate_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/cm3leon_generate_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/cm3leon_generate_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/cm3leon_generate_Infer/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused__to_copy_add_addmm_embedding__7` | 5.599183 / 3008.0 | 2.32% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_addmm_8` | 2.977751 / 1536.0 | 1.23% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_addmm_4` | 2.826955 / 1536.0 | 1.17% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_per_fused__to_copy_add_addmm_embedding__7`：同名/截断名对应 2 个定义（NPU_g06_output_code.py:580, NPU_g07_output_code.py:626）；以下尺寸仅展示首个入口：NPU_g06_output_code.py:580；输入 `f32[1, 1, 1536][1536, 1536, 1]npu:0`；size_hints=`{'r0': 1536}`；split_axis=`[0]`。GPU 检查入口：`triton_red_fused_add_addmm_native_layer_norm_view_13`, `triton_red_fused_add_addmm_native_layer_norm_view_9`, `triton_poi_fused_addmm_mul_view_2`。
- `triton_poi_fused_addmm_8`：同名/截断名对应 2 个定义（NPU_g06_output_code.py:649, NPU_g07_output_code.py:695）；以下尺寸仅展示首个入口：NPU_g06_output_code.py:649；输入 `f32[6144][1]npu:0`；size_hints=`{'x0': 6144}`；split_axis=`[0]`。GPU 检查入口：`triton_red_fused_add_addmm_native_layer_norm_view_13`, `triton_red_fused_add_addmm_native_layer_norm_view_9`, `triton_poi_fused_addmm_mul_view_2`。
- `triton_poi_fused_addmm_4`：同名/截断名对应 2 个定义（NPU_g06_output_code.py:332, NPU_g07_output_code.py:332）；以下尺寸仅展示首个入口：NPU_g06_output_code.py:332；输入 `f32[1536][1]npu:0`；size_hints=`{'x0': 1536}`；split_axis=`[0]`。GPU 检查入口：`triton_red_fused_add_addmm_native_layer_norm_view_13`, `triton_red_fused_add_addmm_native_layer_norm_view_9`, `triton_poi_fused_addmm_mul_view_2`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

本批目录没有此场景，无法给出性能结论。
