---
title: "CICD 性能分析：lennard_jones"
---

# lennard_jones：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=4096；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 0.022059 | 0.017641 | 20.03% |
| GPU | 0.034382 | 0.029444 | 14.36% |

达到 GPU **整网**提升比时，NPU 整步目标为 0.018891 ms；当前相对该目标的差额为 -0.001250 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/lennard_jones_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/lennard_jones_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/lennard_jones_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/lennard_jones_Infer/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_addmm_tanh_1` | 0.006367 / 3.0 | 36.09% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_addmm_t_tanh_0` | 0.002052 / 0.9 | 11.63% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_poi_fused_addmm_tanh_1`：NPU_g01_output_code.py:120；输入 `f32[16][1]npu:0`；size_hints=`{'y0': 4096, 'x1': 16}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_tanh_1`, `triton_poi_fused_addmm_t_tanh_0`。
- `triton_poi_fused_addmm_t_tanh_0`：NPU_g01_output_code.py:62；输入 `f32[16][1]npu:0`；size_hints=`{'y0': 4096, 'x1': 16}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_tanh_1`, `triton_poi_fused_addmm_t_tanh_0`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=4096；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 0.119734 | 0.295936 | -147.16% |
| GPU | 0.147570 | 0.118586 | 19.64% |

达到 GPU **整网**提升比时，NPU 整步目标为 0.096218 ms；当前相对该目标的差额为 0.199719 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/lennard_jones_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/lennard_jones_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/lennard_jones_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/lennard_jones_Train/GPU_wrapper_calls.txt)。

### 1. tanh backward 与 sum 梯度归约

**优化入口：NPU Inductor scheduler、codegen/reduction tiling。** 该融合包含 tanh 导数与按列梯度归约；GPU 为 tanh_backward 与 sum 生成独立或分阶段 kernel。检查输出通道少、归约维大时是否分派过少 program。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused_sum_tanh_backward_2` | 3.0 | 0.178655 | 60.37% |

本项 kernel/区间合计 **0.178655 ms，占整步 60.37%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.035622 | 0.178655 | ≤-401.52% |
| GPU（Eager 下界 / compiled 上界） | 0.040021 | 0.018727 | ≥53.21% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **454.73 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnReduceSum_ReduceSumOpAiCore_ReduceSum`, `aclnnReduceSum_BroadcastToAiCore_BroadcastTo`, `aclnnTanhBackward_TanhGradAICore_TanhGrad`。

GPU Eager 对账项：`sum`, `tanh_backward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_sum_2` | 4.0 | 0.005594 | 4.72% |
| `triton_per_fused_sum_3` | 4.0 | 0.004650 | 3.92% |
| `triton_poi_fused_tanh_backward_4` | 3.0 | 0.003648 | 3.08% |
| `sum` | 1.0 | 0.002317 | 1.95% |
| `triton_red_fused_sum_0` | 1.0 | 0.001357 | 1.14% |
| `triton_poi_fused_mm_t_tanh_backward_1` | 1.0 | 0.001162 | 0.98% |

**贡献粗估：0.161986 ms/step，相当于当前 NPU Inductor 整步耗时的 54.74%，使整网提升比增加 135.288 个百分点，单独达到该预算后整网提升比约为 -11.87%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_per_fused_sum_tanh_backward_2`：NPU_g01_output_code.py:187；输入 `f32[4096, 16][16, 1]npu:0`；size_hints=`{'x0': 16, 'r1': 4096}`；split_axis=`[0]`。

### 额外具体检查点

`triton_per_fused_sum_tanh_backward_2` 对 f32[4096,16] 计算 tanh 导数与列和；size_hints={x0:16,r1:4096}、split_axis=[0]，最多只有 16 个 x0 program（配置合并通道后还会更少），整个 4096 维在一个 persistent tile 内归约。优先优化小输出维情况下的分阶段归约/融合策略。GPU 把导数 pointwise 和 sum 分拆，避免把所有并行度压在 16 个输出上。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused_mm_sum_t_tanh_backward_1` | 0.041563 / 1.0 | 14.04% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_addmm_tanh_1` | 0.006537 / 3.0 | 2.21% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_add_copy__0` | 0.005977 / 6.0 | 2.02% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_per_fused_mm_sum_t_tanh_backward_1`：NPU_g01_output_code.py:118；输入 `f32[4096, 1][1, 1]npu:0`；size_hints=`{'x0': 16, 'r1': 4096}`；split_axis=`[0]`。GPU 检查入口：`triton_red_fused_sum_2`, `triton_per_fused_sum_3`, `triton_poi_fused_tanh_backward_4`。
- `triton_poi_fused_addmm_tanh_1`：NPU_g02_output_code.py:120；输入 `f32[16][1]npu:0`；size_hints=`{'y0': 4096, 'x1': 16}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_tanh_1`, `triton_poi_fused_addmm_t_tanh_0`。
- `triton_poi_fused_add_copy__0`：NPU_g03_output_code.py:58；输入 `f32[16, 1][1, 1]npu:0`；size_hints=`{'x0': 16}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_copy__0`, `triton_poi_fused_add_copy__1`, `triton_poi_fused_add_copy__2`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
