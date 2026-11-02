---
title: "CICD 性能分析：basic_gnn_gcn"
---

# basic_gnn_gcn：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=1；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 0.845399 | 0.737259 | 12.79% |
| GPU | 1.086228 | 0.293174 | 73.01% |

达到 GPU **整网**提升比时，NPU 整步目标为 0.228174 ms；当前相对该目标的差额为 0.509085 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_gcn_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_gcn_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_gcn_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_gcn_Infer/GPU_wrapper_calls.txt)。

### 1. scatter_add 外部调用阻断 gather/mul 融合

**优化入口：NPU Inductor aten.scatter_add lowering、scatter/gather scheduler fusion。** NPU wrapper 保留 aten.scatter_add 外部调用及独立 gather/index_select；GPU 生成含 scatter_add 的 Triton kernel，可连同 index_select/mul/new_zeros 融合。优先补这个具体 lowering/融合边界；原子写入实现再由 Triton_ascend/CANN 承接。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnScatterAdd_ScatterElements_ScatterElementsV2` | 3.0 | 0.100559 | 13.64% |
| `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd` | 3.0 | 0.267468 | 36.28% |

本项 kernel/区间合计 **0.368027 ms，占整步 49.92%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.371113 | 0.368027 | ≤0.83% |
| GPU（Eager 下界 / compiled 上界） | 0.361072 | 0.191119 | ≥47.07% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **46.24 个百分点**。GPU fused kernel 的全部耗时均收费给 scatter_add 核心；GPU Eager 仅计 scatter_add_，因此是保守 GPU 收益下界。

NPU Eager 对账项：`aclnnScatterAdd_ScatterElements_ScatterElementsV2`, `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd`。

GPU Eager 对账项：`scatter_add_`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_expand_index_select_mul_new_zeros_scatter_add_select_view_1` | 3.0 | 0.167938 | 57.28% |
| `triton_poi_fused_new_zeros_ones_scatter_add_select_1` | 3.0 | 0.015111 | 5.15% |
| `triton_poi_fused_expand_index_select_mul_new_zeros_scatter_add_select_view_0` | 3.0 | 0.005248 | 1.79% |
| `triton_poi_fused_new_zeros_ones_scatter_add_select_0` | 3.0 | 0.002823 | 0.96% |

**贡献粗估：0.171593 ms/step，相当于当前 NPU Inductor 整步耗时的 23.27%，使整网提升比增加 20.297 个百分点，单独达到该预算后整网提升比约为 33.09%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnScatterAdd_ScatterElements_ScatterElementsV2`：外部调用；见 wrapper 调用边界。
- `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_index_select_mul_select_view_1` | 0.191376 / 3.0 | 25.96% | NPU Inductor gather/scatter lowering 与 producer-consumer 融合 |
| `triton_poi_fused_eq_index_masked_fill_mul_pow__2` | 0.022782 / 3.0 | 3.09% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_new_zeros_0` | 0.013699 / 6.0 | 1.86% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_poi_fused_index_select_mul_select_view_1`：同名/截断名对应 3 个定义（NPU_g04_output_code.py:112, NPU_g05_output_code.py:112, NPU_g06_output_code.py:112）；以下尺寸仅展示首个入口：NPU_g04_output_code.py:112；输入 `f32[209978][1]npu:0`；size_hints=`{'y0': 209978, 'x1': 64}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_expand_index_select_mul_new_zeros_scatter_add_select_view_1`, `triton_poi_fused_new_zeros_ones_scatter_add_select_1`, `triton_poi_fused_eq_index_masked_fill_mul_pow_select_2`。
- `triton_poi_fused_eq_index_masked_fill_mul_pow__2`：NPU_g03_output_code.py:161；输入 `i64[2, 209978][209978, 1]npu:0`；size_hints=`{'x0': 209978}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_expand_index_select_mul_new_zeros_scatter_add_select_view_1`, `triton_poi_fused_new_zeros_ones_scatter_add_select_1`, `triton_poi_fused_eq_index_masked_fill_mul_pow_select_2`。
- `triton_poi_fused_new_zeros_0`：同名/截断名对应 4 个定义（NPU_g03_output_code.py:54, NPU_g04_output_code.py:54, NPU_g05_output_code.py:54, NPU_g06_output_code.py:54）；以下尺寸仅展示首个入口：NPU_g03_output_code.py:54；size_hints=`{'x0': 10000}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_expand_index_select_mul_new_zeros_scatter_add_select_view_1`, `triton_poi_fused_new_zeros_ones_scatter_add_select_1`, `triton_poi_fused_expand_index_select_mul_new_zeros_scatter_add_select_view_0`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=1；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 2.381309 | 1.266601 | 46.81% |
| GPU | 1.992435 | 0.550495 | 72.37% |

达到 GPU **整网**提升比时，NPU 整步目标为 0.657938 ms；当前相对该目标的差额为 0.608663 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_gcn_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_gcn_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_gcn_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_gcn_Train/GPU_wrapper_calls.txt)。

### 1. scatter_add 外部调用阻断 gather/mul 融合

**优化入口：NPU Inductor aten.scatter_add lowering、scatter/gather scheduler fusion。** NPU wrapper 保留 aten.scatter_add 外部调用及独立 gather/index_select；GPU 生成含 scatter_add 的 Triton kernel，可连同 index_select/mul/new_zeros 融合。优先补这个具体 lowering/融合边界；原子写入实现再由 Triton_ascend/CANN 承接。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnScatterAdd_ScatterElements_ScatterElementsV2` | 3.0 | 0.100862 | 7.96% |
| `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd` | 3.0 | 0.269092 | 21.25% |

本项 kernel/区间合计 **0.369953 ms，占整步 29.21%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.372117 | 0.369953 | ≤0.58% |
| GPU（Eager 下界 / compiled 上界） | 0.362071 | 0.189692 | ≥47.61% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **47.03 个百分点**。GPU fused kernel 的全部耗时均收费给 scatter_add 核心；GPU Eager 仅计 scatter_add_，因此是保守 GPU 收益下界。

NPU Eager 对账项：`aclnnScatterAdd_ScatterElements_ScatterElementsV2`, `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd`。

GPU Eager 对账项：`scatter_add_`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_expand_index_select_mul_scatter_add_select_view_1` | 3.0 | 0.167333 | 30.40% |
| `triton_poi_fused_new_zeros_ones_scatter_add_select_1` | 3.0 | 0.019568 | 3.55% |
| `triton_poi_fused_new_zeros_ones_scatter_add_select_0` | 3.0 | 0.002791 | 0.51% |

**贡献粗估：0.174999 ms/step，相当于当前 NPU Inductor 整步耗时的 13.82%，使整网提升比增加 7.349 个百分点，单独达到该预算后整网提升比约为 54.16%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnScatterAdd_ScatterElements_ScatterElementsV2`：外部调用；见 wrapper 调用边界。
- `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd`：外部调用；见 wrapper 调用边界。

### 2. 独立 sum 生成归约

**优化入口：NPU Inductor codegen/reduction tiling、runtime autotune。** 选中 kernel 仅有 sum 和 view/transpose。检查输入 stride、归约维大小、split_axis、persistent/分阶段选择，与 GPU 归约拆分对照。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_sum_0` | 3.0 | 0.127919 | 10.10% |

本项 kernel/区间合计 **0.127919 ms，占整步 10.10%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.025310 | 0.127919 | ≤-405.41% |
| GPU（Eager 下界 / compiled 上界） | 0.021774 | 0.014708 | ≥32.45% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **437.86 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnReduceSum_ReduceSumOpAiCore_ReduceSum`, `aclnnReduceSum_BroadcastToAiCore_BroadcastTo`。

GPU Eager 对账项：`sum`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_sum_0` | 3.0 | 0.006455 | 1.17% |
| `sum` | 1.0 | 0.004356 | 0.79% |
| `triton_red_fused_sum_1` | 3.0 | 0.003898 | 0.71% |

**贡献粗估：0.110822 ms/step，相当于当前 NPU Inductor 整步耗时的 8.75%，使整网提升比增加 4.654 个百分点，单独达到该预算后整网提升比约为 51.46%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused_sum_0`：同名/截断名对应 3 个定义（NPU_g04_output_code.py:55, NPU_g06_output_code.py:55, NPU_g08_output_code.py:55）；以下尺寸仅展示首个入口：NPU_g04_output_code.py:55；输入 `f32[10000, 64][64, 1]npu:0`；size_hints=`{'x0': 64, 'r1': 10000}`；split_axis=`[0]`。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_expand_gather_index_add_mul_s_2` | 0.265910 / 3.0 | 20.99% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_index_select_mul_select_view_1` | 0.194067 / 3.0 | 15.32% | NPU Inductor gather/scatter lowering 与 producer-consumer 融合 |
| `triton_poi_fused_eq_index_masked_fill_mul_pow__2` | 0.025332 / 3.0 | 2.00% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_poi_fused_expand_gather_index_add_mul_s_2`：同名/截断名对应 3 个定义（NPU_g04_output_code.py:192, NPU_g06_output_code.py:192, NPU_g08_output_code.py:192）；以下尺寸仅展示首个入口：NPU_g04_output_code.py:192；输入 `i64[2, 209978][209978, 1]npu:0`；size_hints=`{'y0': 209978, 'x1': 64}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_expand_gather_index_add_mul_select_view_3`, `triton_poi_fused_expand_index_select_mul_scatter_add_select_view_1`, `triton_poi_fused_new_zeros_ones_scatter_add_select_1`。
- `triton_poi_fused_index_select_mul_select_view_1`：同名/截断名对应 3 个定义（NPU_g05_output_code.py:112, NPU_g07_output_code.py:112, NPU_g09_output_code.py:112）；以下尺寸仅展示首个入口：NPU_g05_output_code.py:112；输入 `f32[209978][1]npu:0`；size_hints=`{'y0': 209978, 'x1': 64}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_expand_gather_index_add_mul_select_view_3`, `triton_poi_fused_expand_index_select_mul_scatter_add_select_view_1`, `triton_poi_fused_new_zeros_ones_scatter_add_select_1`。
- `triton_poi_fused_eq_index_masked_fill_mul_pow__2`：NPU_g03_output_code.py:161；输入 `i64[2, 209978][209978, 1]npu:0`；size_hints=`{'x0': 209978}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_expand_gather_index_add_mul_select_view_3`, `triton_poi_fused_expand_index_select_mul_scatter_add_select_view_1`, `triton_poi_fused_new_zeros_ones_scatter_add_select_1`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
