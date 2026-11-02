---
title: "CICD 性能分析：basic_gnn_sage"
---

# basic_gnn_sage：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=1；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 0.586155 | 0.606804 | -3.52% |
| GPU | 0.793989 | 0.250912 | 68.40% |

达到 GPU **整网**提升比时，NPU 整步目标为 0.185233 ms；当前相对该目标的差额为 0.421571 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_sage_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_sage_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_sage_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_sage_Infer/GPU_wrapper_calls.txt)。

### 1. scatter_add 外部调用阻断 gather/mul 融合

**优化入口：NPU Inductor aten.scatter_add lowering、scatter/gather scheduler fusion。** NPU wrapper 保留 aten.scatter_add 外部调用及独立 gather/index_select；GPU 生成含 scatter_add 的 Triton kernel，可连同 index_select/mul/new_zeros 融合。优先补这个具体 lowering/融合边界；原子写入实现再由 Triton_ascend/CANN 承接。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnScatterAdd_ScatterElements_ScatterElementsV2` | 3.0 | 0.098875 | 16.29% |
| `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd` | 3.0 | 0.254970 | 42.02% |

本项 kernel/区间合计 **0.353845 ms，占整步 58.31%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.355516 | 0.353845 | ≤0.47% |
| GPU（Eager 下界 / compiled 上界） | 0.341784 | 0.181755 | ≥46.82% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **46.35 个百分点**。GPU fused kernel 的全部耗时均收费给 scatter_add 核心；GPU Eager 仅计 scatter_add_，因此是保守 GPU 收益下界。

NPU Eager 对账项：`aclnnScatterAdd_ScatterElements_ScatterElementsV2`, `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd`。

GPU Eager 对账项：`scatter_add_`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_expand_index_select_new_zeros_scatter_add_select_view_1` | 2.0 | 0.105849 | 42.19% |
| `triton_poi_fused_add_addmm_expand_index_select_new_zeros_relu_scatter_add_select_view_6` | 1.0 | 0.056331 | 22.45% |
| `triton_poi_fused_new_ones_new_zeros_scatter_add_select_3` | 1.0 | 0.011671 | 4.65% |
| `triton_poi_fused_expand_index_select_new_zeros_scatter_add_select_view_0` | 3.0 | 0.005194 | 2.07% |
| `triton_poi_fused_new_ones_new_zeros_scatter_add_select_2` | 3.0 | 0.002710 | 1.08% |

**贡献粗估：0.164787 ms/step，相当于当前 NPU Inductor 整步耗时的 27.16%，使整网提升比增加 28.113 个百分点，单独达到该预算后整网提升比约为 24.59%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnScatterAdd_ScatterElements_ScatterElementsV2`：外部调用；见 wrapper 调用边界。
- `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_index_select_select_3` | 0.092409 / 2.0 | 15.23% | NPU Inductor gather/scatter lowering 与 producer-consumer 融合 |
| `triton_poi_fused_add_addmm_index_select_relu_s_7` | 0.073347 / 1.0 | 12.09% | NPU Inductor gather/scatter lowering 与 producer-consumer 融合 |
| `triton_poi_fused_clamp_div_expand_view_4` | 0.011280 / 3.0 | 1.86% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_poi_fused_index_select_select_3`：NPU_g01_output_code.py:197；输入 `i64[2, 200000][200000, 1]npu:0`；size_hints=`{'y0': 200000, 'x1': 64}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_expand_index_select_new_zeros_scatter_add_select_view_1`, `triton_poi_fused_add_addmm_expand_index_select_new_zeros_relu_scatter_add_select_view_6`, `triton_poi_fused_new_ones_new_zeros_scatter_add_select_3`。
- `triton_poi_fused_add_addmm_index_select_relu_s_7`：NPU_g01_output_code.py:450；输入 `i64[2, 200000][200000, 1]npu:0`；size_hints=`{'y0': 200000, 'x1': 64}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_expand_index_select_new_zeros_scatter_add_select_view_1`, `triton_poi_fused_add_addmm_expand_index_select_new_zeros_relu_scatter_add_select_view_6`, `triton_poi_fused_new_ones_new_zeros_scatter_add_select_3`。
- `triton_poi_fused_clamp_div_expand_view_4`：NPU_g01_output_code.py:261；输入 `f32[10000, 64][64, 1]npu:0`；size_hints=`{'y0': 10000, 'x1': 64}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_expand_index_select_new_zeros_scatter_add_select_view_1`, `triton_poi_fused_add_addmm_expand_index_select_new_zeros_relu_scatter_add_select_view_6`, `triton_poi_fused_clamp_div_expand_view_4`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=1；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 1.593750 | 1.140618 | 28.43% |
| GPU | 1.330825 | 0.468717 | 64.78% |

达到 GPU **整网**提升比时，NPU 整步目标为 0.561319 ms；当前相对该目标的差额为 0.579300 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_sage_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_sage_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_sage_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_sage_Train/GPU_wrapper_calls.txt)。

### 1. scatter_add 外部调用阻断 gather/mul 融合

**优化入口：NPU Inductor aten.scatter_add lowering、scatter/gather scheduler fusion。** NPU wrapper 保留 aten.scatter_add 外部调用及独立 gather/index_select；GPU 生成含 scatter_add 的 Triton kernel，可连同 index_select/mul/new_zeros 融合。优先补这个具体 lowering/融合边界；原子写入实现再由 Triton_ascend/CANN 承接。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnScatterAdd_ScatterElements_ScatterElementsV2` | 1.0 | 0.032990 | 2.89% |
| `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd` | 3.0 | 0.255921 | 22.44% |

本项 kernel/区间合计 **0.288911 ms，占整步 25.33%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.357938 | 0.288911 | ≤19.28% |
| GPU（Eager 下界 / compiled 上界） | 0.339635 | 0.164596 | ≥51.54% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **32.25 个百分点**。GPU fused kernel 的全部耗时均收费给 scatter_add 核心；GPU Eager 仅计 scatter_add_，因此是保守 GPU 收益下界。

NPU Eager 对账项：`aclnnScatterAdd_ScatterElements_ScatterElementsV2`, `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd`。

GPU Eager 对账项：`scatter_add_`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_expand_index_select_scatter_add_select_view_3` | 3.0 | 0.158199 | 33.75% |
| `triton_poi_fused_new_ones_new_zeros_scatter_add_select_1` | 1.0 | 0.005463 | 1.17% |
| `triton_poi_fused_new_ones_new_zeros_scatter_add_select_0` | 1.0 | 0.000935 | 0.20% |

**贡献粗估：0.115444 ms/step，相当于当前 NPU Inductor 整步耗时的 10.12%，使整网提升比增加 7.244 个百分点，单独达到该预算后整网提升比约为 35.68%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnScatterAdd_ScatterElements_ScatterElementsV2`：外部调用；见 wrapper 调用边界。
- `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd`：外部调用；见 wrapper 调用边界。

### 2. 独立 sum 生成归约

**优化入口：NPU Inductor codegen/reduction tiling、runtime autotune。** 选中 kernel 仅有 sum 和 view/transpose。检查输入 stride、归约维大小、split_axis、persistent/分阶段选择，与 GPU 归约拆分对照。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_sum_0` | 1.0 | 0.039793 | 3.49% |

本项 kernel/区间合计 **0.039793 ms，占整步 3.49%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.021557 | 0.039793 | ≤-84.59% |
| GPU（Eager 下界 / compiled 上界） | 0.021111 | 0.014490 | ≥31.37% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **115.96 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnReduceSum_ReduceSumOpAiCore_ReduceSum`, `aclnnReduceSum_BroadcastToAiCore_BroadcastTo`。

GPU Eager 对账项：`sum`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_sum_0` | 3.0 | 0.006400 | 1.37% |
| `sum` | 1.0 | 0.004240 | 0.90% |
| `triton_red_fused_sum_1` | 3.0 | 0.003850 | 0.82% |

**贡献粗估：0.024997 ms/step，相当于当前 NPU Inductor 整步耗时的 2.19%，使整网提升比增加 1.568 个百分点，单独达到该预算后整网提升比约为 30.00%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused_sum_0`：NPU_g01_output_code.py:55；输入 `f32[10000, 64][64, 1]npu:0`；size_hints=`{'x0': 64, 'r1': 10000}`；split_axis=`[0]`。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_red_fused_add_sum_threshold_backward_3` | 0.269052 / 2.0 | 23.59% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_index_select_select_3` | 0.159587 / 3.0 | 13.99% | NPU Inductor gather/scatter lowering 与 producer-consumer 融合 |
| `triton_poi_fused_clamp_div_expand_gather_index_2` | 0.148314 / 2.0 | 13.00% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_red_fused_add_sum_threshold_backward_3`：NPU_g01_output_code.py:281；输入 `f32[10000, 64][64, 1]npu:0`；size_hints=`{'x0': 64, 'r1': 10000}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_threshold_backward_4`, `triton_red_fused_sum_0`, `triton_poi_fused_add_copy__0`。
- `triton_poi_fused_index_select_select_3`：NPU_g02_output_code.py:197；输入 `i64[2, 200000][200000, 1]npu:0`；size_hints=`{'y0': 200000, 'x1': 64}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_expand_index_select_scatter_add_select_view_3`, `triton_poi_fused_clamp_div_expand_gather_index_add_select_view_3`, `triton_poi_fused_new_ones_new_zeros_scatter_add_select_1`。
- `triton_poi_fused_clamp_div_expand_gather_index_2`：NPU_g01_output_code.py:207；输入 `i64[2, 200000][200000, 1]npu:0`；size_hints=`{'y0': 200000, 'x1': 64}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_expand_index_select_scatter_add_select_view_3`, `triton_poi_fused_clamp_div_expand_gather_index_add_select_view_3`, `triton_poi_fused_clamp_div_expand_view_4`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
