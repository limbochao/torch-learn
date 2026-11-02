---
title: "CICD 性能分析：deit_tiny_patch16_224.fb_in1k"
---

# deit_tiny_patch16_224.fb_in1k：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=128；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 19.665136 | 20.283000 | -3.14% |
| GPU | 22.124624 | 20.425061 | 7.68% |

达到 GPU **整网**提升比时，NPU 整步目标为 18.154506 ms；当前相对该目标的差额为 2.128495 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/deit_tiny_patch16_224.fb_in1k_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/deit_tiny_patch16_224.fb_in1k_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/deit_tiny_patch16_224.fb_in1k_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/deit_tiny_patch16_224.fb_in1k_Infer/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `aclnnGeluV2_GeluV2_GeluV2` | 1.364944 / 12.0 | 6.73% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_per_fused_add_addmm_native_layer_norm_v_8` | 0.645595 / 10.0 | 3.18% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_addmm_5` | 0.632641 / 12.0 | 3.12% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `aclnnGeluV2_GeluV2_GeluV2`：外部调用；见 wrapper 调用边界。GPU 检查入口：需对照外部 ACLNN/CUDA 调用和 FX 图。
- `triton_per_fused_add_addmm_native_layer_norm_v_8`：NPU_g01_output_code.py:691；输入 `f32[128, 197, 192][37824, 192, 1]npu:0`；size_hints=`{'y0': 128, 'x1': 197, 'r2': 192}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_2`, `triton_per_fused_add_addmm_native_layer_norm_view_4`, `triton_per_fused_add_addmm_native_layer_norm_view_3`。
- `triton_poi_fused_addmm_5`：NPU_g01_output_code.py:418；输入 `f32[768][1]npu:0`；size_hints=`{'y0': 25216, 'x1': 768}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_2`, `triton_per_fused_add_addmm_native_layer_norm_view_4`, `triton_per_fused_add_addmm_native_layer_norm_view_3`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=128；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 58.198861 | 106.215521 | -82.50% |
| GPU | 70.502921 | 64.649292 | 8.30% |

达到 GPU **整网**提升比时，NPU 整步目标为 53.366799 ms；当前相对该目标的差额为 52.848722 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/deit_tiny_patch16_224.fb_in1k_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/deit_tiny_patch16_224.fb_in1k_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/deit_tiny_patch16_224.fb_in1k_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/deit_tiny_patch16_224.fb_in1k_Train/GPU_wrapper_calls.txt)。

### 1. LayerNorm backward 归约调度

**优化入口：NPU Inductor codegen/reduction tiling 与 runtime autotune。** 检查该 kernel 的 reduction size_hints、split_axis 和 program 并行度；GPU 对应归约 kernel 列于本项。优先检查归约维切分与多阶段归约，不能据此断言 Triton_ascend/npuir 有缺陷。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_native_layer_norm_backward_vi_7` | 24.0 | 15.056300 | 14.18% |

本项 kernel/区间合计 **15.056300 ms，占整步 14.18%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 1.248193 | 15.056300 | ≤-1106.25% |
| GPU（Eager 下界 / compiled 上界） | 3.814364 | 4.664185 | ≥-22.28% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **1083.97 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnLayerNormBackward_LayerNormGradV3_LayerNormGradV3`。

GPU Eager 对账项：`native_layer_norm_backward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused_add_native_layer_norm_backward_view_10` | 23.0 | 1.947891 | 3.01% |
| `triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_5` | 10.0 | 0.880223 | 1.36% |
| `triton_red_fused_native_layer_norm_backward_view_11` | 23.0 | 0.758567 | 1.17% |
| `triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_4` | 11.0 | 0.578166 | 0.89% |
| `triton_red_fused_native_layer_norm_backward_select_backward_3` | 73.0 | 0.122144 | 0.19% |
| `triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_3` | 1.0 | 0.092498 | 0.14% |
| `triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_15` | 1.0 | 0.091126 | 0.14% |
| `triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_6` | 1.0 | 0.062645 | 0.10% |
| `triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_1` | 1.0 | 0.056418 | 0.09% |
| `triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_16` | 1.0 | 0.048174 | 0.07% |
| `triton_red_fused_native_layer_norm_backward_select_backward_1` | 1.0 | 0.014445 | 0.02% |
| `triton_red_fused_native_layer_norm_backward_select_backward_2` | 1.0 | 0.007840 | 0.01% |
| `triton_red_fused_native_layer_norm_backward_select_backward_4` | 1.0 | 0.002759 | 0.00% |
| `triton_per_fused_native_layer_norm_backward_select_backward_5` | 1.0 | 0.001289 | 0.00% |

**贡献粗估：13.530016 ms/step，相当于当前 NPU Inductor 整步耗时的 12.74%，使整网提升比增加 23.248 个百分点，单独达到该预算后整网提升比约为 -59.26%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused_native_layer_norm_backward_vi_7`：NPU_g01_output_code.py:681；输入 `f32[25216, 192][192, 1]npu:0`；size_hints=`{'x0': 192, 'r1': 25216}`；split_axis=`[0]`。

### 2. 独立 sum 生成归约

**优化入口：NPU Inductor codegen/reduction tiling、runtime autotune。** 选中 kernel 仅有 sum 和 view/transpose。检查输入 stride、归约维大小、split_axis、persistent/分阶段选择，与 GPU 归约拆分对照。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused_sum_0` | 1.0 | 0.021840 | 0.02% |
| `triton_red_fused_sum_view_3` | 24.0 | 4.664207 | 4.39% |
| `triton_per_fused_sum_10` | 1.0 | 0.026905 | 0.03% |

本项 kernel/区间合计 **4.712951 ms，占整步 4.44%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.818338 | 4.712951 | ≤-475.92% |
| GPU（Eager 下界 / compiled 上界） | 2.280925 | 1.085479 | ≥52.41% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **528.33 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnReduceSum_ReduceSumOpAiCore_ReduceSum`, `aclnnReduceSum_BroadcastToAiCore_BroadcastTo`。

GPU Eager 对账项：`sum`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_gelu_gelu_backward_sum_view_8` | 12.0 | 0.337408 | 0.52% |
| `sum` | 13.0 | 0.291256 | 0.45% |
| `triton_red_fused__unsafe_view_clone_permute_stack_sum_view_13` | 12.0 | 0.231174 | 0.36% |
| `triton_red_fused_sum_view_6` | 24.0 | 0.177352 | 0.27% |
| `triton_red_fused_gelu_gelu_backward_sum_view_9` | 12.0 | 0.021418 | 0.03% |
| `triton_red_fused__unsafe_view_clone_permute_stack_sum_view_14` | 12.0 | 0.016570 | 0.03% |
| `triton_red_fused_sum_17` | 1.0 | 0.007197 | 0.01% |
| `triton_red_fused_sum_0` | 1.0 | 0.001565 | 0.00% |
| `triton_red_fused_slice_sum_18` | 1.0 | 0.001539 | 0.00% |

**贡献粗估：4.323509 ms/step，相当于当前 NPU Inductor 整步耗时的 4.07%，使整网提升比增加 7.429 个百分点，单独达到该预算后整网提升比约为 -75.08%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_per_fused_sum_0`：NPU_g01_output_code.py:55；输入 `f32[128, 1000][1000, 1]npu:0`；size_hints=`{'x0': 1000, 'r1': 128}`；split_axis=`[0]`。
- `triton_red_fused_sum_view_3`：NPU_g01_output_code.py:318；输入 `f32[128, 197, 192][37824, 192, 1]npu:0`；size_hints=`{'x0': 192, 'r1': 25216}`；split_axis=`[0]`。
- `triton_per_fused_sum_10`：NPU_g01_output_code.py:880；输入 `f32[128, 197, 192][37824, 192, 1]npu:0`；size_hints=`{'y0': 197, 'x1': 192, 'r2': 128}`；split_axis=`[0]`。

### 3. GELU backward 与 sum 梯度归约

**优化入口：NPU Inductor scheduler、codegen/reduction tiling。** 具体检查融合 gelu_backward 后的 bias sum；GPU 相应融合/归约成本全部计入上界。优化输出维分块、归约维切分和融合取舍。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_gelu_backward_sum_view_5` | 12.0 | 2.115231 | 1.99% |

本项 kernel/区间合计 **2.115231 ms，占整步 1.99%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 2.637364 | 2.115231 | ≤19.80% |
| GPU（Eager 下界 / compiled 上界） | 5.805904 | 4.268687 | ≥26.48% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **6.68 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnReduceSum_ReduceSumOpAiCore_ReduceSum`, `aclnnReduceSum_BroadcastToAiCore_BroadcastTo`, `aclnnGeluBackwardV2_GeluGradV2AiCore_GeluGradV2`。

GPU Eager 对账项：`gelu_backward`, `sum`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_gelu_gelu_backward_view_7` | 12.0 | 3.183208 | 4.92% |
| `triton_red_fused_gelu_gelu_backward_sum_view_8` | 12.0 | 0.337408 | 0.52% |
| `sum` | 13.0 | 0.291256 | 0.45% |
| `triton_red_fused__unsafe_view_clone_permute_stack_sum_view_13` | 12.0 | 0.231174 | 0.36% |
| `triton_red_fused_sum_view_6` | 24.0 | 0.177352 | 0.27% |
| `triton_red_fused_gelu_gelu_backward_sum_view_9` | 12.0 | 0.021418 | 0.03% |
| `triton_red_fused__unsafe_view_clone_permute_stack_sum_view_14` | 12.0 | 0.016570 | 0.03% |
| `triton_red_fused_sum_17` | 1.0 | 0.007197 | 0.01% |
| `triton_red_fused_sum_0` | 1.0 | 0.001565 | 0.00% |
| `triton_red_fused_slice_sum_18` | 1.0 | 0.001539 | 0.00% |

**贡献粗估：0.176156 ms/step，相当于当前 NPU Inductor 整步耗时的 0.17%，使整网提升比增加 0.303 个百分点，单独达到该预算后整网提升比约为 -82.20%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused_gelu_backward_sum_view_5`：NPU_g01_output_code.py:500；输入 `f32[128, 197, 768][151296, 768, 1]npu:0`；size_hints=`{'x0': 768, 'r1': 25216}`；split_axis=`[0]`。

### 额外具体检查点

`triton_red_fused_native_layer_norm_backward_vi_7` 计算参数梯度，按 192 个通道分 program，在 r1=25216 的循环中累计两个归约，load 索引为 x0+192*r1。这是参数梯度的大外轴归约，并非 LayerNorm 输入梯度的逐行归约。具体应检查 codegen 的归约维切分、多阶段参数梯度与双输出调度；不要泛化成所有 LayerNorm 都慢。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_red_fused_clone_permute_stack_sum_view_9` | 4.419689 / 12.0 | 4.16% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_per_fused_add_native_layer_norm_backwar_6` | 3.924140 / 24.0 | 3.69% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_gelu_backward_view_4` | 2.621630 / 12.0 | 2.47% | NPU Inductor activation backward tiling/融合；核对 GPU 对应激活及相邻 bias 边界 |

- `triton_red_fused_clone_permute_stack_sum_view_9`：NPU_g01_output_code.py:817；输入 `f32[384, 3, 197, 64][37824, 12608, 64, 1]npu:0`；size_hints=`{'x': 1024, 'r0_': 32768}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_clone_permute_stack_12`, `triton_red_fused_gelu_gelu_backward_sum_view_8`, `triton_red_fused__unsafe_view_clone_permute_stack_sum_view_13`。
- `triton_per_fused_add_native_layer_norm_backwar_6`：NPU_g01_output_code.py:589；输入 `f32[25216, 192][192, 1]npu:0`；size_hints=`{'y0': 128, 'x1': 197, 'r2': 192}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_per_fused_add_native_layer_norm_backward_view_10`, `triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_5`, `triton_red_fused_native_layer_norm_backward_view_11`。
- `triton_poi_fused_gelu_backward_view_4`：NPU_g01_output_code.py:399；输入 `f32[25216, 768][768, 1]npu:0`；size_hints=`{'z0': 128, 'y1': 197, 'x2': 768}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_gelu_gelu_backward_view_7`, `triton_red_fused_gelu_gelu_backward_sum_view_8`, `triton_red_fused_gelu_gelu_backward_sum_view_9`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
