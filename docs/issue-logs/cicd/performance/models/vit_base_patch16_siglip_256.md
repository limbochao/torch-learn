---
title: "CICD 性能分析：vit_base_patch16_siglip_256"
---

# vit_base_patch16_siglip_256：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=128；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 282.208470 | 290.055040 | -2.78% |
| GPU | 270.369855 | 247.376048 | 8.50% |

达到 GPU **整网**提升比时，NPU 整步目标为 258.207839 ms；当前相对该目标的差额为 31.847201 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/vit_base_patch16_siglip_256_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/vit_base_patch16_siglip_256_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/vit_base_patch16_siglip_256_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/vit_base_patch16_siglip_256_Infer/GPU_wrapper_calls.txt)。

### 1. GELU 前向外部调用边界

**优化入口：NPU Inductor decomposition/lowering、scheduler 的 bias/GELU 融合。** NPU 保留 ACLNN GELU；GPU 将 GELU 与相邻 bias/pointwise 合入生成 kernel。优化具体外部调用边界，而非仅调整 ACLNN GELU。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnGeluV2_GeluV2_GeluV2` | 13.0 | 7.626856 | 2.63% |

本项 kernel/区间合计 **7.626856 ms，占整步 2.63%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 7.060981 | 7.626856 | ≤-8.01% |
| GPU（Eager 下界 / compiled 上界） | 14.437787 | 14.236444 | ≥1.39% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **9.41 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnGeluV2_GeluV2_GeluV2`。

GPU Eager 对账项：`gelu`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_addmm_gelu_view_2` | 12.0 | 14.233676 | 5.75% |
| `triton_poi_fused_addmm_gelu_view_8` | 1.0 | 0.002768 | 0.00% |

**贡献粗估：0.664344 ms/step，相当于当前 NPU Inductor 整步耗时的 0.23%，使整网提升比增加 0.235 个百分点，单独达到该预算后整网提升比约为 -2.55%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnGeluV2_GeluV2_GeluV2`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_addmm_5` | 6.087939 / 12.0 | 2.10% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_per_fused_add_addmm_native_layer_norm_v_9` | 3.174288 / 10.0 | 1.09% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_per_fused_add_addmm_native_layer_norm_v_8` | 2.304625 / 11.0 | 0.79% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |

- `triton_poi_fused_addmm_5`：NPU_g01_output_code.py:466；输入 `f32[3072][1]npu:0`；size_hints=`{'y0': 32768, 'x1': 3072}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_2`, `triton_per_fused_add_addmm_native_layer_norm_view_4`, `triton_per_fused_add_addmm_native_layer_norm_view_3`。
- `triton_per_fused_add_addmm_native_layer_norm_v_9`：NPU_g01_output_code.py:799；输入 `f32[128, 256, 768][196608, 768, 1]npu:0`；size_hints=`{'y0': 128, 'x1': 256, 'r2': 768}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_2`, `triton_per_fused_add_addmm_native_layer_norm_view_4`, `triton_per_fused_add_addmm_native_layer_norm_view_3`。
- `triton_per_fused_add_addmm_native_layer_norm_v_8`：NPU_g01_output_code.py:700；输入 `f32[128, 256, 768][196608, 768, 1]npu:0`；size_hints=`{'y0': 128, 'x1': 256, 'r2': 768}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_2`, `triton_per_fused_add_addmm_native_layer_norm_view_4`, `triton_per_fused_add_addmm_native_layer_norm_view_3`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=128；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 859.478266 | 880.856709 | -2.49% |
| GPU | 741.976173 | 690.403940 | 6.95% |

达到 GPU **整网**提升比时，NPU 整步目标为 799.738864 ms；当前相对该目标的差额为 81.117845 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/vit_base_patch16_siglip_256_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/vit_base_patch16_siglip_256_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/vit_base_patch16_siglip_256_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/vit_base_patch16_siglip_256_Train/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_gelu_backward_view_12` | 10.690814 / 12.0 | 1.21% | NPU Inductor activation backward tiling/融合；核对 GPU 对应激活及相邻 bias 边界 |
| `triton_red_fused_clone_permute_stack_sum_view_15` | 8.169992 / 12.0 | 0.93% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_add_native_layer_norm_backwar_14` | 7.967106 / 24.0 | 0.90% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |

- `triton_poi_fused_gelu_backward_view_12`：NPU_g01_output_code.py:1029；输入 `f32[32768, 3072][3072, 1]npu:0`；size_hints=`{'z0': 128, 'y1': 256, 'x2': 3072}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_gelu_gelu_backward_view_12`, `triton_red_fused_gelu_gelu_backward_sum_view_13`, `triton_per_fused_gelu_gelu_backward_sum_view_14`。
- `triton_red_fused_clone_permute_stack_sum_view_15`：NPU_g01_output_code.py:1296；输入 `f32[384, 12, 256, 64][196608, 16384, 64, 1]npu:0`；size_hints=`{'x': 4096, 'r0_': 32768}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_clone_permute_stack_16`, `triton_red_fused_gelu_gelu_backward_sum_view_13`, `triton_red_fused__unsafe_view_clone_permute_stack_sum_view_17`。
- `triton_poi_fused_add_native_layer_norm_backwar_14`：NPU_g01_output_code.py:1214；输入 `f32[128, 256, 768][196608, 768, 1]npu:0`；size_hints=`{'y0': 768, 'x2': 128, 'x1': 256}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_red_fused_add_native_layer_norm_backward_view_15`, `triton_red_fused_native_layer_norm_backward_view_9`, `triton_red_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_5`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
