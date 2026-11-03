---
title: "CICD 性能分析：speech_transformer"
---

# speech_transformer：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=1；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 7.046158 | 6.532695 | 7.29% |
| GPU | 5.988075 | 5.247475 | 12.37% |

达到 GPU **整网**提升比时，NPU 整步目标为 6.174695 ms；当前相对该目标的差额为 0.358000 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/speech_transformer_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/speech_transformer_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/speech_transformer_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/speech_transformer_Infer/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused__softmax_div_masked_fill_repe_2` | 0.290110 / 12.0 | 4.44% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_per_fused_add_addmm_mul_native_layer_no_4` | 0.201989 / 30.0 | 3.09% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_addmm_clone_permute_view_1` | 0.119986 / 42.0 | 1.84% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_per_fused__softmax_div_masked_fill_repe_2`：同名/截断名对应 2 个定义（NPU_g02_output_code.py:226, NPU_g07_output_code.py:209）；以下尺寸仅展示首个入口：NPU_g02_output_code.py:226；输入 `b8[10, 204, 204][204, 0, 1]npu:0`；size_hints=`{'y3': 8, 'y4': 10, 'x1': 204, 'r2': 204}`；split_axis=`[0, 1, 2]`。GPU 检查入口：`triton_red_fused__softmax_div_masked_fill_prepare_softmax_online_repeat_2`, `triton_per_fused__softmax_div_masked_fill_prepare_softmax_online_repeat_6`, `triton_per_fused__softmax_div_masked_fill_prepare_softmax_online_repeat_2`。
- `triton_per_fused_add_addmm_mul_native_layer_no_4`：同名/截断名对应 2 个定义（NPU_g02_output_code.py:375, NPU_g07_output_code.py:358）；以下尺寸仅展示首个入口：NPU_g02_output_code.py:375；输入 `f32[512][1]npu:0`；size_hints=`{'y0': 10, 'x1': 204, 'r2': 512}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_per_fused_add_addmm_mul_native_layer_norm_view_4`, `triton_poi_fused_addmm_clone_permute_view_1`, `triton_poi_fused_addmm_relu_view_5`。
- `triton_poi_fused_addmm_clone_permute_view_1`：同名/截断名对应 2 个定义（NPU_g02_output_code.py:150, NPU_g07_output_code.py:133）；以下尺寸仅展示首个入口：NPU_g02_output_code.py:150；输入 `f32[512][1]npu:0`；size_hints=`{'z0': 8, 'y1': 2040, 'x2': 64}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_per_fused_add_addmm_mul_native_layer_norm_view_4`, `triton_poi_fused_addmm_clone_permute_view_1`, `triton_poi_fused_addmm_relu_view_5`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=32；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 21.488555 | 23.387973 | -8.84% |
| GPU | 18.677374 | 17.084066 | 8.53% |

达到 GPU **整网**提升比时，NPU 整步目标为 19.655434 ms；当前相对该目标的差额为 3.732539 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/speech_transformer_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/speech_transformer_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/speech_transformer_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/speech_transformer_Train/GPU_wrapper_calls.txt)。

### 1. prims.fma metadata 重复执行第一组 Mul/Add

**优化入口：PyTorch Inductor inductor_prims.make_prim.meta 与 prims._prim_impl；NPU prims.fma fallback 暴露真实执行。** softmax backward 后连续 Mul/Add/Mul/Add 中第一对来自 metadata 执行真实 impl_aten，随后真实实现再次执行。第一对结果不参与最终输出；GPU 生成 kernel 内执行有效公式，没有 metadata 设备调用。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnMul_MulAiCore_Mul` | 18.0 | 0.114641 | 0.49% |
| `aclnnAdd_AddAiCore_Add` | 18.0 | 0.086621 | 0.37% |

本项 kernel/区间合计 **0.201263 ms，占整步 0.86%**。

仅计算这些相邻窗口中的第一对，不使用整网 Add/Mul 汇总。预算为已证实重复调用的当前设备耗时；修复后整网时间仍须实测。另一个有效 Mul/Add 对不是可全额删除的工作。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__softmax_backward_data_div_masked_fill_native_dropout_backward_threshold_backward_10` | 5.0 | 0.246263 | 1.44% |
| `triton_red_fused__softmax__softmax_backward_data_div_masked_fill_native_dropout_backward_threshold_backward_16` | 1.0 | 0.079941 | 0.47% |
| `triton_per_fused__softmax_backward_data_div_masked_fill_native_dropout_backward_threshold_backward_8` | 6.0 | 0.033950 | 0.20% |
| `triton_per_fused__softmax_backward_data_div_masked_fill_native_dropout_backward_threshold_backward_12` | 5.0 | 0.011363 | 0.07% |
| `triton_per_fused__softmax__softmax_backward_data_div_masked_fill_native_dropout_backward_threshold_backward_17` | 1.0 | 0.002253 | 0.01% |

**贡献粗估：0.201263 ms/step，相当于当前 NPU Inductor 整步耗时的 0.86%，使整网提升比增加 0.937 个百分点，单独达到该预算后整网提升比约为 -7.90%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnMul_MulAiCore_Mul`：外部调用；见 wrapper 调用边界。
- `aclnnAdd_AddAiCore_Add`：外部调用；见 wrapper 调用边界。

历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused_add_mul_native_layer_norm_bac_13` | 0.914926 / 5.0 | 3.91% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_per_fused_native_layer_norm_backward_12` | 0.635622 / 5.0 | 2.72% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_add_copy__1` | 0.372874 / 147.0 | 1.59% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_per_fused_add_mul_native_layer_norm_bac_13`：NPU_g02_output_code.py:1147；输入 `f32[10, 204, 512][104448, 512, 1]npu:0`；size_hints=`{'x0': 512, 'r1': 2040}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_copy__3`, `triton_poi_fused_add_copy__2`, `triton_per_fused_add_addmm_mul_native_dropout_native_layer_norm_native_layer_norm_backward_view_4`。
- `triton_per_fused_native_layer_norm_backward_12`：NPU_g02_output_code.py:1066；输入 `f32[10, 204, 512][104448, 512, 1]npu:0`；size_hints=`{'x0': 512, 'r1': 2040}`；split_axis=`[0]`。GPU 检查入口：`triton_per_fused_add_addmm_mul_native_dropout_native_layer_norm_native_layer_norm_backward_view_4`, `triton_per_fused_add_mul_native_dropout_backward_native_layer_norm_backward_view_12`, `triton_red_fused_add_mul_native_layer_norm_backward_view_15`。
- `triton_poi_fused_add_copy__1`：NPU_g10_output_code.py:115；输入 `f32[512][1]npu:0`；size_hints=`{'x0': 512}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_copy__3`, `triton_poi_fused_add_copy__2`, `triton_per_fused_add_addmm_mul_native_dropout_native_layer_norm_native_layer_norm_backward_view_4`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
