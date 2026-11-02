---
title: "CICD 性能分析：BERT_pytorch"
---

# BERT_pytorch：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=16；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 22.604497 | 21.502626 | 4.87% |
| GPU | 18.754696 | 18.134962 | 3.30% |

达到 GPU **整网**提升比时，NPU 整步目标为 21.857548 ms；当前相对该目标的差额为 -0.354922 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/BERT_pytorch_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/BERT_pytorch_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/BERT_pytorch_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/BERT_pytorch_Infer/GPU_wrapper_calls.txt)。

### 1. GELU 前向外部调用边界

**优化入口：NPU Inductor decomposition/lowering、scheduler 的 bias/GELU 融合。** NPU 保留 ACLNN GELU；GPU 将 GELU 与相邻 bias/pointwise 合入生成 kernel。优化具体外部调用边界，而非仅调整 ACLNN GELU。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnGeluV2_GeluV2_GeluV2` | 12.0 | 0.463677 | 2.16% |

本项 kernel/区间合计 **0.463677 ms，占整步 2.16%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.469546 | 0.463677 | ≤1.25% |
| GPU（Eager 下界 / compiled 上界） | 0.292110 | 0.175004 | ≥40.09% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **38.84 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnGeluV2_GeluV2_GeluV2`。

GPU Eager 对账项：`gelu`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_addmm_gelu_view_7` | 12.0 | 0.175004 | 0.97% |

**贡献粗估：0.182371 ms/step，相当于当前 NPU Inductor 整步耗时的 0.85%，使整网提升比增加 0.807 个百分点，单独达到该预算后整网提升比约为 5.68%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnGeluV2_GeluV2_GeluV2`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_red_fused__log_softmax_view_11` | 0.318582 / 1.0 | 1.48% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_per_fused__softmax_eq_gt_masked_fill_re_4` | 0.253683 / 12.0 | 1.18% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_poi_fused_addmm_7` | 0.203644 / 12.0 | 0.95% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_red_fused__log_softmax_view_11`：NPU_g01_output_code.py:969；输入 `f32[2048, 20005][20005, 1]npu:0`；size_hints=`{'y0': 16, 'x1': 128, 'r2': 20005}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_red_fused__log_softmax_prepare_softmax_online_view_11`, `triton_poi_fused__log_softmax_prepare_softmax_online_10`。
- `triton_per_fused__softmax_eq_gt_masked_fill_re_4`：NPU_g01_output_code.py:403；输入 `b8[16, 128, 128][16384, 128, 1]npu:0`；size_hints=`{'z0': 16, 'y1': 12, 'x2': 128, 'r3': 128}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_red_fused__softmax_div_eq_gt_masked_fill_prepare_softmax_online_repeat_unsqueeze_view_4`, `triton_poi_fused_gt_repeat_unsqueeze_0`。
- `triton_poi_fused_addmm_7`：NPU_g01_output_code.py:650；输入 `f32[3072][1]npu:0`；size_hints=`{'y0': 2048, 'x1': 3072}`；split_axis=`[0]`。GPU 检查入口：`triton_per_fused_add_addmm_div_mean_mul_std_sub_view_8`, `triton_poi_fused_addmm_gelu_view_7`, `triton_poi_fused_addmm_clone_transpose_view_2`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=16；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 69.929201 | 71.175177 | -1.78% |
| GPU | 55.919768 | 51.480214 | 7.94% |

达到 GPU **整网**提升比时，NPU 整步目标为 64.377418 ms；当前相对该目标的差额为 6.797759 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/BERT_pytorch_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/BERT_pytorch_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/BERT_pytorch_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/BERT_pytorch_Train/GPU_wrapper_calls.txt)。

### 1. prims.fma metadata 重复执行第一组 Mul/Add

**优化入口：PyTorch Inductor inductor_prims.make_prim.meta 与 prims._prim_impl；NPU prims.fma fallback 暴露真实执行。** softmax backward 后连续 Mul/Add/Mul/Add 中第一对来自 metadata 执行真实 impl_aten，随后真实实现再次执行。第一对结果不参与最终输出；GPU 生成 kernel 内执行有效公式，没有 metadata 设备调用。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnMul_MulAiCore_Mul` | 12.0 | 0.129903 | 0.18% |
| `aclnnAdd_AddAiCore_Add` | 12.0 | 0.109095 | 0.15% |

本项 kernel/区间合计 **0.238998 ms，占整步 0.34%**。

仅计算这些相邻窗口中的第一对，不使用整网 Add/Mul 汇总。预算为已证实重复调用的当前设备耗时；修复后整网时间仍须实测。另一个有效 Mul/Add 对不是可全额删除的工作。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused__softmax_backward_data_div_eq_masked_fill_native_dropout_backward_view_11` | 11.0 | 0.456818 | 0.89% |
| `triton_per_fused__softmax__softmax_backward_data_div_eq_masked_fill_native_dropout_backward_view_18` | 1.0 | 0.050619 | 0.10% |

**贡献粗估：0.238998 ms/step，相当于当前 NPU Inductor 整步耗时的 0.34%，使整网提升比增加 0.342 个百分点，单独达到该预算后整网提升比约为 -1.44%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnMul_MulAiCore_Mul`：外部调用；见 wrapper 调用边界。
- `aclnnAdd_AddAiCore_Add`：外部调用；见 wrapper 调用边界。

历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused_add_div_mul_sum_view_13` | 2.687045 / 12.0 | 3.78% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_per_fused_add_div_mul_sum_view_6` | 1.590248 / 12.0 | 2.23% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_gelu_backward_view_4` | 0.738327 / 12.0 | 1.04% | NPU Inductor activation backward tiling/融合；核对 GPU 对应激活及相邻 bias 边界 |

- `triton_per_fused_add_div_mul_sum_view_13`：NPU_g01_output_code.py:1073；输入 `f32[2048, 768][768, 1]npu:0`；size_hints=`{'x0': 768, 'r1': 2048}`；split_axis=`[0]`。GPU 检查入口：`triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_5`, `triton_poi_fused_add_copy__4`, `triton_per_fused_add_addmm_div_mean_mul_native_dropout_std_sub_view_8`。
- `triton_per_fused_add_div_mul_sum_view_6`：NPU_g01_output_code.py:488；输入 `f32[2048, 768][768, 1]npu:0`；size_hints=`{'x0': 768, 'r1': 2048}`；split_axis=`[0]`。GPU 检查入口：`triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_5`, `triton_poi_fused_add_copy__4`, `triton_per_fused_add_addmm_div_mean_mul_native_dropout_std_sub_view_8`。
- `triton_poi_fused_gelu_backward_view_4`：NPU_g01_output_code.py:321；输入 `f32[2048, 3072][3072, 1]npu:0`；size_hints=`{'z0': 16, 'y1': 128, 'x2': 3072}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_5`, `triton_per_fused_gelu_gelu_backward_native_dropout_backward_sum_view_6`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
