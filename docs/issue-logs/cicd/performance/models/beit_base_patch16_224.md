---
title: "CICD 性能分析：beit_base_patch16_224"
---

# beit_base_patch16_224：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；本轮仅在本地读取和复算，未连接 NPU_A5、未修复或跑卡。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=128；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 230.820898 | 285.367785 | -23.63% |
| GPU | 200.313101 | 184.412517 | 7.94% |

达到 GPU **整网**提升比时，NPU 整步目标为 212.498646 ms；当前相对该目标的差额为 72.869139 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/beit_base_patch16_224_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/beit_base_patch16_224_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/beit_base_patch16_224_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/beit_base_patch16_224_Infer/GPU_wrapper_calls.txt)。

### 1. GELU 前向外部调用边界

**优化入口：NPU Inductor decomposition/lowering、scheduler 的 bias/GELU 融合。** NPU 保留 ACLNN GELU；GPU 将 GELU 与相邻 bias/pointwise 合入生成 kernel。优化具体外部调用边界，而非仅调整 ACLNN GELU。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnGeluV2_GeluV2_GeluV2` | 12.0 | 5.992801 | 2.10% |

本项 kernel/区间合计 **5.992801 ms，占整步 2.10%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 5.469227 | 5.992801 | ≤-9.57% |
| GPU（Eager 下界 / compiled 上界） | 11.029800 | 10.812802 | ≥1.97% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **11.54 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnGeluV2_GeluV2_GeluV2`。

GPU Eager 对账项：`gelu`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_addmm_gelu_view_4` | 12.0 | 10.812802 | 5.86% |

**贡献粗估：0.631174 ms/step，相当于当前 NPU Inductor 整步耗时的 0.22%，使整网提升比增加 0.273 个百分点，单独达到该预算后整网提升比约为 -23.36%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnGeluV2_GeluV2_GeluV2`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_addmm_10` | 4.479281 / 12.0 | 1.57% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_per_fused_add_addmm_native_layer_norm_12` | 2.377457 / 10.0 | 0.83% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_addmm_matmul_mul_permute_tran_6` | 2.364787 / 12.0 | 0.83% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_poi_fused_addmm_10`：NPU_g01_output_code.py:842；输入 `f32[3072][1]npu:0`；size_hints=`{'y0': 25216, 'x1': 3072}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_4`, `triton_per_fused_add_addmm_mul_native_layer_norm_view_6`, `triton_per_fused_add_addmm_mul_native_layer_norm_view_5`。
- `triton_per_fused_add_addmm_native_layer_norm_12`：NPU_g01_output_code.py:1032；输入 `f32[128, 197, 768][151296, 768, 1]npu:0`；size_hints=`{'y0': 128, 'x1': 197, 'r2': 768}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_4`, `triton_per_fused_add_addmm_mul_native_layer_norm_view_6`, `triton_per_fused_add_addmm_mul_native_layer_norm_view_5`。
- `triton_poi_fused_addmm_matmul_mul_permute_tran_6`：NPU_g01_output_code.py:466；输入 `f32[2304][1]npu:0`；size_hints=`{'y1': 128, 'y0': 768, 'x2': 197}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_4`, `triton_per_fused_add_addmm_mul_native_layer_norm_view_6`, `triton_per_fused_add_addmm_mul_native_layer_norm_view_5`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=128；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 678.767482 | 717.090532 | -5.65% |
| GPU | 581.189263 | 548.963012 | 5.54% |

达到 GPU **整网**提升比时，NPU 整步目标为 641.130636 ms；当前相对该目标的差额为 75.959896 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/beit_base_patch16_224_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/beit_base_patch16_224_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/beit_base_patch16_224_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/beit_base_patch16_224_Train/GPU_wrapper_calls.txt)。

### 1. prims.fma metadata 重复执行第一组 Mul/Add

**优化入口：PyTorch Inductor inductor_prims.make_prim.meta 与 prims._prim_impl；NPU prims.fma fallback 暴露真实执行。** softmax backward 后连续 Mul/Add/Mul/Add 中第一对来自 metadata 执行真实 impl_aten，随后真实实现再次执行。第一对结果不参与最终输出；GPU 生成 kernel 内执行有效公式，没有 metadata 设备调用。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnMul_MulAiCore_Mul` | 12.0 | 4.263129 | 0.59% |
| `aclnnAdd_AddAiCore_Add` | 12.0 | 6.051381 | 0.84% |

本项 kernel/区间合计 **10.314511 ms，占整步 1.44%**。

仅计算这些相邻窗口中的第一对，不使用整网 Add/Mul 汇总。预算为已证实重复调用的当前设备耗时；修复后整网时间仍须实测。另一个有效 Mul/Add 对不是可全额删除的工作。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `_efficient_attention_backward` | 12.0 | 66.255790 | 12.07% |

**贡献粗估：10.314511 ms/step，相当于当前 NPU Inductor 整步耗时的 1.44%，使整网提升比增加 1.520 个百分点，单独达到该预算后整网提升比约为 -4.13%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnMul_MulAiCore_Mul`：外部调用；见 wrapper 调用边界。
- `aclnnAdd_AddAiCore_Add`：外部调用；见 wrapper 调用边界。

历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。

### 额外具体检查点

`triton_per_fused__safe_softmax_add_clone_index_10` 为明确的大 kernel（34.230 ms），但 GPU Eager/Inductor 均使用 efficient_attention，不能拿 NPU softmax 单独对比 GPU 完整 attention。需要以两次 BMM 及 softmax/dropout 构成完整前向区域再比较；已确认的 metadata 重复调用独立计入前项。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused__safe_softmax_add_clone_index_10` | 34.230398 / 12.0 | 4.77% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_poi_fused_gelu_backward_view_8` | 8.617573 / 12.0 | 1.20% | NPU Inductor activation backward tiling/融合；核对 GPU 对应激活及相邻 bias 边界 |
| `triton_per_fused__softmax_backward_data_matmul_14` | 8.579427 / 12.0 | 1.20% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |

- `triton_per_fused__safe_softmax_add_clone_index_10`：NPU_g02_output_code.py:792；输入 `f32[1536, 197, 197][38809, 197, 1]npu:0`；size_hints=`{'z0': 128, 'y1': 12, 'x2': 197, 'r3': 197}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_per_fused_add_mul_native_layer_norm_backward_view_10`, `triton_poi_fused_clone_permute_stack_13`, `triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_view_7`。
- `triton_poi_fused_gelu_backward_view_8`：NPU_g01_output_code.py:624；输入 `f32[25216, 3072][3072, 1]npu:0`；size_hints=`{'z0': 128, 'y1': 197, 'x2': 3072}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_gelu_gelu_backward_view_7`, `triton_red_fused_gelu_gelu_backward_sum_view_8`, `triton_per_fused_gelu_gelu_backward_sum_view_9`。
- `triton_per_fused__softmax_backward_data_matmul_14`：NPU_g01_output_code.py:1128；输入 `f32[1536, 197, 197][38809, 197, 1]npu:0`；size_hints=`{'z0': 128, 'y1': 12, 'x2': 197, 'r3': 197}`；split_axis=`[0, 1]`。GPU 检查入口：需对照外部 ACLNN/CUDA 调用和 FX 图。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
