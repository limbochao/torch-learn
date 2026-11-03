---
title: "CICD 性能分析：OPTForCausalLM"
---

# OPTForCausalLM：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；初筛在本地读取和复算；下文临时修改对照另有 NPU_A5 实测，原始统计不混入复测结果。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=2；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 92.748133 | 69.839784 | 24.70% |
| GPU | 54.743527 | 64.170959 | -17.22% |

达到 GPU **整网**提升比时，NPU 整步目标为 108.720374 ms；当前相对该目标的差额为 -38.880590 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/OPTForCausalLM_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/OPTForCausalLM_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/OPTForCausalLM_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/OPTForCausalLM_Infer/GPU_wrapper_calls.txt)。

### 1. log_softmax→NLLLoss 的完整 log probability 写回

**优化入口：NPU Inductor nll_loss_forward decomposition/lowering、producer-consumer fusion。** 外部 NLLLoss 消费 log_softmax 完整矩阵；GPU loss 融合 kernel 可只提取 target logit 和行归约统计。检查 NLLLoss 唯一消费者边界，避免最后一个完整词表循环及中间矩阵写回。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__log_softmax_clone_matmul_sli_13` | 1.0 | 1.272026 | 1.82% |
| `triton_poi_fused_clone_slice_14` | 1.0 | 0.002442 | 0.00% |
| `aclnnInplaceOne_OnesLikeAiCore_OnesLike` | 1.0 | 0.003351 | 0.00% |
| `aclnnNLLLoss_NLLLossAiCore_NLLLoss` | 1.0 | 0.013173 | 0.02% |

本项 kernel/区间合计 **1.290992 ms，占整步 1.85%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 2.379939 | 1.290992 | ≤45.76% |
| GPU（Eager 下界 / compiled 上界） | 2.447325 | 1.130406 | ≥53.81% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **8.06 个百分点**。NPU 为最终 projection 后至 NLLLoss 的逐步精确窗口，包含 shift/clone/copy；GPU Eager 只计 log_softmax+NLLLoss 下界，compiled 包含 prepare 及全部 loss kernel。预算是保守下界。

NPU Eager 对账项：`aclnnInplaceOne_OnesLikeAiCore_OnesLike`, `aclnnInplaceCopy_SliceAiCore_Slice`, `aclnnLogSoftmax_LogSoftmaxAiCore_LogSoftmaxV2`, `aclnnNLLLoss_NLLLossAiCore_NLLLoss`。

GPU Eager 对账项：`_log_softmax`, `nll_loss_forward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__unsafe_view_prepare_softmax_online_view_8` | 1.0 | 1.124281 | 1.75% |
| `triton_red_fused__log_softmax__unsafe_view_clone_constant_pad_nd_nll_loss_forward_slice_view_9` | 1.0 | 0.006125 | 0.01% |

**贡献粗估：0.191712 ms/step，相当于当前 NPU Inductor 整步耗时的 0.27%，使整网提升比增加 0.207 个百分点，单独达到该预算后整网提升比约为 24.91%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused__log_softmax_clone_matmul_sli_13`：NPU_g01_output_code.py:1334；输入 `f32[4096, 50272][50272, 1]npu:0`；size_hints=`{'x2': 2, 'x3': 2047, 'r1': 50272}`；split_axis=`[0, 1]`。
- `triton_poi_fused_clone_slice_14`：NPU_g01_output_code.py:1418；输入 `i64[2, 2048][2048, 1]npu:0`；size_hints=`{'y0': 2, 'x1': 2047}`；split_axis=`[0, 1]`。
- `aclnnInplaceOne_OnesLikeAiCore_OnesLike`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused__softmax__to_copy_add_arange__4` | 6.589630 / 12.0 | 9.44% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_poi_fused_addmm_relu_8` | 0.481705 / 12.0 | 0.69% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_add_addmm_embedding_mul_sub_v_11` | 0.307379 / 10.0 | 0.44% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_per_fused__softmax__to_copy_add_arange__4`：NPU_g01_output_code.py:394；输入 `f32[24, 2048, 2048][4194304, 2048, 1]npu:0`；size_hints=`{'y0': 24, 'x1': 2048, 'r2': 2048}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_red_fused__scaled_dot_product_efficient_attention_add_addmm_all_arange_bitwise_not_clone_eq_expand_full_gt_masked_fill_mul_ones_transpose_triu_unsqueeze_view_1`, `triton_per_fused__to_copy_add_addmm_cumsum_embedding_mul_native_layer_norm_ones_sub_view_6`, `triton_poi_fused__scaled_dot_product_efficient_attention_add_addmm_all_arange_bitwise_not_clone_eq_expand_full_gt_masked_fill_mul_ones_transpose_triu_unsqueeze_view_2`。
- `triton_poi_fused_addmm_relu_8`：NPU_g01_output_code.py:718；输入 `f32[3072][1]npu:0`；size_hints=`{'y0': 4096, 'x1': 3072}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_relu_4`, `triton_red_fused__scaled_dot_product_efficient_attention_add_addmm_all_arange_bitwise_not_clone_eq_expand_full_gt_masked_fill_mul_ones_transpose_triu_unsqueeze_view_1`, `triton_per_fused__to_copy_add_addmm_cumsum_embedding_mul_native_layer_norm_ones_sub_view_6`。
- `triton_poi_fused_add_addmm_embedding_mul_sub_v_11`：NPU_g01_output_code.py:1020；输入 `f32[2, 2048, 768][1572864, 768, 1]npu:0`；size_hints=`{'y0': 4096, 'x1': 768}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_relu_4`, `triton_red_fused__scaled_dot_product_efficient_attention_add_addmm_all_arange_bitwise_not_clone_eq_expand_full_gt_masked_fill_mul_ones_transpose_triu_unsqueeze_view_1`, `triton_per_fused__to_copy_add_addmm_cumsum_embedding_mul_native_layer_norm_ones_sub_view_6`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=2；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 258.887296 | 301.438821 | -16.44% |
| GPU | 163.739390 | 206.727004 | -26.25% |

达到 GPU **整网**提升比时，NPU 整步目标为 326.854735 ms；当前相对该目标的差额为 -25.415914 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/OPTForCausalLM_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/OPTForCausalLM_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/OPTForCausalLM_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/OPTForCausalLM_Train/GPU_wrapper_calls.txt)。

### 1. ReLU backward 与 bias 梯度归约

**优化入口：NPU Inductor scheduler 融合决策、codegen/reduction tiling。** 具体检查 threshold_backward 输出写回与沿 batch/edge 维 sum 的联合调度；GPU 将 pointwise 与 reduction 分拆或采用不同归约 kernel。重点是归约 kernel 的并行度和多阶段策略。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused_sum_threshold_backward_1` | 12.0 | 18.747805 | 6.22% |

本项 kernel/区间合计 **18.747805 ms，占整步 6.22%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 1.701854 | 18.747805 | ≤-1001.61% |
| GPU（Eager 下界 / compiled 上界） | 3.343242 | 2.502012 | ≥25.16% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **1026.77 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnReduceSum_ReduceSumOpAiCore_ReduceSum`, `aclnnThresholdBackward_ReluGrad_ReluGrad`。

GPU Eager 对账项：`threshold_backward`, `sum`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_threshold_backward_3` | 12.0 | 1.516183 | 0.73% |
| `triton_red_fused_sum_1` | 60.0 | 0.310223 | 0.15% |
| `triton_per_fused_sum_2` | 108.0 | 0.264696 | 0.13% |
| `sum` | 12.0 | 0.198203 | 0.10% |
| `triton_red_fused_sum_4` | 12.0 | 0.183557 | 0.09% |
| `triton_per_fused_sum_5` | 12.0 | 0.029150 | 0.01% |

**贡献粗估：17.474172 ms/step，相当于当前 NPU Inductor 整步耗时的 5.80%，使整网提升比增加 6.750 个百分点，单独达到该预算后整网提升比约为 -9.69%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_per_fused_sum_threshold_backward_1`：NPU_g04_output_code.py:132；输入 `f32[4096, 3072][3072, 1]npu:0`；size_hints=`{'x0': 3072, 'r1': 4096}`；split_axis=`[0]`。

### 2. prims.fma metadata 重复执行第一组 Mul/Add

**优化入口：PyTorch Inductor inductor_prims.make_prim.meta 与 prims._prim_impl；NPU prims.fma fallback 暴露真实执行。** softmax backward 后连续 Mul/Add/Mul/Add 中第一对来自 metadata 执行真实 impl_aten，随后真实实现再次执行。第一对结果不参与最终输出；GPU 生成 kernel 内执行有效公式，没有 metadata 设备调用。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnMul_MulAiCore_Mul` | 12.0 | 6.876388 | 2.28% |
| `aclnnAdd_AddAiCore_Add` | 12.0 | 10.101049 | 3.35% |

本项 kernel/区间合计 **16.977438 ms，占整步 5.63%**。

仅计算这些相邻窗口中的第一对，不使用整网 Add/Mul 汇总。预算为已证实重复调用的当前设备耗时；修复后整网时间仍须实测。另一个有效 Mul/Add 对不是可全额删除的工作。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `_efficient_attention_backward` | 12.0 | 73.902843 | 35.75% |

**贡献粗估：16.977438 ms/step，相当于当前 NPU Inductor 整步耗时的 5.63%，使整网提升比增加 6.558 个百分点，单独达到该预算后整网提升比约为 -9.88%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnMul_MulAiCore_Mul`：外部调用；见 wrapper 调用边界。
- `aclnnAdd_AddAiCore_Add`：外部调用；见 wrapper 调用边界。

历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。

<!-- BEGIN CICD VALIDATION OPTForCausalLM_Train:fma_duplicate -->
<a id="fma-fallback-validation"></a>

#### 临时删除 FMA fallback：整网净省 35.523 ms/step

**已验证的优化入口是 NPU Inductor `lowering_fallback_list.py`，只删除 `prims.fma`、`prims.fma.default` 两个条目。** 无需修改 Triton_ascend 或 npuir，本次即可消除重复调用并恢复有效计算的融合。

实测在 2026-11-03 的 NPU_A5 物理卡 2、`pta_213_cicd` 上完成，2026-11-04 整理结果。场景为 Train、FP32、static、batch=2、sequence=2048，关闭 aclgraph。每份 profile 采样 10 个稳定 step，以下均为设备 kernel duration 之和，首次编译时间不计入。

| 进程内配置 | Eager 实测 ms/step | Inductor 实测 ms/step | 本组提升比 `(E-I)/E` |
| --- | ---: | ---: | ---: |
| 原始 fallback 列表 | 259.228504 | 301.702054 | -16.38% |
| 仅删除 FMA 两项 | 267.535271 | 266.179190 | 0.51% |

**整网实际减少 35.522864 ms/step，设备整步耗时下降 11.77%。** 两组 Inductor 的稳定步范围分别为 301.554–301.862 ms 和 265.985–266.305 ms。
两份 Eager 的调用次数逐 kernel 完全一致，但耗时有 3.20% 波动。因此，归因该修改对编译提升比的贡献时，固定使用基线 Eager=259.228504 ms：增量为 **13.703 个百分点**，折算提升比从 -16.38% 到 -2.68%；不能把表中混入 Eager 波动的全部提升比变化算作修复收益。

**原文的 16.977438 ms 是重复第一对的历史成本，不是删除 fallback 的全部收益上限。** 本次重复第一对为 16.906731 ms，数值复现；但删除 fallback 还改变了有效计算的融合边界。按同一组 12 层调用对账：

| 具体 kernel / 边界，均为每步合计 | 修改前 ms | 修改后 ms | 调用次数/step |
| --- | ---: | ---: | ---: |
| `triton_per_fused__softmax_backward_data_7` | 13.352061 | 7.254623 | 12 → 12 |
| FMA 外部调用触发的 Mul/Add/Mul/Add | 33.528020 | 0 | 48 → 0 个设备调用 |
| 原 `triton_poi_fused_div_masked_fill_threshold_bac_8` → 新 `triton_poi_fused__softmax_backward_data_div_ma_8` | 8.055639 | 11.243889 | 12 → 12 |
| **上述完整区间** | **54.935720** | **18.498512** | — |

四个外部算子的 33.528020 ms 中，第一对 **16.906731 ms** 是 metadata 重复执行，第二对 **16.621289 ms** 是有效 FMA 实现。修改后有效公式仍执行，只是移入新 `_8`；不能把四个调用全部消失理解为有效计算被删除。目标区间净省 **36.437208 ms**，其余 kernel 合计增加 0.914344 ms，最终整网净省 35.522864 ms。

从 Inductor 生成代码看，旧 `_7` 同时写回完整矩阵 `u=p*g`、`-p` 和行和 `s=sum(u)`，供外部 `prims.fma` 消费。移除 fallback 后，现有 pointwise lowering 生效：新 `_7` 只写行和；新 `_8` 在使用元素时重算 `u`，直接执行：

```python
# 新 _8 中的原始 DSL，保持原格式
                    tmp3 = -tmp2
                    tmp6 = tmp5 * tmp2
                    tmp7 = tl.fma(tmp3, tmp4, tmp6)
```

这里 `tmp2=p`、`tmp5=g`、`tmp4=s`，即 `g*p-p*s`，随后继续执行原有 div/masked-fill。归约 kernel 的矩阵中间结果写回减少，FMA 的有效 Mul/Add 也不再经过外部调用。这解释了为什么实际收益超过仅删除 metadata 第一对的 16.907 ms。

**责任结论：NPU fallback 策略造成了外部执行边界；社区 `make_prim.meta` 在这个边界上执行真实实现，使第一对重复。** 本次仅调整 NPU fallback 列表，就复用了现有 `make_pointwise` lowering、scheduler 融合和 `tl.fma` codegen。它验证了这个优化入口的收益，没有修复社区 metadata 实现本身，也没有证据要求先修改 Triton_ascend/npuir。

独立整网 accuracy 使用原 runner 的 FP32、batch=2、训练 2 步、默认 `tol=0.01`，结果为 **`pass_accuracy`**；logits 最大绝对误差 1.3113022e-5，梯度 196/196 通过、最大绝对误差 5.8484264e-5，更新后参数 196/196 通过。accuracy 中包含编译的 step 计时不用于上述性能结论。

环境为 torch `2.13.0+cpu`、torch_npu `2.13.0.dev20261030`、Triton `3.6.0`、transformers `4.36.0`。安装的 codegen、lowering、fallback list 与 `test/cicd_benchmark@b3bd74035189` 一致；decomposition 有一份既存的 matmul backward view/squeeze 改动，两组都保留，见[版本校验](../../../../../artifacts/cicd/performance/fma-seed-validation-20261103/installed_identity.json)及[已有差异](../../../../../artifacts/cicd/performance/fma-seed-validation-20261103/decomposition_difference.patch)。全部修改仅在进程内生效；各组运行结束均确认安装源码和 runner 文件未变化。

证据：[修改前完整 kernel](../../../../../artifacts/cicd/performance/fma-seed-validation-20261103/OPTForCausalLM_baseline/selected_kernels.txt)、[修改后完整 kernel](../../../../../artifacts/cicd/performance/fma-seed-validation-20261103/OPTForCausalLM_no_fma_fallback/selected_kernels.txt)、[旧调用边界](../../../../../artifacts/cicd/performance/fma-seed-validation-20261103/OPTForCausalLM_baseline/wrapper_calls.txt)、[新调用边界](../../../../../artifacts/cicd/performance/fma-seed-validation-20261103/OPTForCausalLM_no_fma_fallback/wrapper_calls.txt)、[逐次四算子窗口](../../../../../artifacts/cicd/performance/fma-seed-validation-20261103/OPTForCausalLM_baseline/fma_windows.csv)、[10 步统计](../../../../../artifacts/cicd/performance/fma-seed-validation-20261103/summary.json)、[accuracy 日志](../../../../../artifacts/cicd/performance/fma-seed-validation-20261103/OPTForCausalLM_no_fma_fallback_accuracy/run.log)、[临时实验脚本](../../../../../scripts/repro/cicd/performance/fma_seed_validation/README.md)。wrapper 中一个调用位置可被多层复用，次数以 profile 的 12 次/step 为准。
<!-- END CICD VALIDATION OPTForCausalLM_Train:fma_duplicate -->

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused__softmax_backward_data_7` | 13.801833 / 12.0 | 4.58% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_poi_fused_div_masked_fill_threshold_bac_8` | 8.007771 / 12.0 | 2.66% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_per_fused__softmax_add_lift_fresh_maxim_4` | 7.793883 / 12.0 | 2.59% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |

- `triton_per_fused__softmax_backward_data_7`：NPU_g04_output_code.py:616；输入 `f32[24, 2048, 2048][4194304, 2048, 1]npu:0`；size_hints=`{'y0': 24, 'x1': 2048, 'r2': 2048}`；split_axis=`[0, 1]`。GPU 检查入口：需对照外部 ACLNN/CUDA 调用和 FX 图。
- `triton_poi_fused_div_masked_fill_threshold_bac_8`：NPU_g04_output_code.py:690；输入 `b8[2, 12, 2048, 2048][50331648, 4194304, 2048, 1]npu:0`；size_hints=`{'t0': 2, 'z1': 12, 'y2': 2048, 'x3': 2048}`；split_axis=`[0, 1, 2]`。GPU 检查入口：`triton_poi_fused_threshold_backward_3`, `triton_red_fused_add_all_arange_bitwise_not_clone_eq_expand_full_gt_masked_fill_mul_triu_unsqueeze_view_0`。
- `triton_per_fused__softmax_add_lift_fresh_maxim_4`：NPU_g05_output_code.py:385；输入 `f32[24, 2048, 2048][4194304, 2048, 1]npu:0`；size_hints=`{'y3': 2, 'y4': 12, 'x1': 2048, 'r2': 2048}`；split_axis=`[0, 1, 2]`。GPU 检查入口：`triton_red_fused__log_softmax__log_softmax_backward_data_add_arange_clone_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_view_0`, `triton_per_fused_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_6`, `triton_poi_fused_add_addmm_native_dropout_view_4`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
