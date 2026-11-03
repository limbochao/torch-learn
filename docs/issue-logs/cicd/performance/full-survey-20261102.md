---
title: CICD 全量性能分析：编译收益差距与共性问题
---

# CICD 全量性能分析：编译收益差距与共性问题

本批共 **40 个模型、78 个场景**（Infer 40、Train 38），已逐场景读取四份设备耗时 CSV，并索引 **4,379 个生成 kernel 定义**及其 wrapper/FX。按模型分别归档，均分 Infer / Train。本轮没有访问 NPU_A5，没有修改编译器。

保守局部筛选与已知重复执行/回退机制共列出 **88 个场景级检查项，覆盖 54 个场景**；每项都给出具体 kernel、代码位置、NPU/GPU 做法、占比及贡献预算。其余场景列出明确候选和缺失证据，不强行认定劣化。

2026-11-03 补充：[MobileBERT 逐 kernel 深挖](models/MobileBertForMaskedLM.md#outer-reduction)已在指定环境与物理卡 1、2 实测；四组确认 IR 分段归约/候选问题，另两组不计入已确认单项缺口。约 50 ms/step 为小图外推。本文原 40 模型统计及粗筛预算保持不变。

## 比较方法与数字边界

`r_N = 1 - N_I/N_E`，`r_G = 1 - G_I/G_E`。只筛选同一语义计算的 `r_N < r_G`；裸 MM/AddMM 若 GPU 编译侧本身没有收益，不以 NPU/GPU 绝对耗时差列为优化点。

局部预算为 `B = N_I - N_E × G_I/G_E`；kernel 占比为 `N_I/T_NI`；达到局部目标对当前整步的耗时贡献为 `B/T_NI`，对整网提升比的贡献为 `100×B/T_NE` 个百分点。**不使用整网 Eager 比例缩放单个 kernel。**

逐调用的 GPU CSV 没有 shape，它是按 kernel 名聚合的。对有混合融合边界的操作族，本轮采用保守办法：NPU Eager 收费整个相关操作族，NPU compiled 只计列出的纯实现/明确融合实现；GPU Eager 只计对应核心操作，GPU compiled 将全部覆盖这些核心操作的生成 kernel 及附带融合工作都收费。故 NPU 提升比是上界、GPU 提升比是下界。若上界仍小于下界，才列入。

这种操作族边界是粗筛，不等于已经逐层逐 shape 对齐。预算依赖核心计算语义可对应；需要用 FX 与逐次 profile 最终确认。文档中的 budget 是目标预算/保守差距估算，**不是已验证可全部回收的收益**。首层 attention 补充了全 12 层收费给首层的保守四方筛选，并另列同设备后续层控制估算；FMA 重复执行使用直接多出的调用成本，不冒充 GPU 局部比例预算。多项预算不可简单相加。

单位为设备 kernel duration 之和的 ms/step，不是主机 wall time。NPU 每场景按 CSV 中 10 个 Step Id 归一化，GPU 按 summary/profile_steps=10 归一化；四份数据 batch 一致。当前 NPU 数据与此前两模型数据有小幅变化，本文全部重新计算。原始 profile 的完整软件提交与构建身份缺失；历史源码/CPU 验证单独引用，不视为当前重新实测。

## 共性问题及优化责任

本表按“占整网缺口”的中位数降序排列。
以下占比按已列出该问题的场景逐个计算，再取中位数及最小–最大范围，不按跨模型毫秒数加权；仅有一个场景时列单值。
**问题耗时占比**为本项 kernel/区间耗时除以当前 NPU Inductor 整步耗时；**劣化预算占比**为本项预算 B 除以同一个整步耗时，表示该项若达到局部目标可减少多少整步耗时。
**占整网缺口**为 B / D，其中 D = T_NI − T_NE × T_GI/T_GE，只统计 D > 0 的场景，表示该项预算可解释多少达到 GPU 整网提升比所需的差距。它与 kernel 耗时占比不同，可能因其他区域的收益抵消而超过 100%；各类存在重叠，不能相加。

| 具体问题 | 场景数 | 问题耗时/整步：中位（范围） | 劣化预算/整步：中位（范围） | 占整网缺口：中位 | 优化入口与 GPU 做法 |
| --- | ---: | ---: | ---: | ---: | --- |
| NFNet avg_pool2d 生成实现 | 1 | 15.0% | 14.4% | 84.4%（1例） | NPU avg_pool2d lowering、pointwise tiling/runtime；先判断是否应回退 ACLNN。 |
| tanh backward+bias sum | 1 | 60.4% | 54.7% | 81.1%（1例） | NPU 输出维很小时的 program 并行度和分阶段归约；GPU pointwise/reduction 分拆。 |
| GELU backward+bias sum | 4 | 4.2%（2.0–4.6%） | 3.4%（0.2–3.7%） | 32.1%（4例） | NPU 融合/归约 tiling；GPU 相应 fused/多阶段归约。 |
| GNN scatter_add 与 gather/mul 分离 | 4 | 39.6%（25.3–58.3%） | 18.5%（10.1–27.2%） | 31.2%（4例） | NPU scatter_add lowering 和 scheduler fusion；GPU 生成含 scatter 的融合 kernel。 |
| MobileBERT outer reduction：分段与候选问题 | 1 | 22.9% | 11.6% | 29.2%（1例） | 四组已确认 IR 分段/候选问题，另有 24 次重复 bias sum；`_20`、`_0` 单独保留边界。见[实测细分](models/MobileBertForMaskedLM.md#outer-reduction)；本行仍为七组历史粗筛。 |
| Infer log_softmax→NLLLoss 完整矩阵写回 | 17 | 1.7%（0.07–10.6%） | 1.0%（0.04–8.3%） | 23.5%（10例） | NPU nll_loss_forward decomposition/lowering 使消费者留在外部调用；GPU 融合 target 提取和归约。 |
| ReLU backward+bias sum | 4 | 6.4%（1.0–50.2%） | 3.0%（0.1–48.7%） | 12.9%（3例） | NPU pointwise/归约融合与并行映射；GPU 分拆或不同归约。 |
| prims.fma metadata 第一组 Mul/Add 重复执行 | 20 | 1.7%（0.2–5.6%） | 1.7%（0.2–5.6%） | 8.7%（18例） | make_prim.meta 调用真实 impl_aten；_prim_impl 随后再次执行；GPU 融合在 kernel 内。已核对本批 wrapper 的 prims.fma 调用与 profile 连续调用窗口。 |
| 首层 attention constexpr/checker 回退 | 2 | 8.9%（7.2–10.7%） | 3.0%（2.5–3.5%） | 8.6%（2例） | NPUIndexTritonKernel.codegen_kernel 的 constexpr 常量绑定及 checker 回退；GPU 无此首层异常。 |
| 具体 sum kernel 的归约调度 | 5 | 3.8%（1.5–10.1%） | 3.6%（1.2–8.7%） | 8.2%（5例） | NPU codegen 归约维切分、多阶段选择；GPU sum kernel 作为保守费用上界。 |
| LayerNorm backward 归约映射 | 5 | 1.6%（1.3–14.2%） | 0.6%（0.2–12.7%） | 5.9%（5例） | NPU reduction codegen/tiling/runtime autotune；GPU 采用不同归约拆分。 |
| GELU 前向外部调用阻断 bias/pointwise 融合 | 16 | 2.0%（1.2–3.7%） | 0.2%（0.05–0.8%） | 2.6%（10例） | NPU activation lowering/fusion；GPU 含 GELU 的融合 kernel。 |
| Train NLLLossBackward→LogSoftmaxBackward 完整 dlogP | 8 | 1.0%（0.05–3.9%） | 0.08%（0.01–0.8%） | 0.4%（8例） | NPU 两条 decomposition/lowering 边界；GPU loss derivative 融合。 |

归约类具体代码入口为 `NPUIndexTritonKernel.reduction`、`codegen_kernel`，配置侧为 `triton_config_npu_index` 与 `NPUCachingAutotuner.autotune_to_one_config`；融合检查入口为 NPU scheduling 的 `can_fuse` / `candidate_tilings`。这些入口定位到 Inductor 的具体功能模块；除已复核的 constexpr/FMA 机制及 MobileBERT 专项实测外，不把首查模块写成已证实根因。相同 DSL/config 仍慢才能继续判断 Triton_ascend/npuir；外部 scatter/broadcast 则还涉及 CANN。

## 优先检查的具体 kernel

下表每个场景取本轮最大一个预算项，按预算排序。相邻的单个 kernel 成本和融合操作族预算不是同一个概念，完整明细见模型文档。

| 场景 | 具体 NPU kernel / 点 | 本项 ms / 整步占比 | 预算 ms | 整网耗时贡献 | 提升比贡献 pp |
| --- | --- | ---: | ---: | ---: | ---: |
| [MobileBertForMaskedLM_Train](models/MobileBertForMaskedLM.md) | `triton_red_fused_add_mul_sum_view_11` | 67.462 / 22.85% | 34.280 | 11.61% | 15.449 |
| [AlbertForMaskedLM_Train](models/AlbertForMaskedLM.md) | `triton_per_fused_sum_view_16` | 24.253 / 1.52% | 18.998 | 1.19% | 1.315 |
| [nfnet_l0_Train](models/nfnet_l0.md) | `triton_poi_fused_avg_pool2d_42` | 19.710 / 14.96% | 18.959 | 14.39% | 13.499 |
| [OPTForCausalLM_Train](models/OPTForCausalLM.md) | `triton_per_fused_sum_threshold_backward_1` | 18.748 / 6.22% | 17.474 | 5.80% | 6.750 |
| [deit_tiny_patch16_224.fb_in1k_Train](models/deit_tiny_patch16_224.fb_in1k.md) | `triton_red_fused_native_layer_norm_backward_vi_7` | 15.056 / 14.18% | 13.530 | 12.74% | 23.248 |
| [MBartForCausalLM_Train](models/MBartForCausalLM.md) | `aclnnAdd_AddAiCore_Add` | 11.397 / 3.36% | 11.397 | 3.36% | 4.069 |
| [BartForCausalLM_Train](models/BartForCausalLM.md) | `triton_per_fused_gelu_backward_sum_view_4` | 13.092 / 4.48% | 10.624 | 3.64% | 4.210 |
| [PegasusForCausalLM_Train](models/PegasusForCausalLM.md) | `triton_per_fused_gelu_backward_sum_view_3` | 13.107 / 4.64% | 10.442 | 3.69% | 4.364 |
| [beit_base_patch16_224_Train](models/beit_base_patch16_224.md) | `aclnnAdd_AddAiCore_Add` | 10.315 / 1.44% | 10.315 | 1.44% | 1.520 |
| [RobertaForCausalLM_Train](models/RobertaForCausalLM.md) | `aclnnAdd_AddAiCore_Add` | 8.738 / 2.11% | 8.738 | 2.11% | 2.447 |
| [LayoutLMForMaskedLM_Train](models/LayoutLMForMaskedLM.md) | `triton_per_fused__softmax_all_amax_div_isfinite_matmul_native_dropout_sub_5` | 25.092 / 7.18% | 8.708 | 2.49% | 2.848 |
| [GPT2ForSequenceClassification_Train](models/GPT2ForSequenceClassification.md) | `aclnnAdd_AddAiCore_Add` | 8.688 / 4.54% | 8.688 | 4.54% | 5.172 |
| [BertForMaskedLM_Train](models/BertForMaskedLM.md) | `aclnnAdd_AddAiCore_Add` | 8.664 / 2.49% | 8.664 | 2.49% | 2.770 |
| [PLBartForCausalLM_Train](models/PLBartForCausalLM.md) | `aclnnAdd_AddAiCore_Add` | 8.475 / 3.40% | 8.475 | 3.40% | 3.701 |
| [basic_gnn_edgecnn_Train](models/basic_gnn_edgecnn.md) | `triton_red_fused_sum_threshold_backward_3` | 8.405 / 50.15% | 8.160 | 48.68% | 105.127 |
| [TrOCRForCausalLM_Train](models/TrOCRForCausalLM.md) | `aclnnAdd_AddAiCore_Add` | 5.929 / 1.18% | 5.929 | 1.18% | 1.222 |
| [ElectraForCausalLM_Train](models/ElectraForCausalLM.md) | `aclnnAdd_AddAiCore_Add` | 5.650 / 3.60% | 5.650 | 3.60% | 4.659 |
| [MegatronBertForCausalLM_Train](models/MegatronBertForCausalLM.md) | `aclnnAdd_AddAiCore_Add` | 5.390 / 2.02% | 5.390 | 2.02% | 2.377 |
| [DistillGPT2_Train](models/DistillGPT2.md) | `aclnnAdd_AddAiCore_Add` | 4.305 / 1.89% | 4.305 | 1.89% | 1.814 |
| [DistilBertForMaskedLM_Train](models/DistilBertForMaskedLM.md) | `aclnnAdd_AddAiCore_Add` | 2.217 / 0.66% | 2.217 | 0.66% | 0.683 |
| [vit_base_patch14_dinov2.lvd142m_Infer](models/vit_base_patch14_dinov2.lvd142m.md) | `aclnnGeluV2_GeluV2_GeluV2` | 37.928 / 2.07% | 1.965 | 0.11% | 0.107 |
| [DistilBertForMaskedLM_Infer](models/DistilBertForMaskedLM.md) | `triton_red_fused__log_softmax_7` | 3.100 / 2.95% | 1.847 | 1.76% | 1.735 |
| [TrOCRForCausalLM_Infer](models/TrOCRForCausalLM.md) | `triton_red_fused__log_softmax_7` | 2.674 / 1.63% | 1.607 | 0.98% | 0.998 |
| [PLBartForCausalLM_Infer](models/PLBartForCausalLM.md) | `triton_red_fused__log_softmax_7` | 2.621 / 3.53% | 1.490 | 2.00% | 1.943 |
| [XGLMForCausalLM_Infer](models/XGLMForCausalLM.md) | `triton_red_fused__log_softmax_10` | 1.839 / 3.22% | 1.356 | 2.37% | 2.382 |
| [torch_multimodal_clip_Train](models/torch_multimodal_clip.md) | `triton_per_fused_native_layer_norm_backward_vi_9` | 3.868 / 4.57% | 1.312 | 1.55% | 1.646 |
| [BertForMaskedLM_Infer](models/BertForMaskedLM.md) | `triton_red_fused__log_softmax_6` | 1.599 / 1.73% | 0.983 | 1.06% | 0.969 |
| [LayoutLMForMaskedLM_Infer](models/LayoutLMForMaskedLM.md) | `triton_red_fused__log_softmax_6` | 1.591 / 1.72% | 0.968 | 1.05% | 0.950 |
| [MT5ForConditionalGeneration_Infer](models/MT5ForConditionalGeneration.md) | `triton_red_fused__log_softmax_16` | 1.106 / 10.65% | 0.861 | 8.28% | 7.371 |
| [BartForCausalLM_Infer](models/BartForCausalLM.md) | `triton_red_fused__log_softmax_6` | 1.397 / 1.69% | 0.819 | 0.99% | 0.997 |
| [MBartForCausalLM_Infer](models/MBartForCausalLM.md) | `triton_red_fused__log_softmax_9` | 1.396 / 1.46% | 0.814 | 0.85% | 0.840 |
| [PegasusForCausalLM_Infer](models/PegasusForCausalLM.md) | `triton_red_fused__log_softmax_10` | 1.391 / 1.74% | 0.810 | 1.01% | 1.031 |
| [ElectraForCausalLM_Infer](models/ElectraForCausalLM.md) | `triton_red_fused__log_softmax_clone_slice_view_6` | 3.265 / 10.21% | 0.746 | 2.33% | 1.843 |
| [vit_base_patch16_siglip_256_Infer](models/vit_base_patch16_siglip_256.md) | `aclnnGeluV2_GeluV2_GeluV2` | 7.627 / 2.63% | 0.664 | 0.23% | 0.235 |
| [deit_base_distilled_patch16_224_Infer](models/deit_base_distilled_patch16_224.md) | `aclnnGeluV2_GeluV2_GeluV2` | 5.981 / 2.73% | 0.641 | 0.29% | 0.299 |
| [beit_base_patch16_224_Infer](models/beit_base_patch16_224.md) | `aclnnGeluV2_GeluV2_GeluV2` | 5.993 / 2.10% | 0.631 | 0.22% | 0.273 |
| [RobertaForCausalLM_Infer](models/RobertaForCausalLM.md) | `triton_red_fused__log_softmax_clone_slice_view_7` | 2.710 / 2.60% | 0.498 | 0.48% | 0.432 |
| [DistillGPT2_Infer](models/DistillGPT2.md) | `triton_red_fused__log_softmax_clone_matmul_sli_8` | 2.742 / 4.17% | 0.457 | 0.69% | 0.574 |
| [XGLMForCausalLM_Train](models/XGLMForCausalLM.md) | `aclnnMul_MulAiCore_Mul` | 0.376 / 0.19% | 0.376 | 0.19% | 0.205 |
| [MT5ForConditionalGeneration_Train](models/MT5ForConditionalGeneration.md) | `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad` | 1.617 / 3.94% | 0.339 | 0.83% | 0.781 |
| [MegatronBertForCausalLM_Infer](models/MegatronBertForCausalLM.md) | `aclnnGeluV2_GeluV2_GeluV2` | 1.232 / 1.76% | 0.280 | 0.40% | 0.387 |
| [AlbertForMaskedLM_Infer](models/AlbertForMaskedLM.md) | `triton_red_fused__log_softmax_6` | 0.443 / 0.10% | 0.266 | 0.06% | 0.054 |
| [BERT_pytorch_Train](models/BERT_pytorch.md) | `aclnnMul_MulAiCore_Mul` | 0.239 / 0.34% | 0.239 | 0.34% | 0.342 |
| [speech_transformer_Train](models/speech_transformer.md) | `aclnnMul_MulAiCore_Mul` | 0.201 / 0.86% | 0.201 | 0.86% | 0.937 |
| [OPTForCausalLM_Infer](models/OPTForCausalLM.md) | `triton_red_fused__log_softmax_clone_matmul_sli_13` | 1.291 / 1.85% | 0.192 | 0.27% | 0.207 |
| [BERT_pytorch_Infer](models/BERT_pytorch.md) | `aclnnGeluV2_GeluV2_GeluV2` | 0.464 / 2.16% | 0.182 | 0.85% | 0.807 |
| [basic_gnn_gcn_Train](models/basic_gnn_gcn.md) | `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd` | 0.370 / 29.21% | 0.175 | 13.82% | 7.349 |
| [basic_gnn_gcn_Infer](models/basic_gnn_gcn.md) | `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd` | 0.368 / 49.92% | 0.172 | 23.27% | 20.297 |
| [basic_gnn_sage_Infer](models/basic_gnn_sage.md) | `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd` | 0.354 / 58.31% | 0.165 | 27.16% | 28.113 |
| [lennard_jones_Train](models/lennard_jones.md) | `triton_per_fused_sum_tanh_backward_2` | 0.179 / 60.37% | 0.162 | 54.74% | 135.288 |
| [basic_gnn_sage_Train](models/basic_gnn_sage.md) | `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd` | 0.289 / 25.33% | 0.115 | 10.12% | 7.244 |
| [soft_actor_critic_Train](models/soft_actor_critic.md) | `triton_red_fused_sum_threshold_backward_3` | 0.688 / 6.54% | 0.029 | 0.28% | 0.277 |
| [GPT2ForSequenceClassification_Infer](models/GPT2ForSequenceClassification.md) | `triton_per_fused__to_copy_argmax_eq_8` | 0.030 / 0.07% | 0.017 | 0.04% | 0.030 |
| [dlrm_Train](models/dlrm.md) | `triton_per_fused_sum_threshold_backward_3` | 0.051 / 0.97% | 0.008 | 0.14% | 0.145 |

还应优先核对 Albert Train 的 `_clone_matmul_backward_permute_20`（90.679 ms）与 `_add_mul_pow_sum_tanh_tanh_bac_17`（51.366 ms），以及 CLIP QKV projection 引出的 BroadcastTo。它们已有具体代码节点，但尚未闭合四方语义边界，所以没有放进“可与 GPU 提升比对齐”的预算表。Albert 的 `_20` 实际是 sum/permute/累加，不能被名字中的 matmul_backward 误导为裸 MM。

## 40 个模型的完整入口

| 模型 | Infer：NPU / GPU 提升比 | Train：NPU / GPU 提升比 | 列出项数 Infer / Train |
| --- | ---: | ---: | ---: |
| [AlbertForMaskedLM](models/AlbertForMaskedLM.md) | 5.23% / 6.34% | -10.34% / 8.19% | 1 / 3 |
| [BERT_pytorch](models/BERT_pytorch.md) | 4.87% / 3.30% | -1.78% / 7.94% | 1 / 1 |
| [BartForCausalLM](models/BartForCausalLM.md) | -0.81% / -1.79% | -15.72% / -11.73% | 2 / 2 |
| [BertForMaskedLM](models/BertForMaskedLM.md) | 8.76% / -2.10% | -11.33% / 3.69% | 2 / 2 |
| [DistilBertForMaskedLM](models/DistilBertForMaskedLM.md) | 1.14% / 6.25% | -3.17% / 3.67% | 2 / 2 |
| [DistillGPT2](models/DistillGPT2.md) | 17.28% / 17.35% | 4.11% / 22.07% | 1 / 1 |
| [ElectraForCausalLM](models/ElectraForCausalLM.md) | 21.01% / 31.34% | -29.40% / 25.05% | 2 / 3 |
| [GPT2ForSequenceClassification](models/GPT2ForSequenceClassification.md) | 24.50% / 23.21% | -14.02% / 21.82% | 1 / 2 |
| [LayoutLMForMaskedLM](models/LayoutLMForMaskedLM.md) | 9.25% / 14.43% | -14.24% / 17.85% | 2 / 3 |
| [MBartForCausalLM](models/MBartForCausalLM.md) | 1.14% / -1.86% | -21.01% / -11.47% | 2 / 3 |
| [MT5ForConditionalGeneration](models/MT5ForConditionalGeneration.md) | 11.02% / 24.96% | 5.39% / 44.02% | 1 / 2 |
| [MegatronBertForCausalLM](models/MegatronBertForCausalLM.md) | 3.78% / 11.51% | -17.92% / 20.57% | 2 / 2 |
| [MobileBertForMaskedLM](models/MobileBertForMaskedLM.md) | 8.46% / 35.04% | -33.04% / 19.82% | 0 / 2 |
| [OPTForCausalLM](models/OPTForCausalLM.md) | 24.70% / -17.22% | -16.44% / -26.25% | 1 / 2 |
| [PLBartForCausalLM](models/PLBartForCausalLM.md) | 3.06% / -5.67% | -8.77% / -10.97% | 2 / 1 |
| [PegasusForCausalLM](models/PegasusForCausalLM.md) | -1.84% / 1.78% | -18.12% / -0.72% | 2 / 3 |
| [RobertaForCausalLM](models/RobertaForCausalLM.md) | 9.50% / 1.45% | -15.84% / 2.26% | 2 / 2 |
| [TrOCRForCausalLM](models/TrOCRForCausalLM.md) | -2.15% / 5.42% | -3.65% / 3.86% | 2 / 1 |
| [XGLMForCausalLM](models/XGLMForCausalLM.md) | -0.30% / 1.67% | -5.40% / 1.27% | 1 / 1 |
| [basic_gnn_edgecnn](models/basic_gnn_edgecnn.md) | 31.32% / 48.40% | -115.94% / 50.07% | 0 / 1 |
| [basic_gnn_gcn](models/basic_gnn_gcn.md) | 12.79% / 73.01% | 46.81% / 72.37% | 1 / 2 |
| [basic_gnn_sage](models/basic_gnn_sage.md) | -3.52% / 68.40% | 28.43% / 64.78% | 1 / 2 |
| [beit_base_patch16_224](models/beit_base_patch16_224.md) | -23.63% / 7.94% | -5.65% / 5.54% | 1 / 1 |
| [cm3leon_generate](models/cm3leon_generate.md) | -5.81% / 5.91% | 数据缺失 | 0 / — |
| [dcgan](models/dcgan.md) | -36.03% / -96.43% | 2.63% / 1.37% | 0 / 0 |
| [deit_base_distilled_patch16_224](models/deit_base_distilled_patch16_224.md) | -2.21% / 7.12% | -0.05% / 5.04% | 1 / 0 |
| [deit_tiny_patch16_224.fb_in1k](models/deit_tiny_patch16_224.fb_in1k.md) | -3.14% / 7.68% | -82.50% / 8.30% | 0 / 3 |
| [dlrm](models/dlrm.md) | 0.26% / 1.38% | -0.21% / 0.92% | 0 / 1 |
| [functorch_maml_omniglot](models/functorch_maml_omniglot.md) | 23.14% / 11.22% | 13.87% / 23.97% | 0 / 0 |
| [lennard_jones](models/lennard_jones.md) | 20.03% / 14.36% | -147.16% / 19.64% | 0 / 1 |
| [llama](models/llama.md) | 9.58% / 10.02% | 数据缺失 | 0 / — |
| [nanogpt](models/nanogpt.md) | 21.04% / 19.50% | 4.90% / 16.07% | 0 / 0 |
| [nfnet_l0](models/nfnet_l0.md) | 19.88% / 28.90% | 6.18% / 22.16% | 0 / 1 |
| [nvidia_deeprecommender](models/nvidia_deeprecommender.md) | 0.68% / -12.83% | 0.98% / -3.54% | 0 / 0 |
| [pytorch_stargan](models/pytorch_stargan.md) | 6.42% / 0.70% | 15.03% / 3.86% | 0 / 0 |
| [soft_actor_critic](models/soft_actor_critic.md) | 3.48% / 4.43% | 0.48% / 3.68% | 0 / 1 |
| [speech_transformer](models/speech_transformer.md) | 7.29% / 12.37% | -8.84% / 8.53% | 0 / 1 |
| [torch_multimodal_clip](models/torch_multimodal_clip.md) | -10.94% / 8.36% | -6.31% / 10.67% | 0 / 1 |
| [vit_base_patch14_dinov2.lvd142m](models/vit_base_patch14_dinov2.lvd142m.md) | -0.42% / 7.27% | 0.45% / 4.36% | 1 / 0 |
| [vit_base_patch16_siglip_256](models/vit_base_patch16_siglip_256.md) | -2.78% / 8.50% | -2.49% / 6.95% | 1 / 0 |

## 证据、复算及待补验证

- [逐 kernel 四方原始汇总](../../../../artifacts/cicd/performance/full-survey-20261102/profiles.json)、[检查项与候选 JSON](../../../../artifacts/cicd/performance/full-survey-20261102/issues.json)、[预算 CSV](../../../../artifacts/cicd/performance/full-survey-20261102/issues.csv)。
- 每个模型链接到该批 NPU/GPU 的完整选中 generated kernel（保留原格式），以及实际 run/外部调用边界；只去除了环境路径、backend_hash 等信息。
- [decomposition/fallback 源码参考](../../../../artifacts/cicd/performance/full-survey-20261102/component_policy_reference.txt)：历史安装源码的 loss/GELU 排除策略，以及本地 checkout 的 scatter fallback 和归约入口；入口参考不等于 profile 构建身份。
- [输入文件 SHA256 清单](../../../../artifacts/cicd/performance/full-survey-20261102/source_manifest.json)覆盖原始 CSV、wrapper、FX，原始输入不上传，防止大体积/环境路径公开。
- [本地复算脚本及用法](../../../../scripts/tools/cicd/full_survey_20261102/README.md)。脚本不导入 torch、不连接远端，只读用户提供数据。
- 当前校验范围为全部 CSV 的单位/步数/batch、预算公式、40 模型/78 场景覆盖、已选择 kernel 的代码/调用证据和链接检查；没有修复性能/精度验证。
- 后续只对选中的最大点补 GPU 逐调用 shape/stride、NPU/GPU selected config、固定输入单 kernel 与 IR；预算最大的 kernel 优先，避免再次泛化成整条反向链路。
- PLBart Train、TrOCR Train 各有 3 个 GPU profile 名称未在本批 wrapper 匹配到定义，均已在模型页列明；需要补对应编译图，不基于这些缺失 DSL 新增算子族收益归因。

历史：[首层 attention 根因](electra-layoutlm-train-first-attention-20261031.md)。此前[两模型绝对耗时差分析](electra-layoutlm-train-gpu-inductor-gap-20261102.md)已标记口径被替代，其中 MM/AddMM 70.049 ms 不再作为编译收益优化依据。
