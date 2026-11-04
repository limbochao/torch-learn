---
title: "CICD 性能分析：ElectraForCausalLM"
---

# ElectraForCausalLM：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；初筛在本地读取和复算；下文临时修改对照另有 NPU_A5 实测，原始统计不混入复测结果。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=32；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 40.491541 | 31.985749 | 21.01% |
| GPU | 49.488627 | 33.976500 | 31.34% |

达到 GPU **整网**提升比时，NPU 整步目标为 27.799536 ms；当前相对该目标的差额为 4.186214 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/ElectraForCausalLM_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/ElectraForCausalLM_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/ElectraForCausalLM_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/ElectraForCausalLM_Infer/GPU_wrapper_calls.txt)。

### 1. log_softmax→NLLLoss 的完整 log probability 写回

**优化入口：NPU Inductor nll_loss_forward decomposition/lowering、producer-consumer fusion。** 外部 NLLLoss 消费 log_softmax 完整矩阵；GPU loss 融合 kernel 可只提取 target logit 和行归约统计。检查 NLLLoss 唯一消费者边界，避免最后一个完整词表循环及中间矩阵写回。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__log_softmax_clone_slice_view_6` | 1.0 | 3.240925 | 10.13% |
| `triton_poi_fused_clone_slice_7` | 1.0 | 0.003278 | 0.01% |
| `aclnnInplaceOne_OnesLikeAiCore_OnesLike` | 1.0 | 0.003038 | 0.01% |
| `aclnnNLLLoss_NLLLossAiCore_NLLLoss` | 1.0 | 0.017951 | 0.06% |

本项 kernel/区间合计 **3.265192 ms，占整步 10.21%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 5.648195 | 3.265192 | ≤42.19% |
| GPU（Eager 下界 / compiled 上界） | 6.084750 | 2.713477 | ≥55.41% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **13.21 个百分点**。NPU 为最终 projection 后至 NLLLoss 的逐步精确窗口，包含 shift/clone/copy；GPU Eager 只计 log_softmax+NLLLoss 下界，compiled 包含 prepare 及全部 loss kernel。预算是保守下界。

NPU Eager 对账项：`aclnnInplaceOne_OnesLikeAiCore_OnesLike`, `aclnnInplaceCopy_SliceAiCore_Slice`, `aclnnLogSoftmax_LogSoftmaxAiCore_LogSoftmaxV2`, `aclnnNLLLoss_NLLLossAiCore_NLLLoss`。

GPU Eager 对账项：`_log_softmax`, `nll_loss_forward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_prepare_softmax_online_view_8` | 1.0 | 2.701483 | 7.95% |
| `triton_red_fused__log_softmax_clone_constant_pad_nd_nll_loss_forward_slice_view_9` | 1.0 | 0.010739 | 0.03% |
| `triton_per_fused__log_softmax_clone_constant_pad_nd_nll_loss_forward_slice_view_10` | 1.0 | 0.001254 | 0.00% |

**贡献粗估：0.746396 ms/step，相当于当前 NPU Inductor 整步耗时的 2.33%，使整网提升比增加 1.843 个百分点，单独达到该预算后整网提升比约为 22.85%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused__log_softmax_clone_slice_view_6`：NPU_g01_output_code.py:518；输入 `f32[16384, 30522][30522, 1]npu:0`；size_hints=`{'x2': 32, 'x3': 511, 'r1': 30522}`；split_axis=`[0, 1]`。
- `triton_poi_fused_clone_slice_7`：NPU_g01_output_code.py:602；输入 `i64[32, 512][512, 1]npu:0`；size_hints=`{'y0': 32, 'x1': 511}`；split_axis=`[0, 1]`。
- `aclnnInplaceOne_OnesLikeAiCore_OnesLike`：外部调用；见 wrapper 调用边界。

### 2. GELU 前向外部调用边界

**优化入口：NPU Inductor decomposition/lowering、scheduler 的 bias/GELU 融合。** NPU 保留 ACLNN GELU；GPU 将 GELU 与相邻 bias/pointwise 合入生成 kernel。优化具体外部调用边界，而非仅调整 ACLNN GELU。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnGeluV2_GeluV2_GeluV2` | 13.0 | 1.194616 | 3.73% |

本项 kernel/区间合计 **1.194616 ms，占整步 3.73%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 1.206088 | 1.194616 | ≤0.95% |
| GPU（Eager 下界 / compiled 上界） | 2.099346 | 2.017051 | ≥3.92% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **2.97 个百分点**。保守筛选：NPU Eager 将该操作族所有调用收费，NPU Inductor 只计列出的纯 kernel/指定融合 kernel；GPU compiled 计全部覆盖该操作族的生成 kernel（含融合附带工作）。NPU 提升比为上界、GPU 提升比为下界，预算为保守缺口下界；不是单个 kernel 的实测修复收益。

NPU Eager 对账项：`aclnnGeluV2_GeluV2_GeluV2`。

GPU Eager 对账项：`gelu`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_addmm_gelu_view_6` | 12.0 | 2.008654 | 5.91% |
| `triton_per_fused_addmm_gelu_native_layer_norm_view_7` | 1.0 | 0.008397 | 0.02% |

**贡献粗估：0.035808 ms/step，相当于当前 NPU Inductor 整步耗时的 0.11%，使整网提升比增加 0.088 个百分点，单独达到该预算后整网提升比约为 21.09%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnGeluV2_GeluV2_GeluV2`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused_add_addmm_native_layer_norm_v_2` | 0.937029 / 24.0 | 2.93% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_addmm_3` | 0.508739 / 12.0 | 1.59% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_clone_permute_1` | 0.172488 / 12.0 | 0.54% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_per_fused_add_addmm_native_layer_norm_v_2`：NPU_g01_output_code.py:238；输入 `f32[256][1]npu:0`；size_hints=`{'y0': 32, 'x1': 512, 'r2': 256}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_6`, `triton_poi_fused_addmm_clone_transpose_view_1`, `triton_per_fused_add_addmm_native_layer_norm_view_5`。
- `triton_poi_fused_addmm_3`：NPU_g01_output_code.py:317；输入 `f32[1024][1]npu:0`；size_hints=`{'y0': 16384, 'x1': 1024}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_gelu_view_6`, `triton_poi_fused_addmm_clone_transpose_view_1`, `triton_per_fused_add_addmm_native_layer_norm_view_5`。
- `triton_poi_fused_clone_permute_1`：NPU_g01_output_code.py:163；输入 `f32[32, 4, 512, 64][131072, 32768, 64, 1]npu:0`；size_hints=`{'t0': 32, 'z1': 512, 'y2': 4, 'x3': 64}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused_addmm_clone_transpose_view_1`, `triton_poi_fused_addmm_clone_transpose_view_2`, `triton_poi_fused_clone_permute_view_4`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=32；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 121.273399 | 156.933574 | -29.40% |
| GPU | 156.677726 | 117.426933 | 25.05% |

达到 GPU **整网**提升比时，NPU 整步目标为 90.892073 ms；当前相对该目标的差额为 66.041501 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/ElectraForCausalLM_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/ElectraForCausalLM_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/ElectraForCausalLM_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/ElectraForCausalLM_Train/GPU_wrapper_calls.txt)。

### 1. prims.fma metadata 重复执行第一组 Mul/Add

**优化入口：PyTorch Inductor inductor_prims.make_prim.meta 与 prims._prim_impl；NPU prims.fma fallback 暴露真实执行。** softmax backward 后连续 Mul/Add/Mul/Add 中第一对来自 metadata 执行真实 impl_aten，随后真实实现再次执行。第一对结果不参与最终输出；GPU 生成 kernel 内执行有效公式，没有 metadata 设备调用。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnMul_MulAiCore_Mul` | 12.0 | 2.216049 | 1.41% |
| `aclnnAdd_AddAiCore_Add` | 12.0 | 3.433829 | 2.19% |

本项 kernel/区间合计 **5.649878 ms，占整步 3.60%**。

仅计算这些相邻窗口中的第一对，不使用整网 Add/Mul 汇总。预算为已证实重复调用的当前设备耗时；修复后整网时间仍须实测。另一个有效 Mul/Add 对不是可全额删除的工作。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused__softmax__softmax_backward_data_div_mul_native_dropout_backward_sub_view_16` | 12.0 | 7.542361 | 6.42% |

**贡献粗估：5.649878 ms/step，相当于当前 NPU Inductor 整步耗时的 3.60%，使整网提升比增加 4.659 个百分点，单独达到该预算后整网提升比约为 -24.75%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnMul_MulAiCore_Mul`：外部调用；见 wrapper 调用边界。
- `aclnnAdd_AddAiCore_Add`：外部调用；见 wrapper 调用边界。

历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。

### 2. 首层 attention constexpr/checker 回退

**优化入口：NPU Inductor NPUIndexTritonKernel.codegen_kernel、TTIR checker constexpr 参数传递及回退 tiling。** 首层 kernel 为 16.763929 ms/step；后续同类 triton_per_fused__softmax_all_amax_div_isfinit_7 为 0.984687 ms/次。此前源码/TTIR checker 已确认 seed offset constexpr 缺失常量触发回退。GPU 同类 attention 没有这一首层回退。实际 compile_mode 不能仅凭 metadata 标签区分。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused__softmax_all_amax_div_isfinite_matmul_native_dropout_sub_4` | 1.0 | 16.763929 | 10.68% |

本项 kernel/区间合计 **16.763929 ms，占整步 10.68%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 9.794080 | 16.763929 | ≤-71.16% |
| GPU（Eager 下界 / compiled 上界） | 4.991089 | 5.722070 | ≥-14.65% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **56.52 个百分点**。保守四方筛选：NPU 将全部 12 层前向两次 BMM 之间的 Eager 工作收费给首层，Inductor 只计首层；GPU compiled 计全部 12 层前向 softmax/dropout kernel，GPU Eager 计全部 softmax，以及混合 dropout 中 12 次 attention 调用的下界。dropout 下界=max(12×min, total−(count−12)×max)，不按平均值分摊混合形状。因此这里是很保守的提升比缺口预算；恢复到 NPU 后续层实现的独立控制估算为 15.779243 ms，与本预算口径不同，不能相加。根因仍采用历史 checker/source 证据。

NPU Eager 对账项：`aclnnAdd_AddAiCore_Add`, `aclnnDivs_Muls_Muls`, `aclnnDropoutV3_DropoutV3AiCore_DropOutV3`, `aclnnMatmul_TransposeAiCore_Transpose`, `aclnnSoftmax_SoftmaxAiCore_SoftmaxV2`。

GPU Eager 对账项：`_softmax`, `native_dropout (12 calls, min/max lower bound)`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused__softmax_abs_all_amax_div_eq_mul_native_dropout_ne_sub_view_7` | 11.0 | 5.245107 | 4.47% |
| `triton_per_fused__softmax_abs_all_amax_div_eq_mul_native_dropout_ne_sub_view_3` | 1.0 | 0.476963 | 0.41% |

**贡献粗估：5.535434 ms/step，相当于当前 NPU Inductor 整步耗时的 3.53%，使整网提升比增加 4.564 个百分点，单独达到该预算后整网提升比约为 -24.84%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_per_fused__softmax_all_amax_div_isfinite_matmul_native_dropout_sub_4`：NPU_g02_output_code.py:416；输入 `f32[128, 512, 512][262144, 512, 1]npu:0`；size_hints=`{'x': 65536, 'r0_': 512}`；split_axis=`[0]`。

历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。

<!-- BEGIN CICD VALIDATION ElectraForCausalLM_Train:attention_seed -->
<a id="attention-constexpr-validation"></a>

#### 补齐 seed constexpr：首层从 16.790 ms 降到 0.974 ms

**已验证的责任模块是 NPU Inductor `NPUIndexTritonKernel.codegen_kernel`：遗漏了 signature 中等于 1 的 SizeArg 常量绑定。** 仅补齐这段元数据生成逻辑，就消除首层 checker 失败，恢复多轴 kernel；无需改动 Triton_ascend 或 npuir。

2026-11-03 在 NPU_A5 物理卡 2、`pta_213_cicd` 实测，2026-11-04 整理。场景为 Train、FP32、static、batch=32、sequence=512，attention scores 为 `[32,4,512,512]`，关闭 aclgraph。两组分别采样 10 个稳定 step。

| 进程内配置 | Eager 实测 ms/step | Inductor 实测 ms/step | 本组提升比 `(E-I)/E` |
| --- | ---: | ---: | ---: |
| 原始 codegen | 126.474829 | 156.810495 | -23.99% |
| 仅补齐 constexpr 常量 | 126.288455 | 141.827617 | -12.30% |

**整网净省 14.982878 ms/step，设备整步耗时下降 9.55%。** 固定用基线 Eager=126.474829 ms 归因，编译提升比增加 **11.847 个百分点**；统一该分母后为 -23.99% → -12.14%。两组 Inductor 的稳定步范围分别为 156.761–156.887 ms 和 141.678–142.120 ms。

| 具体 kernel / 对照 | 修改前 ms/step | 修改后 ms/step | 次数/step |
| --- | ---: | ---: | ---: |
| 首层：`triton_per_fused__softmax_all_amax_div_isfinite_matmul_native_dropout_sub_4` → `triton_per_fused__softmax_all_amax_div_isfinit_4` | 16.790405 | 0.974007 | 1 → 1 |
| 后续层：`triton_per_fused__softmax_all_amax_div_isfinit_7` 合计 | 10.704739 | 10.704843 | 11 → 11 |

首层自身减少 **15.816398 ms/step**，耗时下降 94.20%、加速 17.24 倍；修复后与后续层每次约 0.973168 ms 一致。其余 kernel 合计增加 0.833520 ms，其中 `_addmm_matmul_backward_permute_3` 增加约 0.626 ms，因此整网净省略低于首层收益；没有将这部分变化计入首层收益。

profile 将旧首层名字截为 `all_amax_div_isfinite_matmul_native_dropout_sub_4`，wrapper 中是上表完整名称。名字中的 `matmul` 来自节点 provenance；该生成 DSL 本身没有执行矩阵乘。

本次直接记录了 **Index codegen 失败再进入 no-linear 回退**的异常：原始组在 `tl.load(in_ptr1 + load_seed_offset)` 报 `'NoneType' object has no attribute 'type'`；补齐常量后该事件消失。总 Index codegen 失败记录从 37 降为 36，其他 36 个既有事件未处理，不能表述为取消所有 no-linear 回退。

临时补丁在 `NPUIndexTritonKernel.codegen_kernel` 的 `self.triton_meta = triton_meta` 前加入：

```python
from torch._inductor.codegen.triton_utils import equal_1_arg_indices
for arg_num in equal_1_arg_indices(signature):
    triton_meta["constants"][signature[arg_num].name] = 1
```

对应的生成路径变化为：

| 检查点 | 原始组 | 补齐常量组 |
| --- | --- | --- |
| Index checker 的 `load_seed_offset` | signature 为 constexpr，但 constants 没有 1，TTIR 生成失败 | `constants={'load_seed_offset': 1}`，该失败消失 |
| 最终首层 DSL | no-linear `Grid1D`，`x=65536, r0_=512`，`requires_no_linear_block_remap=True` | `GridNpu`，`z0=32,y1=4,x2=512,r3=512` |
| 后续层 DSL | 正常多轴 `_7` | 保持正常多轴 `_7`，耗时基本一致 |

**原始组最终的 Grid1D kernel 自身也含 `constants={'load_seed_offset': 1}`**，因为备用 codegen 已补齐常量；出错的是回退前的 Index checker，不能把最终备用 kernel 的 metadata 当作失败 checker 的 metadata。异常发生在 TTIR 生成阶段，后续设备 IR/npuir 尚未接收这个失败的 Index kernel。上述仅改 Inductor 元数据的 A/B 同时证明了触发原因与实际性能影响。

本节先前的 **5.535434 ms** 是把全部 12 层费用计入首层后的保守四方筛选预算；**15.779243 ms** 是历史同设备后续层控制估算。本次 15.816398 ms 为首层修复实测、14.982878 ms 为整网净收益，四者口径不同，不能相加。

独立整网 accuracy 保持 FP32、batch=32、训练 2 步及原 runner 默认 `tol=0.01`，结果为 **`pass_accuracy`**；logits 最大绝对误差 1.013279e-6，梯度 204/204 通过、最大绝对误差 8.5565262e-9，更新后参数 204/204 通过。accuracy 的编译计时不作为性能结果。

软件版本为 torch `2.13.0+cpu`、torch_npu `2.13.0.dev20261030`、Triton `3.6.0`、transformers `4.36.0`。三个关键安装文件与 `test/cicd_benchmark@b3bd74035189` 一致；既存 matmul decomposition 差异在两组中保持一致，见[版本校验](../../../../../artifacts/cicd/performance/fma-seed-validation-20261103/installed_identity.json)和[已有差异](../../../../../artifacts/cicd/performance/fma-seed-validation-20261103/decomposition_difference.patch)。补丁只在当前进程生效，安装目录和 runner 文件校验均未变化。LayoutLM 有相同历史机制，本次未测其整网修复收益，不将 Electra 数值直接搬过去。

证据：[原始首层及后续层 DSL](../../../../../artifacts/cicd/performance/fma-seed-validation-20261103/ElectraForCausalLM_baseline/selected_kernels.txt)、[修改后 DSL](../../../../../artifacts/cicd/performance/fma-seed-validation-20261103/ElectraForCausalLM_bind_constexpr/selected_kernels.txt)、[失败记录](../../../../../artifacts/cicd/performance/fma-seed-validation-20261103/ElectraForCausalLM_baseline/codegen_failures.json)、[修改后记录](../../../../../artifacts/cicd/performance/fma-seed-validation-20261103/ElectraForCausalLM_bind_constexpr/codegen_failures.json)、[临时 codegen 方法](../../../../../artifacts/cicd/performance/fma-seed-validation-20261103/ElectraForCausalLM_bind_constexpr/codegen_kernel_patched.py)、[10 步统计](../../../../../artifacts/cicd/performance/fma-seed-validation-20261103/summary.json)、[accuracy 日志](../../../../../artifacts/cicd/performance/fma-seed-validation-20261103/ElectraForCausalLM_bind_constexpr_accuracy/run.log)、[实验脚本](../../../../../scripts/repro/cicd/performance/fma_seed_validation/README.md)。
正式补丁现已基于最新 master 建立独立分支；分支、修改范围与修复前后 kernel 对照见
[Index seed constexpr 修复说明](../index-triton-seed-constexpr-fix.md)。本节性能数字仍为上述指定版本的实测。
<!-- END CICD VALIDATION ElectraForCausalLM_Train:attention_seed -->

### 3. NLLLossBackward→LogSoftmaxBackward 的完整中间矩阵

**优化入口：NPU Inductor nll_loss_backward / _log_softmax_backward_data decomposition 与 lowering。** 两条独立外部调用之间生成完整 dlogP；无类别权重时每个 token 的 dlogP 仅 target 类别非零。GPU 将 loss 导数融合生成，目标是直接计算 dlogits，避免完整中间矩阵；最终 dlogits 仍须生成。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnNLLLossBackward_NLLLossGradAiCore_NLLLossGrad` | 1.0 | 1.460205 | 0.93% |
| `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad` | 1.0 | 4.237956 | 2.70% |

本项 kernel/区间合计 **5.698161 ms，占整步 3.63%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 5.640561 | 5.698161 | ≤-1.02% |
| GPU（Eager 下界 / compiled 上界） | 9.220881 | 8.788170 | ≥4.69% |

判定依据：即使用上述保守边界，NPU 提升比仍比 GPU 至少低 **5.71 个百分点**。GPU compiled 上界故意包含全部前向与反向 loss/prepare kernel。需要完整 GPU wrapper 进一步切开前后向后才能给更紧预算。

NPU Eager 对账项：`aclnnNLLLossBackward_NLLLossGradAiCore_NLLLossGrad`, `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad`。

GPU Eager 对账项：`_log_softmax_backward_data`, `nll_loss_backward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused__log_softmax__log_softmax_backward_data_arange_clone_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_view_0` | 1.0 | 6.146171 | 5.23% |
| `triton_red_fused__log_softmax_prepare_softmax_online_10` | 1.0 | 2.631669 | 2.24% |
| `triton_red_fused__log_softmax_clone_nll_loss_forward_slice_view_11` | 1.0 | 0.009197 | 0.01% |
| `triton_per_fused__log_softmax_clone_nll_loss_forward_slice_view_12` | 1.0 | 0.001133 | 0.00% |

**贡献粗估：0.322296 ms/step，相当于当前 NPU Inductor 整步耗时的 0.21%，使整网提升比增加 0.266 个百分点，单独达到该预算后整网提升比约为 -29.14%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnNLLLossBackward_NLLLossGradAiCore_NLLLossGrad`：外部调用；见 wrapper 调用边界。
- `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad`：外部调用；见 wrapper 调用边界。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused__softmax_all_amax_div_isfinit_7` | 10.831557 / 11.0 | 6.90% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_per_fused__softmax__softmax_backward_da_13` | 4.474075 / 12.0 | 2.85% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_poi_fused_add_0` | 4.117995 / 1.0 | 2.62% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_per_fused__softmax_all_amax_div_isfinit_7`：NPU_g02_output_code.py:727；输入 `f32[128, 512, 512][262144, 512, 1]npu:0`；size_hints=`{'z0': 32, 'y1': 4, 'x2': 512, 'r3': 512}`；split_axis=`[0, 1, 2]`。GPU 检查入口：`triton_per_fused__softmax__softmax_backward_data_div_mul_native_dropout_backward_sub_view_16`, `triton_per_fused__softmax_abs_all_amax_div_eq_mul_native_dropout_ne_sub_view_7`, `triton_per_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_view_5`。
- `triton_per_fused__softmax__softmax_backward_da_13`：NPU_g01_output_code.py:1220；输入 `b8[32, 4, 512, 1][2048, 512, 1, 1]npu:0`；size_hints=`{'z0': 32, 'y1': 4, 'x2': 512, 'r3': 512}`；split_axis=`[0, 1, 2]`。GPU 检查入口：`triton_per_fused__softmax__softmax_backward_data_div_mul_native_dropout_backward_sub_view_16`, `triton_per_fused__softmax_abs_all_amax_div_eq_mul_native_dropout_ne_sub_view_7`, `triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_view_19`。
- `triton_poi_fused_add_0`：NPU_g01_output_code.py:56；输入 `f32[32, 512, 30522][15627264, 30522, 1]npu:0`；size_hints=`{'z0': 32, 'y1': 512, 'x2': 30522}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_red_fused_add_native_layer_norm_backward_view_20`, `triton_per_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_view_5`, `triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_view_19`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
