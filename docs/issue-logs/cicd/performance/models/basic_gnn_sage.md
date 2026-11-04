---
title: "CICD 性能分析：basic_gnn_sage"
---

# basic_gnn_sage：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；初始四方数据在本地完成粗筛；2026-11-05 补充了 NPU_A5 原始组/临时改动组验证，实测另列。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=1；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 0.586155 | 0.606804 | -3.52% |
| GPU | 0.793989 | 0.250912 | 68.40% |

达到 GPU **整网**提升比时，NPU 整步目标为 0.185233 ms；当前相对该目标的差额为 0.421571 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_sage_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_sage_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_sage_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_sage_Infer/GPU_wrapper_calls.txt)。

### 1. Inductor scatter fallback 阻断 index_select / ReLU 融合

**已验证的入口：`torch_npu/_inductor/lowering_fallback_list.py`。** 与 GCN 相同的五项 scatter 注册屏蔽了已有社区 lowering；本模型消息聚合是 index_select→scatter，最后一路还包含 bias/add/ReLU，不应泛称带权重 mul。仅放开注册，现有 scheduler/codegen 即可生成融合 atomic kernel，具体实测在下方。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnScatterAdd_ScatterElements_ScatterElementsV2` | 3.0 | 0.098875 | 16.29% |
| `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd` | 3.0 | 0.254970 | 42.02% |

本项 kernel/区间合计 **0.353845 ms，占整步 58.31%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.355516 | 0.353845 | ≤0.47% |
| GPU（Eager 下界 / compiled 上界） | 0.341784 | 0.181755 | ≥46.82% |

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

<!-- BEGIN CICD VALIDATION basic_gnn_sage_Infer:scatter -->
<a id="scatter-infer-validation"></a>

#### 2026-11-05 实测：只移除五项 fallback，Infer 整步净省 0.286 ms

修改范围为导入 lowering 前，临时过滤 `aten.scatter_add` 的 packet/default/out，
以及 `aten.scatter_add_` 的 packet/default，共五项。安装文件不改动。
社区 lowering 原本就支持 `scatter_add → scatter_add_ → scatter_reduce_ → IR.Scatter`，
现有 NPU scheduler 可融合 producer；本次没有新增 scheduler 或下层编译器实现。
消息融合 kernel 的最终配置为 `simt_only, Y0BLOCK=3576, Y0BLOCK_SUB=32, X1BLOCK_SUB=64, num_warps=32`。

该模型聚合的核心计算是 `out[dst] += x[src]`，另算 degree 并除以 degree 完成 mean 聚合。
`x=[10000,64]`，`edge_index=[2,200000]`，FP32；三层各做一次消息 scatter。
两层使用 `triton_poi_fused_expand_index_select_new_zeros_1`，
最后一层为 `triton_poi_fused_add_addmm_expand_index_select_6`，包含 bias/add/ReLU 后的 gather 与 atomic add。
消息张量 `[200000,64]` 不再单独写出，约 48.8 MiB/层的中间张量读写被省掉；输出清零仍有独立 kernel。

另有三路 degree。原来是三次 CANN 调用，新 `_new_ones_new_zeros_scatter_ad_3` 一次调用写出三个结果。
这里是横向融合，**不是删除了三路 degree 中的两路计算**。实际 DSL 如下，保留原排版：

```python
        tl.atomic_add(out_ptr0 + (tmp0), tmp2, x0_mask)
        tl.atomic_add(out_ptr1 + (tmp0), tmp2, x0_mask)
        tl.atomic_add(out_ptr2 + (tmp0), tmp2, x0_mask)
```

| 选定同语义区间 | 原 ms/step | 新 ms/step |
| --- | ---: | ---: |
| message scatter + index_select/末层 ReLU、degree scatter、清零/ones 与 CANN copy | 0.547044 | 0.268064 |

原区间包含：`aclnnScatterAdd_ScatterAddAiCore_ScatterAdd` 3 次 **0.255713 ms**，
`aclnnScatterAdd_ScatterElements_ScatterElementsV2` 3 次 **0.098303 ms**，
`_index_select_select_3` 2 次 **0.096755 ms**，`_add_addmm_index_select_relu_s_7` 1 次 **0.076893 ms**，
以及 clear/ones/copy 合计 **0.019381 ms**。
新 message 两类 kernel 合计 **0.220930 ms**，degree 横向融合 **0.036608 ms**，
两类清零合计 **0.010527 ms**。
区间省 **0.278979 ms**；其他 kernel 合计再省约 **0.006521 ms**，整步净省 **0.285501 ms**。

| FP32/static，batch=1，关闭 aclgraph | Eager ms/step | Inductor ms/step |
| --- | ---: | ---: |
| 原安装版 | 0.598875 | 0.618005 |
| 仅放开五项 scatter 注册 | 0.610346 | 0.332504 |

整步耗时下降 **46.20%**；固定原始组 Eager，编译提升比从 **-3.19% 到 44.48%**，增加 **47.67 pp**。
历史 GPU 提升比为 **68.40%**，本次尚未追平；GPU 未在本轮重测。
历史 0.164787 ms 是旧四方数据上的局部预算，本次 0.285501 ms 是完整区间和整网实测，两者不相加。

正式修改优先在 NPU fallback 注册中按能力放开已验证路径；本次覆盖 FP32、静态形状、NPU950。
不能据此直接全量放开其他 dtype、动态形状、deterministic 或 out/mutation 用法。
准确率结论与 Train、环境版本及复现入口统一见本页后面的验证记录。

[原 Infer DSL/FX/IR](../../../../../artifacts/cicd/performance/small-change-validation-20261105/basic_gnn_sage_inference_baseline/graphs/basic_gnn_sage__0_inference_0.0/output_code.py)、
[修改后 Infer DSL/FX/IR](../../../../../artifacts/cicd/performance/small-change-validation-20261105/basic_gnn_sage_inference_scatter/graphs/basic_gnn_sage__0_inference_0.0/output_code.py)、
[逐步统计](../../../../../artifacts/cicd/performance/small-change-validation-20261105/summary.json)。
<!-- END CICD VALIDATION basic_gnn_sage_Infer:scatter -->

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_clamp_div_expand_view_4` | 0.011280 / 3.0 | 1.86% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_poi_fused_clamp_div_expand_view_4`：NPU_g01_output_code.py:261；输入 `f32[10000, 64][64, 1]npu:0`；size_hints=`{'y0': 10000, 'x1': 64}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_expand_index_select_new_zeros_scatter_add_select_view_1`, `triton_poi_fused_add_addmm_expand_index_select_new_zeros_relu_scatter_add_select_view_6`, `triton_poi_fused_clamp_div_expand_view_4`。

除上述已完成的 scatter 实测外，其他候选待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=1；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 1.593750 | 1.140618 | 28.43% |
| GPU | 1.330825 | 0.468717 | 64.78% |

达到 GPU **整网**提升比时，NPU 整步目标为 0.561319 ms；当前相对该目标的差额为 0.579300 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_sage_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_sage_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_sage_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_sage_Train/GPU_wrapper_calls.txt)。

### 1. Inductor scatter fallback 阻断 index_select 融合

**已验证的入口：`torch_npu/_inductor/lowering_fallback_list.py`。** 与 GCN 相同的五项 scatter 注册屏蔽了已有社区 lowering；本模型消息聚合是 index_select→scatter，不包含 GCN 的消息权重 mul；本次 Train 中 add/addmm/ReLU 仍为独立 kernel。仅放开注册，现有 scheduler/codegen 即可生成融合 atomic kernel，具体实测在下方。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnScatterAdd_ScatterElements_ScatterElementsV2` | 1.0 | 0.032990 | 2.89% |
| `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd` | 3.0 | 0.255921 | 22.44% |

本项 kernel/区间合计 **0.288911 ms，占整步 25.33%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.357938 | 0.288911 | ≤19.28% |
| GPU（Eager 下界 / compiled 上界） | 0.339635 | 0.164596 | ≥51.54% |

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

<!-- BEGIN CICD VALIDATION basic_gnn_sage_Train:scatter -->
<a id="scatter-train-validation"></a>

#### 2026-11-05 实测：五项 fallback 改动，Train 复测净省 0.200–0.222 ms

修改仍只涉及上文列出的五项 scatter fallback 注册。原 forward 中三次
`triton_poi_fused_index_select_select_3` 合计 **0.205088 ms**，
三次 `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd` 合计 **0.259445 ms**；
新 forward 为三次 `triton_poi_fused_expand_index_select_scatter_a_3`，合计 **0.212804 ms**。
这组消息 gather→scatter 从 **0.464533 → 0.212804 ms**，省 **0.251729 ms**。
输出清零、degree 和其他反向计算另计；不能把两个原 kernel 与新 kernel 当作逐名对应。

首轮整步 **1.218834 → 0.957647 ms**，净省 0.261187 ms。
但该轮变体 Eager 为 **5.805776 ms**，明显偏离原始组 **1.633214 ms**；
十个 Eager step 均偏高，原因未确认，不能用这个异常分母宣称提升比。
随后在另一张空闲卡按 baseline→scatter→baseline 同卡复测，并复用原变体最终配置：

| FP32/static、batch=1、关闭 aclgraph | Eager ms/step | Inductor ms/step |
| --- | ---: | ---: |
| 原安装版，复测前 | 1.608466 | 1.152910 |
| 仅放开五项 scatter 注册 | 1.612284 | 0.953353 |
| 原安装版，复测后 | 1.614286 | 1.175722 |

以两个 baseline 为边界，整步净省 **0.199558–0.222369 ms（17.31–18.91%）**。
取前后 baseline 均值，净省 **0.210963 ms**；固定这两次 Eager 均值 1.611376 ms，
编译提升比从 **27.74% → 40.84%**，增加 **13.09 pp**。
历史 GPU 提升比为 **64.78%**，本次尚未追平；GPU 未重测。
单项实测已经包含三层调用，不能再乘三。

本次没有改变 `triton_red_fused_add_sum_threshold_backward_3`：首轮仍有两次，
合计 **0.267308 → 0.298213 ms**，以及独立 `_sum_0` **0.041664 → 0.041652 ms**。
这两项仍是后续独立候选，不能把 scatter 收益写成 bias sum 已优化。
现有 scheduler/codegen 已能完成本项融合，本次无需修改 Triton_ascend/npuir；
正式优化应在 NPU fallback 注册中按 dtype、mutation、动态形状等能力条件放开现有 lowering。

Infer、Train 的 baseline/scatter **四组独立 accuracy 全部 `pass_accuracy`**，
沿用 runner 阈值，没有放宽精度；均为 `calls_captured=58`、一个 unique graph、零 graph break；
`calls_captured` 是 Dynamo 捕获的算子数，不是每步设备 kernel 调用次数。
performance 每组按 10 个真实 Step Id 的设备 kernel duration 求均值，不使用 runner 的主机计时。
首轮使用物理卡 3，复测使用物理卡 2；每个 A/B 或前后基线组内设备相同，未使用物理卡 4、5、6。

软件为 torch **2.13.0+cpu**、torch_npu **2.13.0.dev20261030**、Triton **3.6.0**，NPU950。
涉及的七个安装源码文件 SHA256 与 `test/cicd_benchmark@b3bd74035189` 一致；
runner checkout 单独为 `master@52b8a09a5fdb`，不能把两者混写成同一分支。
所有已结束模型实验均确认 `source_unchanged=true`，本次改动仅在进程内生效。

[原 Train forward DSL/FX/IR](../../../../../artifacts/cicd/performance/small-change-validation-20261105/basic_gnn_sage_training_baseline/graphs/basic_gnn_sage__0_forward_1.0/output_code.py)、
[修改后 Train forward DSL/FX/IR](../../../../../artifacts/cicd/performance/small-change-validation-20261105/basic_gnn_sage_training_scatter/graphs/basic_gnn_sage__0_forward_1.0/output_code.py)、
[逐步统计与前后复测](../../../../../artifacts/cicd/performance/small-change-validation-20261105/summary.json)、
[安装源码身份](../../../../../artifacts/cicd/performance/small-change-validation-20261105/source_identity.json)、
[最终配置](../../../../../artifacts/cicd/performance/small-change-validation-20261105/selected_configs.json)、
[复现入口](../../../../../scripts/repro/cicd/performance/gnn_validation/README.md)。
<!-- END CICD VALIDATION basic_gnn_sage_Train:scatter -->

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
| `triton_poi_fused_clamp_div_expand_gather_index_2` | 0.148314 / 2.0 | 13.00% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_red_fused_add_sum_threshold_backward_3`：NPU_g01_output_code.py:281；输入 `f32[10000, 64][64, 1]npu:0`；size_hints=`{'x0': 64, 'r1': 10000}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_threshold_backward_4`, `triton_red_fused_sum_0`, `triton_poi_fused_add_copy__0`。
- `triton_poi_fused_clamp_div_expand_gather_index_2`：NPU_g01_output_code.py:207；输入 `i64[2, 200000][200000, 1]npu:0`；size_hints=`{'y0': 200000, 'x1': 64}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_expand_index_select_scatter_add_select_view_3`, `triton_poi_fused_clamp_div_expand_gather_index_add_select_view_3`, `triton_poi_fused_clamp_div_expand_view_4`。

除上述已完成的 scatter 实测外，其他候选待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
