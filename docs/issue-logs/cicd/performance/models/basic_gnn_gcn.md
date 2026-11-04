---
title: "CICD 性能分析：basic_gnn_gcn"
---

# basic_gnn_gcn：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；初始四方粗筛在本地完成；2026-11-04 已补充 NPU_A5 整网 A/B、配置复测和精度验证，详见各项实测。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=1；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 0.845399 | 0.737259 | 12.79% |
| GPU | 1.086228 | 0.293174 | 73.01% |

达到 GPU **整网**提升比时，NPU 整步目标为 0.228174 ms；当前相对该目标的差额为 0.509085 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_gcn_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_gcn_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_gcn_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_gcn_Infer/GPU_wrapper_calls.txt)。

### 1. Inductor fallback list 阻断 scatter_add 与 gather/mul 融合

**已验证入口：`torch_npu/_inductor/lowering_fallback_list.py`。** 五个 `scatter_add` 条目覆盖了已有社区 lowering，使 gather/mul 的消费者落入外部调用。临时移除这五项后，现有 scheduler/codegen 已能生成融合的 `tl.atomic_add` kernel；本次无需新增 scheduler 规则或修改下层编译器。以下为历史四方粗筛，实测另列。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnScatterAdd_ScatterElements_ScatterElementsV2` | 3.0 | 0.100559 | 13.64% |
| `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd` | 3.0 | 0.267468 | 36.28% |

本项 kernel/区间合计 **0.368027 ms，占整步 49.92%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.371113 | 0.368027 | ≤0.83% |
| GPU（Eager 下界 / compiled 上界） | 0.361072 | 0.191119 | ≥47.07% |

NPU Eager 对账项：`aclnnScatterAdd_ScatterElements_ScatterElementsV2`, `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd`。

GPU Eager 对账项：`scatter_add_`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_expand_index_select_mul_new_zeros_scatter_add_select_view_1` | 3.0 | 0.167938 | 57.28% |
| `triton_poi_fused_new_zeros_ones_scatter_add_select_1` | 3.0 | 0.015111 | 5.15% |
| `triton_poi_fused_expand_index_select_mul_new_zeros_scatter_add_select_view_0` | 3.0 | 0.005248 | 1.79% |
| `triton_poi_fused_new_zeros_ones_scatter_add_select_0` | 3.0 | 0.002823 | 0.96% |

**贡献粗估：0.171593 ms/step，相当于当前 NPU Inductor 整步耗时的 23.27%，使整网提升比增加 20.297 个百分点，单独达到该预算后整网提升比约为 33.09%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnScatterAdd_ScatterElements_ScatterElementsV2`：外部调用；见 wrapper 调用边界。
- `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd`：外部调用；见 wrapper 调用边界。

<!-- BEGIN CICD VALIDATION basic_gnn_gcn_Infer:scatter -->
<a id="scatter-fusion-validation"></a>

#### 2026-11-04 实测：恢复现有 scatter lowering，Infer 整步净省 0.285 ms

**具体问题是 fallback 注册策略屏蔽了可用实现。** `lowering_fallback_list.py` 包含以下五项：

```python
    aten.scatter_add,
    aten.scatter_add.default,
    aten.scatter_add.out,
    aten.scatter_add_,
    aten.scatter_add_.default,
```

导入 NPU lowering 前仅过滤这五项，社区 `scatter_add → scatter_add_ → scatter_reduce_` lowering
便可产生 `IR.Scatter`，现有 scheduler 将 gather、乘权重和 atomic add 一起生成。
原先每层的计算可写成：

```python
message = x[src] * weight[:, None]             # [209978,64]
out = zeros([10000,64]).scatter_add(0, dst[:, None].expand_as(message), message)
```

独立 message 的临时 buffer 有 13,438,592 个 FP32 元素（约 51.3 MiB）。融合后在 kernel 内读源节点、乘权重，
直接累加到目标节点，省去该中间张量的写入/再读。输出清零仍有独立 kernel，不能一并当作消除。
另一路是 degree 计算：把边上的 ones 按目标节点归约到 `[10000]`。
因此三层共有 **3 次 message scatter + 3 次 degree scatter**，不是六次相同形状的调用。

生成的 message kernel 为 `triton_poi_fused_expand_index_select_mul_new_z_1`。
以下保留原 DSL 行，仅省略 index 校验等中间语句：

```python
        tmp0 = tl.load(in_ptr0 + (209978 + y0), y0_mask)
        tmp2 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp3 = tl.load(in_ptr0 + (y0), y0_mask)
```

```python
        tmp9 = tl.load(in_ptr2 + (x1 + 64*tmp7), x1_mask & y0_mask)
        tmp10 = tmp2 * tmp9
        tl.atomic_add(out_ptr0 + (x1 + 64*tmp0), tmp10, x1_mask & y0_mask)
```

| 同语义前向区间 | 次数/step | ms/step |
| --- | ---: | ---: |
| 原外部 `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd` | 3 | 0.268657 |
| 原外部 `aclnnScatterAdd_ScatterElements_ScatterElementsV2` | 3 | 0.099211 |
| 原 `triton_poi_fused_index_select_mul_select_view_1` | 3 | 0.218821 |
| 原 `triton_poi_fused_new_zeros_0` | 6 | 0.013124 |
| 新 message 融合 `_expand_index_select_mul_new_z_1` | 3 | 0.234704 |
| 新 degree 融合 `_new_zeros_ones_scatter_add_se_1` | 3 | 0.093072 |
| 新 message 清零 `_expand_index_select_mul_new_z_0` | 3 | 0.008746 |
| 新 degree 清零 `_new_zeros_ones_scatter_add_se_0` | 3 | 0.003982 |

原链路合计 **0.599812 ms**，新链路 **0.340504 ms**，区间节省 **0.259308 ms**。
其余 kernel 合计减少约 0.025661 ms，整步净省 **0.284970 ms**。
这里同时计入原 gather/mul 和清零，不能拿两个外部 scatter 的 0.367868 ms 直接减去新融合 kernel。

| FP32/static、batch=1、关闭 aclgraph | Eager ms/step | Inductor ms/step |
| --- | ---: | ---: |
| 原实现 | 0.859507 | 0.767982 |
| 仅移除五项 scatter fallback | 0.867339 | 0.483013 |

**整步下降 37.11%**；固定基线 Eager 折算，编译提升比增加 **33.16 pp**，由 10.65% 到 43.80%。
历史 GPU 提升比为 73.01%，本项取得收益后仍有整网差距，不能说已追平 GPU。
历史局部预算 0.171593 ms 与本次不同基线、完整融合边界的实测不同，不能相加。

**runtime 配置也必须保留正确候选。** 新 message kernel 的 metadata 为 `npu_kernel_type=simd_simt_mix`，
最终保存配置是 `Y0BLOCK=3752, Y0BLOCK_SUB=32, X1BLOCK_SUB=64, compile_mode=simt_only,
num_warps=32, multibuffer=False`，实际 grid 为 56。
同语义手写融合小图包含清零，`simt_only / block=1024` 为 **84.025 μs**；
相同手写 DSL 用 `simd_simt_template / block=1024` 则为 **13.428 ms**，原 fallback 小图 **167.149 μs**。
这证明“允许融合”还需配合合适的执行模式；本次现有 autotune 已选到快实现，
不应据慢的手写配置就判定 Triton_ascend/npuir 有缺陷，也不把 84 μs 直接外推整网。

正式修改优先对 `lowering_fallback_list.py` 的注册加能力条件，恢复本次已验证的 FP32/NPU950 路径；
其他 dtype、动态 shape、deterministic、mutation/out 约束及其他芯片需另行回归。
本次五项一起移除是恢复 packet/overload 注册的诊断手段，不是建议无条件全量删除。
`lowering.py` 的社区 scatter lowering、NPU scheduling 与 atomic codegen 已能工作，暂不需先重写它们。

推理修改组为 **`pass_accuracy`**；使用原 runner 阈值。
[整网与逐 kernel 统计][g-summary]、[实际融合 DSL / FX / IR][g-scatter-code]、
[最终配置][g-config]、[同语义小图结果][g-scatter-micro]保留了边界与配置证据。
<!-- END CICD VALIDATION basic_gnn_gcn_Infer:scatter -->

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_eq_index_masked_fill_mul_pow__2` | 0.022782 / 3.0 | 3.09% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_poi_fused_eq_index_masked_fill_mul_pow__2`：NPU_g03_output_code.py:161；输入 `i64[2, 209978][209978, 1]npu:0`；size_hints=`{'x0': 209978}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_expand_index_select_mul_new_zeros_scatter_add_select_view_1`, `triton_poi_fused_new_zeros_ones_scatter_add_select_1`, `triton_poi_fused_eq_index_masked_fill_mul_pow_select_2`。

除上述已实测项目外，其他候选待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=1；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 2.381309 | 1.266601 | 46.81% |
| GPU | 1.992435 | 0.550495 | 72.37% |

达到 GPU **整网**提升比时，NPU 整步目标为 0.657938 ms；当前相对该目标的差额为 0.608663 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_gcn_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_gcn_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_gcn_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_gcn_Train/GPU_wrapper_calls.txt)。

### 1. Inductor fallback list 阻断 scatter_add 与 gather/mul 融合

**已验证入口：`torch_npu/_inductor/lowering_fallback_list.py`。** 五个 `scatter_add` 条目覆盖了已有社区 lowering，使 gather/mul 的消费者落入外部调用。临时移除这五项后，现有 scheduler/codegen 已能生成融合的 `tl.atomic_add` kernel；本次无需新增 scheduler 规则或修改下层编译器。以下为历史四方粗筛，实测另列。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnScatterAdd_ScatterElements_ScatterElementsV2` | 3.0 | 0.100862 | 7.96% |
| `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd` | 3.0 | 0.269092 | 21.25% |

本项 kernel/区间合计 **0.369953 ms，占整步 29.21%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.372117 | 0.369953 | ≤0.58% |
| GPU（Eager 下界 / compiled 上界） | 0.362071 | 0.189692 | ≥47.61% |

NPU Eager 对账项：`aclnnScatterAdd_ScatterElements_ScatterElementsV2`, `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd`。

GPU Eager 对账项：`scatter_add_`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_expand_index_select_mul_scatter_add_select_view_1` | 3.0 | 0.167333 | 30.40% |
| `triton_poi_fused_new_zeros_ones_scatter_add_select_1` | 3.0 | 0.019568 | 3.55% |
| `triton_poi_fused_new_zeros_ones_scatter_add_select_0` | 3.0 | 0.002791 | 0.51% |

**贡献粗估：0.174999 ms/step，相当于当前 NPU Inductor 整步耗时的 13.82%，使整网提升比增加 7.349 个百分点，单独达到该预算后整网提升比约为 54.16%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnScatterAdd_ScatterElements_ScatterElementsV2`：外部调用；见 wrapper 调用边界。
- `aclnnScatterAdd_ScatterAddAiCore_ScatterAdd`：外部调用；见 wrapper 调用边界。

<!-- BEGIN CICD VALIDATION basic_gnn_gcn_Train:scatter -->
<a id="scatter-training-validation"></a>

#### 2026-11-04 实测：同一 fallback 改动覆盖 Train 前向

Train 仍是 `lowering_fallback_list.py` 五项注册阻断前向 gather/mul→scatter 融合。
恢复社区 lowering 后，三层前向外部 scatter 消失，生成
`triton_poi_fused_expand_index_select_mul_scatt_1` 和 degree 的
`triton_poi_fused_new_zeros_ones_scatter_add_se_1`，各 3 次/step；
原有反向 `expand_gather_index_add_mul` atomic kernel 保留。
**前向 scatter 的移除不等于把反向 scatter 一起优化了。**

第一轮同环境整网 A/B：Eager 为 2.401377 / 2.429127 ms，
Inductor **1.287585 → 0.958072 ms**，净省 **0.329513 ms（25.59%）**；
固定基线 Eager，编译提升比增加 **13.72 pp**，达到约 **60.10%**，历史 GPU 为 **72.37%**。
单独 split 和组合实测见下一项，不能把两轮数字混合求和。
Train baseline、仅 scatter、仅 split、两项组合均通过 **`pass_accuracy`**。
正式修改仍是按能力放开 fallback，无需为该收益新增 scatter fusion scheduler 规则或修改下层组件。

[Train 前向融合代码][g-train-scatter]、[所有组统计][g-summary]、[精度与复现命令][g-run]。
<!-- END CICD VALIDATION basic_gnn_gcn_Train:scatter -->

### 2. Inductor IR 未拆分 linear bias 梯度的 outer sum

**已验证入口：`torch_npu/_inductor/ir.py::num_splits`。** 该 sum 来自 linear 的 bias 梯度，输入 `[10000,64]` 沿第 0 维归约；它与 scatter backward 共用上游梯度，但不是 scatter_add 内部的归约。输出 64 个元素时仍返回单阶段计划，形成低并行度的长列归约。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_sum_0` | 3.0 | 0.127919 | 10.10% |

本项 kernel/区间合计 **0.127919 ms，占整步 10.10%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.025310 | 0.127919 | ≤-405.41% |
| GPU（Eager 下界 / compiled 上界） | 0.021774 | 0.014708 | ≥32.45% |

NPU Eager 对账项：`aclnnReduceSum_ReduceSumOpAiCore_ReduceSum`, `aclnnReduceSum_BroadcastToAiCore_BroadcastTo`。

GPU Eager 对账项：`sum`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_sum_0` | 3.0 | 0.006455 | 1.17% |
| `sum` | 1.0 | 0.004356 | 0.79% |
| `triton_red_fused_sum_1` | 3.0 | 0.003898 | 0.71% |

**贡献粗估：0.110822 ms/step，相当于当前 NPU Inductor 整步耗时的 8.75%，使整网提升比增加 4.654 个百分点，单独达到该预算后整网提升比约为 51.46%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused_sum_0`：同名/截断名对应 3 个定义（NPU_g04_output_code.py:55, NPU_g06_output_code.py:55, NPU_g08_output_code.py:55）；以下尺寸仅展示首个入口：NPU_g04_output_code.py:55；输入 `f32[10000, 64][64, 1]npu:0`；size_hints=`{'x0': 64, 'r1': 10000}`；split_axis=`[0]`。

<!-- BEGIN CICD VALIDATION basic_gnn_gcn_Train:sum -->
<a id="bias-sum-validation"></a>

#### 2026-11-04 实测：拆分 bias sum，局部省约 0.12 ms；整步收益受其他 kernel 波动影响

**责任模块是 NPU Inductor `ir.py::num_splits` 的归约计划选择。**
FX 明确把这一项标在 `out = out + self.bias` 的反向：

```python
        sum_1: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True)
        view_2: "f32[64]" = torch.ops.aten.view.default(sum_1, [64]);  sum_1 = None
```

`input=[10000,64]`、stride `[64,1]`，即对每个输出通道，把 10000 个节点贡献的梯度相加。
同一份 `tangents_1` 还给另一条 gather/index_add 反向分支，但本 sum 不执行 atomic scatter。
三层各一个 `triton_red_fused_sum_0`，一步 3 次。

安装版 `num_splits` 的关键分支如下（省略单元素输出内部细节）：

```python
    if numel_hint == 1:
        # ... calculate split for a single output ...
        return ReductionHint.INNER, split
    return ReductionHint.DEFAULT, 1
```

本项输出元素数为 64，即使 `config.split_reductions=True` 仍返回 1。
实际整网 winner 是 `X0BLOCK=2, X0BLOCK_SUB=1, R1BLOCK_SUB=1024`、
`simd_simt_template, num_warps=1`，只有 32 个 program，逐列遍历 10000 行。
GPU 历史 DSL 已拆为 79 个部分和再做第二阶段；其确切分块不应直接硬套 NPU。

实验在 IR 层把本项改为 **20 份 × 500 行**，已有 multilayer reduction 生成两阶段归约。
第一阶段逻辑轴为 `[20,64,500]`，使通道连续访存并增大 grid；第二阶段归约 20 个部分和。
仅替换 `num_splits`，不改 model、scheduler、Triton 后端或 npuir。

| 第一轮同语义 kernel 链 | 次数/step | ms/step |
| --- | ---: | ---: |
| 原 `triton_red_fused_sum_0` | 3 | 0.134403 |
| 新 `triton_per_fused_sum_0`，部分和 | 3 | 0.010042 |
| 新 `triton_per_fused_sum_1`，最终和 | 3 | 0.005056 |

局部 **0.134403 → 0.015098 ms**，减少 **0.119305 ms（88.77%）**。
但第一轮整网 **1.287585 → 1.259494 ms**，只净省 **0.028091 ms（2.18%）**。
其中其他 kernel 合计增加约 0.091214 ms，主要是反向
`triton_poi_fused_expand_gather_index_add_mul_s_2`（新编号 `_3`）
**0.269369 → 0.336127 ms**，前向 gather/mul 也增加约 0.013345 ms。
比对确认该反向 atomic kernel 的 **DSL 函数体和保存的 selected config 相同**，
都是 `Y0BLOCK=3752, Y0BLOCK_SUB=32, X1BLOCK_SUB=64, simt_only, num_warps=32`。
因此不能把这次成本变化归因于新生成了更差 DSL 或 autotune 改选；当前证据只确认运行上下文中的波动，
尚不能细分到缓存、调度等原因。

为检查整步收益，复用各变体第一轮已选配置，按 baseline→split→scatter→both→baseline 顺序重跑。
FX graph cache 关闭，保留生成代码和新 profile；下表不含主机 wall time：

| 同轮复测 | Eager ms/step | Inductor ms/step | 对前后基线的整步净省 |
| --- | ---: | ---: | ---: |
| baseline，开始 | 2.402775 | 1.332182 | — |
| 仅 split | 2.354866 | 1.193621 | **0.138561–0.179805 ms** |
| 仅 scatter | 2.362831 | 0.956486 | **0.375696–0.416940 ms** |
| split + scatter | 2.361428 | 0.878719 | **0.453463–0.494707 ms** |
| baseline，结束 | 2.404178 | 1.373426 | — |

复测 baseline 的 sum 合计为 0.132716 / 0.134300 ms，split 后为 **0.017424 ms**，
局部收益可重复。仅 scatter→两项叠加，sum 链 **0.121978 → 0.020451 ms**，
整步再省 **0.077767 ms**；组合总收益已实测，不能把独立测试节省量相加。
以同轮前后基线 Eager 的均值作分母，组合后的编译提升比约 **63.44%**，仍低于历史 GPU 的 **72.37%**。
第一轮组合组 Eager=3.498700 ms、Inductor=1.579355 ms，连 Eager 控制都明显变慢，
该组保留在原始统计中，不用它推断组合策略无收益。

**可执行的优化目标**：正式修改 `ir.py::num_splits`，让这类小输出、长外轴 sum 进入成本可接受的多阶段计划；
以自动生成的 3 次部分和 + 3 次最终和 **约 0.015–0.020 ms/step** 为局部验收目标。
第一轮单独优化整步只省 0.028 ms，复测约省 0.139–0.180 ms；不能承诺局部 0.119 ms 全部等额落到整网。
在已恢复 scatter 的路径上，本次增加 split 实际再省约 0.078 ms，适合作为联合优化的当前整步参考。
手写同语义小图为原 kernel 40.594 μs、较宽 tile 20.225 μs、两阶段 4.311 μs；
它支持先改 IR 计划而非只调 tile，但不能作为额外收益再加一次。

正式策略应根据输出 program 数、R 长度、连续通道访存、临时 buffer 和第二阶段开销判断，
而不是硬编码 10000×64 或 20 份。`NPUIndexTritonKernel.reduction` 与 runtime tiling
负责把已有计划落到 DSL/config；本次它们无需修改就获得局部收益，没有证据要求先修 Triton_ascend/npuir。
若后续要追究未改 atomic kernel 的波动，需要另做固定输入/配置、固定上下文的重复测试，当前不据此定责下层。

环境为 NPU_A5 **物理卡 3**、`pta_213_cicd`，FP32/static、batch=1、关闭 aclgraph。
torch `2.13.0+cpu`、torch_npu `2.13.0.dev20261030`、Triton `3.6.0`；
七个关键 NPU 安装文件与 `test/cicd_benchmark@b3bd74035189` 的 SHA256 一致。
实际 runner checkout 为 `master@52b8a09a5fdb`，不把 runner HEAD 当作安装版编译器版本。
所有实验改动为进程内 hook，安装文件未变化；采样为预热后的 10 个 Step Id，按设备 kernel duration 求和。
GCN 仍有原先 5 次 graph break，accuracy baseline 与变体均相同，本次没有消除这些 break。
训练四组和推理修改组均为 **`pass_accuracy`**，沿用 runner 阈值；训练检查 2 步。

[完整统计及每步明细][g-summary]、[原 backward DSL/FX/IR][g-sum-before]、
[拆分后 backward DSL/FX/IR][g-sum-after]、[配置][g-config]、[版本对应][g-identity]、[实验入口][g-run]。
完整原始产物只去敏路径和构建 hash，保留 DSL 排版。

[g-summary]: ../../../../../artifacts/cicd/performance/gnn-validation/results/summary.json
[g-scatter-code]: ../../../../../artifacts/cicd/performance/gnn-validation/results/basic_gnn_gcn_inference_scatter/graphs/basic_gnn_gcn__4_inference_4.4/output_code.py
[g-config]: ../../../../../artifacts/cicd/performance/gnn-validation/results/selected_configs.json
[g-scatter-micro]: ../../../../../artifacts/cicd/performance/gnn-validation/results/micro_scatter_verified/result.json
[g-train-scatter]: ../../../../../artifacts/cicd/performance/gnn-validation/results/basic_gnn_gcn_training_scatter/graphs/basic_gnn_gcn__5_forward_10.5/output_code.py
[g-sum-before]: ../../../../../artifacts/cicd/performance/gnn-validation/results/basic_gnn_gcn_training_baseline/graphs/basic_gnn_gcn__5_backward_12.6/output_code.py
[g-sum-after]: ../../../../../artifacts/cicd/performance/gnn-validation/results/basic_gnn_gcn_training_split/graphs/basic_gnn_gcn__5_backward_12.6/output_code.py
[g-identity]: ../../../../../artifacts/cicd/performance/gnn-validation/results/source_identity.json
[g-run]: ../../../../../scripts/repro/cicd/performance/gnn_validation/README.md
<!-- END CICD VALIDATION basic_gnn_gcn_Train:sum -->

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_expand_gather_index_add_mul_s_2` | 0.265910 / 3.0 | 20.99% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_eq_index_masked_fill_mul_pow__2` | 0.025332 / 3.0 | 2.00% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_poi_fused_expand_gather_index_add_mul_s_2`：同名/截断名对应 3 个定义（NPU_g04_output_code.py:192, NPU_g06_output_code.py:192, NPU_g08_output_code.py:192）；以下尺寸仅展示首个入口：NPU_g04_output_code.py:192；输入 `i64[2, 209978][209978, 1]npu:0`；size_hints=`{'y0': 209978, 'x1': 64}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_expand_gather_index_add_mul_select_view_3`, `triton_poi_fused_expand_index_select_mul_scatter_add_select_view_1`, `triton_poi_fused_new_zeros_ones_scatter_add_select_1`。
- `triton_poi_fused_eq_index_masked_fill_mul_pow__2`：NPU_g03_output_code.py:161；输入 `i64[2, 209978][209978, 1]npu:0`；size_hints=`{'x0': 209978}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_expand_gather_index_add_mul_select_view_3`, `triton_poi_fused_expand_index_select_mul_scatter_add_select_view_1`, `triton_poi_fused_new_zeros_ones_scatter_add_select_1`。

除上述已实测项目外，其他候选待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
