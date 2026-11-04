---
title: "CICD 性能分析：lennard_jones"
---

# lennard_jones：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；初筛在本地复算；2026-11-05 补充 NPU_A5 单项改动的整网 A/B，原始四方统计不混入本次结果。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=4096；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 0.022059 | 0.017641 | 20.03% |
| GPU | 0.034382 | 0.029444 | 14.36% |

达到 GPU **整网**提升比时，NPU 整步目标为 0.018891 ms；当前相对该目标的差额为 -0.001250 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/lennard_jones_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/lennard_jones_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/lennard_jones_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/lennard_jones_Infer/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_addmm_tanh_1` | 0.006367 / 3.0 | 36.09% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_addmm_t_tanh_0` | 0.002052 / 0.9 | 11.63% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_poi_fused_addmm_tanh_1`：NPU_g01_output_code.py:120；输入 `f32[16][1]npu:0`；size_hints=`{'y0': 4096, 'x1': 16}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_tanh_1`, `triton_poi_fused_addmm_t_tanh_0`。
- `triton_poi_fused_addmm_t_tanh_0`：NPU_g01_output_code.py:62；输入 `f32[16][1]npu:0`；size_hints=`{'y0': 4096, 'x1': 16}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_tanh_1`, `triton_poi_fused_addmm_t_tanh_0`。

除上述实测项外，其他候选待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=4096；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 0.119734 | 0.295936 | -147.16% |
| GPU | 0.147570 | 0.118586 | 19.64% |

达到 GPU **整网**提升比时，NPU 整步目标为 0.096218 ms；当前相对该目标的差额为 0.199719 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/lennard_jones_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/lennard_jones_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/lennard_jones_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/lennard_jones_Train/GPU_wrapper_calls.txt)。

### 1. Inductor IR 的 bias sum 单阶段计划限制 tanh backward 访存

**已验证入口：NPU Inductor `torch_npu/_inductor/ir.py::num_splits`。** 输出只有 16 个元素时仍选单阶段归约；tanh 导数与 bias sum 融进逐列的大 persistent tile。仅改变归约计划就能触发现有 pointwise + 两阶段 sum 生成，具体配置与整网收益见实测。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused_sum_tanh_backward_2` | 3.0 | 0.178655 | 60.37% |

本项 kernel/区间合计 **0.178655 ms，占整步 60.37%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.035622 | 0.178655 | ≤-401.52% |
| GPU（Eager 下界 / compiled 上界） | 0.040021 | 0.018727 | ≥53.21% |

NPU Eager 对账项：`aclnnReduceSum_ReduceSumOpAiCore_ReduceSum`, `aclnnReduceSum_BroadcastToAiCore_BroadcastTo`, `aclnnTanhBackward_TanhGradAICore_TanhGrad`。

GPU Eager 对账项：`sum`, `tanh_backward`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_sum_2` | 4.0 | 0.005594 | 4.72% |
| `triton_per_fused_sum_3` | 4.0 | 0.004650 | 3.92% |
| `triton_poi_fused_tanh_backward_4` | 3.0 | 0.003648 | 3.08% |
| `sum` | 1.0 | 0.002317 | 1.95% |
| `triton_red_fused_sum_0` | 1.0 | 0.001357 | 1.14% |
| `triton_poi_fused_mm_t_tanh_backward_1` | 1.0 | 0.001162 | 0.98% |

**贡献粗估：0.161986 ms/step，相当于当前 NPU Inductor 整步耗时的 54.74%，使整网提升比增加 135.288 个百分点，单独达到该预算后整网提升比约为 -11.87%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_per_fused_sum_tanh_backward_2`：NPU_g01_output_code.py:187；输入 `f32[4096, 16][16, 1]npu:0`；size_hints=`{'x0': 16, 'r1': 4096}`；split_axis=`[0]`。

<!-- BEGIN CICD VALIDATION lennard_jones_Train:tanh_grad_reduce -->
<a id="tanh-bias-validation"></a>

#### 2026-11-05 实测：只改 num_splits，整步 0.269425 → 0.100832 ms

每处计算先求 `dZ = dY * (1 - tanh_output**2)`，再沿 batch 求和得到 `dBias[16]`；
`dZ[4096,16]` 还会被后续 mm 使用。三次 `triton_per_fused_sum_tanh_backward_2`
及一次 `triton_per_fused_mm_sum_t_tanh_backward_1` 都包含这一归约。
后者还融合了由最后一层权重广播形成的乘法，不能漏掉这一次或把名字里的 mm 当作裸 MM。

安装版 `ir.py::num_splits` 只对输出元素数等于 1 的归约考虑拆分，输出为 16 时返回
`(ReductionHint.DEFAULT, 1)`；实际 `split_reductions=True`。
原三个 `_sum_tanh_backward_2` 的最终配置是
`X0BLOCK=1, X0BLOCK_SUB=1, R1BLOCK_SUB=4096, simd_simt_template, num_warps=1`，
即 16 个 program，每个 program 处理一列的 4096 个值。原 DSL 如下，仅省略中间算术：

```python
        tmp0 = tl.load(in_out_ptr0 + (x0 + 16*r1), x0_mask, other=0.0)
        tmp1 = tl.load(in_ptr0 + (x0 + 16*r1), x0_mask, other=0.0)
```

```python
        tmp8 = tl.sum(tmp7, 0).reshape(1, X0BLOCK_SUB)
        tl.store(in_out_ptr0 + (x0 + 16*r1), tmp5, x0_mask)
        tl.store(out_ptr0 + (x0 ), tmp8, x0_mask)
```

`X0BLOCK_SUB=1` 表示不能同时处理连续的 16 个通道，load/store 沿 batch 跨 stride=16。
pointwise 导数的完整写回也被绑定在按列的归约 kernel 内。

实验沿用 GNN 的进程内分段 hook，仅把实验触发下限由 8192 降至 4096：
对 NPU sum、输出 2–128 个元素，使用 `min(256,ceil(R/512))` 份。
实际只命中 `sum_2(mul_2)`、`sum_3(mul_4)`、`sum_4(mul_6)`、`sum_5(mul_8)` 四处，
各拆为 **8 份 × 512 行**。原有 scheduler 自动生成 pointwise 与两阶段 sum，未另写手工 kernel。

新部分和 `_sum_2` 的 logical shape 为 `[8,16,512]`，配置
`Y0BLOCK=1, X1BLOCK=16, X1BLOCK_SUB=16, R2BLOCK_SUB=512`，grid 为 `8×1`；
其原始 load 为：

```python
            tmp0 = tl.load(in_ptr0 + (x1 + 16*r2 + 8192*y0), x1_mask & y0_mask, other=0.0)
```

新 pointwise `_tanh_backward_4` 选中 `Y0BLOCK=256, Y0BLOCK_SUB=256, X1BLOCK_SUB=16`，
沿 batch 分 16 个 program，同时处理连续通道。
因此这里不能简单归因于“program 更多”：部分和甚至只有 8 个 program；
主要改变是导数写回与长列归约解耦、连续通道一起访存，归约工作也缩短为每份 512 行。

| 同语义链路，一步调用次数 | 原 ms/step | 新 ms/step |
| --- | ---: | ---: |
| 原 `_sum_tanh_backward_2`，3 次 | 0.150380 | — |
| 原 `_mm_sum_t_tanh_backward_1`，1 次 | 0.038934 | — |
| 新 `_tanh_backward_4`，3 次 | — | 0.005248 |
| 新 `_mm_t_tanh_backward_1`，1 次 | — | 0.002342 |
| 新 `_sum_2` 部分和，4 次 | — | 0.008263 |
| 新 `_sum_3` 最终和，4 次 | — | 0.004474 |
| **合计** | **0.189314** | **0.020327** |

局部减少 **0.168987 ms（89.26%）**。其余 kernel 合计增加约 0.000395 ms，
整步净省 **0.168592 ms（62.57%）**；该数已经包含四处归约，不能再乘层数。

| FP32/static、batch=4096、关闭 aclgraph | Eager ms/step | Inductor ms/step |
| --- | ---: | ---: |
| 原安装版 | 0.126628 | 0.269425 |
| 仅 IR 分段 | 0.128046 | 0.100832 |

固定原始组 Eager，编译提升比从 **-112.77% 到 20.37%**，增加 **133.14 pp**；
历史 GPU 为 **19.64%**，本次均值略高，但余量仅约 0.73 pp，需要结合下方复测看稳定性。
GPU 本轮未重测；原文 0.161986 ms 仅为旧四方筛选预算，不与本次收益相加。

正式优化入口是 `ir.py::num_splits` 的小输出 outer reduction 成本策略，
需要同时考虑输入连续轴、pointwise 输出的消费者、临时部分和与额外 launch 的成本。
本次 `4096`、`8` 和 `512` 是诊断配置，不能直接按模型或 shape 硬编码。
scheduler、`NPUIndexTritonKernel.reduction` 和 runtime tiling 已能完成新计划，
本次无需修改 Triton_ascend/npuir，也没有证据把该性能差距归为下层编译缺陷。

随后在物理卡 2 上按原实现→split→原实现复测，组内使用同一卡，并复用原变体缓存：

| FP32/static、batch=4096、关闭 aclgraph | Eager ms/step | Inductor ms/step |
| --- | ---: | ---: |
| 原安装版，复测前 | 0.127909 | 0.273076 |
| 仅 IR 分段 | 0.128447 | 0.101753 |
| 原安装版，复测后 | 0.129134 | 0.276211 |

前后基线给出的整步净省范围为 **0.171323–0.174458 ms（62.74–63.16%）**。
固定两个 baseline Eager 均值 0.128521 ms，编译提升比从 **-113.69% → 20.83%**；
相对历史 GPU 的 19.64% 高约 **1.19 pp**。首轮与复测均略高于历史 GPU，
但 GPU 没有同步重测，且余量很小；这只说明当前固定场景达到了参考目标，不代表所有配置稳定超过。

原始/split **两组独立训练 accuracy 均 `pass_accuracy`**，阈值未改，
各 `calls_captured=9`、一个 unique graph、零 graph break；这是捕获的算子数，不是设备 kernel 次数。performance 每组统计 10 个真实 Step Id，
按设备 kernel duration 求均值。首轮物理卡 3、复测物理卡 2，未使用物理卡 4、5、6。
软件为 torch 2.13.0+cpu、torch_npu 2.13.0.dev20261030、Triton 3.6.0，NPU950。
七个相关安装源码文件与 `test/cicd_benchmark@b3bd74035189` SHA256 一致，
runner 为 `master@52b8a09a5fdb`；所有已结束实验 `source_unchanged=true`。

[原 backward DSL/FX/IR](../../../../../artifacts/cicd/performance/small-change-validation-20261105/lennard_jones_training_baseline/graphs/lennard_jones__0_backward_3.1/output_code.py)、
[分段后 backward DSL/FX/IR](../../../../../artifacts/cicd/performance/small-change-validation-20261105/lennard_jones_training_split/graphs/lennard_jones__0_backward_3.1/output_code.py)、
[最终配置](../../../../../artifacts/cicd/performance/small-change-validation-20261105/selected_configs.json)、
[逐步统计](../../../../../artifacts/cicd/performance/small-change-validation-20261105/summary.json)、
[安装源码身份](../../../../../artifacts/cicd/performance/small-change-validation-20261105/source_identity.json)、
[复现入口](../../../../../scripts/repro/cicd/performance/gnn_validation/README.md)。
<!-- END CICD VALIDATION lennard_jones_Train:tanh_grad_reduce -->

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_addmm_tanh_1` | 0.006537 / 3.0 | 2.21% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_add_copy__0` | 0.005977 / 6.0 | 2.02% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_poi_fused_addmm_tanh_1`：NPU_g02_output_code.py:120；输入 `f32[16][1]npu:0`；size_hints=`{'y0': 4096, 'x1': 16}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_tanh_1`, `triton_poi_fused_addmm_t_tanh_0`。
- `triton_poi_fused_add_copy__0`：NPU_g03_output_code.py:58；输入 `f32[16, 1][1, 1]npu:0`；size_hints=`{'x0': 16}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_copy__0`, `triton_poi_fused_add_copy__1`, `triton_poi_fused_add_copy__2`。

除上述实测项外，其他候选待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
