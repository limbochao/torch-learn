---
title: "CICD 性能分析：basic_gnn_edgecnn"
---

# basic_gnn_edgecnn：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；初始四方粗筛仅在本地复算；2026-11-04 已补充 NPU_A5 整网 A/B 与精度验证，见 Train 第一项。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=1；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 1.987049 | 1.364681 | 31.32% |
| GPU | 5.353945 | 2.762886 | 48.40% |

达到 GPU **整网**提升比时，NPU 整步目标为 1.025411 ms；当前相对该目标的差额为 0.339270 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_edgecnn_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_edgecnn_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_edgecnn_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_edgecnn_Infer/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_index_select_relu_select_sub_3` | 0.280373 / 2.0 | 20.54% | NPU Inductor gather/scatter lowering 与 producer-consumer 融合 |
| `aclnnInplaceCopy_BroadcastToAiCore_BroadcastTo` | 0.147457 / 3.0 | 10.81% | NPU Inductor 布局/materialization 与 CANN 广播路径；先定位触发的外部调用 |
| `triton_poi_fused_index_select_select_sub_0` | 0.144060 / 1.0 | 10.56% | NPU Inductor gather/scatter lowering 与 producer-consumer 融合 |

- `triton_poi_fused_index_select_relu_select_sub_3`：NPU_g01_output_code.py:246；输入 `i64[2, 200000][200000, 1]npu:0`；size_hints=`{'y0': 200000, 'x1': 64}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_index_select_relu_select_sub_3`, `triton_poi_fused_addmm_relu_1`, `triton_poi_fused_index_select_select_sub_0`。
- `aclnnInplaceCopy_BroadcastToAiCore_BroadcastTo`：外部调用；见 wrapper 调用边界。GPU 检查入口：需对照外部 ACLNN/CUDA 调用和 FX 图。
- `triton_poi_fused_index_select_select_sub_0`：NPU_g01_output_code.py:64；输入 `i64[2, 200000][200000, 1]npu:0`；size_hints=`{'y0': 200000, 'x1': 64}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_index_select_relu_select_sub_3`, `triton_poi_fused_index_select_select_sub_0`, `triton_poi_fused_expand_new_zeros_scatter_reduce_select_view_2`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=1；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 7.761574 | 16.760050 | -115.94% |
| GPU | 11.795400 | 5.889982 | 50.07% |

达到 GPU **整网**提升比时，NPU 整步目标为 3.875708 ms；当前相对该目标的差额为 12.884342 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_edgecnn_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_edgecnn_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_edgecnn_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/basic_gnn_edgecnn_Train/GPU_wrapper_calls.txt)。

### 1. Inductor IR 未拆分 ReLU backward 后的长 sum

**已验证的优化入口：NPU Inductor `torch_npu/_inductor/ir.py::num_splits`。** 输出只有 64 个通道、归约长度为 200000 时仍返回 `split=1`；scheduler 随后把 ReLU backward 融进长归约，pointwise 输出也受同一个低并行度 grid 限制。下表为历史四方筛选，后面的整网实测确认了具体改动与收益。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_red_fused_sum_threshold_backward_3` | 3.0 | 8.405490 | 50.15% |

本项 kernel/区间合计 **8.405490 ms，占整步 50.15%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.347957 | 8.405490 | ≤-2315.67% |
| GPU（Eager 下界 / compiled 上界） | 0.918399 | 0.649202 | ≥29.31% |

NPU Eager 对账项：`aclnnReduceSum_ReduceSumOpAiCore_ReduceSum`, `aclnnThresholdBackward_ReluGrad_ReluGrad`。

GPU Eager 对账项：`threshold_backward`, `sum`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_threshold_backward_2` | 3.0 | 0.476144 | 8.08% |
| `triton_red_fused_sum_0` | 6.0 | 0.148996 | 2.53% |
| `triton_red_fused_sum_1` | 6.0 | 0.009949 | 0.17% |
| `threshold_backward` | 2.0 | 0.009635 | 0.16% |
| `sum` | 1.0 | 0.004477 | 0.08% |

**贡献粗估：8.159524 ms/step，相当于当前 NPU Inductor 整步耗时的 48.68%，使整网提升比增加 105.127 个百分点，单独达到该预算后整网提升比约为 -10.81%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_red_fused_sum_threshold_backward_3`：NPU_g01_output_code.py:243；输入 `f32[200000, 64][64, 1]npu:0`；size_hints=`{'x0': 64, 'r1': 200000}`；split_axis=`[0]`。

<!-- BEGIN CICD VALIDATION basic_gnn_edgecnn_Train:relu_grad_reduce -->
<a id="relu-bias-validation"></a>

#### 2026-11-04 实测：只修改 IR 拆分策略，整步净省 8.162 ms

**需要修改的模块已经收敛到 NPU Inductor IR 的 `num_splits`；本次收益不需要修改 Triton_ascend / npuir。**
这类 kernel 是 `threshold_backward` 的 pointwise 输出与后续 `sum(dim=0)` 的融合。
每条边、每个通道的梯度先按 ReLU 条件置零，再沿边数维求和，得到 linear 层的 bias 梯度：

```python
dZ = torch.where(relu_output <= 0, 0., dY)  # [200000, 64]，还会被后续 mm 使用
dBias = dZ.sum(dim=0, keepdim=True)        # [1, 64]
```

原 kernel `triton_red_fused_sum_threshold_backward_3` 一步调用 3 次，对应三层。
实际回放选中 `X0BLOCK=2, X0BLOCK_SUB=1, R1BLOCK_SUB=8192`、
`compile_mode=simd_simt_template, num_warps=1`，grid 只有 `ceil(64/2)=32` 个 program。
每个 program 逐列循环，并在内部遍历全部 200000 行；`X0BLOCK_SUB=1` 还使一个 tile 只取一列，
不能一起处理行内连续的 64 个通道。原 DSL 的两个关键位置如下，省略中间运算：

```python
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
```

```python
            tl.store(in_out_ptr0 + (x0 + 64*r1), tmp4, r1_mask & x0_mask)
        tmp6 = tl.sum(_tmp6, 0).reshape(1, X0BLOCK_SUB)
        tl.store(out_ptr0 + (x0 ), tmp6, x0_mask)
```

这说明 ReLU backward 的完整输出也在长归约的 program 内写回。仅说“融合不好”不够：
真正缺少的是沿归约维派发独立工作的 IR 计划。安装版 `num_splits` 只有在输出元素数为 1 时考虑拆分，
输出为 64 时直接返回 `(ReductionHint.DEFAULT, 1)`；本次 `config.split_reductions=True`，打开这个开关也解决不了。

实验仅为 NPU sum 增加一个临时拆分规则，并限制输入为 `aten.where.self`，实际命中
`sum_2(where)`、`sum_4(where_2)`、`sum_6(where_4)` 三项。
每项拆成 256 份，每份最多 782 行，最终再对 256 个部分和求和。
生成器已有的 multilayer reduction 与 scheduler 随之生成下面三种 kernel，无需另写模型算子：

| 实际生成 kernel | 次数/step | 合计 ms/step | 含义 |
| --- | ---: | ---: | --- |
| 原 `triton_red_fused_sum_threshold_backward_3` | 3 | 8.435799 | ReLU backward 与完整 200000 行 sum 融合 |
| 新 `triton_poi_fused_threshold_backward_3` | 3 | 0.165885 | 沿边维并行写出 dZ |
| 新 `triton_red_fused_sum_4` | 3 | 0.068864 | `y0=256, x1=64, r2=782`，分段归约；连续通道一起处理 |
| 新 `triton_per_fused_sum_5` | 3 | 0.009190 | 对 256 个部分和做最终归约 |

新链路合计 **0.243939 ms/step**，同语义区间减少 **8.191859 ms（97.11%）**；
其余 kernel 合计增加约 0.029473 ms，因此整网净省为 **8.162386 ms**。
另外三次 gather/mul 长归约仍走原实现，未计作本次修复收益。

| 整网 A/B，FP32/static、batch=1、关闭 aclgraph | Eager ms/step | Inductor ms/step |
| --- | ---: | ---: |
| 原安装版 | 8.048755 | 17.210226 |
| 只拆分 ReLU backward 后的 sum | 8.030439 | 9.047840 |

**设备整步耗时下降 47.43%。** 固定基线 Eager=8.048755 ms 归因，编译提升比增加 **101.41 个百分点**；
修复后仍为负提升比，尚未达到历史 GPU 整网 50.07% 的提升比，剩余瓶颈没有一并解决。
原文 8.159524 ms 是历史四方筛选预算，本次 8.162386 ms 是不同日期基线上的整网 A/B 实测，不能相加。

同语义小图还验证了“只调 tile”和“修改归约计划”的区别：原 DSL 改为较宽 tile 仍是毫秒级，
两阶段 DSL 约 0.082 ms/次。可执行目标应采用上表**由 Inductor 自己生成的三 kernel 链路**，
不能直接把手写 DSL 时间乘调用次数当作整网保证。

正式实现应在 `ir.py::num_splits` 增加小输出、长 outer reduction 的成本判断，
同时考虑 program 数、连续通道访存、中间 buffer、第二阶段开销与已有用户。
本次 `256` / `782` 是验证配置，不建议硬编码这些 shape 或只识别 EdgeCNN；
`NPUTritonScheduling.can_fuse`、`NPUIndexTritonKernel.reduction` 与 runtime tiling 是配合模块，
本次没有修改它们就取得收益，因此无需先把问题归给下层编译器。

验证环境为 NPU_A5 物理卡 3、`pta_213_cicd`，torch `2.13.0+cpu`、torch_npu `2.13.0.dev20261030`、
Triton `3.6.0`；七个关键 NPU 安装文件 SHA256 与 `test/cicd_benchmark@b3bd74035189` 一致。
改动只在进程内生效，安装文件校验未变化；性能按 10 个稳定 Step Id 的设备 kernel duration 求和，
不使用 runner 的主机 speedup。原始组和修改组训练 2 步均为 **`pass_accuracy`**，未放宽原阈值。

证据：[整网逐步统计与 kernel 对账][gnn-summary]、[原 backward DSL / FX / IR][gnn-before]、
[修改后 backward DSL / FX / IR][gnn-after]、[实验入口及规则][gnn-run]、[修改组 accuracy][gnn-accuracy]。
完整生成代码对个人路径和构建 hash 做去敏，保留 DSL 排版。

[gnn-summary]: ../../../../../artifacts/cicd/performance/gnn-validation/results/summary.json
[gnn-before]: ../../../../../artifacts/cicd/performance/gnn-validation/results/basic_gnn_edgecnn_training_baseline/graphs/basic_gnn_edgecnn__0_backward_3.1/output_code.py
[gnn-after]: ../../../../../artifacts/cicd/performance/gnn-validation/results/basic_gnn_edgecnn_training_split/graphs/basic_gnn_edgecnn__0_backward_3.1/output_code.py
[gnn-run]: ../../../../../scripts/repro/cicd/performance/gnn_validation/README.md
[gnn-accuracy]: ../../../../../artifacts/cicd/performance/gnn-validation/results/basic_gnn_edgecnn_training_split_accuracy/accuracy.csv
<!-- END CICD VALIDATION basic_gnn_edgecnn_Train:relu_grad_reduce -->

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_red_fused_add_div_expand_gather_mul_sum_6` | 3.408717 / 2.0 | 20.34% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_red_fused_div_expand_gather_mul_sum_vie_2` | 1.070179 / 1.0 | 6.39% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_add_index_add_neg_slice_5` | 0.394489 / 2.0 | 2.35% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_red_fused_add_div_expand_gather_mul_sum_6`：NPU_g01_output_code.py:481；输入 `b8[200000, 64][64, 1]npu:0`；size_hints=`{'x0': 64, 'r1': 200000}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_threshold_backward_2`, `triton_poi_fused_add_index_add_neg_new_zeros_slice_4`, `triton_poi_fused__to_copy_expand_scatter_add_view_1`。
- `triton_red_fused_div_expand_gather_mul_sum_vie_2`：NPU_g01_output_code.py:171；输入 `b8[200000, 64][64, 1]npu:0`；size_hints=`{'x0': 64, 'r1': 200000}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused__to_copy_expand_scatter_add_view_1`, `triton_red_fused_sum_0`, `triton_poi_fused_div_expand_gather_mul_view_2`。
- `triton_poi_fused_add_index_add_neg_slice_5`：NPU_g01_output_code.py:393；输入 `i64[200000][1]npu:0`；size_hints=`{'y0': 200000, 'x1': 64}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_add_index_add_neg_new_zeros_slice_4`, `triton_poi_fused_index_add_new_zeros_slice_3`, `triton_poi_fused_add_copy__1`。

除上述已实测 ReLU/bias sum 外，其他候选待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
