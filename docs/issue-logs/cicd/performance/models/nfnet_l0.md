---
title: "CICD 性能分析：nfnet_l0"
---

# nfnet_l0：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；初筛在本地复算；2026-11-05 补充 NPU_A5 单 kernel 实现选择实测，整网尚未完成。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

## Infer

Batch=128；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 58.527466 | 46.892745 | 19.88% |
| GPU | 221.127549 | 157.224468 | 28.90% |

达到 GPU **整网**提升比时，NPU 整步目标为 41.613764 ms；当前相对该目标的差额为 5.278981 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/nfnet_l0_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/nfnet_l0_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/nfnet_l0_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/nfnet_l0_Infer/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_poi_fused_silu_30` | 2.222617 / 18.0 | 4.74% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_add_mul_sigmoid_41` | 1.713988 / 2.0 | 3.66% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_add_mul_sigmoid_silu_16` | 1.425677 / 1.0 | 3.04% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_poi_fused_silu_30`：NPU_g01_output_code.py:2350；输入 `f32[128, 384, 18, 18][124416, 324, 18, 1]npu:0`；size_hints=`{'t0': 128, 'z1': 384, 'y2': 18, 'x3': 18}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused__native_batch_norm_legit_add_convolution_mean_mul_relu_sigmoid_silu_view_18`, `triton_poi_fused__native_batch_norm_legit_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_view_26`, `triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_view_10`。
- `triton_poi_fused_add_mul_sigmoid_41`：NPU_g01_output_code.py:3211；输入 `f32[128, 1536, 18, 18][497664, 324, 18, 1]npu:0`；size_hints=`{'y1': 128, 'y0': 1536, 'x2': 324}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused__native_batch_norm_legit_add_convolution_mean_mul_relu_sigmoid_silu_view_18`, `triton_poi_fused__native_batch_norm_legit_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_view_26`, `triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_view_10`。
- `triton_poi_fused_add_mul_sigmoid_silu_16`：NPU_g01_output_code.py:1292；输入 `f32[128, 256, 72, 72][1327104, 5184, 72, 1]npu:0`；size_hints=`{'y0': 32768, 'x1': 5184}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused__native_batch_norm_legit_add_convolution_mean_mul_relu_sigmoid_silu_view_18`, `triton_poi_fused__native_batch_norm_legit_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_view_26`, `triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_view_10`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=128；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 140.453107 | 131.778025 | 6.18% |
| GPU | 349.804233 | 272.279674 | 22.16% |

达到 GPU **整网**提升比时，NPU 整步目标为 109.325510 ms；当前相对该目标的差额为 22.452515 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/nfnet_l0_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/nfnet_l0_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/nfnet_l0_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/nfnet_l0_Train/GPU_wrapper_calls.txt)。

### 1. Inductor avg_pool2d lowering 的实现选择与 template 配置

**已验证的改动入口：Inductor `lowering.lowerings[aten.avg_pool2d.default]`。** 该 shape 的生成实现明显慢于 ACLNN；临时换成现有 fallback_handler 后，编译小图达到接近 Eager 的性能。同 DSL 改 tile/mode 也有收益，但仍慢于 ACLNN。整网验证尚未完成，详见下方。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_avg_pool2d_16` | 1.0 | 0.485396 | 0.37% |
| `triton_poi_fused_avg_pool2d_26` | 1.0 | 0.368915 | 0.28% |
| `triton_poi_fused_avg_pool2d_42` | 1.0 | 18.855676 | 14.31% |

本项 kernel/区间合计 **19.709987 ms，占整步 14.96%**。

| 局部口径 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU（Eager 上界 / 选中 kernel 成本下界） | 0.752289 | 19.709987 | ≤-2520.00% |
| GPU（Eager 下界 / compiled 上界） | 1.635334 | 1.631657 | ≥0.22% |

NPU Eager 对账项：`aclnnAvgPool2d_AvgPoolV2AiCore_AvgPoolV2`。

GPU Eager 对账项：`avg_pool2d`。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_poi_fused_avg_pool2d_17` | 1.0 | 0.838084 | 0.31% |
| `triton_poi_fused_avg_pool2d_29` | 1.0 | 0.441634 | 0.16% |
| `triton_poi_fused_avg_pool2d_46` | 1.0 | 0.351939 | 0.13% |

**贡献粗估：18.959389 ms/step，相当于当前 NPU Inductor 整步耗时的 14.39%，使整网提升比增加 13.499 个百分点，单独达到该预算后整网提升比约为 19.68%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `triton_poi_fused_avg_pool2d_16`：NPU_g02_output_code.py:1290；输入 `f32[128, 256, 56, 56][802816, 3136, 56, 1]npu:0`；size_hints=`{'y0': 917504, 'x1': 28}`；split_axis=`[0]`。
- `triton_poi_fused_avg_pool2d_26`：NPU_g02_output_code.py:2063；输入 `f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0`；size_hints=`{'y0': 917504, 'x1': 14}`；split_axis=`[0]`。
- `triton_poi_fused_avg_pool2d_42`：NPU_g02_output_code.py:3330；输入 `f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0`；size_hints=`{'y0': 1376256, 'x1': 7}`；split_axis=`[0]`。

<!-- BEGIN CICD VALIDATION nfnet_l0_Train:pool -->
<a id="pool-selection-validation"></a>

#### 2026-11-05 局部实测：`_avg_pool2d_42` 12.535 → compiled fallback 0.127 ms/次

本次只验证旧 profile 中最慢的 `triton_poi_fused_avg_pool2d_42`。
输入 FP32 `[128,1536,14,14]`、stride `[301056,196,14,1]`，
2×2、stride=2、无 padding，输出 `[128,1536,7,7]`。
`y0=1376256` 合并 N/C/输出行，`x1=7` 是输出列。
从原 wrapper 提取完整 kernel 后直接回放，原 DSL 的核心如下，保留排版：

```python
        tmp0 = tl.load(in_ptr0 + (2*x1 + 28*y0), y0_mask)
        tmp1 = tl.load(in_ptr0 + (1 + 2*x1 + 28*y0), y0_mask)
        tmp3 = tl.load(in_ptr0 + (14 + 2*x1 + 28*y0), y0_mask)
        tmp5 = tl.load(in_ptr0 + (15 + 2*x1 + 28*y0), y0_mask)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tl.full([1, 1], 4.0, tl.float32)
        tmp8 = (tmp6 / tmp7)
        tl.store(out_ptr0 + (x1 + 7*y0), tmp8, y0_mask)
```

原 autotune winner 为 `Y0BLOCK=24576, Y0BLOCK_SUB=1024, X1BLOCK_SUB=7`，
`compile_mode=simd_simt_template, multibuffer=True, num_warps=1, num_stages=2`，grid=56。
同输入通过当前 lowering 重新 codegen 仍约 12.5 ms，说明直接提取回放能够重现异常量级。

| 同语义输出，物理卡 1，同一批次 | ms/次 |
| --- | ---: |
| 原 generated kernel + 原 winner | 12.534630 |
| 当前 lowering 重新生成 | 12.508224 |
| 同 DSL，template，SUB=32 | 7.510695 |
| 同 DSL，template，SUB=128 | 7.386366 |
| 同 DSL，SIMT-only，SUB=32 | 0.491514 |
| 同 DSL，SIMT-only，SUB=128 | 0.303762 |
| 同语义 ACLNN Eager | 0.126680 |
| **仅替换 lowering 的 compiled ACLNN fallback** | **0.127196** |

固定 DSL 对照均使用 grid=56、Y0BLOCK=24576、X1BLOCK_SUB=7、num_stages=2、multibuffer=False；
template 的 num_warps=1，SIMT-only 为 32。这是编译模式及对应 launch 配置的对照，
不能把收益全部归于 SUB，也不能据此区分 Triton_ascend 与 npuir 内部的根因。
SIMT-only 比原 winner 快约 41 倍，**仍比 ACLNN 慢约 2.40 倍**。
因此只扩大 SIMT 候选不足以达到当前可执行目标，应先解决 Inductor 的实现选择。

最小改动是进程内替换一个 lowering 注册，关闭 FX graph cache 后编译相同小图：

```python
op = torch.ops.aten.avg_pool2d.default
lowering.lowerings[op] = lowering.fallback_handler(op, add_to_fallback_set=False)
```

新 wrapper 中的实际调用是：

```python
            buf0 = torch.ops.aten.avg_pool2d.default(arg0_1, [2, 2], [2, 2])
```

这说明收益可以由现有 CANN 实现取得；本次没有修改 Triton_ascend/npuir。
**第一入口是 Inductor avg_pool2d lowering 的生成实现/外部实现选择**，
正式方案需按布局、shape、dtype、参数和融合边界建立能力或成本规则，不能无条件把所有 pooling 回退。
**第二入口是 NPU kernel type 与 runtime 候选生成**：原 metadata 为 `simt_template`，
实际映射 `simd_simt_template`；它没有同时搜索 SIMT-only。
现有 `SIMD_SIMT_MIX` 路径才会同时加入两个类型，不能把名称里的 SIMT 当作已搜索 `simt_only`。
template 内为什么仍慢，需要继续比较后端 IR/指令；当前不能具体归责到 Triton_ascend 或 npuir。

本批局部可执行目标为 **约 0.127 ms/次**，相对原 kernel 减少 **12.407434 ms（98.99%）**。
旧 profile 中该 kernel **每步一次**；在相同调用布局和成本成立时，才可把单次节省量近似用于整步估算。
原文 **18.855676 ms** 是历史整网中的 `_42`，**19.709987 ms** 还含另外两个 shape 的 pooling；
均与本批小图基线不同。不能把 12.407434 ms 与旧预算 18.959389 ms 相加，
也不能把本次局部结果写成整网已省 12.4 ms 或 18.9 ms。

完整 NFNet Train 基线编译约 18 分钟后主动停止，尚未进入正式 performance 采样，
没有完成整网 A/B 或整网 accuracy；退出码 143 对应主动 SIGTERM，状态单独归档。
本轮八个局部方案均通过 CPU FP64 参考及 NaN 填充后的 graph replay 输出检查，
最大绝对误差约 2.98e-7，`rtol=1e-4, atol=2e-3`。
计时为 NPUGraph 捕获 16 次调用、9 次 replay 的设备 Event 中位数，包含正常输出写回；
该口径与整网 profiler duration 之和不同。
fallback 与同批 Eager 基本持平（约 -0.41% 编译提升比），
还不能宣称超过历史 GPU 操作族约 0.22% 的编译收益；GPU 本轮没有重测。

环境为 `pta_213_cicd`、物理卡 1、NPU950，torch 2.13.0+cpu、
torch_npu 2.13.0.dev20261030、Triton 3.6.0；未使用物理卡 4、5、6。
安装版相关七个源码文件与 `test/cicd_benchmark@b3bd74035189` 一致；
runner checkout 为 `master@52b8a09a5fdb`，实验没有修改安装文件。

[原 kernel 完整 DSL](../../../../../artifacts/cicd/performance/gnn-validation/original/pool.py)、
[同批八组测量与最终配置](../../../../../artifacts/cicd/performance/small-change-validation-20261105/micro_pool_fallback_verified/result.json)、
[compiled fallback wrapper](../../../../../artifacts/cicd/performance/small-change-validation-20261105/micro_pool_fallback_verified/graphs/model__1_inference_0.0/output_code.py)、
[整网中止状态](../../../../../artifacts/cicd/performance/small-change-validation-20261105/nfnet_l0_training_baseline/status.json)、
[源码身份](../../../../../artifacts/cicd/performance/small-change-validation-20261105/source_identity.json)、
[复现入口](../../../../../scripts/repro/cicd/performance/gnn_validation/README.md)。
<!-- END CICD VALIDATION nfnet_l0_Train:pool -->

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `aclnnAvgPool2dBackward_AvgPoolV2Grad_AvgPoolV2Grad` | 5.148650 / 3.0 | 3.91% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_per_fused_add_mul_sigmoid_sigmoid_backw_27` | 1.479441 / 1.0 | 1.12% | NPU Inductor reduction codegen：核对本 kernel 的轴、stride、输出并行度与多阶段归约 |
| `triton_poi_fused_silu_31` | 1.245492 / 18.0 | 0.95% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `aclnnAvgPool2dBackward_AvgPoolV2Grad_AvgPoolV2Grad`：外部调用；见 wrapper 调用边界。GPU 检查入口：需对照外部 ACLNN/CUDA 调用和 FX 图。
- `triton_per_fused_add_mul_sigmoid_sigmoid_backw_27`：NPU_g01_output_code.py:3062；输入 `f32[128, 256, 56, 56][802816, 3136, 56, 1]npu:0`；size_hints=`{'y0': 128, 'x1': 256, 'r2': 3136}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused__native_batch_norm_legit_add_convolution_mul_sigmoid_silu_view_16`, `triton_per_fused__native_batch_norm_legit_convolution_mean_mul_view_36`, `triton_red_fused_add_avg_pool2d_backward_mul_sigmoid_sigmoid_backward_silu_backward_sum_29`。
- `triton_poi_fused_silu_31`：NPU_g02_output_code.py:2450；输入 `f32[128, 384, 14, 14][75264, 196, 14, 1]npu:0`；size_hints=`{'t0': 128, 'z1': 384, 'y2': 14, 'x3': 14}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_poi_fused__native_batch_norm_legit_add_convolution_mul_sigmoid_silu_view_16`, `triton_poi_fused__native_batch_norm_legit_convolution_mul_silu_view_5`, `triton_poi_fused_add_mul_sigmoid_silu_28`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
