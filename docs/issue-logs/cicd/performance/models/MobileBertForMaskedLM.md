---
title: "CICD 性能分析：MobileBertForMaskedLM"
---

# MobileBertForMaskedLM：Infer / Train 编译收益差距

按各设备自身的 Eager→Inductor 提升比筛选。四份 profile 和本批 wrapper/FX 为用户提供的数据；初筛在本地读取和复算，Train 参数梯度归约另有下文列明的 NPU_A5 单 kernel 实测。

局部预算、上界/下界的定义见[全量分析与共性问题](../full-survey-20261102.md)。以下“百分点”均指 NPU **整网提升比**的变化。操作族汇总用于保守筛选，实际要优化的 kernel 逐个列出；未对齐的候选单列，不将其绝对耗时当作收益。

2026-11-03 更新：Train 六组归约已在物理卡 1、2、`pta_213_cicd`、指定分支对应安装版本复测；逐 kernel 证据、责任模块和整网外推统一见下方 Train 第一项。

## Infer

Batch=128；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 71.405178 | 65.363471 | 8.46% |
| GPU | 84.108132 | 54.636697 | 35.04% |

达到 GPU **整网**提升比时，NPU 整步目标为 46.384850 ms；当前相对该目标的差额为 18.978621 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/MobileBertForMaskedLM_Infer/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/MobileBertForMaskedLM_Infer/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/MobileBertForMaskedLM_Infer/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/MobileBertForMaskedLM_Infer/GPU_wrapper_calls.txt)。

本轮保守筛选没有确认局部提升比缺口。下面仍列出具体检查点；“没有确认”不等于没有优化空间。

### 额外具体检查点

`triton_red_fused__log_softmax_12` 的 4.669 ms 是具体候选，但 GPU 的 prepare_softmax_online 已耗时约 6.105 ms，不能看到 NPU 比自身 Eager 慢就认定 GPU 有更高收益；该局部保守筛选未通过，暂不承诺优化预算。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_red_fused__log_softmax_12` | 4.668885 / 1.0 | 7.14% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_poi_fused_addmm_6` | 2.120228 / 96.0 | 3.24% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |
| `triton_poi_fused_add_addmm_mul_view_7` | 0.933994 / 96.0 | 1.43% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_red_fused__log_softmax_12`：NPU_g01_output_code.py:877；输入 `f32[16384, 30522][30522, 1]npu:0`；size_hints=`{'x0': 16384, 'r1': 30522}`；split_axis=`[0]`。GPU 检查入口：`triton_red_fused__log_softmax__unsafe_view_add_nll_loss_forward_view_14`, `triton_per_fused__log_softmax__unsafe_view_add_nll_loss_forward_view_15`。
- `triton_poi_fused_addmm_6`：NPU_g01_output_code.py:476；输入 `f32[512][1]npu:0`；size_hints=`{'y0': 16384, 'x1': 512}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused_addmm_relu_view_8`, `triton_poi_fused_add_addmm_mul_view_9`, `triton_poi_fused_add_addmm_mul_view_10`。
- `triton_poi_fused_add_addmm_mul_view_7`：NPU_g01_output_code.py:542；输入 `f32[128][1]npu:0`；size_hints=`{'y0': 16384, 'x1': 128}`；split_axis=`[0]`。GPU 检查入口：`triton_red_fused__unsafe_view_add_prepare_softmax_online_13`, `triton_poi_fused_addmm_relu_view_8`, `triton_per_fused__softmax_abs_all_amax_div_eq_mul_ne_sub_view_5`。

本场景待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。

## Train

Batch=128；四份数据均按 10 个采样 step 归一化为 ms/step。

| 设备 | Eager ms | Inductor ms | 提升比 |
| --- | ---: | ---: | ---: |
| NPU | 221.888467 | 295.193592 | -33.04% |
| GPU | 268.925023 | 215.610848 | 19.82% |

达到 GPU **整网**提升比时，NPU 整步目标为 177.899253 ms；当前相对该目标的差额为 117.294340 ms。该整步差额只作背景，不分摊给单个 kernel。

[NPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/MobileBertForMaskedLM_Train/NPU_kernels.txt)、[GPU 完整 DSL](../../../../../artifacts/cicd/performance/full-survey-20261102/MobileBertForMaskedLM_Train/GPU_kernels.txt)、[NPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/MobileBertForMaskedLM_Train/NPU_wrapper_calls.txt)、[GPU 调用边界](../../../../../artifacts/cicd/performance/full-survey-20261102/MobileBertForMaskedLM_Train/GPU_wrapper_calls.txt)。

<!-- BEGIN MOBILEBERT REDUCTION ANALYSIS -->
<a id="outer-reduction"></a>

### 1. outer reduction：IR 分段计划缺失与 tiling 候选漏搜

**责任模块：NPU Inductor IR `ir.py::num_splits`，以及 runtime `tile_generator.py::TileGenerator`。**
`_11`、`_9`、`_7`、`_13` 都是 `R=16384、X=128` 的 outer reduction：IR 未生成分段计划，
runtime 候选又只覆盖很窄的 X tile。相同后端改用更宽 tile 或显式分段，实测均显著变快。
四组自动分段的整网外推约 **50.59 ms/step**；尚未修改框架或做整网替换验证。

**这里的 kernel 类型是 outer reduction，可融合 Add/Mul 等 pointwise。**
把 contiguous 输入展平为 `[R,X]`，`R` 是归约维，`X` 是连续且保留到输出的维度。
这类 kernel 沿外侧的 R 求和，输出 X 个结果；“参数梯度”只是该计算在模型中的用途。
例如 `_11` 的数学表达式如下，代码用于解释语义，并非原始 DSL：

```python
# a, b, h: [R=16384, X=128]
g = a + b
out0 = g.sum(dim=0)        # [X=128]
out1 = (g * h).sum(dim=0)  # [X=128]
```

七组都属于 outer reduction，但问题不能统一归因：

| kernel 后缀 | R / X | 已确认的问题或分析边界 |
| --- | --- | --- |
| `_11`、`_9`、`_7`、`_13` | 16384 / 128 | IR 缺少分段计划；runtime 遗漏更快 tile |
| `_9` 中的 24 次调用 | 16384 / 128 | 与 `_7` 重复求同一 Sum；IR 规范化 / scheduler / codegen 中的具体阻断点尚未定界 |
| `_20` | 16384 / 512 | IR 分段方案有实测收益；当前 tile 搜索没有显示相同漏搜问题，GPU 单项收益差距仍待确认 |
| `_0` | 16384 / 30522 | 当前实验未确认需要修复的模块；分段和所试 tile 均未胜过原实现 |
| `_5` | 16384 / 512 | 按要求不分析 |

本轮在 NPU_A5 Docker、`pta_213_cicd` 运行；诊断轮使用物理卡 **2**，同编译路径复核轮使用空闲物理卡 **1**，
未使用卡 4、5、6。安装的 IR、codegen、runtime、tile generator、config 五个关键文件确认与
`test/cicd_benchmark@b3bd7403518924a7cc1e7c470aa80ccfb4e38120` 一致，见[环境核验][env]。
torch `2.13.0+cpu`、torch_npu `2.13.0.dev20261030`、Triton `3.6.0`，
目标 `Ascend950PR_9579`。原整网 CSV 未提供完整构建身份，单 kernel 复测与历史整网数据分别列示。

#### 67.462484 ms 如何拆开

下表“原成本”为原整网 CSV 的设备 duration 汇总，已经乘上一步内的全部调用次数。
实测列来自独立重放，不能拿其中一个新值直接减整网历史成本。

| NPU kernel（省略 `triton_red_fused_`） | 次/step | 原成本 / ms | 本轮原始实现 / μs | 同路径仅换 tile / μs | 两阶段 ATen 自动生成 / μs | 当前结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `add_mul_sum_view_11` | 96 | 25.847447 | 247.761 | 87.502 | 22.487 | 已确认，优先优化 |
| `mul_sum_view_9` | 168 | 15.300343 | 93.397 | 29.860 | 7.343 | 已确认，优先优化 |
| `mul_sum_view_7` | 48 | 9.252472 | 173.668 | 61.922 | 20.005 | 已确认，优先优化 |
| `add_mul_sum_view_13` | 24 | 8.921167 | 323.062 | 134.058 | 25.957 | 已确认，优先优化 |
| `add_mul_sum_view_20` | 24 | 5.468952 | 184.514 | 183.639 | 97.704 | 有可达性实测，GPU 收益缺口待确认 |
| `add_sum_view_0` | 1 | 2.585491 | 2547.143 | 3540.150 | 2733.418 | 已试方案均未胜过原实现 |
| `mul_sum_view_5` | 1 | 0.086612 | — | — | — | 用户指定排除 |

四组优先点原成本合计 **59.321429 ms，占七组成本 87.93%**。
这不表示 59.321429 ms 可以全部回收；后面的目标仍保留归约计算、数据读取和最终写回成本。
七组 67.462484 ms 与原粗筛 34.279800 ms 预算分别是“当前成本”和“保守筛选预算”，都不是已实测收益。

完整数值、真实 winner 和候选见[同路径复核][verified]、[诊断轮][diagnostic]。

#### 两个已定位模块分别有什么问题

**模块一：NPU Inductor IR，`torch_npu/_inductor/ir.py::num_splits`。**
`patch_num_splits()` 将它安装到社区 `Reduction.num_splits`。
源码在 `numel_hint == 1` 时才进入分段判断；保留输出元素数大于 1 时，直接返回
`ReductionHint.DEFAULT, 1`。本次四组的 `numel_hint=128`，因此无论 R 长达 16384，
IR 都只生成一阶段归约。仅开启 `config.split_reductions` 也无法改变这个分支。
源码及补丁接入位置见[源码摘录][source]。

生成结果是 `split_axis=[0]`，program 只按 X 分派，每个 program 内循环处理完整 R。
下面是 `_9` 的原始 DSL 连续摘录，保留原始格式：

```python
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        _tmp2 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        for loop_r1 in range(loops_r1):
            r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[:,None]
            r1_mask = r1 < r1_numel
            tmp0 = tl.load(in_ptr0 + (x0 + 128*r1), r1_mask & x0_mask, other=0.0)
            tmp1 = tl.reshape(tmp0, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp3 = _tmp2 + tmp1
            _tmp2 = tl.where(tl.reshape((r1_mask & x0_mask), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp3, _tmp2)
        tmp2 = tl.sum(_tmp2, 0).reshape(1, X0BLOCK_SUB)
        tl.store(out_ptr0 + (x0 ), tmp2, x0_mask)
```

`tl.program_id(0)` 只影响 `x0`；`r1` 完全由 program 内的 `loop_r1` 遍历。
最后 `tl.sum(_tmp2, 0)` 才生成完整结果。调整 `R1BLOCK_SUB` 只改变循环内 tile 大小，
不会把 R 分给多个 program。
需要 IR 提供的另一种计划是 **partial/final**：先让不同 program 处理不同 R 段并写部分和，
再用 final reduction 合并部分和。显式两阶段 ATen 小图已经让同一后端生成了这种更快实现。
因此这里定位的是 **IR 归约计划缺失**；上述 DSL 的循环结构是该计划的执行结果。

**模块二：NPU Inductor runtime，`torch_npu/_inductor/runtime/tile_generator.py::TileGenerator`。**
入口为 `runtime.triton_heuristics.triton_config_npu_index`，调用
`TileGenerator.descend_split_tiling()`；具体候选策略在
`descend_split_axis_inner()` / `descend_tiling_axis()`。
本次小 X、长 R 场景的 `tiny_kernel=False`，`program_threshold=56/2=28`。
`descend_split_axis_inner()` 的初始候选接纳条件包含
`initial_total_programs > self.program_threshold or self.dual_reduction`；
常规下降过程中，加入 `candidate_blocks` 的条件同样包含
`total_programs > self.program_threshold or self.dual_reduction`，见[源码 350–435 行摘录][source]。
本例 `dual_reduction=False`，X0BLOCK=16、32 分别只有 8、4 个 program，未通过该常规门槛。
结合后续候选展开与资源过滤，四组最终各 24 个候选的 X0BLOCK 只有 3、4；
更快的 `16/16/256`、`32/32/128` 没有进入最终搜索列表。

本文配置简写依次为 **`X0BLOCK / X0BLOCK_SUB / R1BLOCK_SUB`**：
第一项决定一个 program 覆盖多少 X，第二项决定循环内实际连续处理多少 X，第三项为循环内的 R tile。
例如 `3/1/4096` 表示一个 program 最多串行处理 3 个 X，每次只处理 1 个连续 X。
地址 `x0 + 128*r1` 下，单通道 tile 沿 R 的相邻值相隔 128 个 FP32 元素。
改成较宽 X tile 可以同时访问多个连续元素；即使 program 数减少，实测仍明显更快。
已知更快候选没有进入搜索，因此首先是 **候选覆盖不足**，还不能归为 autotune 在已有候选中计时选错。

两个模块处在不同层次：IR 决定是否拆成多个阶段，runtime 决定某个阶段如何分 tile。
同一批 kernel 的收益存在重叠，不能把“换 tile”和“分段”的完整收益相加。

#### `_11`：IR 未拆分双输出归约；runtime 漏掉宽 X tile

**模块一：IR `num_splits`。** 两个输出各有 128 个元素，落入 `DEFAULT,1` 返回分支。
生成 DSL 在同一个 `loop_r1` 中维护两个 accumulator，分别累加 `a+b` 和 `(a+b)*h`，
完整 R 都由同一 program 处理，缺少跨 program 的 partial/final 计划。

**模块二：runtime `TileGenerator`。** 原 winner 为 `3/1/4096`，grid 为 `ceil(128/3)=43`；
一个 program 最多依次处理 3 个 X，每个 X 再做 4 轮 R 循环。
24 个候选只覆盖 X0BLOCK 3、4，遗漏更快的宽 X tile。
原 TTAdapter 也保留了 `memref<4096x1xf32, strided<[128,1]>>` 和循环，见[固定配置 IR][ir11]。

**kernel 边界与证据。** 完整名称为 `triton_red_fused_add_mul_sum_view_11`，
三个输入均对应 contiguous FP32 `[128,128,128]`，展平后是 `[16384,128]`。
其数学表达式就是本节开头的两路 Sum；两个输出的实际 shape 为 `[1,1,128]`。
原 wrapper 有 96 处 launch，对应 24 层、每层 4 次 NoNorm 参数梯度计算；
相邻 `triton_poi_fused_add_mul_view_12` 另算 `dh=g*gamma`，不在本 kernel 成本内。
见[原始 DSL][original]、[wrapper 调用边界][calls]。

**模块问题的实测验证。** 保持原 signature、metadata、launcher 路径，仅换 `32/32/128`：
**247.761 → 87.502 μs，快 2.83 倍**；显式分段后自动 codegen 为
**22.487 μs，快 11.02 倍**，实际为 **1 partial + 2 final，共 3 次 launch**。
工程目标先用 **≤25 μs**；按每步 96 次外推，约节省 **21.626 ms/step**。
见[自动 codegen][staged11]及[同路径复核][verified]。

首轮诊断的原始实现、固定 `16/16/256`、自动分段、手写两阶段分别为
246.390、103.624、22.544、13.211 μs，见[首轮结果][result11]。
手写版合并了两路 final，仅 2 次 launch，partial/final 为 11.544 / 1.667 μs，作为后续调优参考。
本文整网外推统一采用复核轮差值。实际后端 `parallel_mode=simd`，
不能根据 `npu_kernel_type=simt_template` 的名称推定在跑纯 SIMT。

#### `_9`：IR 未拆分单输出归约；runtime 漏候选；另有重复 Sum

**模块一：IR `num_splits`。** 单输出仍保留 X=128，因而不分段。
前述原始 DSL 只有一个 accumulator，按 `x0+128*r1` 读取整个 R，再 `tl.sum(...,0)`。
完整名称虽为 `triton_red_fused_mul_sum_view_9`，实际 DSL 已没有 Mul，
只计算 contiguous FP32 `[16384,128]` 的 `sum(dim=0)`，输出 `[1,128]`，每步 168 次。

**模块二：runtime `TileGenerator`。** 原 winner `3/1/8192`，grid 43；
实际连续 X tile 仅 1，24 个候选的 X0BLOCK 只有 3、4。
同路径补入 `16/16/256`，**93.397 → 29.860 μs，快 3.13 倍**；
显式分段自动生成 1 partial + 1 final，合计 **7.343 μs，快 12.72 倍**。
工程目标 **≤8 μs**；168 次外推约节省 **14.457 ms/step**。
见[原始 DSL][dsl9]、[自动 codegen][stage9]。

**另一个优化点：等价归约复用，具体阻断模块尚未定界。**
原 wrapper 第一层第 2176 行 `_9.run(buf64,buf67,...)` 与第 2299 行
`_7.run(buf64,addmm_346,buf98,buf99,...)` 的第一路都计算同一个 `buf64.sum`。
输入在两次调用之间没有改写，输出分别为 `[1,128]` 和 `[1,1,128]`。
这种关系在 24 层重复。GPU `_12` 的 DSL 将同一个 partial `tmp24` 写给两个输出，复用了归约。
见[FX / wrapper / GPU DSL 映射证据][mapping]。

已确认的是 **NPU 生成了两次等价扫描**；尚未确定是 IR 对 reshape/keepdim 的规范化未识别等价，
还是 scheduler 的融合条件阻止合并，或 codegen 没有复用相同表达式，不能把三者都写成已确认根因。
需要沿这三处继续定界；优化后仍需保留两个梯度各自的存储，不能让不同参数的 `.grad` 共享可变 buffer。

对应 24 次 `_9` 的原 profile 成本为 **2.168733 ms/step，占整网缺口 1.85%**，
这是取消重复扫描的成本上限，还需扣除必要的独立写回。
若先做分段，残余成本只有约 `24×7.343 μs=0.176 ms`；不能再把原 2.169 ms 加到分段收益中。

#### `_7`：IR 未拆分双输出归约；runtime 漏掉宽 X tile

**模块一：IR `num_splits`。** 两路输出保留 X=128，均没有 partial/final 计划。
DSL 在同一 `loop_r1` 内累加输入及输入乘另一张量，并在循环后分别 `tl.sum(...,0)`。
输入 `g,h` 为 contiguous FP32 `[128,128,128]`，数学表达式为：

```python
out0 = g.sum((0, 1), keepdim=True)
out1 = (g * h).sum((0, 1), keepdim=True)
```

**模块二：runtime `TileGenerator`。** 原 winner `3/1/4096`，24 个候选的 X0BLOCK 只有 3、4，
遗漏更快的 `32/32/128`，与 `_11` 属于同一种候选覆盖问题。

**kernel 边界与实测。** 完整名称 `triton_red_fused_mul_sum_view_7`，每层 2 次、每步 48 次。
仅换 tile：**173.668 → 61.922 μs，快 2.80 倍**；
显式分段自动 codegen 为 **20.005 μs，快 8.68 倍**，实际 1 partial + 2 final。
工程目标 **≤22 μs**；48 次外推约节省 **7.376 ms/step**。
两路 final 合并的手写诊断版为 10.995 μs，仅作为后续调优参考。
首轮与复核轮的自动结果分别为 19.875、20.005 μs，量级一致。
见[原始 DSL][dsl7]、[自动 codegen][stage7]；GPU `_8` 和 `_12` 对应部分均先做 partial 再 final。

#### `_13`：IR 未拆分带 pointwise 重计算的归约；runtime 漏候选

**模块一：IR `num_splits`。** 输出 X=128 未分段，DSL 把全部 pointwise 和两路归约放在完整 R 循环中。
完整名称 `triton_red_fused_add_mul_sum_view_13`，每步 24 次；四个大输入为
contiguous FP32 `[128,128,128]`，另有两个 `[128]` 参数。其数学表达式为：

```python
g = a + b
h = c + (d * scale + bias)
out0 = g.sum((0, 1), keepdim=True)
out1 = (g * h).sum((0, 1), keepdim=True)
```

DSL 中的 `tmp9=tmp7*tmp8`、`tmp11=tmp9+tmp10`、`tmp12=tmp6+tmp11` 对应 `h` 的重计算，
随后 `tmp13=tmp2*tmp12` 进入第二路归约。完整原始格式见[DSL][dsl13]。
问题是这组表达式没有跨 program 分段；保留所有表达式不变的两阶段实测已明显更快。

**模块二：runtime `TileGenerator`。** 原 winner `3/1/4096`，24 个候选只覆盖 X0BLOCK 3、4。
保持同一 DSL，仅补入 `16/16/256`：**323.062 → 134.058 μs，快 2.41 倍**；
显式分段自动 codegen 为 **25.957 μs，快 12.45 倍**，1 partial + 2 final。
24 次外推约节省 **7.131 ms/step**，见[自动 codegen][stage13]。

**尚未确认为问题的模块：AOTAutograd 重计算策略。**
`d*scale+bias` 与残差 Add 是 FX graph 已有的前向表达式重计算，GPU `_12` 也含相同工作。
本轮没有定界具体 AOT 分区规则，不能因为这里多出 Add/Mul 就认定为额外缺陷。
上述分段和换 tile 实验均保留这些计算，收益不依赖删除重计算。

自动分段两轮为 **40.851、25.957 μs**，selected config 相同，运行波动尚未定界。
初期按 **≤41 μs** 验收，稳定测量后再争取 26–30 μs。
手写诊断版为 15.349 μs，仅作后续参考，不能直接采用 `_11` 的目标。

#### `_20`：IR 分段计划有优化空间；未确认 runtime 候选问题

**模块：IR `num_splits`。** 此处保留 X=512，同样直接返回不分段。
DSL 在一个完整 R 循环内读入五个大输入、合并四路输入，再执行两路 Sum；
地址随 R 前进的步长为 512，数学表达式为：

```python
g = ((a + b) + c) + d
out0 = g.sum((0, 1), keepdim=True)
out1 = (g * h).sum((0, 1), keepdim=True)
```

完整名称 `triton_red_fused_add_mul_sum_view_20`，输入 FP32 `[128,128,512]`，每步 24 次。
原实现 **184.514 μs**；显式分段自动 codegen 为 **97.704 μs，快 1.89 倍**，
说明 IR 应有分段与不分段的可比较计划。见[原始 DSL][dsl20]、[自动 codegen][stage20]。

**runtime 边界：没有实测支持相同的候选漏搜归因。**
原 20 个候选的 X0BLOCK 已为 16，winner `16/16/128`，grid 32。
同路径换 `32/32/128` 为 **183.639 μs**，改善不足 1%。
所以这里的已验证空间主要来自分段，而非 `_11` 等四组的宽 X tile 漏搜。

24 次同轮差值外推约 **2.083 ms/step，占整步 0.71%**。
当前还缺 GPU 侧同边界数据，暂不计入前述四组优先收益。
首轮手写版 78.655 μs 仅作参考；首轮原始基线波动较大，不用于收益外推。

#### `_0`：未确认 IR 或 runtime 缺陷；宽输出场景不宜直接套用分段

**模块结论：当前证据不支持把 IR 不分段或 runtime 选定 tile 判为缺陷。**
完整名称 `triton_red_fused_add_sum_view_0`，两个 contiguous FP32 输入 `[128,128,30522]`，
展平后为 `[R=16384,X=30522]`；计算 `(a+b).sum((0,1),keepdim=True)`，每步一次。
逻辑输入读取约 4.001 GB。

**IR 行为与实验。** 当前仍为一阶段归约，但显式分段自动 codegen 为 **2733.418 μs**，
比原实现 **2547.143 μs 慢 7.31%**；三种手写分段约为 2741.898、2664.295、2705.270 μs，均未胜出。
R 虽然同样长，X 已很大，原计划有足够的 X 方向并行度；增加部分和还会引入中间存储与 final 阶段。
因此未来修改 `num_splits` 时，应保留不分段选项，并考虑输出宽度、布局与中间流量。

**runtime 行为与实验。** 84 个候选的 X0BLOCK 为 552、720、960，winner `552/552/32`，
grid 已有 56 个 program，连续 X tile 也是 552。
这里不存在前四组“连续 X 只有 1、候选只覆盖 3/4”的证据。
同配置重放为 2547.690 μs，本轮其他 tile 最好仍需 3540.150 μs。

本轮没有找到胜过原实现的方案，**优化收益计 0**；也未确认应由 Triton_ascend/npuir 修复的问题。
见[原始 DSL][dsl0]、[分段 codegen][stage0]。

#### 实测如何排除编译入口和计时干扰

首轮普通 JIT 诊断中，`_20` 原始基线与同 tile 重放有较大差异；该轮不用于 `_20` 的收益承诺。
复核轮使用 **`NPUCachingAutotuner._precompile_config` → 原 `TritonCompileResultNpu.make_launcher`**，
保留原 signature、metadata、编译选项及 runtime block，只改变指定 tile。
六组原始 kernel 与选中配置重放的 **TTIR SHA256 全部一致**，排除了将不同编译入口的收益误算成 tiling 收益。
复核 `_20` 为 184.514 / 187.383 μs，相近，没有证据指向一个额外的编译入口缺陷。

每项 warmup 5 次，NPUGraph 捕获 32 次，重放 7 轮取 Event 中位；大词表 `_0` 为 4 次捕获，以限制 Eager 临时内存。
每项另 profile 10 次，设备时间收齐全部阶段；多输出自动方案实际是 3 次 launch，单输出为 2 次。
所有 capture 输出先填 NaN，再 replay、校验，避免测到空图；launcher 每次读取当前 stream。
输入固定随机种子，FP32 pointwise 结果后接 CPU FP64 累加参考，检查 `rtol=1e-4, atol=2e-3`。
复核 **42 个 case** 全部通过；逐次 profile duration、Event 样本和误差均保存在[结果][verified]。
共享环境出现过其他任务，未宣称持续独占；原始重放一致性和双口径结果用于发现明显干扰。

同一个 Triton_ascend/npuir 后端已经执行了显著更快的上游计划，因此首先优化 Inductor IR 和候选生成。
本轮没有证明下层已无优化空间，也没有确认固定 DSL/config 的下层缺陷。

可复现脚本为 [run_family.py][run]，命令及计时说明见[README][readme]；
原始和固定配置 TTIR、TTAdapter、生成 C++ 见[复核 IR][ir]，身份和精度汇总见[validation][validation]。

#### 整网预期收益与改动范围

使用复核轮 `单次原耗时 − 单次方案耗时`，乘历史 `调用次数/step`；不直接跨批次相减，也不缩放其他算子。

| 已确认点 | 仅换候选：外推节省 / ms | 自动分段：累计外推节省 / ms | 自动分段占整网缺口 | 建议首阶段验收目标 / μs |
| --- | ---: | ---: | ---: | ---: |
| `_11` | 15.385 | 21.626 | 18.44% | ≤25 |
| `_9` | 10.674 | 14.457 | 12.33% | ≤8 |
| `_7` | 5.364 | 7.376 | 6.29% | ≤22 |
| `_13` | 4.536 | 7.131 | 6.08% | ≤41，进一步争取 26–30 |
| 四组合计 | **35.959** | **50.590** | **43.13%** | — |

“仅换候选”与“自动分段”是同一批 kernel 的两个累计状态，不能相加。
自动分段表列采用本轮较快实测；按右列较保守验收值估计，四组合计约 **49.78 ms/step**。
因此当前合理工程预期为 **约 50 ms/step**，不是全部 67.462 ms，也不是承诺完全收回整网缺口。
四组小图若在整网均兑现复核轮差值，NPU Inductor 约 **295.194 → 244.604 ms**，
整步下降 **17.14%**；提升比约 **−33.04% → −10.24%**，仍未达到 GPU 的 **+19.82%**。
没有做原 wrapper 替换与整网复测，因此缓存、实际输入、autotune 选择和融合变化仍可能改变兑现值。

| 步骤 | 应改的组件和行为 | 改动范围估计 | 收益口径 |
| --- | --- | --- | --- |
| 1 | `TileGenerator.descend_split_axis_inner/descend_tiling_axis`：给小通道长归约保留更宽的连续通道候选，允许少于阈值的 program 数参与比较 | 较小；主要在候选策略及资源过滤，不能硬编码 kernel 名或 shape | 四组累计约 36 ms；`_20` 不宣称有相同漏搜收益 |
| 2 | `ir.py::num_splits`、partial/final IR 创建与布局：对合适的小输出长归约产生两阶段计划，保留 pointwise 与多输出融合 | 中等；涉及中间 buffer、两阶段布局与融合边界，需要同时保留 `_0` 不分段的选择 | 四组累计约 50 ms；相对步骤 1 新增约 14.6 ms |
| 3 | final sum 多输出融合、等价归约规范化和共享 partial | 中等，需要维护各参数梯度独立存储；属于后续完善 | 手写诊断值仅作参考；重复 `_9` 在分段后剩余成本约 0.18 ms，不再加原 2.17 ms |

以上范围是工程判断，不是已完成的框架补丁。本轮止于问题定位和可达性验证。

[env]: ../../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/environment.json
[verified]: ../../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/verified_result.json
[diagnostic]: ../../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/diagnostic_result.json
[mapping]: ../../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/mapping_evidence.txt
[source]: ../../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/source_evidence.txt
[run]: ../../../../../scripts/repro/cicd/performance/mobilebert_reduction_20261103/run_family.py
[readme]: ../../../../../scripts/repro/cicd/performance/mobilebert_reduction_20261103/README.md
[ir]: ../../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/ir_verified
[validation]: ../../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/validation.json
[dsl9]: ../../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/mul_sum_view_9.py
[dsl7]: ../../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/mul_sum_view_7.py
[dsl13]: ../../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/add_mul_sum_view_13.py
[dsl20]: ../../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/add_mul_sum_view_20.py
[dsl0]: ../../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/add_sum_view_0.py
[stage9]: ../../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/staged_9_output_code.py
[stage7]: ../../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/staged_7_output_code.py
[stage13]: ../../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/staged_13_output_code.py
[stage20]: ../../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/staged_20_output_code.py
[stage0]: ../../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/staged_0_output_code.py

[original]: ../../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/original_kernel.py
[calls]: ../../../../../artifacts/cicd/performance/full-survey-20261102/MobileBertForMaskedLM_Train/NPU_wrapper_calls.txt
[staged11]: ../../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/staged_output_code.py
[result11]: ../../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/result.json
[ir11]: ../../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/ir
<!-- END MOBILEBERT REDUCTION ANALYSIS -->

### 2. prims.fma metadata 重复执行第一组 Mul/Add

**优化入口：PyTorch Inductor inductor_prims.make_prim.meta 与 prims._prim_impl；NPU prims.fma fallback 暴露真实执行。** softmax backward 后连续 Mul/Add/Mul/Add 中第一对来自 metadata 执行真实 impl_aten，随后真实实现再次执行。第一对结果不参与最终输出；GPU 生成 kernel 内执行有效公式，没有 metadata 设备调用。

| NPU 具体 kernel / 选定调用 | 调用次数/step | ms/step | 占 NPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `aclnnMul_MulAiCore_Mul` | 24.0 | 0.515588 | 0.17% |
| `aclnnAdd_AddAiCore_Add` | 24.0 | 0.546524 | 0.19% |

本项 kernel/区间合计 **1.062112 ms，占整步 0.36%**。

仅计算这些相邻窗口中的第一对，不使用整网 Add/Mul 汇总。预算为已证实重复调用的当前设备耗时；修复后整网时间仍须实测。另一个有效 Mul/Add 对不是可全额删除的工作。

GPU 对应实现/保守收费项：

| GPU kernel | 调用次数/step | ms/step | 占 GPU Inductor 整步 |
| --- | ---: | ---: | ---: |
| `triton_per_fused__softmax__softmax_backward_data_div_mul_native_dropout_backward_sub_view_14` | 24.0 | 2.156044 | 1.00% |

**贡献粗估：1.062112 ms/step，相当于当前 NPU Inductor 整步耗时的 0.36%，使整网提升比增加 0.479 个百分点，单独达到该预算后整网提升比约为 -32.56%。**

该数是局部目标预算/控制估算；不是修复后的实测收益，多项有操作族或融合边界重叠时不能直接相加。

直接代码证据：

- `aclnnMul_MulAiCore_Mul`：外部调用；见 wrapper 调用边界。
- `aclnnAdd_AddAiCore_Add`：外部调用；见 wrapper 调用边界。

历史机制证据：[首层 attention 定位](../electra-layoutlm-train-first-attention-20261031.md)、[FMA 源码调用链](../../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt)。历史源码验证与当前 profile 版本身份未完整绑定；本批 wrapper 和调用序列复核见本节证据。

### 未计入收益的具体候选

以下按当前非裸 MM kernel 成本列出检查顺序。这些项尚未完成相同语义区间的四方对齐，**不判定为已经确认的劣化，不填写“跟 GPU 一致可回收”的数字**。GPU 所列仅是共享 ATen 的入口，可能覆盖其他形状/融合工作。

| NPU kernel | ms/step / 调用次数 | 整步占比 | 具体检查入口 |
| --- | ---: | ---: | --- |
| `triton_per_fused__softmax_all_amax_div_isfinit_8` | 11.544797 / 23.0 | 3.91% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `triton_red_fused__log_softmax_18` | 4.683339 / 1.0 | 1.59% | NPU Inductor softmax/attention lowering 与 reduction tiling；先统一完整 attention 或 loss 边界 |
| `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad` | 4.229120 / 1.0 | 1.43% | NPU Inductor 本 kernel 的 lowering/布局与 pointwise tiling；需固定 DSL/config 定界下层 |

- `triton_per_fused__softmax_all_amax_div_isfinit_8`：NPU_g02_output_code.py:598；输入 `f32[512, 128, 128][16384, 128, 1]npu:0`；size_hints=`{'z0': 128, 'y1': 4, 'x2': 128, 'r3': 128}`；split_axis=`[0, 1]`。GPU 检查入口：`triton_per_fused__softmax_abs_all_amax_div_eq_mul_native_dropout_ne_sub_view_6`, `triton_per_fused__softmax__softmax_backward_data_div_mul_native_dropout_backward_sub_view_14`, `triton_per_fused__softmax_abs_all_amax_div_eq_mul_native_dropout_ne_sub_view_12`。
- `triton_red_fused__log_softmax_18`：NPU_g02_output_code.py:1383；输入 `f32[16384, 30522][30522, 1]npu:0`；size_hints=`{'x0': 16384, 'r1': 30522}`；split_axis=`[0]`。GPU 检查入口：`triton_poi_fused__log_softmax__log_softmax_backward_data__unsafe_view_add_arange_eq_expand_mm_nll_loss_backward_nll_loss_forward_scalar_tensor_view_2`, `triton_red_fused__log_softmax__log_softmax_backward_data__unsafe_view_add_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_view_0`, `triton_red_fused__log_softmax__log_softmax_backward_data__unsafe_view_add_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_sum_view_1`。
- `aclnnLogSoftmaxBackward_LogSoftmaxGrad_LogSoftmaxGrad`：外部调用；见 wrapper 调用边界。GPU 检查入口：需对照外部 ACLNN/CUDA 调用和 FX 图。

除上述已完成的归约实测外，其他候选待补：逐次 GPU shape/stride 与时间关联、两个设备最终 selected config、必要的固定输入/固定配置单 kernel 测量。若要区分 Triton_ascend 与 npuir，还需相同 DSL/config 的 TTIR→后端 IR 与生成指令；当前资料只支持定位到上述 Inductor 入口，不能越级定责。
