---
title: CICD 性能分析：MobileBERT 参数梯度归约深挖
---

# MobileBERT Train：参数梯度归约的两处 Inductor 问题

**`triton_red_fused_add_mul_sum_view_11` 的主要瓶颈已定位到 NPU Inductor：
IR 没有拆分长归约，tiling 候选又遗漏了更快的宽通道 tile。**
原 profile 中该 kernel 一步 96 次，共 25.847447 ms，占 NPU Inductor 整步 8.76%。
在指定分支重放时，设备 profile 为 246.390 μs/次；只改同一 DSL 的 tiling 降至 103.624 μs；
等价两阶段 ATen 小图自动 codegen 为 22.544 μs；两阶段 Triton 诊断实现为 13.211 μs。
没有修改框架实现，没有做整网修复验证。

2026-11-03 在 NPU_A5 Docker、`pta_213_cicd`、**物理卡 2** 上验证，未使用卡 4、5、6。
安装的 IR、codegen、runtime、tile generator 和 config 五个文件与
`test/cicd_benchmark@b3bd7403518924a7cc1e7c470aa80ccfb4e38120` 一致。
版本为 torch `2.13.0+cpu`、torch_npu `2.13.0.dev20261030`、Triton `3.6.0`，
目标为 `Ascend950PR_9579`；见[环境身份][env]。
原整网 CSV 未提供完整构建身份，本次单 kernel 复测与历史整网数据分别列示。

本轮后续已完成除 `_5` 外其余五组的分析，见[剩余 kernel 深挖](mobilebert-remaining-reductions-20261103.md)。
其中 `_9`、`_7`、`_13` 确认存在编译收益缺口；`_20` 暂未完成 GPU 提升比定量确认，
`_0` 的实测不支持强行分段。下文 GPU 对位同步修正为 `_11` + `_18`，原单 kernel 实测不变。

## kernel 做什么，96 次如何产生

这是 MobileBERT NoNorm 的参数梯度。NoNorm 前向为 `y = h * gamma + beta`；
反向将两路梯度相加得到 `g = a + b`，再计算：

```python
dbeta = g.sum((0, 1), keepdim=True)
dgamma = (g * h).sum((0, 1), keepdim=True)
```

三个输入的物理布局均对应 contiguous FP32 `[128,128,128]`，
其中一个 wrapper 输入先从 `[16384,128]` reshape；stride 为 `[16384,128,1]`。
输出为两个 `[1,1,128]`，每个通道归约 `128 × 128 = 16384` 个 token。
输入来自 MM 的结果缓冲区，但该 kernel 内没有矩阵乘。

原 backward wrapper 静态列出 **96 处 launch**，即 24 层、每层 4 次对应的 NoNorm 梯度归约；
CSV 每个采样 step 也是 96 次，平均 269.244 μs/次。
相邻 `triton_poi_fused_add_mul_view_12` 另做 `dh = g * gamma`，不在本 kernel 的 25.847 ms 内。
完整[原始 DSL][original]、[原 wrapper 调用边界][calls]可对照上述输入和输出。

这里的 add/mul/sum 都来自原模型反向语义。
本问题涉及这些运算如何归约和调度；与此前 `prims.fma` metadata 重复执行 Mul/Add 是两个独立问题。

## 为什么符合“NPU 编译提升比小于 GPU”的筛选条件

此前[模型汇总](models/MobileBertForMaskedLM.md#1-addmulsum-参数梯度归约)的 67.462 ms、
34.280 ms 预算覆盖七组 kernel，不能全部归给 `_11`。
这里进一步对 `_11` 做保守四方核验，继续使用 `r = 1 - Inductor/Eager`，
不拿两个设备的绝对 kernel 时间直接判定劣化。

GPU 的对应实现为：

| GPU 实现 | 次/step | ms/step | 与本点的关系 |
| --- | ---: | ---: | --- |
| `triton_red_fused_add_mul_sum_view_11` | 72 | 1.727827 | 对应每层 3 次；同时融合相邻 bias sum 和 `dh` 写回 |
| `triton_red_fused_add_mul_sum_view_18` | 24 | 0.563310 | 对应每层 attention 分支的另外 1 次，并融合输入梯度和 bias sum |
| `triton_red_fused_sum_view_9` | 576 | 0.802718 | 含上述部分结果的 final sum；保守地把全部 576 次都收费 |

GPU 先按 **128 组、每组 128 个 token** 生成部分梯度，再合并部分结果。
例如 `_11` 的第一阶段输出包含 `[128,128]` 部分结果，wrapper 随后调用 `_9` 做 final sum。
对照 [GPU 完整 DSL][gpu-dsl] 和 [GPU 调用边界][gpu-calls]。

由于 GPU CSV 没有逐次 shape，四方核验刻意使用不利于确认问题的上下界：

- NPU Eager：将全部 `Add → Sum → Mul → Mul → Sum` 连续窗口计入。
  每步 168 个窗口，包含目标 96 次及其他同类工作，还多收了输入梯度 Mul；合计 **9.841717 ms 上界**。
  它不是目标 96 次 Eager 的精确耗时，也不能除以 96 当作单次 Eager 时间。
- GPU Eager：只收目标 96 次各自必需的 `1 Add + 1 Mul + 2 Sum`，
  每次按整份 CSV 中该操作的最小值收费。
  `96 × (6.816 + 1.120 + 2×11.360) μs = 2.942976 ms 下界`。
- GPU Inductor：把表中三组 kernel 全收，包含附带工作，得 **3.093854 ms 上界**。
- NPU Inductor：只收 `_11` 的 **25.847447 ms**。

| 四方边界 | Eager / ms | Inductor / ms | 局部提升比边界 |
| --- | ---: | ---: | ---: |
| NPU | ≤9.841717 | 25.847447 | ≤−162.63% |
| GPU | ≥2.942976 | ≤3.093854 | ≥−5.13% |

**NPU 提升比的上界仍比 GPU 的下界低至少 157.50 个百分点，因而可判定本点存在编译收益缺口。**
这里没有声称 GPU 此区间的真实提升比是负值；GPU 的真实边界更窄，收益可以更高。
相应保守缺口为
`25.847447 − 9.841717 × 3.093854 / 2.942976 = 15.501171 ms/step`。
约占当前整步 **5.25%**、整网提升比缺口 **13.22%**，对应整网提升比 **6.99 个百分点**。
这是保守核验预算，不能与七组 kernel 的 34.280 ms 预算相加。

逐窗口 CSV 行号、四份文件 SHA256 与复算结果见[窗口明细][windows]、[四方边界][bound]，
提取脚本为[analyze_profile.py][analyze]。
它仍采用窗口超集和 GPU 耗时下界；没有伪造逐层 GPU shape 对齐。

## 根因一：IR 对保留多个输出元素的长归约没有拆分

分支源码 `torch_npu/_inductor/ir.py::num_splits` 被
`patch_num_splits()` 安装到社区 `Reduction.num_splits`。
它只在 `numel_hint == 1` 时考虑分段；其余情况末尾直接：

```python
return ReductionHint.DEFAULT, 1
```

本例 `numel_hint=128`，因此归约长达 16384 也不分段。
**仅开启 `config.split_reductions` 无法改变这个返回分支。**
直接责任入口是 **NPU Inductor IR 的 `num_splits` / outer reduction 拆分策略**，
需要让长 token 归约生成部分结果与 final sum，而不是只给单个 DSL 调一个更小的 R tile。
源码及安装调用链见[源码证据][source]。

最终 DSL 的 `split_axis=[0]`，只有通道轴分 program；token 归约全部在 program 内完成。
当前 winner 为：

```text
X0BLOCK=3, X0BLOCK_SUB=1, R1BLOCK_SUB=4096
grid=(ceil(128/3),1,1)=(43,1,1)
compile_mode=simd_simt_template, num_warps=1, num_stages=2
```

每个 program 串行处理最多 3 个通道；每个通道再做 4 次归约循环。
每轮三路 load 的地址为 `x0 + 128*r1`，tile 为 `[4096,1]`，
相邻 token 的地址相隔 128 个 FP32 元素，连续通道宽度仅 1。
这同时限制了访问连续性和归约并行度；不能只用“43 个 program 少于 56 个核”解释全部差距。

同一后端显式两阶段 ATen 小图已自动生成更快的代码：
`value.sum(1, keepdim=True).sum(0, keepdim=True)`，两路参数梯度同样改写。
其实际调用为 **1 次 `triton_per_fused_add_mul_sum_0` + 2 次 `triton_per_fused_sum_1`**，
共 3 次 launch，22.544 μs。两种 kernel 名不等于两次调用。
见[两阶段自动 codegen][staged]。

## 根因二：TileGenerator 的候选范围遗漏更快的配置

`runtime.triton_heuristics.triton_config_npu_index` 调用 `TileGenerator.descend_split_tiling()`。
本例 `tiny_kernel=False`，`program_threshold=56/2=28`。
`descend_split_axis_inner()` 的候选保留规则偏向超过该阈值的 program 数，
在本例最终只留下 **24 个候选**：

```text
X0BLOCK ∈ {3,4}
X0BLOCK_SUB ∈ {1,2,3,4}
R1BLOCK_SUB ∈ {1024,2048,4096}（并非所有组合）
multibuffer ∈ {False,True}
```

手动固定到 `X0BLOCK=16, X0BLOCK_SUB=16, R1BLOCK_SUB=256`，
**完全相同的原 DSL、相同后端编译模式**，即可由 246.390 降至 **103.624 μs**。
这个 tile 只有 8 个 program，却更快；自动候选里没有它。
后续保持原 signature、metadata 和 launcher 路径重放，`32/32/128` 实测 87.502 μs，
显式分段为 22.487 μs；原始与选中配置重放的 TTIR SHA256 一致，
见[补充实测](mobilebert-remaining-reductions-20261103.md#实测如何排除编译入口和计时干扰)。
故这里首先是 **候选生成漏搜**，不能只归为 autotune 计时选错 winner。
`TileGenerator.descend_split_axis_inner` / `descend_tiling_axis` 是具体优化入口。

只改 tiling 仍有长归约与并行不足。用相同 `[128,32]` tile 比较：
原单阶段约 103.51 μs；把 token 切成 128 组并增加 final sum 后约 13.03 μs（Event）。
两个效应分别有实验支持，收益存在重叠，不能相加。

## 实测、责任边界与收益

相同三个随机输入，与 CPU FP64 参考结果比较，全部实验通过
`rtol=1e-4, atol=2e-3`；最大绝对误差小于 `1.3e-4`。
固定输出版还先填 NaN、重放捕获图并校验，确认 graph 中确实执行了 kernel。

| 实现 | 设备 kernel 合计 / μs | Event 中位 / μs | 相对原实现 |
| --- | ---: | ---: | ---: |
| 原 kernel，自动 winner | 246.390 | 246.473 | 1× |
| 同 DSL，固定 `16/16/256` | 103.624 | 103.144 | 2.38× |
| 显式两阶段 ATen，小图自动 codegen，3 次 launch | 22.544 | 21.736 | 10.93× |
| 两阶段 Triton 诊断实现，2 次 launch | 13.211 | 13.034 | 18.65× |

设备列为每项 warmup 后 10 次调用的 kernel duration 之和除以 10。
Event 列为捕获图 64 次调用、重放 7 轮取中位数，包含 graph 调度开销。
两阶段诊断实现的 partial 为 **11.544 μs**，final 为 **1.667 μs**；没有漏收第二阶段。
同边界 NPU Eager 本次为 26.312 μs，仅作因果控制，不替代前面的四方筛选。
完整选中配置、候选列表、逐组次数/耗时见[实测结果][result]。

| 组件 | 本次结论 |
| --- | --- |
| NPU Inductor IR：`ir.py::num_splits` | 主要问题：本类 outer reduction 无分段计划；显式分段后自动 codegen 已快 10.93 倍 |
| NPU Inductor runtime：`TileGenerator` | 独立确认漏搜更快 tile；同 DSL 固定 tile 快 2.38 倍 |
| NPU Inductor 多输出 final sum 融合 | 后续检查入口；自动版两路 final 分别 launch，诊断版合并一次。不是本轮主要根因 |
| Triton_ascend / npuir | 相同后端及编译模式下，改变上游计划即可显著变快；未发现需要先修下层的证据，不将主要差距归给下层 |

已保存固定配置 [TTIR/TTAdapter 产物][ir]。
原配置 TTAdapter 保留了 `memref<4096x1xf32, strided<[128,1]>>` 及循环；
metadata 的实际 `parallel_mode=simd`，不能根据 `npu_kernel_type=simt_template` 名称认定在跑纯 SIMT。
本次没有分析最终指令吞吐或证明下层完全没有优化空间。

按本次设备对照差值乘原每步 96 次，预计只优化 `_11`：

- 更换候选覆盖：约 **13.71 ms/step**。
- 引入两阶段计划：约 **21.49–22.39 ms/step**，相当于原整步 **7.3–7.6%**，
  整网提升比约增加 **9.7–10.1 个百分点**，覆盖整网缺口约 **18–19%**。

后两项是同一问题的不同实现，不相加。这些是独立小图向 96 次调用外推的收益量级，
不是原整网替换后的实测；原整网 269.244 μs/次与本次 246.390 μs/次也存在上下文差异。
因此不承诺收回整组 34.280 ms，更不能凭该点宣称整网提升比已达到 GPU 的 19.82%。

可先优化通用 outer reduction 分段策略及候选覆盖；本轮止于问题定位。
复现命令和计时说明见[脚本说明][readme]。

[env]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/environment.json
[original]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/original_kernel.py
[calls]: ../../../../artifacts/cicd/performance/full-survey-20261102/MobileBertForMaskedLM_Train/NPU_wrapper_calls.txt
[gpu-dsl]: ../../../../artifacts/cicd/performance/full-survey-20261102/MobileBertForMaskedLM_Train/GPU_kernels.txt
[gpu-calls]: ../../../../artifacts/cicd/performance/full-survey-20261102/MobileBertForMaskedLM_Train/GPU_wrapper_calls.txt
[windows]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/eager_windows.csv
[bound]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/profile_bound.json
[analyze]: ../../../../scripts/repro/cicd/performance/mobilebert_reduction_20261103/analyze_profile.py
[source]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/source_evidence.txt
[staged]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/staged_output_code.py
[result]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/result.json
[ir]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/ir
[readme]: ../../../../scripts/repro/cicd/performance/mobilebert_reduction_20261103/README.md
