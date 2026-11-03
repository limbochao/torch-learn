---
title: CICD 性能分析：两个 Train 对 GPU Inductor 的差距与优化模块
---

# CICD 性能分析：两个 Train 对 GPU Inductor 的差距与优化模块

> **历史分析，比较口径已被替代。** 本文按 NPU/GPU Inductor 绝对耗时差排序，
> 不足以判断编译器收益差距。尤其是 MM/AddMM 的 70.049 ms，不再作为 Inductor 优化依据。
> 当前应比较同一语义区域的 `r_N=1−N_I/N_E` 与 `r_G=1−G_I/G_E`。
> 最新数据和具体 kernel、局部预算见 [全量分析](full-survey.md)、
> [Electra](models/ElectraForCausalLM.md)、[LayoutLM](models/LayoutLMForMaskedLM.md)。
> 以下保留历史数字及证据，不作为当前优先级结论。

**Electra Train 当前最明确的问题是首层 attention codegen 回退和反向 FMA 拆分；
LayoutLM Train 除这两项外，最大尚未解释的差距是线性层矩阵乘组，约 70.05 ms/步。**

本文汇总截至 2026-11-02 的分析，包含 [首层 attention 已定位问题][first-analysis]。
全文直接比较 NPU Inductor 与 GPU Inductor；Eager 仅用于整步提升比目标的计算。
目前没有实施编译器修复，预计收益均不是修复后的实测收益。

## 比较口径与整步目标

- 单位统一为 ms/步，使用设备 kernel duration 之和，不包含 Wait Time，也不是主机 wall time。
- NPU 使用稳定的 `Step Id=10…19`，将 `Duration(us)` 求和后除以 `10 × 1000`。
- GPU 使用原始聚合 CSV 的 `Total Time(us)`，按同样 10 步归一化。
- 各部分在自身采样中稳定出现；GPU 侧目前只有聚合 CSV，没有 wrapper、逐次 shape/stride 或 selected config。
- 原始 CSV 的完整构建版本未知。2026-10-31 的 codegen/source capture 和 11-02 的 CPU 对照是另外的证据来源。

提升比定义为 `r = (T_eager - T_inductor) / T_eager`。
要达到 GPU 提升比，NPU 的整步目标为：

```text
T_npu_inductor_target = T_gpu_inductor × T_npu_eager / T_gpu_eager
```

| Train 场景 | NPU Inductor | GPU Inductor | 直接差额 | NPU / GPU 提升比 | NPU 目标 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Electra | 156.901 | 117.427 | +39.474 | -29.50% / 25.05% | 90.804 |
| LayoutLM | 348.988 | 249.548 | +99.440 | -14.12% / 17.85% | 251.226 |

因此只把 NPU 整步降到 GPU 的绝对耗时，也不一定满足提升比目标，Electra 尤其如此。
数据见 [对比汇总][comparison]、[NPU kernel 汇总][npu-summary]，
GPU 原始聚合记录见 [Electra][electra-gpu] 和 [LayoutLM][layoutlm-gpu]。

## ElectraForCausalLM Train

| 问题 | NPU / GPU Inductor | NPU 多耗时 | 应优化或检查的模块 | 状态 |
| --- | ---: | ---: | --- | --- |
| 首层 softmax + dropout | 16.764 / 0.477 | 16.287，35.15 倍 | NPU Inductor codegen 常量绑定 | 已定位 |
| attention backward，12 层 | 16.946 / 7.542 | 9.403，2.25 倍 | lowering/fallback、公共 prims runtime | 部分原因已定位 |
| 后续 11 层 softmax + dropout | 10.831 / 5.245 | 5.585，2.06 倍 | tiling/autotune、生成循环、下层编译 | 待定责 |
| 线性层 Mm + Addmm | 60.758 / 58.230 | 2.528 | GEMM lowering、布局/实现选择、CANN MatMul | 待逐调用对齐 |

1. **首层 kernel** 为
   `triton_per_fused__softmax_all_amax_div_isfinite_matmul_native_dropout_sub_4`。
   与 GPU 对应融合 kernel 比较，多花 16.287 ms。
   已定位的是 NPU `NPUIndexTritonKernel.codegen_kernel` 漏填 constexpr 值，触发 no-linear 回退。
   按恢复到本模型后续层实现水平估计可节省 **15.78 ms/步**；这还没有达到 GPU 的 0.477 ms。
2. **反向前半段** 为 `triton_per_fused__softmax__softmax_backward_da_13`，
   后接两组 `aclnnMul_MulAiCore_Mul → aclnnAdd_AddAiCore_Add`，
   再执行 `triton_poi_fused_div_14`。
   生成 wrapper、源码和 profile 已建立对应，重复乘加成本估计 **5.76 ms/步**。
   这只能解释 9.403 ms 差距的一部分；完整链路还需要处理融合边界和中间结果读写。
3. **后续层 kernel** 为 `triton_per_fused__softmax_all_amax_div_isfinit_7`。
   NPU 已是单个多轴融合 kernel，GPU CSV 也记录为融合 kernel；尚不能将差距直接解释成缺少融合。
   首查 Inductor 的 tile 候选、最终配置、循环和访存生成，再定界下层编译质量。
   **5.585 ms 是观察到的差距，尚不是确认可回收的收益。**
4. **线性层矩阵乘** 使用外部 Mm/Addmm 路径，当前是算子族总账。
   GPU 和 NPU 的调用数及融合边界并不完全相同，必须逐层匹配后才可指定某一次 MatMul 劣化。
   相对 NPU Eager 持平，不能作为排除本项的理由。

## LayoutLMForMaskedLM Train

| 问题 | NPU / GPU Inductor | NPU 多耗时 | 应优化或检查的模块 | 状态 |
| --- | ---: | ---: | --- | --- |
| 线性层 Mm + Addmm | 233.816 / 163.767 | 70.049 | GEMM lowering、布局/实现选择、CANN MatMul | 最大待定位项 |
| 首层 softmax + dropout | 25.097 / 0.710 | 24.387，35.34 倍 | NPU Inductor codegen 常量绑定 | 已定位 |
| attention backward，12 层 | 26.087 / 11.428 | 14.659，2.28 倍 | lowering/fallback、公共 prims runtime | 相同模式待补 wrapper |
| 后续 11 层 softmax + dropout | 14.117 / 7.792 | 6.326，1.81 倍 | tiling/autotune、生成循环、下层编译 | 待定责 |

1. **矩阵乘组** 主要对应 `aclnnMm_MatMulV3Common_MatMulV3`、
   `aclnnAddmm_MatMulV3Common_MatMulV3`，是当前最大的未解释差距。
   [前向 wrapper 摘录][layoutlm-mm] 确认存在 `extern_kernels.mm` 和 `aten.addmm` 外部调用。
   但 **70.049 ms 尚未定位到单个 kernel，也不能承诺全部回收**。
   应先匹配每层矩阵尺寸、stride、精度开关、调用次数和融合边界，再区分布局/实现选择与 CANN 执行成本。
2. **首层 kernel** 为
   `triton_per_fused__softmax_all_amax_div_isfinite_matmul_native_dropout_sub_5`。
   与 Electra 共用常量绑定问题和优化入口；估计恢复正常后续层水平可节省 **23.81 ms/步**。
   25.097 ms 来自原始 CSV；本次未重新测量该模型的首层 kernel。
3. **反向前半段** 为 `triton_per_fused__softmax__softmax_backward_da_11`，
   随后同样出现两组外部 Mul/Add，再执行 `triton_poi_fused_div_12`。
   按相同 FMA 执行模式，重复乘加成本估计 **8.54 ms/步**。
   目前 profile 模式与 Electra 一致，尚缺该模型完整反向 wrapper 的直接确认。
4. **后续层 kernel** 为 `triton_per_fused__softmax_all_amax_div_isfinit_8`。
   应检查的模块与 Electra 相同，对 GPU 的 **6.326 ms** 差距尚未拆成可验证的收益项。

## 四类问题的模块归属与证据

### 首层 attention：Inductor codegen 生成了不完整的常量 metadata

首层 seed offset 实际为 1，signature 将其标为 `constexpr`，但 `constants` 未提供值。
生成 TTIR 时在 seed load 处失败，`NPUCombinedScheduling.codegen_node` 捕获异常，回退至 no-linear 实现。
仅绑定常量值或将参数改为运行时 i32，均可使同一 checker 通过 TTIR 生成。

需要优化 `torch_npu/_inductor/codegen/triton.py::NPUIndexTritonKernel.codegen_kernel`，
使 signature 与 constants 一致。Triton 前端是报错位置；触发根因已经在 Inductor，
没有证据归因到尚未进入的后续设备 IR / npuir lowering。
完整产物、源码摘录和验证范围保留在 [首层专题][first-analysis]。

### 反向 FMA：fallback 切断融合，运行时乘加又计算两遍

softmax 反向的核心表达式为：

```text
u = grad × P
s = sum(u)
dScores = (u - P × s) / 8
```

Electra 的 [完整前半段 DSL][backward-kernel]、[缩放 DSL][div-kernel]
和 [wrapper 摘录][backward-wrapper] 显示：

```text
Triton：重算 P，计算 u、s、-P，写回中间结果
prims.fma(-P, s, u)：外部调用
另一个 Triton kernel：/8
```

对应 GPU CSV 中的融合 kernel 为：

```text
Electra:
triton_per_fused__softmax__softmax_backward_data_div_mul_native_dropout_backward_sub_view_16
LayoutLM:
triton_per_fused__softmax__softmax_backward_data_mul_native_dropout_backward_sub_view_11
```

NPU 反向区间边界为每层 dP/dV BMM 之后、dQ/dK BMM 之前；
GPU 侧按上述融合 kernel 归组，目前尚没有其 DSL 用于逐指令核对。
NPU 细项见 [区间明细][backward-windows] 和 [汇总][backward-summary]。

[安装源码摘录][fma-source] 中有三段相互对应的逻辑：

- `lowering_fallback_list.py` 包含 `prims.fma`，`lowering.py` 将其注册为外部 fallback。
- 公共 `_prims._make_prim` 的 `_prim_impl` 先调用 `meta()`，再调用 `impl_aten()`。
- `inductor_prims.make_prim` 的 `meta()` 实际执行 `impl_aten()` 获取 TensorMeta；FMA 实现为 `(a*b)+c`。

因而一次 FMA 执行两组 Mul/Add，第一组仅用于提取元数据，数值结果丢弃。
2026-11-02 的 [CPU dispatch 对照][cpu-result] 重现 1 次 FMA、2 次 Mul、2 次 Add，数值比较通过。
[复现脚本说明][repro] 已归档。

应优化的入口是 NPU Inductor 的 fallback/lowering 接入和 softmax backward decomposition，
以及公共 prims runtime 的 metadata 实现；同时应使乘加、缩放能进入适当的融合范围。
这是分析方向，不是已验证的具体生产补丁。
源码中虽有 no-FMA override，但其调用属于 `triton_experimental` 注册路径，
不能据此认定本次 default 路径已避免 FMA；实际生成 wrapper 仍存在外部调用。
这部分乘加没有作为完整融合表达式交给 Triton，当前不能归因为 Triton_ascend/npuir kernel 慢。

### 正常多轴 softmax：先定界 tiling/codegen，再判断下层编译

首先检查 Inductor 的 `runtime/tile_generator.py`、`runtime/triton_heuristics.py`，
以及 `codegen/triton.py` 的分块、循环和访存生成。
tile 指一次处理的数据块；autotune 从候选配置中选择实测实现。

已有 Electra [运行时配置记录][tiling]，首层与后续层实际 `compile_mode` 都是 `simd_simt_template`。
不能仅凭 metadata 中 `npu_kernel_type` 的标签差异判断底层编译模式和责任。

下一步需要固定候选并检查生成 IR，判断是否有上游分块/循环/中间结果安排不合理，
再检查 Triton_ascend、npuir 的访存、向量化、临时数据溢出和调度质量。
目前两层均未完成定责，不能将它们写成已确认的底层编译缺陷。

### 线性层矩阵乘：Inductor 调用和 CANN 实现分开定位

Inductor 入口包括 `kernel/mm.py` 的 `tuned_mm/tuned_addmm`、布局和实现选择，
必要时追溯 matmul decomposition。
[入口参考摘录][mm-entry] 来自本地 checkout，仅用于指出检查位置，不代表原采样的安装源码版本。

应先对齐 NPU/GPU 的 M/N/K、batch、输入和输出 stride、dtype、矩阵乘精度开关、次数和融合边界。
若差距源于 Inductor 选择了不利布局、增加物化或选择了较差实现，优化对应的 lowering/layout/选择逻辑；
若相同输入条件下独立 aclnn 调用本身仍慢，再落实到 CANN MatMul 实现或 tiling。
当前观察到的是外部 aclnn 路径，不能直接把这组差距归到 Triton_ascend 或 npuir。

## 收益、优先级与验证边界

| 已有具体原因的项 | Electra 预计收益 | LayoutLM 预计收益 | 估计依据 |
| --- | ---: | ---: | --- |
| 首层恢复正常多轴实现水平 | 15.78 | 23.81 | 首层与本模型后续层耗时之差 |
| 消除 FMA 第一组重复乘加 | 5.76 | 8.54 | 原 profile 中第一组 Mul/Add 之和 |

这两个估计不能与表中的 GPU 差额重复相加。
后续层与矩阵乘组尚没有经实验确认的可回收收益；全部降到 GPU 耗时也不是已验证结果。

对于最终提升比目标，[汇总 JSON][comparison] 还记录按整步 Eager 比例缩放的 GPU 参考预算。
这只是分摊目标差额的算术方法，不意味着每个 kernel 都必须服从相同缩放比例。
例如矩阵乘组相对参考预算的差额为 Electra 15.73、LayoutLM 68.95 ms，
因此 Electra 矩阵乘绝对耗时接近 GPU，也不能直接认为它对提升比目标没有优化价值。

attention BMM 上 NPU 分别比 GPU 快 4.905 / 9.729 ms，会抵消部分正差额。
已列四类正差额与 BMM 合并后，Electra 为 +28.898 ms，其他未展开部分净差额 +10.576 ms；
LayoutLM 为 +105.692 ms，其他未展开部分净差额 -6.252 ms。
两者分别回到整步差额 +39.474 / +99.440 ms，不能把所有正差额直接当作整步收益。

确定可着手的缺陷是首层常量绑定和反向 FMA；最大待定位项是 LayoutLM 的矩阵乘组。
本轮未执行编译器修复或修复后整模型回归。11-02 没有满足限制的空闲健康卡，
因此仅补充 CPU dispatch 对照，没有新增 NPU 设备测量。
归档时检查数据汇总、公式、链接、脚本语法和去敏；版本、原始文件校验值见 [manifest][manifest]。

[first-analysis]: electra-layoutlm-train-first-attention-20261031.md
[comparison]:
  ../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/gpu_inductor_comparison.json
[npu-summary]:
  ../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/npu_inductor_kernel_summary.csv
[electra-gpu]:
  ../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/electra_gpu_inductor.csv
[layoutlm-gpu]:
  ../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/layoutlm_gpu_inductor.csv
[layoutlm-mm]:
  ../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/layoutlm_mm_wrapper_excerpt.txt
[backward-kernel]:
  ../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/electra_backward_softmax.py
[div-kernel]:
  ../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/electra_backward_div.py
[backward-wrapper]:
  ../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/electra_backward_wrapper_excerpt.txt
[backward-windows]:
  ../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/backward_attention_windows.csv
[backward-summary]:
  ../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/backward_attention_summary.json
[fma-source]:
  ../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_source_evidence.txt
[cpu-result]:
  ../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/fma_cpu_dispatch_result.json
[repro]:
  ../../../../scripts/repro/cicd/performance/attention_fma_fallback_20261102/README.md
[tiling]:
  ../../../../artifacts/cicd/performance/attention-seed-constexpr-20261031/electra_selected_tiling.json
[mm-entry]:
  ../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/mm_entrypoint_reference.txt
[manifest]:
  ../../../../artifacts/cicd/performance/train-gpu-inductor-gap-20261102/manifest.json
