---
title: CICD 性能分析：Electra / LayoutLM Train 首层 attention 劣化
---

# CICD 性能分析：Electra / LayoutLM Train 首层 attention 劣化

**两个 Train 当前最突出的局部劣化点，是首层 softmax + dropout 融合 kernel：
Electra 为 16.764 ms/步，LayoutLM 为 25.097 ms/步。
已确认的触发根因属于 NPU Inductor codegen：seed offset 的 constexpr 常量值遗漏，
使多轴实现生成 TTIR 失败，随后运行明显更慢的 no-linear 回退实现。**

本文记录 2026-10-31 的问题定位，不包含编译器修复或修复后的整模型验证。
四个场景的基线用于说明提升比目标；根因分析只聚焦两个 Train 的首层 attention。

## 数据口径和提升比目标

原始数据包括 `ElectraForCausalLM`、`LayoutLMForMaskedLM` 的 Infer / Train，
每个场景均有 NPU/GPU、Eager/Inductor 四份 CSV。
原始文件名和 SHA256 保存在 [manifest][manifest]，汇总值保存在 [profile_summary.json][summary]。

- NPU 使用 `Step Id=10…19`，将 `Duration(us)` 求和后除以 `10 × 1000`。
- GPU CSV 已按算子聚合，将 `Total Time(us)` 求和后按同样的 10 步归一化。
- 本文的单步耗时是设备 kernel duration 之和，不包含 `Wait Time(us)`，也不是主机端 wall time。
- 原 CSV 未附完整环境版本；下文重新采集的框架版本只标识本次 codegen / kernel 对照环境。

定义提升比为 `r = (T_eager - T_inductor) / T_eager`。
若要使 NPU 提升比达到 GPU，需满足 `T_npu_inductor ≤ T_npu_eager × (1 - r_gpu)`。

| 场景 | NPU Eager / ms | NPU Inductor / ms | NPU 提升比 | GPU 提升比 | NPU 目标上限 / ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Electra Infer | 40.599 | 31.804 | 21.66% | 31.34% | 27.874 |
| Electra Train | 121.155 | 156.901 | -29.50% | 25.05% | 90.804 |
| LayoutLM Infer | 101.827 | 92.553 | 9.11% | 14.43% | 87.133 |
| LayoutLM Train | 305.813 | 348.988 | -14.12% | 17.85% | 251.226 |

这四行只说明当前差距；本文没有据此给两个 Infer 场景下根因结论。

## 哪个 kernel 劣化，劣化多少

Electra Train 的首层 kernel，完整 DSL 见 [Electra 首层生成产物][electra-first]：

```text
triton_per_fused__softmax_all_amax_div_isfinite_matmul_native_dropout_sub_4
```

LayoutLM Train 的首层 kernel，完整 DSL 见 [LayoutLM 首层生成产物][layoutlm-first]：

```text
triton_per_fused__softmax_all_amax_div_isfinite_matmul_native_dropout_sub_5
```

原始 CSV 中显示为 `all_amax_div_isfinite_matmul_native_dropout_sub_4` / `_5`。
根据生成 wrapper 中的调用位置，与上述完整名称对应。
名称中的 `matmul` 是来源标记；这两个 kernel 的 DSL 没有矩阵乘，QK BMM 是前一个独立调用。

| Train 场景 | Eager 对应区间 / ms | Inductor 首层 kernel / ms | 耗时倍数 | 每步增加 / ms |
| --- | ---: | ---: | ---: | ---: |
| Electra | 0.809096 | 16.764052 | 20.72 倍 | 15.954956 |
| LayoutLM | 1.222625 | 25.096688 | 20.53 倍 | 23.874063 |

比较边界是 **QK BMM 结束后，到与 V 相乘的 BMM 开始前**。
Eager 区间包括缩放、mask 加法、softmax、dropout 和相应布局处理，
Inductor 在这个区间运行上述融合 kernel。
本次生成图还消除了零值 mask 加法；没有让 Inductor 承担更多模型语义。

这些数字来自稳定采样的 10 步，不是首轮编译或 warmup。
[attention_profile_windows.csv][windows] 保存两个模型、两种 backend、10 步和 12 层的区间明细，
保留原 CSV 行号，去除了绝对起始时间。

同模型后 11 层的相同 shape、相同融合计算没有出现这个数量级的耗时：

| 场景 | attention shape / FP32 | 后续层 kernel | 单层均值 / ms |
| --- | --- | --- | ---: |
| Electra | `[32, 4, 512, 512]` | `triton_per_fused__softmax_all_amax_div_isfinit_7` | 0.984592 |
| LayoutLM | `[16, 12, 512, 512]` | `triton_per_fused__softmax_all_amax_div_isfinit_8` | 1.283408 |

后续层完整 DSL 分别见 [Electra][electra-later]、[LayoutLM][layoutlm-later]。
首层和后续层都包含两路 max、有限值检查、softmax 和 dropout，
因此不能仅用“融合计算比较复杂”解释首层独有的劣化。

## 首层为什么进入另一条生成路径

正常多轴实现保留 batch、head、query、reduction 四个轴，metadata 使用 `GridNpu`。
首层最终产物将 batch/head/query 展平为 x，使用 `Grid1D`。
这里的 no-linear 是备用 codegen 路径，仍然生成并执行 NPU kernel。

两个模型的 [Electra 回退日志][electra-failure]、[LayoutLM 回退日志][layoutlm-failure]
都记录了以下调用链：

```text
NPUCombinedScheduling.codegen_node
  -> NPUTritonScheduling.codegen_node_schedule
  -> make_ttir_for_check
  -> _triton_make_ttir
  -> ast_to_ttir
  -> tl.load(in_ptr1 + load_seed_offset)
  -> AttributeError: 'NoneType' object has no attribute 'type'
  -> falling back to no-linear codegen
```

TTIR 是 Triton 前端生成的中间表示。
上述异常发生在生成 TTIR 时，并不是设备 kernel 运行到 seed 读取时才报错。
异常被 Inductor 捕获，模型因此仍能进入备用实现。

[失败 checker 的完整 DSL 和 metadata][checker] 中，关键矛盾为：

```python
triton_meta = {
    "signature": {"load_seed_offset": "constexpr", ...},
    "constants": {},
}
```

`constexpr` 表示编译时必须知道的常量。这里首层 `load_seed_offset` 实际为 1，
但常量表没有提供其值；前端使用该参数时拿到缺失值，最终在地址表达式处失败。
后续层的 offset 不为 1，以普通 `i32` 参数传入，没有触发这个缺失常量问题。

## 为什么责任归于 Inductor codegen

[安装环境源码摘录][source] 保留了相互对应的三段逻辑：

1. 社区 `torch/_inductor/codegen/triton_utils.py::signature_of`：
   在当前 Triton 接口下，将等于 1 的 `SizeArg` 标记为 `constexpr`。
2. 社区 `torch/_inductor/codegen/triton.py::TritonKernel.codegen_kernel`：
   用 `equal_1_arg_indices(signature)` 给这些参数填入常量值 1。
3. NPU `torch_npu/_inductor/codegen/triton.py::NPUIndexTritonKernel.codegen_kernel`：
   使用同一个 `signature_to_meta`，初始化 `constants={}`，但遗漏了上述填值步骤。

社区对应的填值语句是：

```python
for arg_num in equal_1_arg_indices(signature):
    triton_meta["constants"][signature[arg_num].name] = 1
```

这与失败 checker 中“signature 声明常量、constants 没有值”的实际产物直接对应。
源码摘录同时保留了 `NPUCombinedScheduling.codegen_node` 捕获异常并回退的分支。

| 层级 | 当前结论 | 依据 |
| --- | --- | --- |
| NPU Inductor codegen | **已确认的触发根因** | constexpr signature 与 constants 不一致；元数据对照可消除失败 |
| Triton_ascend 前端 | 报错发生位置 | 接收到缺失常量的 metadata，在生成 TTIR 时失败 |
| 后续设备 IR / npuir lowering | 没有证据指向它是本次触发根因 | 最初的失败发生在 `ast_to_ttir`，尚未进入后续 lowering |
| 回退 kernel 内部性能 | 尚未进一步拆解 | 已测得回退实现明显更慢，但未分析其内部指令、访存或调度成本 |

不能仅凭 traceback 出现在 Triton 就认定为 Triton bug，
也不能把备用 kernel 的全部耗时归因于一个尚未验证的底层缺陷。

## 最小对照与实际验证范围

本次采集环境为 Ascend950PR_9579、torch `2.13.0+cpu`、
torch_npu `2.13.0.dev20261030`、Triton `3.6.0`、transformers `4.36.0`。
模型配置为 static、FP32、序列长度 512、attention dropout 0.1；
Electra batch 32，LayoutLM batch 16，关闭 aclgraph，使用 Inductor 的 NPU Triton/default 路径。

[manifest][manifest] 记录实际版本、runner HEAD、runner 文件校验值和源码快照校验值。
runner HEAD 只标识 runner checkout，不代表安装的 torch_npu 与该提交完全相同；
责任判断以本次实际安装代码的摘录和生成产物为依据。

使用捕获的 Electra 首层多轴 checker，只改变 seed 参数元数据，结果见 [TTIR 对照结果][checks]：

| 对照 | 改动 | 结果 |
| --- | --- | --- |
| 原始产物 | `constexpr`，`constants={}` | 重现相同失败 |
| 常量绑定对照 | 仅填入 `constants["load_seed_offset"]=1` | 通过 TTIR 编译 |
| 运行时参数对照 | 仅将 signature 中该参数改为 `i32` | 通过 TTIR 编译 |

这三项使用相同计算体、shape 和 TTIR 检查配置。
它们是定位常量绑定问题的临时对照，不是已安装的编译器修复。
[最小复现说明和脚本][repro] 已随仓库归档；本次归档没有重新执行 NPU 测试。

另外，对原始 Electra 首层与后续层生成 kernel 做独立 NPU replay，
使用 3 次预热、15 次采样，以 NPU Event 测量并取中位数：

| 原始实现 | 本次实测 / ms | 原始 CSV / ms |
| --- | ---: | ---: |
| 首层回退 kernel | 16.798096 | 16.764052 |
| 后续层多轴 kernel | 0.979769 | 0.984592 |

原始采样保存在 [kernel_timings][timings]，两者均通过有限值输入下的输出语义检查。
[selected_tiling][tilings] 来自运行时 recorder，而非候选日志推测；
两者最终 `compile_mode` 都是 `simd_simt_template`。
这也说明不能只看 metadata 的 `npu_kernel_type` 标签就给底层编译模式判责。

模型级采集范围分别是：Electra 已执行前向并生成前反向产物，随后停止完整反向 autotune；
LayoutLM 生成前向产物后停止，未执行整模型。
LayoutLM 的 25.097 ms 来自原始 CSV，本次确认了它相同的首层回退异常，未重新测该模型单 kernel。
这些记录均不能表述为修复后整模型性能或整模型精度验证。

## 本项收益估计与状态

若首层恢复到同模型后续层的实现水平，按原始 profile 估计：

- Electra：`16.764052 - 0.984592 ≈ 15.78 ms/步`。
- LayoutLM：`25.096688 - 1.283408 ≈ 23.81 ms/步`。

这与“相对 Eager 增加 15.955 / 23.874 ms”是不同参照，不能相加。
上述收益用于判断本问题的优先级，不是修复后的实测收益，
也不表示仅处理这一项就能使 NPU 提升比达到 GPU。

当前状态：**问题已定位，未实施编译器修复，未完成修复后的整模型回归。**

[manifest]:
  ../../../../artifacts/cicd/performance/attention-seed-constexpr-20261031/manifest.json
[summary]:
  ../../../../artifacts/cicd/performance/attention-seed-constexpr-20261031/profile_summary.json
[windows]:
  ../../../../artifacts/cicd/performance/attention-seed-constexpr-20261031/attention_profile_windows.csv
[electra-first]:
  ../../../../artifacts/cicd/performance/attention-seed-constexpr-20261031/electra_first_attention.py
[electra-later]:
  ../../../../artifacts/cicd/performance/attention-seed-constexpr-20261031/electra_later_attention.py
[layoutlm-first]:
  ../../../../artifacts/cicd/performance/attention-seed-constexpr-20261031/layoutlm_first_attention.py
[layoutlm-later]:
  ../../../../artifacts/cicd/performance/attention-seed-constexpr-20261031/layoutlm_later_attention.py
[electra-failure]:
  ../../../../artifacts/cicd/performance/attention-seed-constexpr-20261031/electra_fallback_traceback.txt
[layoutlm-failure]:
  ../../../../artifacts/cicd/performance/attention-seed-constexpr-20261031/layoutlm_fallback_traceback.txt
[checker]:
  ../../../../artifacts/cicd/performance/attention-seed-constexpr-20261031/failed_first_attention_checker.py
[source]:
  ../../../../artifacts/cicd/performance/attention-seed-constexpr-20261031/source_evidence.txt
[checks]:
  ../../../../artifacts/cicd/performance/attention-seed-constexpr-20261031/ttir_check_results.json
[timings]:
  ../../../../artifacts/cicd/performance/attention-seed-constexpr-20261031/electra_kernel_timings.json
[tilings]:
  ../../../../artifacts/cicd/performance/attention-seed-constexpr-20261031/electra_selected_tiling.json
[repro]:
  ../../../../scripts/repro/cicd/performance/attention_seed_constexpr_20261031/README.md
