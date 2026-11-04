---
title: BF16 融合计算的中间舍入与 emulate_precision_casts
---

# BF16 融合计算的中间舍入与 emulate_precision_casts

## 问题与验证范围

一组从 scheduler IR 恢复的 ATen 子图，在 NPU 上与对应的 `torch.compile` 输出比较时，
出现 `rtol=1e-3、atol=1e-3` 精度失败。代表计算包括 BF16 连续加法、
BF16 `add → sigmoid → mul`，以及用于对照的单次加法和 FP32 计算链。

已经确认的数值原因是：eager 每个 BF16 算子结束时发生的舍入，没有在融合计算的相同位置保留。
开启 combo 后仍能出现这个问题；关闭 Inductor 的 load upcast 也不足以恢复 eager 的结果。
对代表用例开启 `emulate_precision_casts` 后，生成代码出现显式的 BF16 舍入边界，结果恢复一致。

本文区分两种概念：与 eager 一致，以及与无限精度的数学结果接近。
FP32 中间结果保留更多尾数，可能更接近数学结果，却仍然无法通过严格的 eager 对照。
一次 eager 精度检查失败，不能直接推定数学计算更不准确。

本轮测试使用合成输入，保留用例的 shape、stride 和 dtype；没有使用原模型的真实中间张量，
也没有进行整网精度或性能验证。所有设备结果均来自 2026-11-04 的本会话 NPU 实测。

## 一个连续加法例子

`triton_poi_fused_40` 包含以下计算模式：

```python
y = (a + b) + c  # 三个输入都是 BF16
```

BF16 的小数部分有 7 位。在 `[1, 2)` 区间内，相邻可表示值的间距为：

```text
2^-7 = 0.0078125
```

舍入的含义是把结果变为 BF16 可表示的值。下面三个输入都可以被 BF16 精确表示，
并且已经用于设备复现：

```text
a = 0.5
b = 0.50390625
c = 0.50390625
```

| 步骤 | eager：每次加法输出 BF16 | FP32 融合链：末尾输出 BF16 |
| --- | --- | --- |
| 第一次加法 | `1.00390625 → 1.0` | 保留 `1.00390625` |
| 第二次加法 | `1.0 + 0.50390625 = 1.50390625 → 1.5` | `1.00390625 + 0.50390625 = 1.5078125` |
| 最终输出 | **1.5** | **1.5078125** |

两个 eager 中间值恰好落在相邻 BF16 数的中点。这里采用“最近值，中点取偶数”的舍入：
`1.00390625` 位于 `128/128` 与 `129/128` 之间，选择 `128/128 = 1.0`；
`1.50390625` 位于 `192/128` 与 `193/128` 之间，选择 `192/128 = 1.5`。

因此，差异来自算子边界的舍入次数。用 `R(x)` 表示转为 BF16 的舍入：

```text
eager：      R(R(a + b) + c)
FP32 融合： R(a + b + c)
```

该例的绝对差为 `0.0078125`，大于参考值为 `1.5` 时允许的
`atol + rtol × |reference| = 0.0025`，所以会精度失败。

## 默认融合为什么会改变结果

FX graph 中可以只有两个 BF16 `aten.add`，不一定存在独立的 BF16 cast 节点。
在 eager 中，第一个算子返回 BF16 Tensor，返回时已经发生舍入；第二个算子读取的是舍入后的值。

Inductor 将两个算子放进同一个 kernel 后，中间值可以一直留在寄存器中。
默认的 FP32 计算链可以直接把第一次加法的 FP32 结果交给第二次加法，最后在 store 时转成 BF16。
因此，不能把这个现象简单归因于“Dynamo 或某个 FX pass 删除了 cast 节点”。

当前安装版本的 `torch/_inductor/config.py:836–845` 明确说明：融合低精度算子时，
Inductor 默认会省略算子之间的 downcast/upcast 对；`emulate_precision_casts` 用于保留这些转换。
对应安装版本的源码节选保存在[源码证据][source]中。

## emulate_precision_casts 做了什么

在触发 tracing 和编译之前设置：

```python
import torch
import torch_npu
import torch._inductor.config as config

config.emulate_precision_casts = True
```

也可以在启动进程前设置环境变量：

```bash
TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1 python repro.py
```

它允许继续使用 FP32 计算和算子融合，但在对应的低精度算子边界插入真实的舍入。
连续加法的行为可以写成以下等价示意，省略了生成代码中的重复转换：

```python
s = a_fp32 + b_fp32
s = s.to(tl.bfloat16).to(tl.float32)
y = s + c_fp32
tl.store(out_ptr, y)
```

其中 `.to(tl.bfloat16)` 执行舍入。随后转回 FP32，只是用 FP32 表示已经舍入的数，
不会恢复被舍掉的尾数。转换可以在寄存器内完成，不要求拆成多个 kernel 或写回全局内存。

当前安装版本的实现分为两步：

1. `torch/fx/experimental/proxy_tensor.py::_maybe_record_pointwise_barrier`，
   第 937–966 行：配置开启时，在 tracing 中记录低精度算子的
   `low_precision_pointwise_barrier` 元数据；分解算子内部的中间步骤不逐一记录。
2. `torch/_inductor/lowering.py::make_pointwise`，第 657–688 行：读取该标记，
   用 `ops.to_dtype(..., use_compute_types=False)` 强制转到实际低精度类型，
   再转换回计算类型。输入和输出边界都可能插入转换。

这里 tracing 的作用是记录“应该在哪里舍入”，Inductor lowering 的作用是把标记实现为转换。
现有证据不支持把默认精度差异判定为 Dynamo 或 FX graph pass 错误删除算子。

[39 的实际生成函数][dsl39]在 add 和 sigmoid 之后均出现 BF16 转换；
[40 的实际生成函数][dsl40]在两次加法之间出现 BF16 转换。
两份产物保留完整函数体，仅省略导入、装饰器及环境 metadata。

## 为什么只关闭 upcast 仍然失败

需要区分 Inductor 生成代码和下层设备编译两个阶段。
这里 upcast 指将低精度值转换为 FP32 后计算；DSL 指生成的 Triton 代码，
TTIR 是 Triton 前端的中间表示，HIVM 产物展示了更接近设备指令的中间表示。

| 配置与阶段 | 本次观察到的行为 |
| --- | --- |
| upcast 开启，默认融合 | DSL 显式将 BF16 load 转成 FP32，连续计算，缺少 eager 的中间舍入 |
| upcast 关闭 | DSL 不再对 load 显式转 FP32；直接 Triton 复现的 TTIR 中，两次加法仍标为 BF16 |
| 下层编译产物 | 连续加法最终变成两个 FP32 向量加法，两者之间没有 BF16 转换 |
| upcast 开启且 emulate 开启 | DSL 显式写出舍入边界，代表用例与 eager 完全一致 |

独立 [Triton repro][repro]绕过 Inductor，包含 `tl.static_assert(ab.dtype == tl.bfloat16)`。
使用 129 个 BF16 值构成的全部 16,641 组输入对，直接复现结果如下：

| 计算模式 | 不满足 `1e-3` 容差的元素数 | 最大绝对差 | 输出是否逐元素等于“加法不舍入”的模型 |
| --- | ---: | ---: | --- |
| `(a + b) + c` | 3,072 | 0.015625 | 是 |
| `(a + b) × sigmoid(a + b)` | 4,868 | 0.015625 | 是；该路径仍保留 sigmoid 输出的 BF16 转换 |

[连续加法 TTIR][ttir-add]中的两次 `arith.addf` 类型为 `tensor<256xbf16>`。
[最终 HIVM 产物][hivm-add]的第一组向量指令则是：

```text
BF16 输入 → FP32
%33 = vadd.s.x(...)       : vector<64xf32>
%35 = vadd.s.x(%33, ...)  : vector<64xf32>
%37 = vcvtff.f322bf16.x(%35, ...)
```

上面是对产物的阅读示意，完整原始语句及操作数类型见链接中的第 41–51 行；IR 的个人路径已泛化，归档时清理了行尾空格。
第一次 FP32 加法的结果直接流入第二次加法，中间不存在 BF16 转换。
这解释了为什么关闭 Inductor 的 upcast 后仍然失败。

下层编译器还提供 `hfusion-legalize-bf16`，说明为“normalize BF16 to FP32”；
但当前未取得该 pass 前后的对应 IR，不能仅凭 pass 名称断言它就是遗漏中间舍入的具体入口。
已确认的是 TTIR 到最终向量指令之间发生了上述变化，尚未定位到具体编译器实现函数。

## 用例分类与 combo 对照

本次实际用例包中，19 个 kernel 的主要计算可分为：

- BF16 `add → sigmoid → mul`：39、47、52、53、58、59、86、88、116、155、156、157、158、159、160。
- 包含 BF16 连续两次加法：40、55、56、60。
- 对照输出：39 中的单次 BF16 加法，53、55、56 中的 FP32 计算链。

这里的编号均指 `triton_poi_fused_<编号>`，不是其他用例包中恰好同名的 kernel。
例如 55 的 BF16 连续加法失败，而其 FP32 sigmoid/mul 输出通过，最大绝对差约为 `5.96e-8`；
39 的四个单次加法输出与 eager 完全一致。

同一组合成输入下，关闭 upcast、关闭 emulate 时：

| combo | 通过用例数 | 失败输出数 | 最大绝对差 | 是否确认实际生成多子 kernel 的 combo |
| --- | ---: | ---: | ---: | --- |
| off | 0 / 19 | 43 / 80 | 0.015625 | 不适用 |
| on | 0 / 19 | 43 / 80 | 0.015625 | 19 / 19 |

两个配置的每个输出是否失败、失败元素数均相同。
这证明问题在 combo 开启后仍存在，而且在 combo 关闭时也已存在；
不能把这些差异归因于 combo 新引入了错误。[基线结果][baseline]保存逐用例对照。

## 开启 emulate 的复测

代表用例的 combo 关闭对照已经完成：upcast 开启、emulate 开启时，
39 的 8 个输出、40 的 2 个输出均与 eager 完全一致，最大绝对差为 0。
见 [39 结果][result39]和 [40 结果][result40]。

本轮 19 个用例复测采用以下配置：

```python
config.triton.codegen_upcast_to_fp32 = True
config.emulate_precision_casts = True
config.combo_kernels = True
```

验证入口为用例工作目录中的 `verify_upcast.py`，调用方式：

```bash
cd <case-workspace>
ASCEND_RT_VISIBLE_DEVICES=<authorized-device> python -u verify_upcast.py \
  --combo on --upcast on --emulate on \
  --output results/emulate_on_upcast_on_combo_on_19
```

该脚本逐个加载 `cases/triton_poi_fused_<编号>/case.py`，为每例启动独立进程和缓存，
使用同一输入分别执行 NPU eager 与 `torch.compile(fullgraph=True, dynamic=False)`，
检查所有输出的 `torch.testing.assert_close(rtol=1e-3, atol=1e-3)`，并记录实际生成的 combo。

输入由 CPU seed 0 生成：`torch.rand(...) * 0.5 + 0.5`，再转成用例指定的 dtype，
保持原 shape 和 stride。所用 `case_runtime` 来自共享的用例运行辅助文件，
不是从原模型采集数据。

复测结果：**19 / 19 个用例通过，80 / 80 个输出通过，失败元素总数为 0**。
所有 19 个用例均从生成代码中确认实际包含多子 kernel 的 combo，未以普通独立 kernel 的通过结果代替。

| 输出 dtype | 输出数 | 最大绝对差 |
| --- | ---: | ---: |
| BF16 | 76 | 0 |
| FP32 | 4 | 5.960464477539063e-8 |

逐用例结果见 [CSV 汇总][retest-csv]，逐输出指标及生成的 combo 信息见 [JSON 证据][retest-json]。
[39 的 combo 函数体][combo-dsl]同时保留了 combo 分支和显式 BF16 转换。

还补做了只改变 emulate 的 39 用例对照，两个配置均开启 upcast 和 combo：

| emulate_precision_casts | 39 的通过输出数 | 最大绝对差 |
| --- | ---: | ---: |
| off | 4 / 8 | 0.0078125 |
| on | 8 / 8 | 0 |

关闭 emulate 后，`buf62、buf63、buf88、buf89` 的 BF16 SiLU 输出再次失败；
其余四个单次加法输出仍通过。[该对照结果][combo-off-casts]确认：即使保持默认 upcast，
combo 开启后仍存在原精度问题，显式恢复舍入边界才让该用例与 eager 对齐。

实测环境为 Ascend 950，编译产物 target 为 `Ascend950PR_9579`，
`torch=2.10.0+cpu`，`torch_npu=2.10.0.post7+gite32cdeb`。
安装包对应 `v2.10.0-1102-combo` 的 `e32cdebf77`。
配置仅在测试进程中修改，没有改动安装包源码。

## 独立复现与限制

在具有对应 torch_npu、Triton-Ascend 的环境中，从仓库根目录运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=<authorized-device> \
  python scripts/repro/bf16_pointwise_rounding_triton_repro.py
```

脚本输入为两个连续一维 BF16 Tensor，各 16,641 个元素；连续加法的第三个输入固定为
`0.50390625`。Triton tile 固定为 256，尾块使用 mask，无 autotune。
输出 eager 对照、最大误差、失败元素数及数值模型匹配结果，并生成 `direct_triton_probe.json`。
脚本用于观察差异，不因观察到预期差异而返回失败状态。

开启 emulate 是验证 eager 舍入语义的配置，不能据此称下层 BF16 运算路径已经修复。
本轮没有测量它对性能、寄存器占用或其他计算模式的影响，也没有推定全部其他模型都会通过。

[source]:
  https://github.com/limbochao/torch-learn/blob/main/artifacts/bf16-rounding-20261104/installed-precision-source.txt
[repro]:
  https://github.com/limbochao/torch-learn/blob/main/scripts/repro/bf16_pointwise_rounding_triton_repro.py
[baseline]:
  https://github.com/limbochao/torch-learn/blob/main/artifacts/bf16-rounding-20261104/baseline-summary.json
[ttir-add]:
  https://github.com/limbochao/torch-learn/blob/main/artifacts/bf16-rounding-20261104/two-adds.ttir.mlir
[hivm-add]:
  https://github.com/limbochao/torch-learn/blob/main/artifacts/bf16-rounding-20261104/two-adds.hivm.mlir
[dsl39]:
  https://github.com/limbochao/torch-learn/blob/main/artifacts/bf16-rounding-20261104/case-39-kernel-bodies.txt
[dsl40]:
  https://github.com/limbochao/torch-learn/blob/main/artifacts/bf16-rounding-20261104/case-40-kernel-bodies.txt
[result39]:
  https://github.com/limbochao/torch-learn/blob/main/artifacts/bf16-rounding-20261104/case-39-combo-off-emulate-on.json
[result40]:
  https://github.com/limbochao/torch-learn/blob/main/artifacts/bf16-rounding-20261104/case-40-combo-off-emulate-on.json
[retest-csv]:
  https://github.com/limbochao/torch-learn/blob/main/artifacts/bf16-rounding-20261104/combo-on-emulate-on-summary.csv
[retest-json]:
  https://github.com/limbochao/torch-learn/blob/main/artifacts/bf16-rounding-20261104/combo-on-emulate-on-results.json
[combo-dsl]:
  https://github.com/limbochao/torch-learn/blob/main/artifacts/bf16-rounding-20261104/case-39-combo-kernel.txt
[combo-off-casts]:
  https://github.com/limbochao/torch-learn/blob/main/artifacts/bf16-rounding-20261104/case-39-combo-on-emulate-off.json
