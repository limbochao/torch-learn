---
title: CICD 性能分析：MobileBERT 剩余参数梯度归约
---

# MobileBERT Train：除 `_5` 外其余归约的逐 kernel 结论

**新增确认 `_9`、`_7`、`_13` 三组编译收益缺口，主要责任仍在 NPU Inductor 的归约计划和候选生成。**
三组采用显式两阶段 ATen 自动 codegen，单次设备耗时分别为 **7.343、20.005、25.957 μs**；
按原调用次数外推，新增约 **28.96 ms/step** 空间。加上此前 `_11` 的最新复测，四组合计约 **50.59 ms/step**。
这是独立小图外推，没有修改框架、替换整网或实测整网提速。

另外两组也完成了实验：`_20` 有分段优化空间，但现有 GPU Eager 聚合数据不足以确认其单项编译收益缺口；
`_0` 的分段方案反而更慢，当前不列为已确认优化点。按要求，本轮未分析 `mul_sum_view_5`。

本轮在 NPU_A5 Docker、`pta_213_cicd` 运行；诊断轮使用物理卡 **2**，同编译路径复核轮使用空闲物理卡 **1**，
未使用卡 4、5、6。安装的五个关键文件再次确认与
`test/cicd_benchmark@b3bd7403518924a7cc1e7c470aa80ccfb4e38120` 一致，见[环境核验][env]。

## 67.462484 ms 如何拆开

下表“原成本”为原整网 CSV 的设备 duration 汇总，已经乘上一步内的全部调用次数。
实测列来自独立重放，不能拿其中一个新值直接减整网历史成本。

| NPU kernel（省略 `triton_red_fused_`） | 次/step | 原成本 / ms | 本轮原始实现 / μs | 同路径仅换 tile / μs | 两阶段 ATen 自动生成 / μs | 当前结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `add_mul_sum_view_11` | 96 | 25.847447 | 247.761 | 87.502 | 22.487 | 此前主问题，已补做同路径复核 |
| `mul_sum_view_9` | 168 | 15.300343 | 93.397 | 29.860 | 7.343 | 新增确认，优先优化 |
| `mul_sum_view_7` | 48 | 9.252472 | 173.668 | 61.922 | 20.005 | 新增确认，优先优化 |
| `add_mul_sum_view_13` | 24 | 8.921167 | 323.062 | 134.058 | 25.957 | 新增确认，优先优化 |
| `add_mul_sum_view_20` | 24 | 5.468952 | 184.514 | 183.639 | 97.704 | 有可达性实测，GPU 收益缺口待确认 |
| `add_sum_view_0` | 1 | 2.585491 | 2547.143 | 3540.150 | 2733.418 | 已试方案均未胜过原实现 |
| `mul_sum_view_5` | 1 | 0.086612 | — | — | — | 用户指定排除 |

四组优先点原成本合计 **59.321429 ms，占七组成本 87.93%**。
这不表示 59.321429 ms 可以全部回收；后面的目标仍保留归约计算、数据读取和最终写回成本。
七组 67.462484 ms 与原粗筛 34.279800 ms 预算分别是“当前成本”和“保守筛选预算”，都不是已实测收益。

完整数值、真实 winner 和候选见[同路径复核][verified]、[诊断轮][diagnostic]。

## 为什么符合“NPU 编译提升比低于 GPU”

仍使用 `r = 1 - Inductor/Eager`，比较相同数学输出，按完整 GPU 融合实现收费。
原 GPU CSV 没有逐次 shape，因此使用保守上下界，不用全部 `sum` 的平均耗时冒充目标 shape。

| 新增确认点 | NPU Eager 上界 / ms | NPU Inductor / ms | NPU 提升比上界 | GPU Eager 下界 / ms | GPU Inductor 上界 / ms | GPU 提升比下界 | 保守缺口 B / ms | 占整网缺口 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `_9` | 3.655412 | 15.300343 | −318.57% | 1.908480 | 4.170746 | −118.54% | 7.311896 | 6.23% |
| `_13` | 0.959258 | 8.921167 | −830.01% | 0.735744 | 1.879609 | −155.47% | 6.470545 | 5.52% |
| `_7` | 1.635958 | 9.252472 | −465.57% | 1.144320 | 2.406344 | −110.29% | 5.812283 | 4.96% |

三项均满足 **NPU 提升比的上界仍低于 GPU 的下界**，所以能够确认收益缺口。
表内 GPU 数值是刻意放宽的下界，不表示 GPU 的真实提升比就是负值。
`B = N_I - N_E × G_I/G_E`；“占整网缺口”分母为原整网 **117.294340 ms**。
这些筛选边界与后面独立小图的可达性实测分开使用。

NPU Eager 的取法有明确边界：

- `_9`：收取每步全部 **362 个**“两次矩阵乘后跟随的线性层 bias sum”，包括其他通道宽度。
  这是目标 168 次的费用超集，只收 sum，得到 3.655412 ms 上界。
- `_7`、`_13`：每个 step 都有 193 个 `Sum → Mul → Mul → Sum` 窗口。
  首个为 512 通道 head；随后 24 层各 8 个 NoNorm 窗口，按 FX 与 wrapper 的分支顺序定位。
  每层顺序为 `_7, _11, _11, _11, _13, _11, _7, _20`。
  `_7` 取第 1、7 个，`_13` 取第 5 个及其紧邻的梯度 Add；均额外收取了不在目标输出内的 `dh` Mul。
  这是依据执行序列匹配，不声称原 CSV 本身记录了逐次 shape。

GPU Eager 下界：每次必需操作分别为 `_9: 1 Sum`、`_7: 1 Mul + 2 Sum`、
`_13: 1 Add + 1 Mul + 2 Sum`，乘原 GPU CSV 的全局最小耗时。
GPU 编译费用则收以下完整 kernel，包含附带输出和其他工作的成本：

| NPU 点 | GPU 对应完整实现，均包含全部 `sum_view_9` final sum |
| --- | --- |
| `_9` | `add_mul_sum_view_11`、`add_mul_sum_view_12`、`add_mul_sum_view_18` |
| `_7` | `mul_sum_view_8`、`add_mul_sum_view_12` |
| `_13` | `add_mul_sum_view_12` |

GPU `_11` 和 `_18`、`_12` 内部分别包含多个参数梯度或 bias 梯度，不能按名称后缀一一对位。
前一版把 NPU `_11` 的 attention 分支对到了 GPU `_12`，本轮纠正为 GPU `_18`。
修正后的 `_11` 保守预算为 **15.501171 ms**，见[已修正的首项分析](mobilebert-parameter-reduction-20261103.md)。
这项修正影响四方费用映射，原 NPU 单 kernel 根因和性能实测不变。

计算脚本、每个窗口的原 CSV 行号、文件 SHA256 见[分析脚本][analyze]、[窗口明细][windows]、[边界结果][bound]；
FX 与两侧 wrapper 摘录见[映射证据][mapping]。GPU 融合费用及 NPU 超集存在重叠，不能直接相加这些分项预算。

## `_9`：纯 bias sum，长归约之外还有重复扫描

**完整名称：`triton_red_fused_mul_sum_view_9`。** 名称含 `mul`，但实际 DSL 没有乘法。
它读取 contiguous FP32 `[16384,128]`，按 token 求和，输出 `[1,128]`，一步 **168 次**。
乘法由前一个 pointwise kernel 完成；这里只剩 `sum_r A[r,x]`。

原 winner 为 `X0BLOCK=3, X0BLOCK_SUB=1, R1BLOCK_SUB=8192`，grid 为 43。
每个 program 最多循环处理 3 个通道，每个通道再串行处理 16384 个 token。
实际 tile 只有 1 个连续通道，load 地址为 `x + 128*r`。
**24 个自动候选仅覆盖 X0BLOCK 3、4**，遗漏实测更快的 `16/16/256`。
同 signature、metadata 和编译入口，仅换此 tile 就从 93.397 降到 29.860 μs，快 **3.13 倍**。

继续显式分段为 `[128 个 token 组,128 个通道]` 的部分和，再 final sum，
自动生成两个 kernel，合计 **7.343 μs，快 12.72 倍**。
因此先优化 **NPU Inductor `ir.py::num_splits` 的分段计划、`TileGenerator` 的候选范围**。
已证明下层能够执行更快实现，没有将此问题归给 Triton_ascend 或 npuir 的证据。

另有 **24 次重复归约**：第一层 wrapper 的 `_9.run(buf64,buf67,...)` 在原文件第 2176 行；
第 2299 行 `_7.run(buf64,addmm_346,buf98,buf99,...)` 的第一路再次求同一个 `buf64.sum`。
两个输出的 shape 分别为 `[1,128]` 和 `[1,1,128]`，均最终返回参数梯度；输入在两次调用之间没有改写。
这组关系在 24 层重复。GPU `_12` 则将同一个 partial sum 值 `tmp24` 写到两个输出，复用归约计算。

这部分应检查 **Inductor IR 对 reshape/keepdim 后等价归约的规范化、scheduler 融合及 codegen 表达式复用**。
现有证据定位了重复计算边界，尚未在 scheduler 上插桩确定具体哪条拒绝规则。
优化应保留两个梯度输出各自的存储，不能直接让两个参数的 `.grad` 共享可变 buffer。

按对应 24 次原 profile 精确收费，这些 `_9` 调用合计 **2.168733 ms/step**，占整网缺口 **1.85%**；
这是取消重复扫描的成本上限，尚未扣除必要的独立写回。
若先完成分段，剩下只约 `24 × 7.343 μs = 0.176 ms`，不能再把原 2.169 ms 加到分段收益上。

原始 [DSL][dsl9] 与[自动两阶段 codegen][stage9]均已归档。

## `_7`：两路 NoNorm 参数梯度，缺少分段

**完整名称：`triton_red_fused_mul_sum_view_7`。** 输入 `g,h` 均为 FP32 contiguous `[128,128,128]`：

```python
dbeta = g.sum((0, 1), keepdim=True)
dgamma = (g * h).sum((0, 1), keepdim=True)
```

一步 **48 次**，每层 2 次。GPU 对应 `_8` 的 24 次和 `_12` 内另 24 次，均先做 128-token partial。
NPU 原 winner 为 `3/1/4096`，24 个候选仍只有 X0BLOCK 3、4。
同路径固定 `32/32/128` 后 **173.668 → 61.922 μs，快 2.80 倍**；
两阶段 ATen 自动生成 **20.005 μs，快 8.68 倍**。

具体责任为 **IR `num_splits` 未把小输出长归约展开成 partial/final，以及 `TileGenerator` 漏候选**。
自动方案实际为 **1 次 partial + 2 次 final，共 3 次 launch**；不能把两个 kernel 名当成两次调用。
手写两路 final 合并的诊断版为 **10.995 μs**，是进一步调优参考；先以自动 codegen 的约 20–22 μs 为工程目标。
首轮与复核轮的自动结果分别为 19.875、20.005 μs，量级一致。

见[原始 DSL][dsl7]、[自动两阶段 codegen][stage7]。不同输出并非模型外新增运算；
GPU 将 `dh=g*gamma` 等相邻输出融合，NPU 采用不同边界，四方费用已对这些附带工作保守收费。

## `_13`：额外重计算不是 323 μs 的充分解释

**完整名称：`triton_red_fused_add_mul_sum_view_13`。** 每层 1 次，一步 **24 次**。
四个大输入为 FP32 contiguous `[128,128,128]`，另有两个 `[128]` 通道参数：

```python
g = a + b
h = c + (d * scale + bias)
dbeta = g.sum((0, 1), keepdim=True)
dgamma = (g * h).sum((0, 1), keepdim=True)
```

`d*scale+bias` 和残差 Add 是 FX graph 已存在的前向表达式重计算，Eager 反向通常使用保存的 `h`；
GPU `_12` 同样含这些重计算，不能仅因 NPU 多见 Add/Mul 就断定其错误或为新增模型计算。
本轮没有进一步确认具体 AOT 分区规则，也不把重计算本身单列为已确认缺陷。

真正的性能证据是保持全部表达式不变：原 winner `3/1/4096` 的 **323.062 μs**，
仅换 `16/16/256` 降到 **134.058 μs，快 2.41 倍**；
显式两阶段 ATen 自动 codegen 降到 **25.957 μs，快 12.45 倍**。
诊断版仍执行全部重计算，合计 **15.349 μs**。

因此主要优化入口是 **NPU Inductor IR 长归约拆分与 TileGenerator**。
它比 `_7` 多读两个大输入并执行更多 pointwise，合理目标应单独实测，不能复用 `_11` 的 13 μs 目标。
自动方案两轮为 **40.851、25.957 μs**，记录到的 selected config 相同，运行上下文存在波动，
本轮未继续定界该波动，不能归为候选变化。初期按 **不高于 41 μs** 验收，
约 26–30 μs 作为进一步稳定测量后的目标。均为 1 partial + 2 final。

见[原始 DSL][dsl13]、[两阶段 codegen][stage13]、两轮[完整配置][verified]。

## `_20`：分段可快约一倍，但还不能按 GPU 缺口立项

**完整名称：`triton_red_fused_add_mul_sum_view_20`。** 五个大输入均为 FP32 `[128,128,512]`：

```python
g = ((a + b) + c) + d
dbeta = g.sum((0, 1), keepdim=True)
dgamma = (g * h).sum((0, 1), keepdim=True)
```

一步 **24 次**。GPU 对应 `_19` 的 23 次，以及带 embedding 附加工作的 `_21` 的 1 次，再完成 final sum。
NPU 原 20 个候选的 X0BLOCK 已是 **16**，复核 winner 为 `16/16/128`，grid 32。
原实现 **184.514 μs**；同路径换 `32/32/128` 仅 **183.639 μs**，不足 1%，
不支持把它归为前三组相同的“漏掉宽通道 tile”问题。

显式两阶段自动 codegen 为 **97.704 μs**，约快 **1.89 倍**，说明 **IR 分段计划**仍有优化空间。
相同输入的 Eager 对照为 225.300 μs；这个 NPU 控制实验用于说明可达性，不能代替 GPU 提升比对比。

原整网 schedule 匹配的 NPU Eager 费用上界为 **5.522489 ms**，含 3 次梯度 Add、两次 Sum、
参数梯度 Mul，并额外收取一次 `dh` Mul；NPU Inductor 为 **5.468952 ms**，提升比上界约 **+0.97%**。
GPU Eager 只按全局最小耗时收费，得到 1.062912 ms 下界；完整融合实现与 final sum 收到 7.050452 ms 上界。
这些边界**不能证明** NPU 提升比低于 GPU；需要 GPU 侧这些 512 通道 Add/Mul/Sum 的逐次 shape/timing 或同边界实测。

若单独采用自动分段，同轮差值外推为 **2.083 ms/step**，约占整步 0.71%；
这是有条件的 NPU 优化空间，**不计入当前已确认 GPU 缺口的四组优先收益**。
手写诊断版首轮 78.655 μs，仅作进一步上限参考，不能把不同轮的原始基线混用来承诺收益。
见[原始 DSL][dsl20]、[自动 codegen][stage20]。

## `_0`：通道很宽，当前分段没有收益

**完整名称：`triton_red_fused_add_sum_view_0`。** 两个输入为 FP32 contiguous `[128,128,30522]`，
输出 `(a+b).sum((0,1),keepdim=True)`，一步 **1 次**。
逻辑输入读取约 4.001 GB，远大于前述 128 通道 kernel。

84 个候选的 X0BLOCK 为 552、720、960；winner 为 **552/552/32**，grid 已有 **56 个 program**，
连续通道 tile 也很宽。这里并不存在“只有单通道 tile、program 不足”的同一证据。

- 原实现及同配置重放分别为 **2547.143、2547.690 μs**。
- 仅换本轮测试的 tile，最好也要 **3540.150 μs**，更慢。
- 显式两阶段 ATen 为 **2733.418 μs**，比原实现慢 **7.31%**。
- 三种手写分段分别约 2741.898、2664.295、2705.270 μs，均未胜过原实现。
- 同边界 Eager Add+Sum 为 5689.651 μs；本次 NPU 编译已有约 **55.23%** 提升。

GPU 对应的大融合 bias-sum kernel 还重算 loss/log-softmax backward 表达式，且输入布局有 30528 的行 stride；
不能把其 2.686653 ms 与 NPU 2.585491 ms 直接相减判定问题，更不能据此判断两个设备的编译提升比。
本轮未找到胜过原实现的方案，也未确认单项 GPU 收益缺口，**优化收益计 0**。
IR 策略应对输出宽度、布局和额外中间流量做选择，不能给所有长 outer reduction 无条件分段。
见[原始 DSL][dsl0]、[分段 codegen][stage0]。

## 实测如何排除编译入口和计时干扰

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

原 IR `num_splits` 对非单元素输出直接返回 `DEFAULT,1`，使这些小输出归约缺少 partial/final 计划；
原源码证据见[首项归档的源码摘录][source]。
同一个 Triton_ascend/npuir 后端已经执行了显著更快的上游计划，当前首先应改 Inductor。
这不等于证明下层已无优化空间；本轮没有发现必须先修下层的固定 DSL/config 缺陷。

可复现脚本为 [run_family.py][run]，命令及计时说明见[README][readme]；
原始和固定配置 TTIR、TTAdapter、生成 C++ 见[复核 IR][ir]，身份和精度汇总见[validation][validation]。

## 整网预期收益与改动范围

使用复核轮 `单次原耗时 − 单次方案耗时`，乘历史 `调用次数/step`；不直接跨批次相减，也不缩放其他算子。

| 已确认点 | 仅换候选：外推节省 / ms | 自动分段：累计外推节省 / ms | 自动分段占整网缺口 | 建议首阶段验收目标 / μs |
| --- | ---: | ---: | ---: | ---: |
| 已分析 `_11` | 15.385 | 21.626 | 18.44% | ≤25 |
| 新增 `_9` | 10.674 | 14.457 | 12.33% | ≤8 |
| 新增 `_7` | 5.364 | 7.376 | 6.29% | ≤22 |
| 新增 `_13` | 4.536 | 7.131 | 6.08% | ≤41，进一步争取 26–30 |
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

[env]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/environment.json
[verified]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/verified_result.json
[diagnostic]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/diagnostic_result.json
[bound]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/profile_bound.json
[windows]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/eager_windows.csv
[mapping]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/mapping_evidence.txt
[analyze]: ../../../../scripts/repro/cicd/performance/mobilebert_reduction_20261103/analyze_family.py
[source]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/source_evidence.txt
[run]: ../../../../scripts/repro/cicd/performance/mobilebert_reduction_20261103/run_family.py
[readme]: ../../../../scripts/repro/cicd/performance/mobilebert_reduction_20261103/README.md
[ir]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/ir_verified
[validation]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/validation.json
[dsl9]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/mul_sum_view_9.py
[dsl7]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/mul_sum_view_7.py
[dsl13]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/add_mul_sum_view_13.py
[dsl20]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/add_mul_sum_view_20.py
[dsl0]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/add_sum_view_0.py
[stage9]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/staged_9_output_code.py
[stage7]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/staged_7_output_code.py
[stage13]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/staged_13_output_code.py
[stage20]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/staged_20_output_code.py
[stage0]: ../../../../artifacts/cicd/performance/mobilebert-reduction-20261103/family/staged_0_output_code.py
