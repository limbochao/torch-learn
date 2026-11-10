---
published: false
---

# 2.13 Combo 同步差异修复与失败项复验

优先范围是 2.10 基线表中已通过的 60 个源方法。总清单仍为 105 个源方法、116 行表格，
嵌套函数不独立计数。foreach 沿用调试阶段每个源方法最多两个参数实例的抽样策略，
本次只执行此前失败的参数实例，以及取消旧 NPU skip 后的实例，没有全量重跑。
历史 PASS 保留原验证版本；不能把逐实例合并后的汇总理解为所有方法已在最新 overlay 上复测。

[当前表](inductor-combo-tests-native-v2.13-sync-64dc85c.csv)；
[2.10 基线](inductor-combo-tests-liulm-v2.10.0-combokernel-dev.csv)；
[复测脚本说明](../../../../scripts/tests/combo_kernel_210/README-213.md)。

## 修复范围及实际版本

NPU 分支为 `fix/combo-sync-interfaces-v2.13`，基础包提交 `64dc85c99c`，
保留此前同步接口修复至 `5a5bda0ae3`。本次新增三个独立提交：

| 提交 | 问题 | 修改后的行为 |
| --- | --- | --- |
| `bf2f56c6e4` | 动态 benchmark 调用已移除的 `size_hint` | 使用 2.13 的 `optimization_hint`，保留动态轴顺序和 unbacked fallback |
| `257e31bde7` | persistent reduction body 引用了未定义的静态块参数 | 恢复常量 reduction 轴生成 `R*BLOCK_SUB` 的条件 |
| `bb5e1d0a84` | Combo 回退后输入被重复 `del` | 在普通 kernel 边界重算 `last_usage`，每个输入在最后使用后释放一次 |

只覆盖相关 Python 文件，未重新构建 wheel，版本字符串仍显示 `2.13.0+git64dc85c`。
两批失败项复验先固定使用 `257e31bde7`；结束后才叠加 `bb5e1d0a84` 验证生命周期与环境失败。
每批都核对安装源码执行前后的哈希；版本不能只看本地 HEAD 或 wheel 字符串。
测试快照为 `c1ce068`，直接运行原生 `python -m pytest -vs`，不使用自定义插件。

## size_hint 为什么需要改

`NPUComboKernel.kernel_benchmark_extra_args()` 根据动态轴长度生成 benchmark 的额外标量参数。
原代码仍调用：

```python
V.graph.sizevars.size_hint(length, fallback=config.unbacked_symint_fallback)
```

社区 2.13 的 `SizeVarAllocator` 不再提供该方法，优化提示使用 `optimization_hint()`。
社区 `ComboKernel.kernel_benchmark_extra_args()` 也已采用新接口。
因此这是接口适配遗漏，不是 2.13 不支持动态 shape。修改仅替换方法名，
保留 NPU 收集的动态轴列表、参数顺序、长度表达式和 unbacked fallback，
没有为优化提示额外加入 shape guard，也没有把估算值写进实际调用的运行时参数。

影响范围是动态 benchmark 的代表尺寸选择；实际 launch 仍由原符号参数驱动。
风险主要在动态和 unbacked 表达式的估算行为，需要以相应用例的编译、执行和数值检查验证。

## persistent reduction 条件是漏移植吗

从生成协议看，属于移植时遗漏，找不到能够替代它的参数生成逻辑。
2.10 的条件是：

```python
if tree.is_no_loop_axis or (tree.is_reduction and sub_kernel.persistent_reduction):
```

移植后仅保留 `tree.is_no_loop_axis`。这两个条件并不等价：persistent reduction 轴未必同时标记
`is_no_loop_axis`。这类轴不再生成循环，但 body 仍需要静态 reduction 块大小，
相应 `R1BLOCK_SUB` 或 `R2BLOCK_SUB` 又没有作为 autotune 参数传入，因而出现未定义名称。

此前定向日志已确认 `test_combo_kernel_per_config_subkernel_per` 被分为一个 `[2]` Combo 分区，
随后因 `R1BLOCK_SUB is not defined` 回退为两个普通 kernel。它不是 NPU 主动收紧融合数量。
`test_combo_kernel_per_config_subkernel_red` 在回退前也有同类 `R2BLOCK_SUB` 错误。
[修复前证据](../../../../artifacts/combo-213-baseline-pass-diagnosis-20261110/evidence.json)。

修复恢复第二个条件，仍只对能化简为常量的轴生成定义：unified SIMT 按原规则取 2 的幂，
其他类型保持实际长度，符号 reduction 轴仍走原有处理。社区 2.13 的同名静态参数生成方法
也保留 `tree.is_reduction and sub_kernel.persistent_reduction` 分支；NPU 使用自己的轴命名，
对齐的是生成协议，而不是逐字复制 CUDA DSL。

实际重生成的 `test_combo_kernel_per_config_subkernel_per` wrapper 只包含一次 `async_compile.triton`，
两个子 kernel body 分别有 `R1BLOCK_SUB: tl.constexpr = 64` 和 `= 512`。
这直接证明静态参数定义和单个 Combo 主路径已恢复，不是仅凭放宽数量断言获得 PASS。

恢复该条件本身不增加 kernel 或额外计算；它使本应可编译的候选不再因缺少参数而退出。
融合及 tiling 选择可能随候选恢复而变化，功能复验不能替代独立的性能测量。

## 为什么还需要修复重复释放

恢复静态参数后，`test_combo_kernel_per_config_subkernel_red` 仍报 `UnboundLocalError`。
它存在另一层独立问题：Combo 不能通过编译校验时会进入普通 kernel 回退；修复前定向日志还记录了
scalar pointer 与 block value 不匹配的候选失败，
这类 Combo 候选在 2.10 也曾回退，并不能认为 2.10 PASS 就说明主路径已支持。

2.13 普通非线性回退会按重新分组后的 `last_usage` 释放输入；移植的 Combo 外层却保存整组
`pending_frees`，逐个生成普通 kernel 后又恢复并释放一次。社区 `WrapperCodeGen.codegen_free()`
对 `InputBuffer` 直接写出 `FreeLine`，不登记到可复用 buffer 的 `freed` 集合，所以外层无法自动去重：

```python
kernel.run(arg0_1, ...)
del arg0_1
kernel.run(arg1_1, ...)
del arg1_1
del arg0_1  # 再次释放，触发 UnboundLocalError
```

修复沿用现有非线性分组的生命周期算法：从外层实际还需使用的 buffer 集合开始，
逆序为普通子 kernel 计算 `last_usage`，考虑 alias 与 mutation 的真实名称；
再正序仅提交当前子 kernel 的最后使用集合。普通 codegen 负责释放，外层不恢复已消费的集合。
这不是禁止释放或全局去重，也没有新加回退来掩盖错误。

实际重生成的 reduction 回退 wrapper 中 `del arg0_1`、`del arg1_1` 各只有一次，
原失败方法的数值比较已通过。该 wrapper 可复用同一定义执行两个普通 kernel launch，
不能把 `async_compile.triton` 定义数 1 误认为它恢复了单次 Combo launch。

风险集中在共享输入、mutation 和 alias 的生命周期；重算必须保留后续 kernel 仍需使用的输入。
算法与本分支已有 `_codegen_nolinear_groups()` 以及社区调度器的逆序最后使用分析一致；
社区 CUDA 没有完全相同的 NPU 两层回退组合，因此是 NPU 调度适配，不是直接搬运上游补丁。

## 用例适配同步的边界

`c1ce068` 取消随机数 Combo=True 和 filter-CPU 用例的旧 NPU skip，并尝试此前跳过的 store-mask 用例。
动态 grid 用例在 NPU 上仍执行两组输入并比较 eager/compiled 输出，但不按 CUDA 的 `traceEvents`
及 `args.grid` 结构读取 NPU profiler。与 2.10 基线通过方法对应的数量及 CUDA `FileCheck` 检查，
仅在非 NPU 后端保留；NPU 数值、mutation 和梯度检查不变。

这与旧 2.10 功能验证策略一致，功能 PASS 不等于证明融合数、CUDA DSL 或新的分块算法均已支持。
未重合的新功能用例继续保留原检查；不通过删除断言把能力缺失改成 PASS。

## 复验结果

2026-11-10 在 NPU 环境完成三批定向复验：

| 批次 | 代码 overlay | 范围 | 方法结果 | 参数实例结果 |
| --- | --- | --- | --- | --- |
| 2.10 重合优先批次 | `257e31bde7` | 19 个失败方法及 2 个取消 skip 的方法，仅执行未通过实例 | 17 PASS / 4 FAIL | 17 PASS / 4 FAIL |
| 其余失败项 | `257e31bde7` | 35 个失败方法及取消 skip 的 store-mask | 1 PASS / 35 FAIL | 1 PASS / 37 FAIL |
| 生命周期与环境重试 | `bb5e1d0a84` | 4 个重复释放方法、2 个设备异常方法 | 6 PASS | 6 PASS |

前两批共覆盖 57 个独立方法、59 个不同参数实例；最后追加 6 个已失败实例。
共 65 次参数实例执行，没有重跑历史 PASS。每批安装的 Inductor Python 源码前后哈希均一致。
三个阶段使用独立新缓存；最后一批使用物理卡 6、2、3，设备异常未在换卡后复现。
最初的动态二维 blocking 空句柄错误不能仅凭一次换卡通过就确定底层根因，但不再作为当前稳定失败分析。

合并后的 105 个方法为 **68 PASS、33 FAIL、4 SKIP**；314 个已选参数实例为
274 PASS、35 FAIL、5 SKIP。原清单还有 150 个未选择的泛化实例，未把它们记为 SKIP 或 PASS。
其中 14 个方法仍是此前每方法两个代表实例的抽样 PASS，不能声明全参数通过。
初始 44 个 PASS 方法没有重跑，本次新增转为 PASS 的方法有 24 个。

2.10 已通过的 60 个优先方法现在是 **59 PASS、1 FAIL、0 SKIP**；
唯一失败是下一节的 `test_helper_fn_defined`。原先 12 个方法的 `size_hint` 接口异常均已消除，
其中 11 个通过、1 个进入 cumsum/helper 的后续代码检查。
随机数 Combo=True、filter-CPU、动态 grid、类型提升、reduction 回退及 store-mask 均已通过相应定向实例。

[逐实例与分阶段证据](../../../../artifacts/inductor-combo-tests-native-v2.13-sync-64dc85c-failure-fixes/)；
[60 方法对照](../../../../artifacts/inductor-combo-tests-native-v2.13-sync-64dc85c-failure-fixes/baseline-pass-comparison.json)；
[生成代码：静态 reduction 参数](../../../../artifacts/inductor-combo-tests-native-v2.13-sync-64dc85c-failure-fixes/generated/per-config-persistent-output-code.py)；
[生成代码：输入释放修复后](../../../../artifacts/inductor-combo-tests-native-v2.13-sync-64dc85c-failure-fixes/generated/red-fallback-output-code.py)。

表格保留 116 行原顺序和原有五列。实例列注明复验提交和数量，PASS 当前分析为“—”。
逐报告的 `validation` 保留实际代码版本；旧 PASS 与两批 overlay 的结果没有伪装成同一次全量运行。

## 重合项中 cumsum 的特殊差异

`test_helper_fn_defined` 在修复尺寸提示后通过数值比较，但仍失败于 helper 定义数量预期 1、实际 0。
实际生成 wrapper 调用 `torch.ops.aten.cumsum.default`，没有 `tl.associative_scan` 或相应 helper。
当前 `lowering.py::_register_npu_inductor_fallbacks` 为非标量 cumsum 注册了 `fallback_cumsum`，
2.10 基线没有这项覆盖，因此两者的 lowering 路线不同，不能通过重新打印 helper 修复。

上游 NPU 仓库已有 `ac0e1564ca`（port Triton cumsum support to 2.13.0），当前同步分支尚未包含它。
该提交取消 cumsum 的 fallback/override，并适配 scan 的 mask、shape propagation 和 `_lift_helper` 新接口。
所以不能说 2.13 上游没有支持；准确说法是当前验证基线尚未包含这项能力提交。
本轮没有随意移除 helper 断言，也没有把普通 ATen cumsum 的数值正确当作 Combo scan 能力通过。

## 未重合失败的判读

- CUDA TMA descriptor、PDL：NPU 生成器没有对应 CUDA 功能；测试头部模拟 GPU capability，
  使 CUDA 条件未能准确排除 NPU。当前失败不能作为通用 Combo 数值错误。
- 独立子 kernel block/default-config、逐子 kernel reduction-hint 和 metadata：
  NPU 使用 `combo_subkernel_meta`、自身轴名及 grid 协议，未完整实现社区新增的 CUDA 格式约定。
- sequential autotune、metadata grouping、coordinate descent：测试观察社区 runtime 的日志或行为，
  NPU 实际使用自己的 runtime。需分别核对功能接入和测试日志目标，不能仅以未出现社区日志认定算错。
- optimize_mem、disable_ftz、带宽/FLOP metadata：NPU 当前 Combo 元数据没有全部携带社区新增字段，
  其中训练前向的 optimize_mem 还涉及 benchmark 输入保护，值得进一步核对，不能统称无关字段。
- profile-bandwidth：`NPUDebugAutotuner` 透传 `with_profiler`，父类 `NPUCostModelAutotuner`
  不接受该关键字。这是实际 Python 构造接口错误，需单独修复，不能归为仅格式不同。
- channels_last：用例 eager 的 `contiguous(memory_format=channels_last)` 就失败，未到 Combo 编译。
- provenance：`cppCodeToPost` 映射没有达到测试预期，需要补追踪接入；不是已确认的数值错误。
- profiler/grid、Grid1D 和轴名文本断言：部分用例比较 CUDA 生成代码格式；须核对数值断言是否先通过。
- 峰值内存：阈值收紧与禁用 Combo 的实测峰值未形成预期差距；是否与融合决策、回退或计量有关仍待定位，
  不能单凭峰值相等断言 NPU 不支持。


## 剩余处理优先级

1. 优先补齐重合项需要的 cumsum 能力：先评估合入上游已有 `ac0e1564ca`，而不是删除 helper 断言。
   当前数值是外部 ATen 路径通过，Triton scan/Combo 覆盖仍未完成。本轮已完成原因定位，未额外搬入整项能力。
2. 未重合项优先修复 profile-bandwidth 的 `with_profiler` 构造接口错误，这是确定的执行阻塞。
   provenance 的追踪接入、训练前向 optimize_mem 的语义也值得单独核对。
3. 独立分块、sequential/grouping autotune、资源/带宽 metadata 属于后续能力对齐，
   先明确 NPU 对应协议，不能只为通过字符串检查照搬 CUDA 字段。
4. TMA/PDL 和 CUDA profiler/grid 专用检查应区分后端适用性；
   channels_last 是 eager 支持问题；峰值内存差异继续保留为待定位的性能/策略问题。

本轮没有性能 benchmark 结论。尺寸接口与静态参数修复不直接增加计算，
但恢复候选会改变 autotune 或融合选择；生命周期修复也需继续关注共享输入和 mutation 的回归范围。
