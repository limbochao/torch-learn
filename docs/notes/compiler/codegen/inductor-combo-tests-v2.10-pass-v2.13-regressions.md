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

当前 NPU 开发分支统一使用 `sync/combo-kernel-master`。
2026-11-11 将 `fix/combo-sync-interfaces-v2.13` 的 7 个修复提交汇总到该分支后，
再将基线 `6c43e4db23` 之后的 8 个 Combo 提交通过 rebase 合并为 `38f959f435`，并更新 origin。
新提交的父提交是 `6c43e4db23`，源码树与合并前 `bb5e1d0a84` 完全一致；后续生产修复也提交到该分支。
合并前提交保留在本地备份分支 `backup/sync-combo-kernel-master-before-squash-20261111-bb5e1d0a84`。
随后将这一个汇总提交 rebase 到新拉取的 master `4939db51db`，当前顶端提交为 **`d12230b556`**，
其父提交就是该 master，分支只领先一个 Combo 提交，并已通过带旧提交校验的 force-with-lease 更新 origin。
唯一冲突位于 Triton 常量/config 生成：保留上游提前绑定 equal-to-one 常量及所有 kernel 生成 configs 的实现，
移除本地重复导入和旧的条件生成逻辑。`range-diff` 已核对其他 Combo 改动得到保留。

当前基线同时带入上游 cumsum `ac0e1564ca` 和带宽调试器 `6fe3d7b969`，
不再需要在 Combo 提交内重复实现这两项上游修复。
已完成 13 个改动文件的 Python 语法、差异空白及上游 scan/DebugAutotuner/常量生成保留检查；
本地缺少 torch、sympy 和 Triton，未运行依赖这些库的用例，也未新增设备复测。
原汇总提交保留在 `backup/sync-combo-kernel-master-before-rebase-20261111-38f959f435`。

以下测试结果和源码评估对应原验证快照，保留实际验证版本；本次 rebase 不将历史 FAIL 改为 PASS。

验证时的分支为 `fix/combo-sync-interfaces-v2.13`，基础包提交 `64dc85c99c`，
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

## 剩余失败是否可修复：2026-11-11 源码评估

本节使用上一节的实际执行结果，另行核对当前 NPU 分支 `bb5e1d0a84`、
本地社区源码 `887a5ee68a4`、Triton Ascend 源码 `872fa1ed6` 及已有上游补丁。
**这次只做可修复性分析，没有新增生产修复，也没有重新进行设备测试。**
“可修复”表示已有可落地的修改路径，不表示修改已完成或用例已转 PASS。

重合的 60 个源方法仍是 **59 PASS、1 FAIL**。未重合的 45 个源方法是
**9 PASS、32 FAIL、4 SKIP**。以下分类按源方法计数，不按参数实例计数。

### 重合的唯一历史失败：当前已同步 scan，待设备复验

`test_helper_fn_defined` 有现成 scan 实现可供评估，不需要自行重写算法；
远端最新 master 已采用这套方案；需要同步完整实现并验证，不能仅凭 helper 数量断言单独删除 fallback。
本地已取得的上游提交 `ac0e1564ca` 包含 5 个生产文件，合计增加 18 行、删除 19 行；
另外的测试文件不计入这一修改量。生产修改包括：

1. 移除 `aten.cumsum` 的 override/fallback，使支持场景重新进入 Triton scan lowering。
2. 将 `_lift_helper` 的旧参数数量接口改为 2.13 所需的实际值列表。
3. 按 NPU 发出的轴名生成 mask，传递 scan 中间值的 shape，并使用正确的 scan 维度。
4. 在 NPU runtime 重新导出社区 `split_scan`。

只移除 fallback 会把原本未适配的 scan 生成路径暴露出来，不能视为完整修复。
本次对补丁执行了只读 `git apply --check`：前四个生产文件的补丁上下文可应用；
runtime 文件的导入块因当前分支已有 grid 别名等改动而冲突，需要手动加入 `split_scan`，
所以不建议未经检查直接整体 cherry-pick。

修改与 NPU 上游已有修复对齐。风险集中在 scan 的 dtype 提升、非整块尾部 mask、维度和动态 shape；
已有数值 PASS 是外部 ATen 路径的结果，不能替代修复后 Triton scan 的验证。
验收至少应重新执行原 `test_helper_fn_defined`，同时核查生成代码确实进入 scan、helper 定义一次且数值正确。

### cumsum fallback 的历史用途与回退记录

这部分在 2026-11-11 进一步核对 git 历史、GitCode 公开 PR/Issue 正文与讨论。
来源与已知用途如下；“没有公开原因”表示查到的记录未提供，不能推定不存在历史问题。

| PR / 提交 | 已确认的变更与用途 |
| --- | --- |
| [!20936](https://gitcode.com/Ascend/pytorch/merge_requests/20936) / `511183ea67` | 本地可追溯历史中最早加入专用 cumsum lowering；整数/bool 未指定 dtype 时用 int32，非标量调用 fallback_cumsum。PR 为整体 Inductor 2.6 集成，没有说明 cumsum 专属故障或为什么不能使用 scan |
| [!25033](https://gitcode.com/Ascend/pytorch/merge_requests/25033) / `7f7312eca8` | 将这段专用 lowering 带入 master；是当前主体代码的 blame 来源 |
| [!29301](https://gitcode.com/Ascend/pytorch/merge_requests/29301) / `0b6e500d8f` | 明确修复 A5 cumsum 输出 dtype，把设备分支改为 A5 使用 int64、其他设备仍 int32；说明 override 承担了真实兼容逻辑 |
| [!41858](https://gitcode.com/Ascend/pytorch/merge_requests/41858) / `72ddcb1b5b` | 将 Inductor 适配到 2.13，保留 fallback；当前 dtype 判断使用 is_ascend950 |
| [!45701](https://gitcode.com/Ascend/pytorch/merge_requests/45701) / `944989f5e9` | 2.10 曾开放 Triton cumsum，移除专用 lowering/fallback |
| [!45800](https://gitcode.com/Ascend/pytorch/merge_requests/45800) / `ac90b06393` | 随后回退该能力，恢复 fallback 与 dtype 逻辑；关联 [Issue #4612](https://gitcode.com/Ascend/pytorch/issues/4612)，公开正文和讨论均未说明具体故障场景、错误类型或根因 |
| [!48259](https://gitcode.com/Ascend/pytorch/merge_requests/48259) / `11396bdd77` | 后续重新开放 2.10 Triton cumsum，并补 shape/helper 等接口；现有公开记录没有把这些修改与 !45800 的具体触发原因逐一对应 |
| [!49102](https://gitcode.com/Ascend/pytorch/merge_requests/49102) / `ac0e1564ca` | 移植到 2.13，继续处理 mask 与实际 scan 维度；可作为候选实现，不能代替历史兼容性验证 |

当前 NPU override 的实际规则是：

```python
if (is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())) and dtype is None:
    dtype = torch.int64 if is_ascend950 else torch.int32
```

而社区 `torch/_inductor/lowering.py::cumsum` 对整数/bool 的默认 dtype 一律是 int64。
因此，删除整个 override 会改变非 Ascend950 设备当前 compiled 路径的默认 dtype 策略。
这是可由源码直接确认的行为差异，尚不能断言修改后一定错误，也不能宣称没有兼容风险。
原 helper 用例使用浮点输入，无法验证这项差异。

社区 cumsum 在 `ir.Scan.create()` 返回 None 时仍会调用 `fallback_cumsum`。
所以候选补丁移除的是 NPU 的强制回退，不是所有 fallback；但这也不自动保证 NPU 所有不支持场景
都能在该分支被正确识别，后续 codegen/编译失败未必会返回到这一层。

进一步联网核对后，结论是：**以最新 master 已合入的完整 cumsum 改动为同步依据。**
历史 dtype 差异应作为回归验证范围，不能据此认定最新 master 仍要求保留 NPU 强制 fallback。
应确认默认与显式 dtype、bool、标量、out 及目标设备的 eager/compiled 行为；
!45800 的具体回退原因仍未公开，不能声称已逐一证明所有历史触发场景都被解决。
本次没有单独删除 fallback，没有增加用例专用开关，也没有修改生产代码。

[历史核对摘要](../../../../artifacts/inductor-combo-tests-native-v2.13-sync-64dc85c-failure-fixes/cumsum-fallback-history.json)。

2026-11-11 通过 `git ls-remote` 与 `git fetch upstream master` 核对远端最新 master，
HEAD 为 `694f3b371432032d5477fc0d5e47d83aafa7cc09`，已包含 `ac0e1564ca`。
主路径三个注册文件的实际状态如下：

| 文件 | 当时拉取的 master | 历史验证快照 bb5e1d0a84 |
| --- | --- | --- |
| `lowering_fallback_list.py` | 不含 aten.cumsum 条目 | 仍有 aten.cumsum.out |
| `lowering_override_list.py` | 不含 aten.cumsum | 仍在 override 列表 |
| `lowering.py` | 已删除专用 cumsum lowering | 仍强制调用 fallback_cumsum |

master 同时采用 `_lift_helper(combine_fn, values, dtypes)` 并导出 `split_scan`。
因此原验证快照缺少 master 已合入的能力提交；当前 d12230b556 已继承这套完整改动，尚未设备复验。
社区通用 scan lowering 在无法创建 Scan 时仍有条件 fallback，不能把“移除 NPU 强制 fallback”
解释成“任何场景都不允许 fallback”。此次仅更新远端引用，没有切换或修改生产工作分支。


### 未重合的 32 个 FAIL 分类

| 分类 | 源方法数 | 处理判断 |
| --- | ---: | --- |
| 明确可着手补齐的接口、追踪和统计接入 | 10 | 优先做范围明确的修复；统计必须使用真实 NPU 数据 |
| 独立分块和逐子 kernel 调优协议尚未对齐 | 9 | 需要能力接入；其中 autotune 日志失败还需核对 NPU 等价行为 |
| 后端专用检查或资源策略 | 7 | 做适用性判断或等价语义适配，不照搬 CUDA/HIP 实现 |
| eager 内存格式支持限制 | 1 | 属于 eager 能力，不是 Combo 生成器的小修 |
| FTZ 数值语义缺少现成接入路径 | 1 | 需确认后端语义与选项，不能只补 metadata |
| 融合数量和峰值内存仍待定位 | 4 | 有差异，但证据不足以定性为功能缺失 |
| 合计 | 32 | 不包含下文的 4 个 SKIP |

#### 可以优先修复或补齐的 10 个方法

| 源方法 | 已确认原因 | 修改方向及风险 |
| --- | --- | --- |
| `test_combo_inductor_meta_has_kernel_num_gb_under_profile_bandwidth` | `NPUDebugAutotuner` 未消费 `with_profiler` 和 `with_bandwidth_info`，前者传给父类立即报 TypeError；`run` 仍要求旧 `input_grid` | 对齐当前 NPU run、最佳候选、runtime block 和计时协议，再补带宽数据；不能只删除一个关键字。风险在 autotune/benchmark 的参数与输入保护 |
| `test_combo_inductor_meta_has_optimize_mem` | Combo 没有输出该字段，runtime 使用默认 True；推理场景的默认值恰好符合预期 | 采用社区 `is_inference or is_backward`，与训练前向一起验证 |
| `test_combo_inductor_meta_optimize_mem_false_in_training_forward` | 同一字段在训练前向仍默认 True，而社区要求 False | 是实际输入复制/恢复策略的接入问题，值得优先修复；当前未证明发生数值错误 |
| `test_triton_kernel_to_post_grad_tracing_combo_kernel` | NPU `define_kernel` 未接 provenance hook，`cppCodeToPost` 映射为空 | 展开实际子节点登记 origins，并向 wrapper 写 debug handle；覆盖源码复用路径，避免只传 `node` 为空的融合外壳。主要影响追踪信息 |
| `test_combo_kernel_num_gb_and_flop_match_standalone_sum` | 普通 NPU kernel 有字节估算，Combo 未汇总带宽/FLOP | 统一普通 kernel 和 Combo 的真实统计，再求和；核对 FLOP 估算，不填 0 凑断言 |
| `test_benchmark_combo_kernel_emits_real_num_gb` | Combo benchmark 未得到真实的聚合字节估计 | 将聚合估计传给 benchmark，并保留正确设备字符串；风险主要是计量失真 |
| `test_combo_pointwise_combo_grid_meta_has_per_subkernel_fields` | 缺少社区 `inductor_meta_i` 的逐子 kernel 计数 | 在 NPU 现有逐子 kernel 结构上补真实 load/store/reduction 等统计；先核对计数是否随 NPU 生成路径正确更新 |
| `test_combo_reduction_combo_grid_meta_has_per_subkernel_fields` | 同上，reduction 路径未输出相同的统计契约 | 与普通 kernel 共用统计出口，不因添加字段宣称调优能力已经完整支持 |
| `test_combo_reduction_deterministic_has_contiguous_rdim_per_subkernel` | 首个失败是缺少 `inductor_meta_i`；连续 reduction 访问标记也需要核对 | 按 NPU 实际地址轴推导连续性，不能直接照搬 CUDA 的轴假设 |
| `test_combo_per_kernel_inductor_meta_matches_standalone` | standalone 解析出 2 组，Combo 的逐子 kernel 元数据为 0 组 | 核对 `no_x_dim`、计数、atomic 和 tiling 的 NPU 语义，统一两种生成方式的统计；不是简单增加一段空字典 |

这些项都有社区已有协议可参考，但 NPU 使用自己的轴、tiling 和 runtime，属于语义对齐，
并不保证逐字复制社区函数就能正确工作。特别是后四项，补齐统计输出与让调优算法消费约束是两项工作：
前者可以单独修复，后者不能因 FileCheck 通过就宣布完成。

带宽调试器的后续问题已有明确源码依据：当前普通 runtime 的 `run` 接受 `stream` 和关键字参数，
而调试器要求额外的 `input_grid`；普通执行选择 `best_launcher` 并构造运行时 block，
调试器却假定 `self.launchers` 只有一个。构造接口修复后仍可能触发这些旧协议问题。
另外，NPU `bench(with_profiler=...)` 的参数没有直接传入底层计时函数，后者按 metadata 和当前 profiler
状态决定计时方式，因此也要保持 NPU 现有 profiler 处理，不能直接换成社区 CUDA 计时器。

#### 需要补齐或核对能力的 9 个方法

NPU 并非完全没有逐子 kernel 信息：`autotune_subkernel_meta()` 已保存每个子 kernel 的真实尺寸、
轴约束和 persistent 标记。但当前默认配置、运行时 block 和搜索仍采用 NPU 自身的共享 split 协议，
与社区 `XBLOCK_i`、逐子 kernel 调优组的契约不等价。

| 源方法 | 当前差异 | 是否可作为小修 |
| --- | --- | --- |
| `test_combo_kernel_no_bench_stitched_config` | 数值通过，缺少独立 `XBLOCK_0/1/2` 默认配置 | 不宜；需贯通独立分块、grid、参数与启动逻辑 |
| `test_combo_kernel_no_bench_persistent_reduction` | 数值通过，persistent 路径也缺少测试要求的独立默认分块 | 不宜；需一起核对静态 reduction 块与资源限制 |
| `test_combo_kernel_per_subkernel_reduction_hint` | 数值通过，缺少 `reduction_hint_0/1`；NPU decorator 当前选择首个子 kernel 的 hint | 不宜只打印字段；应接到逐子 kernel 候选生成和筛选 |
| `test_combo_rms_norm_forwards_add_persistent_rblock` | 未按社区 `inductor_meta_i` 接入大 RBLOCK 的特殊候选 | 先核对 NPU 是否已有等价候选及资源约束，再决定补接入或新增调优能力 |
| `test_combo_kernel_max_autotune` | 未通过社区逐子 kernel autotune 日志/行为检查 | 先记录 NPU 实际候选和赢家；不应只补日志 |
| `test_combo_kernel_max_autotune_with_reduction` | 同上，涉及 reduction 候选 | 需要确认 NPU 对应行为和资源约束 |
| `test_combo_autotune_many_subkernels` | 未满足社区多子 kernel 调优协议检查 | 需要确认是否真正缺少顺序调优，再决定能力接入 |
| `test_combo_autotune_grouping` | 未满足社区按子 kernel metadata 合并调优组的检查 | 社区调优分组与 NPU 符号输入分组不是同一个概念，不能混为一谈 |
| `test_combo_kernel_coordesc_tunes_largest_subkernel_first` | 未满足社区优先调最大子 kernel 的 coordinate-descent 顺序检查 | 需确认 NPU 是否采用该算法；若未采用，这是算法对齐工作 |

后五项的数值断言位于当前失败的检查之后，不能记作“数值已通过”。
NPU 的 `_disable_grouped_autotune()` 禁用的是自身符号输入分组，
这并不能单独证明社区按子 kernel metadata 分组的所有能力都缺失。
准确结论是当前社区协议未对齐，哪些可以映射到 NPU 现有算法仍需候选级证据。

#### 后端检查与资源策略：7 个方法

| 源方法 | 判断及处理方向 |
| --- | --- |
| `test_persistent_reduction_size_hint` | 检查写死 CUDA `x/r0_` 文本，NPU 轴命名不同；可适配 NPU 等价语义。数值断言在失败检查之后，还不能声称数值通过 |
| `test_single_combo_kernels` | 数值与前置数量检查已通过，失败于 `Grid1D` 对 `GridNpu`；适配 NPU grid 检查 |
| `test_combo_kernel_yz_overflow` | 按 CUDA `traceEvents/args.grid` 读取 NPU profiler 事件列表而失败；应验证 NPU 实际启动维度，不能只删除 overflow 检查 |
| `test_combo_max_persistent_rblock_gated_on_per_subkernel` | 失败的 False 参数要求 HIP 资源过滤 metadata；上游用于避免 `XBLOCK*RBLOCK` 过大导致 ROCm 编译膨胀。NPU 不应直接复制 HIP 的 4096 阈值 |
| `test_combo_kernel_tma_descriptor_block_name` | CUDA TMA descriptor 专用代码检查；应恢复准确的后端适用条件，不为该断言在 NPU 模拟 CUDA TMA |
| `test_pdl_codegen_in_combo_kernel` | CUDA PDL 启动协议检查，当前 NPU 没有这条 CUDA 路径 |
| `test_pdl_combo_kernel_pointwise` | 同上，应按后端能力筛选，不以补一个 `launch_pdl` 字段宣称支持 |

这 7 个方法不等于 7 个 NPU 内核错误。前 3 个可以做 NPU 等价验证适配；
后 4 个应先明确适用性。TMA/PDL 用例会进入执行还与测试头部模拟 GPU capability 有关。

#### 另外 6 个方法：eager、FTZ 与待定位差异

| 源方法 | 判断及下一步 |
| --- | --- |
| `test_combo_kernel_per_config_subkernel_block_size` | eager 的 `channels_last contiguous` 在 compile 前失败；该输入场景受 eager 内存格式支持限制，需要独立能力工作。换成普通 contiguous 会改变测试覆盖 |
| `test_combo_triton_meta_has_disable_ftz` | NPU 未发出该字段，也未映射对应编译选项；社区仅在 CUDA 分支将它转换为 `enable_reflect_ftz=False`。本地 Ascend 编译器选项未见等价开关，需进一步核对后端/硬件数值语义，不能只补字段 |
| `test_fuse_mix_order_reductions_combo_kernels` | 数值通过，kernel 数预期 3、实际 4；需要分区、候选编译和回退日志，尚不能定性为主动收紧融合或缺能力 |
| `test_combo_kernel_scalar_store_broadcast` | 数值通过，kernel 数预期 1、实际 2；同样需定位拒绝融合的首个原因 |
| `test_combo_kernel_no_bench_carve_out` | 数值通过，kernel 数预期 2、实际 1；需查大子 kernel carve-out 的条件和 NPU 策略，当前不能直接归入功能缺失 |
| `test_combo_kernel_peak_memory_wide_resnet` | tight 与 disabled 都是 616101888 字节；需联系实际融合决策、fallback 与 buffer 生命周期定位，现有峰值差异不能证明内存策略完全未实现 |

FTZ 指把非正规数冲刷为零的处理方式，涉及真实数值语义；缺少现成配置通路不等于已证明硬件不支持。
本次对 Ascend 编译器的判断来自本地源码选项检查，未在目标设备核实相应底层行为。

### 4 个 SKIP 不作为功能缺失证据

- `test_dynamic_shapes_persistent_reduction_mixed_x_dim_xpu` 与
  `test_combo_kernel_amd_special_config_args` 分别限定 XPU 和 ROCm，当前保持平台筛选。
- `test_snode_runtime.py` 的 `test_foreach_basic`、`test_foreach_after_extern` 带有原始 NPU skip。
  源码分别检查图输入 foreach、mm 后 foreach 的 scheduler 输入距离，尚未实际进入失败路径。
  不能由这两个装饰器推断 foreach 算子不支持；可后续单独尝试适用性验证，现阶段仍记 SKIP。

### 源码依据与建议顺序

以下路径相对于对应源码仓库，行号以本次核对版本为准，不写入总表列中。

| 仓库与位置 | 支撑结论 |
| --- | --- |
| NPU `ac0e1564ca` 的 5 个生产文件补丁 | cumsum 的 fallback、scan 接口、mask、shape 与 split_scan 是一组完整适配 |
| NPU `torch_npu/_inductor/codegen/triton_combo_kernel.py::jit_line`（180 行附近） | 缺少 optimize_mem、带宽/FLOP 聚合及公共 triton metadata；当前禁止自身符号分组 |
| NPU 同文件 `autotune_subkernel_meta`（141 行）、`combo_grid_meta`（465 行） | 已有逐子 kernel 轴约束，但默认配置采用共享 split block |
| NPU `torch_npu/_inductor/runtime/triton_heuristics.py::NPUDebugAutotuner`（3115 行）、`cached_autotune`（3159 行附近） | 两个构造参数及旧 run 协议不一致；optimize_mem 默认 True |
| NPU 同文件 `run`（1873 行）、`bench`（1427 行） | 当前候选、runtime blocks 和 profiler 计时的 NPU 协议 |
| NPU `torch_npu/_inductor/codegen/scheduling.py::define_kernel`（574 行） | 未接社区 provenance/debug handle 协议 |
| 社区 `torch/_inductor/debug.py::set_kernel_post_grad_provenance_tracing`（1142 行） | 只登记实际 node 的 origins，融合外壳需要展开 |
| 社区 `torch/_inductor/codegen/triton.py::inductor_meta_per_kernel`（6235 行） | 普通 kernel 与 Combo 共用真实统计的参考契约 |
| 社区 `torch/_inductor/codegen/triton_combo_kernel.py::jit_line`（715 行附近） | optimize_mem 图阶段判定、带宽聚合和 ROCm 资源保护的来源 |
| 社区 `torch/_inductor/runtime/triton_heuristics.py`（1097 行附近） | disable_ftz 到后端选项的转换限定在 CUDA |
| Triton Ascend `third_party/ascend/backend/compiler.py::NPUOptions`、`parse_options` | 本地源码未见与 CUDA enable_reflect_ftz 等价的现成选项 |
| 用例 `test_combo_kernels.py`（1831 行附近） | max_persistent_rblock 测试自身明确说明 HIP 过滤用途 |

当前先定向复验已随 master 带入的 **cumsum 和带宽调试器**，确认是否仍有后续失败。
其余可评估的修复顺序为 **optimize_mem → provenance → 真实统计与聚合**；均以新基线复验结果为准。
每项应独立验证原失败方法，再讨论是否继续扩展分块和 autotune 能力。
融合数量与峰值内存先保留为待定位项，不用放宽断言代替定位。

本轮没有新增设备验证或性能 benchmark 结论，也没有将任何 FAIL/SKIP 改为 PASS。
