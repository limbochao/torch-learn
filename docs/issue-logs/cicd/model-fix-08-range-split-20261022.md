---
title: 修复 ⑧：range 拆分的拒绝协议，及 current_group 检查为什么必须提前
published: false
---

# 修复 ⑧：range 拆分的拒绝协议，及 current_group 检查为什么必须提前

对应整合提交 `5b18fac65380a14a404251de2eb6b89072c60911`，主题为
`fix(inductor): preserve CantSplit compatibility rejections`。
生产修改在 `torch_npu/_inductor/codegen/triton.py` 的
`NPUIndexTritonKernel._split_iteration_ranges()`，完整代码变化和三项单元见[补丁][patch]。
八项关系见[总览][overview]。

## 这个提交解决什么问题

**解决不兼容的迭代范围本应被正常拒绝，却因异常构造错误或数组越界导致编译中断的问题。**

- **触发场景**：调度器尝试把节点的迭代范围映射到候选 kernel，遇到不能整除或 group 已用完的情况。
  正常行为应是报告 `CantSplit`，由兼容性检查返回 False，继续选择其他合法安排。
- **修复前**：部分分支直接 `raise CantSplit`，但异常类要求两个参数，实际抛出的是 `TypeError`；
  另一个分支先读取 `remaining[current_group]`，之后才检查下标是否越界，可能直接抛 `IndexError`。
  这两种异常都无法按原有 `CantSplit` 拒绝流程处理。
- **修复后**：为 `CantSplit` 提供失败时的操作数，并在访问数组前检查 group 是否耗尽，
  让不兼容候选正常被拒绝；合法范围的拆分与索引重建公式继续沿用。

两部分证据需要区分：双 shape 训练小图实际暴露了 `TypeError`；
提前下标检查解决的 `IndexError` 由 `(2,), ([2,3],[])` 专项输入复现，尚未证明它单独阻塞原整模型。
详见[旧版记录][old]、[新版检查][new]和[训练结果][training]。

## 为什么融合前还需要拆分 iteration ranges

一个 kernel 和其中节点可能用不同方式描述同一迭代空间。
kernel 可以把总量 30 分成三组 `[2,3,5]`，某个节点却只用一个长度为 30 的扁平索引。
要把节点放进 kernel，编译器必须知道如何把 kernel 的坐标重建成节点索引。

这里的 range 是逻辑迭代范围，不是 Python `range()` 对象，也不是内存分配大小。
`_split_iteration_ranges(groups, lengths)` 的两个输入与返回值是：

| 名称 | 例子 | 含义 |
| --- | --- | --- |
| `groups` | `(2,3,5)` | kernel 各组可提供的迭代元素量 |
| `lengths` | `([30],[])` | 节点的逐点、归约长度列表；空列表表示该类轴为空 |
| `new_ranges` | `[[2],[3],[5]]` | 每个 kernel group 中实际拆出的子范围 |
| `return_getters_groups` | 一组索引重建函数 | 从新坐标计算原节点各轴索引 |

`is_compatible()` 用它预先判断节点能否放进候选布局。
真正建立映射时，`split_and_set_ranges()` 等流程也会用到范围映射逻辑。
这解释了为什么函数既计算索引，又可能在正式 DSL 生成之前被调用。

## 先理解内部几个变量和 helper

`remaining` 是每个 group 尚未被分配掉的因子量，初始等于化简后的 groups。
`current_group` 是当前使用的 group 下标，不是还剩多少个 group。
当某个 remaining 变成 1，说明该组可分配的非平凡因子已耗尽。
`new_ranges` 保存分出的子轴长度；`var_count` 为新子轴分配索引编号。

| helper | 作用 |
| --- | --- |
| `add_range(i,expr)` | 从第 i 组分出长度 expr，要求 `remaining[i]` 能整除 expr |
| `add_multiple_range(size,...)` | 一个节点轴跨多个 group，依次消费并合并 |
| `make_combined(strides,index_list)` | 返回函数，用新轴坐标和 stride 重建旧索引 |
| `safe_size_hint_is_one()` | 尝试用已有 hint 判断是否可跳过长度 1 的组，异常时返回 False |

本提交没有重写这些 helper 的成功算法。
它修的是失败分支如何报告“不兼容”，以及何时检查 current_group 已超出数组。

## 正常例子：三个 group 如何重建长度 30 的索引

设 kernel 新坐标为 `i0∈[0,2)`、`i1∈[0,3)`、`i2∈[0,5)`。
一个节点轴 size=30 大于第一个 group 的 2，于是进入 `add_multiple_range()`：

| 步骤 | 进入时 remained_size | 消费 group_size | 除后 remained_size / 记录 stride |
| --- | --- | --- | --- |
| 1 | 30 | 2 | 15 |
| 2 | 15 | 3 | 5 |
| 3 | 5 | 5 | 1 |

`index_list` 记录新坐标编号 `[0,1,2]`，`stride_list` 记录 `[15,5,1]`。
`make_combined()` 的 getter 把它们组合为：

```text
old_index = 15*i0 + 5*i1 + i2
```

这是标准的多维坐标线性化公式。新坐标覆盖全部 30 个位置，且不会引入重复或遗漏。
本提交专门保留三组测试，避免把 NPU 的多组支持误改为只支持两组。

## 调用方为什么只接受 CantSplit

相关 `is_compatible()` 主体为：

```python
try:
    groups = flatten(groups)
    NPUIndexTritonKernel._split_iteration_ranges(groups, lengths)
    return True
except CantSplit:
    return False
```

该方法前面还会按既有规则补足缺失的 reduction_numel；上面只摘录失败协议相关部分。
“不能按当前 groups 拆分”是正常的候选拒绝，不是整个编译流程一定要终止的异常。
因此内部用 CantSplit 通知调用方，外部转换成 False，供调度选择其他合法安排。

社区 `torch/_inductor/codegen/simd.py` 的异常定义要求两个参数：

```python
class CantSplit(Exception):
    def __init__(self, expr, remaining):
        super().__init__()
        self.expr = expr
        self.remaining = remaining
```

`raise CantSplit` 会尝试无参数构造这个类，于是抛的是 TypeError。
调用方只捕获 CantSplit，并不能捕获构造阶段的 TypeError。
这是异常协议不匹配，不是 Python 不允许使用 `raise ExceptionClass` 这种语法。

## 四处异常修改逐项解释

### 1. `add_range()`：当前组不能整除请求长度

```python
elif not sv.statically_known_multiple_of(remaining[i], expr):
    raise CantSplit(remaining[i], expr)
```

例子是 groups=(6,1)、lengths=([4],[1])。
要从长度 6 的组分出长度 4，6 不能被 4 整除，因此拒绝。
传入 6 和 4，异常中 `.expr=6`、`.remaining=4`，与要报告的整除关系一致。
注意异常属性名中的 remaining 不是本函数整个 remaining 列表，而是第二个诊断操作数。

### 2. `add_multiple_range()`：跨组合并到中途发现不可整除

```python
if not sv.statically_known_multiple_of(remained_size, group_size):
    raise CantSplit(remained_size, group_size)
```

例子 groups=(2,3)、lengths=([10],[])。
先消费 group=2，10/2=5；接下来想消费 group=3，但 5 不能整除 3，
所以应报告 CantSplit(5,3)。新代码使用实际失败时的操作数，而不是无参异常。

### 3. `add_multiple_range()`：所有可用组都用完，节点长度仍有余量

```python
if remained_size != 1:
    raise CantSplit(remained_size, 1)
```

例子 groups=(2,)、lengths=([6],[])。
消耗唯一 group 后，节点轴还剩因子 3，没有其他 group 可以继续提供。
这里的 `(3,1)` 是保留异常构造协议的诊断参数；
不能按 `CantSplit.__str__()` 的通用文字机械理解为“3 不能被 1 整除”。
真正失败原因是循环结束后 remained_size 没有完全消费。

### 4. 主循环：group 用完时报告 `CantSplit(size,0)`

0 是“已经没有可用 group”的诊断占位，不是在这里执行一次除零。
关键不只是加两个参数，还包括把检查移到数组访问之前。

## 为什么 `current_group >= len(remaining)` 必须挪位置

修改前关键控制流等价于：

```python
while current_group < len(remaining) and group_is_exhausted:
    current_group += 1
if sv.statically_known_gt(size, remaining[current_group]):
    add_multiple_range(size, return_getters)
else:
    if current_group >= len(remaining):
        raise CantSplit
    ...
```

第一段 while 的短路条件只保护 while 自己读取 `remaining[current_group]`。
它允许退出时 current_group 恰好等于 len(remaining)，
并不会替下一条 if 自动建立“下标仍合法”的保证。
旧检查位于 else 中，但 Python 必须先求值 if 条件，
其中的 `remaining[current_group]` 已经可能越界，根本到不了 else 内的检查。

修复后流程是：

```python
while current_group < len(remaining) and group_is_exhausted:
    current_group += 1
if current_group >= len(remaining):
    raise CantSplit(size, 0)
if sv.statically_known_gt(size, remaining[current_group]):
    add_multiple_range(size, return_getters)
else:
    ...
```

以上两段用 `group_is_exhausted` 简写源码的耗尽条件，属于控制流示意；
完整表达式含 `statically_known_equals(...,1)` 和 `safe_size_hint_is_one(...)`，见补丁及源码上下文。

### 手动走一遍真实专项输入

groups=(2,)、lengths=([2,3],[])：

1. 初始 `remaining=[2]`、`current_group=0`。
2. 第一个请求 size=2，`add_range(0,2)` 成功，把 remaining[0] 改成 1。
3. 第二个请求 size=3，while 看见该组已耗尽，将 current_group 加成 1。
4. 此时 len(remaining)=1，没有 remaining[1]。
5. 旧版先访问 remaining[1]，抛 IndexError；新版先检查并抛 CantSplit(3,0)。
6. `is_compatible()` 捕获新版 CantSplit，正常返回 False。

所以不是把检查挪到“看起来更整齐”的地方，而是修正检查的支配关系：
任何可能越界的读取都必须发生在该检查通过之后。

## 为什么不在外层捕获 TypeError / IndexError

这两类异常还可能表示别处真实的代码缺陷，全部捕获会掩盖无关错误。
内部已经知道“不能拆分”的具体原因，应发出既定 CantSplit，
外层保留只处理正常不兼容的契约。
这比强行返回 True 或删掉合法性检查更符合原设计，拒绝候选也不会生成错误融合 kernel。

## 不修改会不会影响模型测试

会影响实际触发这些拒绝路径的编译：本应返回 False 的候选检查会向外抛 TypeError 或 IndexError。
但是要分开看两部分证据：

| 改动 | 已有直接触发证据 | 能否称为四个整模型都必需 |
| --- | --- | --- |
| 为 CantSplit 传参数 | 双 shape 小图初次 backward 编译的 TypeError | 不能，原整网矩阵未观察到此新错误 |
| 提前 group 耗尽检查 | 上述 `(2,),([2,3],[])` 专项输入的 IndexError | 不能，尚未证明这个分支单独阻塞整模型 |

某些后续重试能数值通过，也不代表第一次候选检查的协议正确。
反过来，模型没有经过这些分支时，不修改也可能完成测试，不能据此否认已复现的缺陷。

## 验证和剩余风险

新增三个单元分别检查四类不兼容输入返回 False、异常操作数、合法三组索引重建。
[旧版记录][old]保留失败，[新版检查][new]为 3/3 通过；
[最终双 shape 训练][training]不再出现 CantSplit TypeError。

成功路径的整除判断、范围分配、索引公式没有被本提交重写。
不兼容候选现在能继续走正常调度，可能改变最终 fusion、kernel 数量和性能，
也可能暴露后面的编译问题；这不是把原先不合法的融合强行接受。
末尾 remaining 未完全消费时的其他 RuntimeError 检查仍保留，
三项测试不是任意符号 range 的完整证明。本次未重跑设备测试。

[overview]: model-fix-dsl-explained-20261021.md
[patch]: ../../../artifacts/cicd/model_fixes_20261020/patches/codegen-range-split.patch
[old]: ../../../artifacts/cicd/model_fixes_20261020/layout_checks_v8/baseline_range_unit.txt
[new]: ../../../artifacts/cicd/model_fixes_20261020/layout_checks_v8/fixed_range_unit.txt
[training]: ../../../artifacts/cicd/model_fixes_20261020/layout_checks_v8/fixed_training.txt
