---
title: 修复 ③：indirect_indexing 如何生成负索引回绕和边界检查
published: false
---

# 修复 ③：indirect_indexing 如何生成负索引回绕和边界检查

对应整合提交 `c32818e10ee1152dcf014bbc0b200134b20faf2a`，主题为
`fix(inductor): preserve negative indexing with debug flags`。
[完整补丁][patch]包含生产代码和一个针对 no-wrap 的单元方法，八项关系见[总览][overview]。

## 这个提交解决什么问题

**解决调试开关改变合法负索引含义，以及未回绕索引的范围记录错误的问题。**

- **触发场景**：长度为 8 的表使用 `table[-1]`，按 PyTorch 语义应读取第 7 项；
  独立回归还覆盖 `-8`、`-3` 等合法负索引，以及 dump/check 调试开关。
- **修复前**：dump/check 路径会覆盖调用方的 `wrap_neg` 要求，可能让本应转换为正索引的值
  继续以负值进入地址计算或边界检查。另一个问题是：即使调用方要求不回绕，
  仍可能把实际含负值的 bounds 记录为非负范围，使后续错误省略下界检查。
- **修复后**：是否回绕由调用方语义决定，调试开关不再覆盖它；
  回绕后的临时量和未回绕的原索引分别保留正确的范围信息，供地址生成和检查使用。

证据来自[独立数值回归][repro]、[no-wrap 单元检查][unit]和[修复后 DSL][dsl]。
目前没有证据证明原四模型都必须经过这条失败路径；不能将它写成四模型共同的阻塞原因。

## 先理解函数的输入和输出

对 `table[indices]`，访问地址由另一个 tensor 中的值决定，这称为 indirect indexing，
即间接索引。与 `table[i]` 中编译器直接生成的循环变量 `i` 不同，
`indices` 中实际存了什么值，通常要到执行时才知道。

PyTorch 的合法负索引会从末尾计数。长度为 8 时，`-1` 表示第 7 项，`-8` 表示第 0 项。
这里的 wrap 是对负值加一次 `size`，不是对任意整数取模；`-9` 仍属于越界输入。

修改点在 `torch_npu/_inductor/codegen/triton.py`：
`NPUIndexTritonKernel.__enter__()` 定义的内部类 `CSEProxy.indirect_indexing()`。
它的参数和后续行为如下：

| 参数或变量 | 含义 |
| --- | --- |
| `var` | 保存索引值的 CSE 临时量，包括名称、dtype、shape、bounds |
| `size` | 被索引维度的长度，可能是符号表达式 |
| `check` | 调用方是否请求对应的检查行为 |
| `wrap_neg` | 调用方是否要求把合法负索引转换成正索引 |
| `sympy_var` | 返回给地址表达式的符号，代表生成后的索引临时量 |

`bounds` 是编译器对索引值范围的保守认识，例如 `[-2,3]`。
它不是一个运行时截断操作；编译器可以用它决定某个检查是否必然成立。
所以 bounds 既影响优化，也影响是否生成安全检查。

## 为什么这个函数藏在 `__enter__()` 中

kernel 生成器用上下文管理器进入当前 kernel 环境。
`__enter__()` 建立操作代理，使图里的 `ops.load`、`ops.indirect_indexing` 等调用转到该 kernel。
`CSEProxy` 是这里定义的内部类；源码已有注释说明，为覆盖这些代理方法，需要覆盖整个内部实现。
因此补丁落在 `__enter__()` 内，并不表示问题发生在设备启动或 Python 模型进入 `with` 时。

职责关系可以这样读：

```text
图中索引操作
  → V.ops.indirect_indexing(var, size, check, wrap_neg)
  → 当前 NPU kernel 安装的 CSEProxy.indirect_indexing
      1. 必要时生成负索引转换
      2. 为转换后的临时量记录 bounds / dtype / shape
      3. parent_handler 生成用于地址表达式的符号
      4. 根据 bounds 决定下界、上界检查是否仍需生成
  → 后续 load 使用该符号计算 table 地址
```

本提交没有修改 `indirect_template` 的既有提前返回分支；
也没有把所有后端和所有索引都改成同一条负索引规则。

## 修改前的两个问题

### 调试开关覆盖了调用方语义

旧代码在 bounds 可能为负时执行：

```python
if (
    os.environ.get("INDUCTOR_ASCEND_DUMP_FX_GRAPH")
    or os.environ.get("INDUCTOR_ASCEND_CHECK_ACCURACY")
):
    wrap_neg = False
```

无论调用方原本传什么，只要环境变量为真，负索引转换就被关闭。
这样 `table[-1]` 的有效地址会因是否开启 dump/check 而不同。
模型数学表达式没有变化，调试选项却改变了访问语义。

### no-wrap 的 bounds 可能与实际值不一致

旧代码即使选择 `stm = var`，仍可能按“负值已加 size”去更新 bounds。
例如 `var` 的范围为 `[-2,3]`、size=8，实际不转换时依然可能出现 -2；
若元数据却记录成 `[0,7]`，后续就可能认为索引必定非负，从而省掉下界检查。
这是独立于调试开关的第二个问题，显式 `wrap_neg=False` 调用也需要正确元数据。

## 按补丁逐段读

### 1. 删除环境变量分支及 `import os`

补丁删除上述五行，函数继续尊重传入的 `wrap_neg`。
删去 `import os` 是因为本文件不再需要这次被移除的环境变量读取。
这不是取消 dump/check 功能，而是取消它们对索引计算的附加干预。

保留的 wrap 生成规则是：

```python
if wrap_neg:
    stm = ops.add(var, ops.index_expr(size, torch.long))
    if var.bounds.upper >= 0:
        lt = ops.lt(var, 0)
        stm = ops.where(lt, stm, var)
else:
    stm = var
```

`ops.*` 在这里表示生成操作，不是立即对真实 NPU tensor 执行加法。
若整个范围都为负，统一加 size 即可；若范围同时含负数和非负数，
则生成条件选择，只有负数加 size，正数保持。

### 2. `new_bounds` 按是否真正 wrap 分开

修改后的关键行是：

```python
new_bounds = ValueRanges.unknown() if wrap_neg else var.bounds
if wrap_neg and var.bounds != ValueRanges.unknown() and isinstance(size, sympy.Number):
    ...
```

第一行：wrap 时先用 unknown，避免在信息不足时误报精确范围；
no-wrap 时值没有改变，直接保留原来的范围。
第二行：只有真正 wrap、原范围已知且 size 是数值常量时，才进入现有的精确范围推导。

原范围 `[-2,3]`、size=8 的 wrap 推导如下：

1. 负数部分 `[-2,-1]` 加 8，得到 `[6,7]`。
2. 非负部分 `[0,3]` 保持。
3. `ValueRanges` 用单个区间包住两部分，得到 `[0,7]`。

`[0,7]` 并不宣称每个整数都一定出现，只说明实际值不会跑出这个区间。
no-wrap 则没有这三步，始终保留 `[-2,3]`。

### 3. CSE 生成显式传入 dtype 和 shape

```python
var = self.cse.generate(
    self.compute, stm, bounds=new_bounds, dtype=var.dtype, shape=var.shape
)
```

`self.compute` 是将来输出到 kernel 计算段的代码缓冲区。
新临时量的值来自 `stm`，bounds 来自上一段推导，dtype/shape 沿用原索引。
只修范围、不携带 dtype/shape，后续 CSE 和索引生成仍可能缺少准确元数据。
这里没有把索引统一强转成另一个 dtype，也没有改变索引 tensor 的形状。

### 4. 未改的检查逻辑解释了为什么 bounds 必须修

在生成索引符号后，原代码按以下条件决定检查：

```python
assert_lower = not (var.bounds.lower >= 0)
assert_upper = (
    not isinstance(size, sympy.Number)
    or not var.bounds.upper < size
)
self.check_bounds(sympy_var, size, assert_lower, assert_upper)
```

若 no-wrap 的范围仍是 `[-2,3]`，下界无法证明非负，`assert_lower=True`。
上界已知小于 8，`assert_upper=False`。
若把 no-wrap 的范围错误写成 `[0,7]`，下界也会被省掉。
因此 bounds 修复不仅关系到调试信息，它直接改变生成器是否保留必要的检查。

## 从真实 DSL 对回 Python 生成器

[归档的修复后 kernel][dsl]包含以下原始语句：

```python
            tmp0 = tl.load(in_ptr0 + (y0), y0_mask)
            tmp1 = tl.full([Y0BLOCK_SUB, X1BLOCK_SUB], 8, tl.int32)
            tmp2 = tmp0 + tmp1
            tmp3 = tmp0 < 0
            tmp4 = tl.where(tmp3, tmp2, tmp0)
            tl.device_assert(((0 <= tmp4) & (tmp4 < 8)) | ~(y0_mask), "index out of bounds: 0 <= tmp4 < 8")
            tmp6 = tl.load(in_ptr1 + (x1 + 16*tmp4), x1_mask & y0_mask)
```

`tmp0` 是读出的原索引；`tmp2/3/4` 就是“加 8、判断负数、选择”的 wrap。
最后地址中的 `16*tmp4` 表示一行 16 个元素，以转换后的行号选择 table。
device assert 的存在还取决于实际 bounds 和检查配置；这个原始例子不能解释成每次调用都生成双边检查。

旧版同一回归的完整 DSL 未留存，旧行为以补丁中删除的代码为证，不能把重建示意称为原始产物。

## 为什么修改这里，而不是在 load 地址上补一个加法

这个入口知道调用方是否要求 wrap、size 是多少，以及索引值的 bounds。
在这里同时决定值和元数据，后续地址计算与检查才会一致。
若在 `tl.load` 地址中无条件加 size，合法正索引会被破坏；
若在那里只改负值，前面已经错误省掉的检查和 CSE bounds 仍未恢复。

## 验证、模型必要性与潜在影响

补丁新增单元 `test_no_wrap_preserves_negative_bounds_and_checks`：
用 `[-2,3]`、size=8 调用真正安装的代理，验证传给 CSE 的 bounds/dtype/shape，
并要求 `check_bounds(..., True, False)`。它不执行设备访存，专门检查生成器契约。
[单元结果][unit]通过；[数值回归][repro]另覆盖静态/动态与 none/dump/check 共六种组合。

该项来自独立回归，未证实四个历史模型必须依赖它才能 PASS。
修复影响 bounds 可能为负的间接索引，恢复调用方语义；bounds 已证明非负时不新增 wrap。
调试模式可能多出原本被错误省掉的加法/选择；no-wrap 更保守的 bounds 可能保留更多检查。
这可能影响少量代码量或性能，但没有依据认为它把合法负索引改成另一种数学行为。
任意越界值、空维、所有符号 size 尚未被六项回归穷举。
本次未新增设备测试，历史结果不能写成全模型负索引覆盖。

[overview]: model-fix-dsl-explained-20261021.md
[patch]: ../../../artifacts/cicd/model_fixes_20261020/patches/codegen-negative-index.patch
[dsl]: ../../../artifacts/cicd/model_fixes_20261020/explanation_20261021/negative_index_v8.py
[unit]: ../../../artifacts/cicd/model_fixes_20261020/final_checks/negative.txt
[repro]: ../../../scripts/repro/cicd/model_fixes_20261020/validate_codegen.py
