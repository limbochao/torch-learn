---
title: 修复 ⑦：Layout 的符号 stride 证明、上游回移与补丁安装恢复
published: false
---

# 修复 ⑦：Layout 的符号 stride 证明、上游回移与补丁安装恢复

对应整合提交 `c74c45c960397879662b8046d320eba9130bb120`，主题为
`fix(inductor): preserve integer modulo in symbolic stride proofs`。
[完整补丁][patch]修改 NPU 侧两个文件并新增四个单元方法；
实际被替换的是社区 `torch._inductor.ir.Layout` 方法。八项关系见[总览][overview]。

2026-10-23 的后续提交 `4b710b75347d02176bfd3e8cc4bac24238bb3800` 回移比较主体，
`089231bede6e3bf3384e1ded83d6ca45ac739012` 补齐其符号整除证明依赖。两者共同回移
[上游 `93879b11e3b5`][upstream]，将原提交的 `PythonMod` 方案改为
“先静态证明整除，再使用 PyTorch `Mod` 构造 guard”。
增量代码见[上游回移补丁][upstream-patch]。
下文区分原始问题、当前比较实现和保留的补丁生命周期机制。

同日后续将上述回移与原始第 7 项归并为 `a4e9a7f975`，并在兜底 guard 单测中
明确断言表达式使用 PyTorch `Mod`；当前整合分支为 `937cdf81e3`。
以下设备验证保留实际加载的历史 SHA，没有将整理提交后的版本冒充为重新实测。

## 这个提交解决什么问题

**解决动态 stride 的符号证明产生负指数表达式，导致 `Invalid ranges [0:-1]` 编译异常的问题。**

- **触发场景**：基于原 2.13 代码的 NFNet 派生训练小图，切换到第二组输入 `(3,16,256,256)` 后，
  反向编译需要比较两个含符号表达式的 stride；比较中会尝试用取模证明整除关系。
- **修复前**：这里的 `%` 操作数是 SymPy 表达式，会构造通用 `Mod` 并触发符号化简。
  特定表达式提取公因子后留下倒数，即指数为 -1 的项；后续整数值域分析无法按当前规则处理它，
  抛出 `Invalid ranges [0:-1]`，编译中断。普通整数取余的数值结果本身没有错。
- **修复后**：当前实现先尝试静态证明整除；仍需构造条件时，显式使用 PyTorch 的 `Mod(left, right)`，
  避开通用 SymPy `Mod` 的上述化简路径。原始 `c74c45c960` 使用的是 `PythonMod`，属于此前已验证的方案。

这发生在 Layout 元数据分析阶段，尚未生成设备 kernel。[基线记录][baseline]保留旧异常，
[历史修复检查][checks]和[双 shape 训练记录][training]覆盖原 PythonMod 方案。
上游回移后的结果见本文末尾的 NPU_A5 验证及[本轮汇总][upstream-status]。
问题范围是已复现的这条符号分析路径，不能据此认定所有 `Invalid ranges` 或所有 CUDA/NPU 模型都需要此补丁。

## Layout 和 stride 在这里解决什么问题

shape 描述每个维度有多少元素；stride 描述该维度索引加 1 时，内存地址跨过多少元素。
例如连续 `[2,3]` tensor 的 stride 通常为 `[3,1]`，
地址是 `base + i*3 + j`。转置可以改变 shape/stride，而不立即复制数据。

Inductor 的 Layout 是编译时的布局描述，不是某个真实 tensor 的数据本体。
当算子要求某种 stride 顺序时，编译器需要证明当前布局是否满足要求，
以决定复用当前布局、冻结布局或安排 copy。
动态 shape 下 stride 也可能是符号表达式，不能总靠当前样例的数字做判断。

相关源码职责关系为：

```text
lowering / 算子布局约束
  → Layout.is_stride_ordered(order)
      忽略 size=1 的维度，按要求整理 stride 顺序
      → Layout._stride_expr_ge_or_false(left, right)
          → sizevars.guard_or_false(条件)
              → ShapeEnv：化简、约束和表达式值域分析
```

这是 Triton DSL 生成之前的元数据路径。
失败不要求先分配 NPU tensor，也不要求先调用 Triton 编译器。
该改动不直接在 kernel 中生成一次新的“取模运算”。

## 原 helper 为什么既比较大小又判断整除

社区方法的顺序是：

```python
if sizevars.guard_or_false(sympy.Eq(right, 0)):
    return True
if sizevars.guard_or_false(sympy.Eq(left, 0)):
    return False
return sizevars.guard_or_false(
    sympy.Ge(left, right)
) or sizevars.guard_or_false(sympy.Eq(left % right, 0))
```

它是面向 tensor stride 的特殊证明 helper，不是适用于所有实数表达式的通用大小比较器。
right=0 先返回 True；因此 left/right 都为 0 时也返回 True。
left=0、right 非零时返回 False。
其余情况先证明 `left >= right`；Python 的 `or` 会短路，已为 True 就不构造后面的取模条件。

整除分支提供另一种 stride 顺序证明：对目标 stride 域中的正整数倍关系，
例如 left 是正的 `u*256`、right 是 256，整除有时比直接证明符号不等式更容易。
该分支是社区原有设计，本补丁保留它，没有发明一套新的 stride 顺序算法。
不能把这种推理脱离 stride 约束推广到任意带负值的表达式。

`guard_or_false` 也不等同于第 6 项的 `statically_known_equals`。
它按现有配置对条件求值，必要时可能形成 guard；缺少可用证明时允许返回 False。
原异常发生在它内部处理表达式的过程中，不能解释成只要名字有 “or_false” 就一定不会抛异常。

## 原异常是怎样形成的

NFNet 派生训练小图第二组 shape 为 `(3,16,256,256)`。
取 `q=FloorDiv(H-1,32)`，约束 H≥33，那么 `a=q+1` 至少为 2，`b=(q+1)**2`。
在目标比较中 left=a、right=b，数学上应返回 False；反向比较应返回 True。

`%` 的操作数是 SymPy 表达式，所以这里调用的是 SymPy 的取模化简，
不是把普通 Python 整数 8 和 64 做一次取余。
通用 `Mod(a,b)` 可能提取公因子，形成类似 `a * Mod(1,b/a)` 的表达式。
当 b 以展开多项式形式存在时，某些化简路径会把含倒数的表达式留下来。
ShapeEnv 后续的整数值域分析遇到负指数 -1，触发 `Invalid ranges [0:-1]`。

H=256 时 a=8、b=64，`8 % 64 = 8` 本身没有歧义。
错误是符号表达式进入了后续整数分析不支持的形式，而不是取模的数值定义不正确。
factored 与 expanded 的 operand 形式会影响复现，不能仅记录数学等价式就认定所有环境必现。
旧异常见[基线记录][baseline]，独立社区表达式对照见[CPU 输出][cpu]。

## 第一个文件：`codegen/_sizevars.py`

### 为什么改 `patch_simplify()`，而不是只改已有 `simplify()`

这个模块原本就用于安装 NPU 所需的尺寸系统调整。
已有 `simplify()` 的主要处理是展开表达式、应用 replacements。
但 Layout 在调用尺寸求值前就通过 `%` 构造出通用 Mod；
等到 simplify 再处理，不能可靠保证去掉所有已经产生的倒数，也可能误伤其他表达式。

本修复直接替换构造取模条件的那一处，同时复用原有安装入口。
后续回移保持安装入口不变，更新比较函数的证明逻辑。
`patch_simplify()` 这个函数名因此不仅包含一行 simplify 赋值，还安装 Layout 的比较方法。
这只是现有接入位置的选择，并不表示异常根因属于 Triton 的算术 codegen。

### 新增 import 各有什么用

| 新增项 | 用途 |
| --- | --- |
| `functools` | `wraps` 保留原函数名称、文档和 `__wrapped__` 等信息 |
| `ir` | 取得公共 `ir.Layout` 类并替换方法 |
| `V` | 在函数实际被调用时读取当前图的 sizevars |
| `Mod` | 当前实现构造 PyTorch 的非负整数取模表达式节点；原提交此处导入 `PythonMod` |
| `free_symbols` | 对分子、分母一起计算符号数量，限制昂贵证明的规模 |
| `simple_floordiv_gcd`、`safe_gcd` | 回移依赖中使用的两种公因子证明入口 |

### 安装前先检查方法是否存在、是否已安装

```python
original = getattr(ir.Layout, "_stride_expr_ge_or_false", None)
if original is None or getattr(original, "_torch_npu_integer_strides", False):
    return
```

旧社区版本可能没有这个方法，此时直接跳过该 Layout 替换。
标记为 True 表示已经安装过，重复调用不会叠出多个 wrapper。
前面的 `SizeVarAllocator.simplify = simplify` 仍按既有逻辑执行，
这里的 return 只终止后续 Layout 补丁安装。

### 当前比较函数采用上游的三段证明

```python
@functools.wraps(original)
def compare_strides(left, right):
    sizevars = V.graph.sizevars
    if sizevars.guard_or_false(sympy.Eq(right, 0)):
        return True
    if sizevars.guard_or_false(sympy.Eq(left, 0)):
        return False
    return (
        sizevars.guard_or_false(sympy.Ge(left, right))
        or _statically_known_stride_multiple_of(sizevars, left, right)
        or sizevars.guard_or_false(sympy.Eq(Mod(left, right), 0))
    )
```

第一行函数体在调用时取得当前图，避免安装时绑定某一个旧图的尺寸环境。
两个零值分支和优先比较大小的行为照旧。大小关系不能证明时，按以下顺序继续：

1. `_statically_known_stride_multiple_of(sizevars, left, right)` 尝试利用表达式结构和已有约束证明整除。
   成功时直接返回 True，无需为这个整除关系增加运行时 guard。
   这也让没有样例值的 unbacked 符号能够利用已知的整除结构。
2. 静态证明仍不成功时，才构造 `Eq(Mod(left, right), 0)` 交给 `guard_or_false`。
   对有样例值的 backed 符号，这条路径可以按尺寸系统的规则生成 guard。

这里导入的是 `torch.utils._sympy.functions.Mod`，不能替换成 `sympy.Mod`。
三个条件使用 Python `or`，前面成功就不会构造后面的表达式。
它没有捕获 ValueRangeError 后一律返回 True，也没有按模型名、H 值特判。

### 为什么还需要回移整除 helper 的依赖

上游比较主体调用社区 `SizeVarAllocator.statically_known_multiple_of()`。
当前 2.13 虽然有这个同名方法，但其符号分母分支只尝试 `Eq(Mod(numerator, denominator), 0)`。
上游修复所在版本已经增加了公因子消去证明，见[对应版本的 sizevars][upstream-sizevars]。
因此接口存在不代表证明能力相同：只替换 Layout 主体时，原始 backed 复现恢复，
但上游 unbacked 复合 stride 测试仍返回 False。

当前在 NPU 文件内回移该依赖，命名为 `_statically_known_stride_multiple_of`，仅供 Layout 补丁调用：

- 相同表达式直接成立；整数分母复用原社区 helper。
- 符号分母先检查符号数量上限，沿用社区阈值，避免对过宽表达式执行昂贵证明。
- 依次尝试 `simple_floordiv_gcd`、`safe_gcd`；只有公因子覆盖整个分母，才判定可整除。
- 若仍不能证明，回到 PyTorch `Mod` 的静态判断；不新增运行时 guard。

例如分子写成 `q**2 + 2*q + 1`，分母是 `q + 1`。
若确认公因子就是整个分母 `q + 1`，即可证明分子是分母的整数倍，
无需先把取模条件交给 ShapeEnv。实际回归还覆盖了含 `FloorDiv` 的复合表达式。

没有全局替换社区的整除方法，因而其他尺寸证明调用继续使用原有实现。
Layout 本身仍按下一节说明的公共类补丁机制安装和恢复。

### 最后安装为 staticmethod

```python
compare_strides._torch_npu_integer_strides = True
ir.Layout._stride_expr_ge_or_false = staticmethod(compare_strides)
```

第一行供幂等检查使用。第二行保留原方法的静态调用协议：参数只有 left/right，
通过 Layout 实例调用时也不应自动多传一个 self。
这会替换进程中的公共类属性，不是给单个 NPU kernel 创建私有副本。

## 当前的 Mod 与历史 PythonMod 方案有什么区别

当前实现沿用上游对 tensor stride 非负整数域的处理。
PyTorch `Mod` 不能直接化简时保留自己的表达式节点，避免把通用 SymPy `Mod` 提取公因子后的
含倒数结果返回给 ShapeEnv。源码内部仍可能临时计算 `p/q`；关键在于传入后续分析的表达式形式。
它只定义非负参数域，不能据此把任意带负值的普通 Python `%` 都改成这个类。

回移不只是将 `PythonMod` 改名成 `Mod`，还增加了前面的静态整除证明。
上游同时验证了 unbacked stride 的布局判断，以及后备 guard 不含负指数、能够正常打印。

原始提交的 `PythonMod` 方案保留如下解释，便于阅读历史补丁和验证记录：

社区 `torch/utils/_sympy/functions.py` 已有 `PythonMod`，无需新增这个符号算子。
它能直接处理常量、0、整数倍等情况；不能安全化简时保留 `PythonMod(p,q)` 节点。
PyTorch 的符号解释器和值域系统有相应的整数取模处理。

需要避免一个过度简化的解释：`PythonMod.eval()` 内部也会尝试 `p/q`，
并用 `sympy.Mod(p,q)==0` 检查整除。因此不是“整个求解过程中绝不构造有理式”。
关键区别是不能化简时不把通用 Mod 的任意化简结果作为最终表达式返回，
从而避免此处将带倒数的结果直接送入后续整数值域分析。
具体分支可在本地社区源码的 `class PythonMod` 中与上述说明逐行对照。

对合法整数、非零除数，“取模为零表示整除”的数学含义保持。
但符号化简能力、guard、耗时和布局决策不保证相同，所以仍需回归检查。

## 第二个文件：`lowering_patch.py` 为什么也必须修改

NPU 后端加载已有 baseline 保存/恢复机制，用于加载其他后端前恢复公共状态。
新增 Layout 方法替换后，也必须把它纳入这个生命周期，否则切换后仍可能残留补丁。

### `_get_inductor_attr_baseline()` 新增一个保存对象

```python
from torch._inductor import ir as inductor_ir
...
_snapshot_inductor_attr(inductor_ir.Layout, "_stride_expr_ge_or_false")
```

baseline 项记录 owner、属性名、是否存在、原始值。
已有 `restore_inductor_baseline()` 会遍历这些项：原来存在就 setattr 恢复，
原来不存在就删除后来新增的属性，因此恢复函数主体不需要另加一个 Layout 专用分支。
这依赖既有加载流程先保存 baseline、再安装补丁。

### `_snapshot_inductor_attr()` 改用 `inspect.getattr_static`

旧 `getattr(Layout,name)` 会触发 Python 描述符机制，把 staticmethod 包装解开，返回函数。
若保存这个普通函数，再用 `setattr(Layout,name,function)` 恢复，
类上属性就不再是 staticmethod，通过实例调用时可能自动绑定 self。

`inspect.getattr_static(owner,name,None)` 保存原始描述符，
恢复后仍保持 `staticmethod`。它改变的是公共 snapshot helper，
所以同一 baseline 列表里其他属性的保存方式也会经过这条规则，审查范围不能只看 Layout 一项。

## 为什么需要 NPU 侧补丁，CUDA 与旧版本是否一样

当前修改范围是 NPU 仓库；出错方法属于安装的社区 torch。
通过现有 patch 入口替换方法，可以在不直接修改社区安装文件的情况下接入修复，
并使后端恢复逻辑知道它被改过。这是接入与维护选择，不意味着社区代码天然没有问题。

本地对照的 2.13 社区 Layout 同位置仍是 `%`，CUDA 也可调用这个公共方法。
因此原始 `c74c45c960` 不是复制 CUDA 已存在的 PythonMod 修复。
上游引入该 helper 的提交为 `98b772f8ab86176b8e0b572d88f4e09510c1b587`，
处于 2.13 开发周期；2.7.1、2.10.0、2.11.0、2.12.0 对应 Layout 没有这个整除 helper。
PythonMod 类本身早已存在，不能把类的存在时间和这个调用位置的引入时间混为一谈。
版本源码入口见[总览的版本证据][overview]。

后续已确认上游存在针对同一根因的[正式修复][upstream]；
[v2.14.1 的 Layout][upstream-layout] 使用静态整除证明加 PyTorch `Mod`，与当前回移后的比较逻辑一致。
旧 2.13 基线通过 NPU 侧入口接入；依赖升级到已包含上游修复的社区 torch 时，应停止覆盖该方法，
避免用本地复制的函数覆盖后续上游改进。当前幂等标记本身不负责识别社区是否已修复。

相同公共源码意味着手动构造相同表达式，理论上不需要 NPU 就能触发底层问题。
但 CUDA 上同一个模型会不会自然构造同样 operand、约束和 layout 路径，仍需实际验证。
不能把现有证据解释成“只有 NPU 线性化导致”或“CUDA 模型一定失败”。

## 测试分别证明什么

原始补丁内四个单元方法覆盖：

| 检查 | 目的 |
| --- | --- |
| 线性与平方 stride 双向比较 | a 与 a² 的顺序恢复 |
| 多项式及展开形式 | 不只支持一个书写形态 |
| 静态 stride 和 0 | 保持原分支顺序与边界行为 |
| 子进程中安装、重复安装、恢复、重装 | 幂等、原描述符和生命周期 |

结果见[最终检查][checks]；[双 shape 小图][training]的静态/动态前向、输入梯度、权重梯度通过。
原 NFNet 整网矩阵没有观察到同一新异常，因此必要性证据属于派生小图。

新增的[CUDA 验证脚本说明][probe]区分纯表达式、当前安装的原生 Layout 方法、真实 CUDA 编译。
本地 CPU 的 torch 2.9.1 没有该 helper，脚本明确 SKIP 原生方法；
但不导入 torch_npu 的展开 `%` 表达式已复现 `Invalid ranges [0:-1]`，PythonMod 对照通过。
这证明底层符号问题不依赖 NPU，不能写成 2.9.1 原生 Layout 也自然触发。
以上 CPU/脚本文档工作对应 2026-10-22，没有新增 CUDA 或 NPU 测试结果；
后续 NPU 上游回移验证单独记录。

后续上游回移增加两个单元方法：`test_unbacked_compound_stride_order` 检查无样例值符号的布局证明，
`test_fallback_guard_has_no_negative_powers` 检查静态证明失败后的 guard 不含负指数且能正常打印。
它们与原有四项共同验证当前实现；新的设备结果与历史 PythonMod 结果分开记录。

## 2026-10-23 NPU_A5 上游回移验证

本轮在 Docker 内的 `pta_213_cicd` 环境实测：社区 torch 为 `2.13.0+cpu`、
`cf30153c4c131c8164ee7798e5022d810682e2cb`，torch_npu 为 `2.13.0.dev20261019`，
SymPy 为 `1.14.0`，设备为 `Ascend950PR_9579`。
框架代码已推送并在远端取到 `089231bede6e3bf3384e1ded83d6ca45ac739012`。

运行使用原 v8 overlay 的独立副本，仅替换本次更新的 `_sizevars.py`；
全部七个覆盖文件均与该提交对应源码逐个核对 SHA256，见[环境记录][upstream-env]和[文件校验][upstream-hashes]。
没有覆盖环境中全局安装的 torch_npu。静态、动态分别使用独立的新 Inductor/Triton 缓存。

| 检查 | 本轮结果 | 证据 |
| --- | --- | --- |
| 恢复社区原 `%` 比较方法，其余整合修复保留 | 预期失败，复现 `Invalid ranges [0:-1]` | [原实现对照][upstream-baseline] |
| 只回移 Layout 主体的 `4b710b7534` | 5/6，unbacked 复合 stride 证明失败 | [依赖缺失对照][upstream-layout-only] |
| 补齐依赖后的六项 stride 单元 | 6/6，包括安装、恢复与重装 | [当前单元结果][upstream-checks] |
| 动态训练，依次执行两组 shape | 前向、输入梯度、权重梯度均通过 | [动态训练][upstream-dynamic] |
| 静态训练，两组 shape 对照 | 前向、输入梯度、权重梯度均通过 | [静态训练][upstream-static] |

训练使用[原小图](../../../scripts/repro/cicd/model_fixes_20261020/symbolic_group_training.py)，
输入为 `(2,16,224,224)`、`(3,16,256,256)`，weight 为 `(16,16,1,1)`，dtype 为 float32。
前向比较使用 `rtol=2e-4, atol=2e-4`，两项梯度使用 `rtol=2e-3, atol=2e-3`。
动态第二组就是此前暴露 Layout 异常的输入。
静态和动态进程分别记录到当前比较函数调用 100 次、异常 0 次；
设置 `torch._dynamo.config.suppress_errors=False`，确保编译异常正常上报。

激活上述环境、加载对应 overlay 后，核心验证命令为：

```bash
python <framework>/test/_inductor/test_symbolic_stride_comparison.py -v
python <repro>/symbolic_group_training.py
python <repro>/symbolic_group_training.py --static
```

其中 `<framework>` 是该框架提交的源码目录，`<repro>` 是上述小图所在目录。
实际执行还包含原方法对照和比较函数计数，完整参数保存在[命令记录][upstream-commands]。
[验证说明][upstream-validation]记录容差、缓存与 overlay 范围；路径和进程信息已去敏。
本轮确认上游方案及其所需依赖能修复目标问题；未重跑四模型的完整 TorchBench 矩阵，也未新增 CUDA 测试。

## 影响范围和剩余风险

替换的是公共 Layout 类，没有 device guard；同进程其他后端也可能调用到它。
潜在回归包括证明能力变保守而增加 copy、guard 变化、编译耗时，以及后端切换恢复问题。
未来社区增加新分支时，复制的比较主体还需同步；幂等标记只能防重复安装，不能自动适配新算法。
静态与多项式单元及小图通过，不是所有后端、stride、符号约束和性能的完整保证。

[overview]: model-fix-dsl-explained-20261021.md
[patch]: ../../../artifacts/cicd/model_fixes_20261020/patches/layout-symbolic-strides.patch
[baseline]: ../../../artifacts/cicd/model_fixes_20261020/layout_checks_v7/baseline_unit.txt
[checks]: ../../../artifacts/cicd/model_fixes_20261020/layout_checks_v8/fixed_stride_unit.txt
[training]: ../../../artifacts/cicd/model_fixes_20261020/layout_checks_v8/fixed_training.txt
[cpu]: ../../../artifacts/cicd/model_fixes_20261020/final_checks/stride_cuda_probe_cpu_20261022.jsonl
[probe]: ../../../scripts/repro/cicd/model_fixes_20261020/symbolic_stride_cuda_probe.md
[upstream]: https://github.com/pytorch/pytorch/commit/93879b11e3b5ce6021d77c808295d9e0d1e24944
[upstream-layout]: https://github.com/pytorch/pytorch/blob/v2.14.1/torch/_inductor/ir.py#L4325
[upstream-sizevars]: https://github.com/pytorch/pytorch/blob/93879b11e3b5/torch/_inductor/sizevars.py#L582
[upstream-patch]: ../../../artifacts/cicd/model_fixes_20261020/patches/layout-symbolic-strides-upstream.patch
[upstream-status]: ../../../artifacts/cicd/model_fixes_20261020/upstream_stride_20261023/status.json
[upstream-env]: ../../../artifacts/cicd/model_fixes_20261020/upstream_stride_20261023/environment.json
[upstream-hashes]: ../../../artifacts/cicd/model_fixes_20261020/upstream_stride_20261023/overlay_hashes.json
[upstream-baseline]: ../../../artifacts/cicd/model_fixes_20261020/upstream_stride_20261023/baseline_original_modulo.txt
[upstream-layout-only]: ../../../artifacts/cicd/model_fixes_20261020/upstream_stride_20261023/layout_only_unit.txt
[upstream-checks]: ../../../artifacts/cicd/model_fixes_20261020/upstream_stride_20261023/upstream_stride_unit.txt
[upstream-dynamic]: ../../../artifacts/cicd/model_fixes_20261020/upstream_stride_20261023/dynamic_training.txt
[upstream-static]: ../../../artifacts/cicd/model_fixes_20261020/upstream_stride_20261023/static_training.txt
[upstream-commands]: ../../../artifacts/cicd/model_fixes_20261020/upstream_stride_20261023/commands.json
[upstream-validation]: ../../../artifacts/cicd/model_fixes_20261020/upstream_stride_20261023/validation.json
