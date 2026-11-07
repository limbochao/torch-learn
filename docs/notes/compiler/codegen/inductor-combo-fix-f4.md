# F4：修正 FloorDiv 拆分、保留无进位优化与 pooling 语义 mask

本修复处理 `test_adaptive_avg_pool2d1` 在 NPU 上的编译失败及随后暴露的数值错误。
它包含三个相互独立但在同一原始方法里逐层出现的问题：SymPy 求值后的类型假设不成立、
加法整除拆分丢失余数进位、pooling 的语义 mask 被按指针轴关系错误删除。
最终提交还加入了编译期无进位证明，在保证语义正确的前提下保留更多拆分机会。

| 项目 | 内容 |
| --- | --- |
| PR 对应项 | 第 4 点 |
| 修复提交 | `53f8af09f144aaaec685562acd4c3a3ffaffb71a`，F4 全部内容只有一个提交 |
| 修改文件 | `torch_npu/_inductor/codegen/ir.py`、`torch_npu/_inductor/codegen/triton.py` |
| 单提交修改量 | 2 个文件，新增 49 行、删除 10 行 |
| 主要入口 | `eliminate_zero_term`、`split_expression`、`_floor_div_terms_do_not_carry`、`filter_masks` |
| 最新验证 | 最终源码 `a8dcca84bf52e8a3e0c46aa46c2a4f663875f10e`，Combo False/True 共 2/2 PASS |
| 上游关系 | 修复 NPU 专有拆分及 mask 过滤逻辑；数学与访问语义对齐，不是上游同名 helper 的直接移植 |

## 原方法覆盖了什么

函数同时返回两个 adaptive average pooling 结果：

```python
def fn(x):
    return aten._adaptive_avg_pool2d(x, (6, 6)), aten._adaptive_avg_pool2d(
        x + 1, (2, 5)
    )
```

输入依次为 `(2,4,16,16)`、`(2,4,3,3)`、`(2,4,6,6)`。
第一组保留原 `check_lowp=False`，后两组保留方法默认的低精度相关检查；
同时运行 Combo 关闭和开启两种参数化实例。原有数值、梯度检查和阈值没有为修复而放宽。

16×16 到 6×6 涉及不能简单整除的窗口边界；3×3 到 6×6 涉及输出尺寸大于输入的窗口有效性；
6×6 到 6×6 还覆盖第一个输出的 no-op 情况。第二个输出仍使用不同尺寸，不能把整组方法都理解成 no-op。
这也是为什么只通过某一输入 shape，不能认为整个 F4 方法已经修好。

## 第一层：FloorDiv 构造结果不一定仍是 FloorDiv

`FloorDiv(a, d)` 是 PyTorch 的 SymPy 表达式，表示 Python 风格的向负无穷取整除法。
构造函数会立即尝试求值，例如：

```python
FloorDiv(0, 3)  # 返回整数 0
FloorDiv(6, 3)  # 返回整数 2
FloorDiv(x, 1) # 可直接返回 x
```

旧 `eliminate_zero_term` 一进入函数就执行 `expr, divisor = term.args`，
假设传入对象必有两个参数。这个假设与 SymPy 的即时化简不兼容。
新增的类型检查只在对象仍为 `FloorDiv` 时继续解包，已经化简好的结果原样返回。

这解决的是辅助函数的输入契约，不会把常量重新包装成一个不可化简的除法。
社区 `FloorDiv.eval` 中零分子、除数为 1 等直接返回逻辑可在
[上游源码摘录](../../../../artifacts/combo-five-fixes-review-20261107/upstream-v2.10.0.md)核查。

## 第二层：为什么不能普遍拆成两个独立除法

设 `d>0`，每个整数项满足 `a_k = q_k*d + r_k`，且 `0≤r_k<d`。
则精确关系为：

```text
floor((a_1 + ... + a_n) / d)
= floor(a_1/d) + ... + floor(a_n/d) + floor((r_1 + ... + r_n)/d)
```

旧实现对每个加项独立做 `FloorDiv` 再相加，丢掉最后一项。
只有余数总和小于除数时，最后一项才为 0，拆分才成立。
例子 `(8+10)//3=6`，而 `8//3+10//3=2+3=5`，差的 1 就是余数 `2+1` 进位。

`(8*i+10)//3` 中的 `i` 可以是输出坐标或循环索引，即使 shape 完全静态，kernel 也要处理不同坐标。
它不必是“动态输入尺寸的符号”。取 `i=0..5`：

| i | 合并 `(8*i+10)//3` | 独立 `(8*i)//3 + 10//3` | 是否相同 |
| --- | --- | --- | --- |
| 0 | 3 | 3 | 是 |
| 1 | 6 | 5 | 否 |
| 2 | 8 | 8 | 是 |
| 3 | 11 | 11 | 是 |
| 4 | 14 | 13 | 否 |
| 5 | 16 | 16 | 是 |

在部分坐标上碰巧正确，不能证明整个索引变换安全。
Adaptive pooling 的边界包含类似 `floor(i*I/O)` 和 `ceil((i+1)*I/O)` 的表达式，
错误拆分会改变窗口边界、访问位置或窗口大小，直接影响结果。

## 先提取整除项，再证明其他项是否能安全拆分

新 `split_expression` 分两步：

1. 用 `statically_known_multiple_of(term, divisor)` 识别已证明整除的项。
   这些项的余数为 0，可直接移出总 FloorDiv。
2. 把其余项放在 `remainder_terms`。若编译期证明所有余数的和严格小于除数，再独立拆分；
   否则将这些项合并在一个 FloorDiv 中。

例如 `(8*x+y+z)//4` 可以先变为 `2*x+(y+z)//4`，不需要先证明 `y`、`z` 是否能拆。
若额外知道 `0≤y≤1`、`0≤z≤2`，余数总上界为 3，可以继续拆；不知道范围时保留合并除法。

这使“修正错误”和“保留安全优化”有明确边界。代码不会为可能进位的表达式建立运行期分支，
也不会为尝试优化增加一个新的 shape guard。

### 证明函数为什么需要循环索引范围

`shape_env.var_to_range` 主要保存图中符号的范围。
拆分表达式还可能包含 range tree 中的有效循环索引，它们的范围是 `[0, length-1]`。
helper 先复制 shape_env 范围，再从 `range_tree_nodes` 和 `range_tree_nodes_removed` 查询这些索引。
后者保存线性化过程中已移除、但表达式分析仍可能引用的节点信息。

`bound_sympy(node.length, ranges).upper` 给出 length 的保守上界，减 1 后得到有效索引上界。
对空长度，`max(0, upper-1)` 将区间扩为至少包含 0，避免建立反向区间；这是保守扩大证明域，
不是让空循环在运行期多执行一次。
分析结果只存在于复制出来的字典，不更新全局 shape_env。

范围证明针对有效逻辑索引；设备上 tile padding 的无效 lane 仍依赖原有 axis mask 保护。
这次验证没有给所有 tiling 与 padding 组合建立通用形式证明，文档不把逻辑范围检查等同于完整内存安全证明。

### 为什么先对系数取模

若除数与系数都是具体整数、其余因子也是整数，有：

```text
(10*x) % 8 = (2*x) % 8
```

先将系数 10 约化成 2，再做区间求界，可能得到更紧的余数上界。
这一步只替换证明过程中的局部变量 `term`，最终表达式仍按原 `remainder_terms` 构造，
并没有把分子 `10*x` 错误改成 `2*x`。

对 `0≤x≤1、0≤y≤5`，`(10*x+y)//8` 的两个余数上界为 2 与 5，总和为 7，小于 8。
因此可以拆成 `(10*x)//8+y//8`，再消去恒为 0 的后项，最终可简化为 `(5*x)//4`。
若 `y` 上界增至 6，总上界变成 8：取 `x=1,y=6` 就实际进位，必须保留合并除法。

### 为什么用严格小于、总和和 PythonMod

严格小于避免漏掉“恰好等于除数”的进位。
所有项的余数必须求总和：例如除数为 5，三个余数分别为 2，各对之和都是 4，但三个相加为 6，仍然进位。

负数也需要一致的商余定义。Python 语义中 `-1//3=-1`，`-1%3=2`，
满足 `-1=(-1)*3+2`；不能混入向 0 截断的余数规则。
`PythonMod` 与 `FloorDiv` 使用配套语义，上游 TritonPrinter 对可能带负号的 FloorDiv
也使用 `div_floor_integer`，见[两版上游摘录](../../../../artifacts/combo-five-fixes-review-20261107/upstream-v2.13.0.md)。

helper 要求正整数除数和整数项。无法证明前置条件，或者出现已捕获的缺失属性、缺失范围、
不支持操作、`ValueRangeError` 时，返回 False，保留合并形式。
它使用每项独立余数上界，可能因为忽略相关性而少证明一些本来安全的情况；这是优化机会不足，不是允许错误拆分。
捕获列表不是任意异常兜底，不能宣称所有未知表达式都绝不会引发其他编译错误。

## 第三层：地址里没有某个轴，为什么仍需要该轴的 mask

mask 是保护 load/store 的布尔条件。这里需要区分两类：

| 类型 | 作用 | 本修复如何处理 |
| --- | --- | --- |
| 普通轴 mask | 限制 tiling 或循环 lane 不越过有效轴范围 | 保留原来的筛选规则 |
| `TritonCSEVariable` 语义 mask | 表达窗口边界、条件访问等算子有效性 | 删除基于地址轴子集关系丢弃它们的循环 |

CSE variable 是编译器为已生成表达式分配的临时变量；这里临时变量的值是一个访问条件。
一个 load 的地址可能只包含输入坐标，但它是否属于当前输出窗口，还依赖输出窗口对应的轴。
“条件依赖的轴没有出现在地址里”不能证明条件无效。

3×3 输入到 6×6 输出时，以一维为例，第 0 个输出窗口：

```text
start = floor(0*3/6) = 0
end   = ceil((0+1)*3/6) = 1
```

有效位置只有 0。二维第一个窗口只包含输入 `[0,0]`。
生成器即使展开了多个候选邻居，窗口外邻居也必须由语义 mask 排除。
旧 `filter_masks` 在地址轴子集条件不成立时清除 `variable_mask_vars`，
会使候选邻居无条件参与 sum，而分母仍按真正窗口元素数计算，造成数值错误。

修复删除这个语义 mask 清除循环。下方仅对 `normal_mask_vars` 的轴 mask 清理仍然存在；
`save_variable_mask` 还被上方轴处理逻辑使用，所以本提交没有删除其计算。
没有把所有 mask 都无条件保留，也没有以 adaptive pooling 算子名或特定 shape 加例外。

### 实际失败证据：修正除法后，3×3 到 6×6 仍然数值错误

保留语义 mask 的依据来自本次实际失败。在 `fix_f4_floor_carry_v2` 阶段，已处理前述
FloorDiv 类型及加法进位问题，但 `test_adaptive_avg_pool2d1` 运行到输入 `(2,4,3,3)` 时，
第一个输出 `(2,4,6,6)` 仍报数值不一致。Combo False 实例的历史日志明确记录：

```text
AssertionError: Tensor-likes are not close!
Mismatched elements: 288 / 288 (100.0%)
Greatest absolute difference: 5.596200466156006
```

这是修复过程中的历史失败，当前正式修复后的该方法已经 PASS。
原始异常见[该阶段日志](../../../../artifacts/combo-fixes-85dec1f-20261106/fix_f4_floor_carry_v2/evidence/test_torchinductor.test_adaptive_avg_pool2d1.log)。
288 正好是 `2*4*6*6`，说明失败覆盖这次输出的所有元素。

该阶段的[完整生成候选](../../../../artifacts/combo-fixes-85dec1f-20261106/fix_f4_floor_carry_v2/evidence/combo_sources/audit.test_torchinductor.test_adaptive_avg_pool2d1/2123bff851e2e8b26eb6c806c3c40990497665e68c9b0728906e32a4c16ad9c9.py)
展开了四个邻居，地址偏移分别为 `0`、`1`、`3`、`4`。
以偏移 1 的邻居为例，实际生成的 load 如下，保留原格式：

```python
                    tmp19 = tl.load(in_ptr0 + (1 + x5 + 3*y3 + 9*z0), x5_mask & y3_mask & z0_mask, other=0.0)
```

该语句只有 `x5_mask & y3_mask & z0_mask`，缺少窗口条件 `tmp18`。
`tmp18` 在同一候选的后面仍被计算，包含邻居的窗口有效性及相关轴 mask；
但它没有用于这个 load。窗口边界表达式包含 `y4`、`x6`，而此 load 的指针地址只包含
`z0`、`y3`、`x5`。旧过滤规则据此丢弃语义 mask，导致不能从“地址轴关系”推出的窗口限制丢失。

分子和分母因此采用了不同的有效元素集合：四个邻居经缺失窗口条件的 load 进入求和，
而计数仍通过 `tl.where(tmp10, tmp37, tmp38)` 等语句按窗口 mask 计算，其中 `tmp37` 为 1、`tmp38` 为 0。
对于前面推导的第一个输出窗口，真正有效的只有 `[0,0]`，其余三个候选邻居应贡献 0；
旧代码却让它们参与求和，再除以有效计数 1，结果自然不等于正确的单元素平均值。
这直接解释了为何仅修正 FloorDiv 还不能让完整方法通过。

### 验证因果关系：恢复四个 load 的窗口条件

单 kernel 的[DSL 诊断快照](../../../../artifacts/combo-fixes-85dec1f-20261106/pool_mask_dsl.py)
保留这组输入和计算，将四个 load 分别放到对应语义 mask 定义之后，使用
`tmp10`、`tmp18`、`tmp27`、`tmp34` 保护访问。上面偏移 1 的访问变为：

```python
                tmp19 = tl.load(in_ptr0 + (1 + x5 + 3*y3 + 9*z0), tmp18, other=0.0)
```

当该邻居不属于当前窗口时，`other=0.0` 使它不再贡献到求和中。
历史实验与 eager `adaptive_avg_pool2d(x, (6,6))` 比较，
[实际结果](../../../../artifacts/combo-fixes-85dec1f-20261106/pool_mask_dsl.log)为：

```json
{"semantic_mask_dsl_check": "passed", "max_abs": 0.0}
```

因此修复删除的是整个“可能丢弃语义 mask”的循环，不是删除窗口 mask。
其中单独被删除的 `if save_variable_mask:` 只是该旧循环的保留分支；
必须结合后面的 `mask_vars.discard(mask_var)` 理解，不能孤立解释成“现在不再需要 mask”。
`save_variable_mask` 的计算及普通轴 mask 的既有处理仍然保留。

正式 codegen 修复后，[审计](../../../../artifacts/combo-f4-nocarry-20261106/codegen_checks.json)
确认语义 mask 在 load 前定义且用于访问；原方法 Combo False/True
[2/2 PASS](../../../../artifacts/combo-f4-nocarry-20261106/f4_nocarry_validation/evidence/functional_results.json)，
后续最终源码全量也再次通过这两个实例。
单 kernel 实验证明这组输入的访问条件问题；完整方法验证覆盖原三组输入。
这些都是已有运行证据，本次文档补充没有重新执行 NPU 测试，也没有从单 kernel 结果推定所有模型均已验证。

## 完整修改与逐行说明

下面的 diff 原样取自本提交相对父提交的变化，包含新增、删除及必要上下文。
表中“旧/新”行号分别属于 diff 两侧的文件；空白行仅用于排版，不逐项计入。
后续提交可能使行号移动，定位时应以本节固定的 commit 为准。

```diff
diff --git a/torch_npu/_inductor/codegen/ir.py b/torch_npu/_inductor/codegen/ir.py
index 4d14b306e0..9a42b34b27 100644
--- a/torch_npu/_inductor/codegen/ir.py
+++ b/torch_npu/_inductor/codegen/ir.py
@@ -10,7 +10,8 @@ from torch._inductor.ir import FloorDiv, ModularIndexing
 from torch._inductor.loop_body import CaptureIndexing, LoopBody, MemoryUsageType
 from torch._inductor.utils import has_free_symbols, sympy_index_symbol, sympy_subs
 from torch._inductor.virtualized import V
-from torch.utils._sympy.value_ranges import IntInfinity, ValueRanges
+from torch.utils._sympy.functions import PythonMod
+from torch.utils._sympy.value_ranges import IntInfinity, ValueRangeError, ValueRanges, bound_sympy
 
 from ..config import inductor_indirect_memory_mode, is_ascend950, log
 from .triton import NPUIndexTritonKernel
@@ -1144,6 +1145,9 @@ def extract_modular_indexing_coefficient(expr):
 
 
 def eliminate_zero_term(term):
+    # FloorDiv can already have evaluated to a constant or another expression.
+    if not isinstance(term, FloorDiv):
+        return term
     expr, divisor = term.args
     if not isinstance(expr, sympy.Symbol):
         return term
@@ -1223,6 +1227,36 @@ def eliminate_modular(term):
     return term
 
 
+def _floor_div_terms_do_not_carry(terms, divisor):
+    sizevars = V.graph.sizevars
+    if divisor.is_integer is not True or not sizevars.statically_known_gt(divisor, 0):
+        return False
+    if not all(term.is_integer is True for term in terms):
+        return False
+
+    ranges = dict(sizevars.shape_env.var_to_range)
+    try:
+        for symbol in sympy.Add(*terms).free_symbols:
+            node = V.kernel.range_tree_nodes.get(symbol)
+            if node is None:
+                node = V.kernel.range_tree_nodes_removed.get(symbol)
+            if node is not None:
+                upper = bound_sympy(sympy.sympify(node.length), ranges).upper
+                ranges[symbol] = ValueRanges(0, max(0, upper - 1))
+
+        remainder_upper = sympy.Integer(0)
+        for term in terms:
+            coefficient, rest = term.as_coeff_Mul()
+            if isinstance(divisor, sympy.Integer) and coefficient.is_Integer and rest.is_integer is True:
+                # Reduce coefficients before bounding: (10*x) % 8 == (2*x) % 8.
+                term = (coefficient % divisor) * rest
+            # PythonMod keeps floor-division semantics for negative terms too.
+            remainder_upper += bound_sympy(PythonMod(term, divisor), ranges).upper
+        return sizevars.statically_known_lt(remainder_upper, divisor)
+    except (AttributeError, KeyError, NotImplementedError, ValueRangeError):
+        return False
+
+
 def split_expression(expr):
     """
     Split expression according to specified logic:
@@ -1250,12 +1284,19 @@ def split_expression(expr):
 
         # Check if arg is Add
         if isinstance(arg, sympy.Add):
-            # Assume denominator is 1: floor(a+b) -> floor(a) + floor(b)
-            split_terms = []
+            # Only exact multiples can move outside floor division. Keep the
+            # remaining terms together so their remainders can carry.
+            split_terms, remainder_terms = [], []
             for term in arg.args:
-                new_term = FloorDiv(term, divisor)
-                new_term = eliminate_zero_term(new_term)
-                split_terms.append(new_term)
+                if V.graph.sizevars.statically_known_multiple_of(term, divisor):
+                    split_terms.append(FloorDiv(term, divisor))
+                else:
+                    remainder_terms.append(term)
+            if len(remainder_terms) > 1 and _floor_div_terms_do_not_carry(remainder_terms, divisor):
+                split_terms.extend(eliminate_zero_term(FloorDiv(term, divisor)) for term in remainder_terms)
+            else:
+                remainder = FloorDiv(sympy.Add(*remainder_terms), divisor)
+                split_terms.append(eliminate_zero_term(remainder))
             return sympy.Add(*split_terms)
 
         # Cannot split, return original expression
diff --git a/torch_npu/_inductor/codegen/triton.py b/torch_npu/_inductor/codegen/triton.py
index 48e158a29f..ce1fbf932a 100644
--- a/torch_npu/_inductor/codegen/triton.py
+++ b/torch_npu/_inductor/codegen/triton.py
@@ -4662,10 +4662,8 @@ class NPUIndexTritonKernel(TritonKernel):
 
             masked_axis_name.append(node.name)
 
-        for mask_var in variable_mask_vars:
-            if save_variable_mask:
-                continue
-            mask_vars.discard(mask_var)
+        # Semantic masks can depend on axes absent from the address (for
+        # example, a pooling window boundary). They must still guard the load.
 
         # Be careful when remove mask from load store
         # 1. Current only leave axis mask from sorted axis
```

| 文件 / 位置 | 修改行 | 为什么这样处理 |
| --- | --- | --- |
| ir.py 旧 13 | `- from torch.utils._sympy.value_ranges import IntInfinity, ValueRanges` | 将原 import 替换为包含范围求界及保守异常处理所需名称的 import。 |
| ir.py 新 13 | `+ from torch.utils._sympy.functions import PythonMod` | 使用 Python floor/mod 语义的取余表达式，确保负数项也符合商余分解公式。 |
| ir.py 新 14 | `+ from torch.utils._sympy.value_ranges import IntInfinity, ValueRangeError, ValueRanges, bound_sympy` | 保留原范围类型，新增 bound_sympy 求上下界和 ValueRangeError 捕获。 |
| ir.py 新 1148 | `+ # FloorDiv can already have evaluated to a constant or another expression.` | 解释类型检查缘由：SymPy 构造 FloorDiv 后可能已返回常量或另一种表达式。 |
| ir.py 新 1149 | `+ if not isinstance(term, FloorDiv):` | 只有仍为 FloorDiv 的对象才允许按两个参数解包。 |
| ir.py 新 1150 | `+ return term` | 对不满足本函数处理条件的表达式原样返回，不强行读取 args，也不改变其值。 |
| ir.py 新 1230 | `+ def _floor_div_terms_do_not_carry(terms, divisor):` | 新增编译期证明函数：判断多个非整除项的余数总和是否不可能进位。 |
| ir.py 新 1231 | `+ sizevars = V.graph.sizevars` | 使用当前 graph 的符号尺寸分析器，复用已知静态关系。 |
| ir.py 新 1232 | `+ if divisor.is_integer is not True or not sizevars.statically_known_gt(divisor, 0):` | 要求除数是已知正整数；无法证明整数性或正性时不采用额外拆分。 |
| ir.py 新 1233 | `+ return False` | 除数的整数性或正性无法证明时停止无进位证明，调用方保留剩余项的合并 FloorDiv。 |
| ir.py 新 1234 | `+ if not all(term.is_integer is True for term in terms):` | 所有被分析项必须可证明为整数，才能使用整数商余恒等式。 |
| ir.py 新 1235 | `+ return False` | 某一项无法证明为整数时停止；不能将整数商余公式套用到未确认整数性的表达式。 |
| ir.py 新 1237 | `+ ranges = dict(sizevars.shape_env.var_to_range)` | 复制现有符号范围；后续补入循环索引范围时不修改 shape_env 原字典。 |
| ir.py 新 1238 | `+ try:` | 将可能不支持的范围求界放入保守回退区域。 |
| ir.py 新 1239 | `+ for symbol in sympy.Add(*terms).free_symbols:` | 遍历剩余和式实际涉及的符号，尝试获取对应循环索引的有效范围。 |
| ir.py 新 1240 | `+ node = V.kernel.range_tree_nodes.get(symbol)` | 优先查当前有效 range tree 中的循环节点。 |
| ir.py 新 1241 | `+ if node is None:` | 当前 range tree 没找到时，再查已移除节点记录，覆盖线性化后仍出现在表达式中的符号。 |
| ir.py 新 1242 | `+ node = V.kernel.range_tree_nodes_removed.get(symbol)` | 获取已移除 range tree 节点保存的 length，供证明使用。 |
| ir.py 新 1243 | `+ if node is not None:` | 只对确实找到循环节点的符号补入索引范围；其他符号沿用原有范围信息。 |
| ir.py 新 1244 | `+ upper = bound_sympy(sympy.sympify(node.length), ranges).upper` | 先将循环 length 转为 SymPy 表达式，再用当前符号范围求其上界。 |
| ir.py 新 1245 | `+ ranges[symbol] = ValueRanges(0, max(0, upper - 1))` | 有效零基循环索引取 [0, length上界-1]；max 将空长度上界保守扩为含 0 的区间。 |
| ir.py 新 1247 | `+ remainder_upper = sympy.Integer(0)` | 以 SymPy 整数 0 初始化全部余数上界之和。 |
| ir.py 新 1248 | `+ for term in terms:` | 逐项累计；需要证明所有项的总和，而不是只比较任意两项。 |
| ir.py 新 1249 | `+ coefficient, rest = term.as_coeff_Mul()` | 分离乘积中的数值系数与其余因子，为系数取模约化准备。 |
| ir.py 新 1250 | `+ if isinstance(divisor, sympy.Integer) and coefficient.is_Integer and rest.is_integer is True:` | 只在除数、系数都是具体整数且剩余因子为整数时应用系数模约化。 |
| ir.py 新 1251 | `+ # Reduce coefficients before bounding: (10*x) % 8 == (2*x) % 8.` | 说明约化的数学依据；较小系数通常能得到更紧的余数上界。 |
| ir.py 新 1252 | `+ term = (coefficient % divisor) * rest` | 仅更换证明过程中的局部项，不修改最终返回表达式的原项。 |
| ir.py 新 1253 | `+ # PythonMod keeps floor-division semantics for negative terms too.` | 注明负数的余数必须与向负无穷取整的商配套，不能替换成截断除法的余数。 |
| ir.py 新 1254 | `+ remainder_upper += bound_sympy(PythonMod(term, divisor), ranges).upper` | 求当前项余数的保守上界并累加；忽略项间相关性可能少优化，但不能凭空缩小上界。 |
| ir.py 新 1255 | `+ return sizevars.statically_known_lt(remainder_upper, divisor)` | 只有总上界严格小于除数才证明成功；等于除数也会进位，必须拒绝。 |
| ir.py 新 1256 | `+ except (AttributeError, KeyError, NotImplementedError, ValueRangeError):` | 这些不支持或缺失范围信息的情况回退为不拆分；并非捕获所有编译异常。 |
| ir.py 新 1257 | `+ return False` | 上述四类范围分析异常发生时放弃额外优化，不因证明失败就执行未证明安全的拆分。 |
| ir.py 旧 1253 | `- # Assume denominator is 1: floor(a+b) -> floor(a) + floor(b)` | 删除原来不适用于一般整数除数的假设，避免丢失余数之和造成的进位。 |
| ir.py 旧 1254 | `- split_terms = []` | 原实现只有拆分项列表，不能保留需要共同参与进位的项；改为两类列表。 |
| ir.py 新 1287 | `+ # Only exact multiples can move outside floor division. Keep the` | 说明第一阶段只无条件提取已证明整除的项。 |
| ir.py 新 1288 | `+ # remaining terms together so their remainders can carry.` | 说明其余项默认保持合并，保证余数进位。后续额外证明成功时才进一步展开。 |
| ir.py 新 1289 | `+ split_terms, remainder_terms = [], []` | 分别保存已经安全提取的商，以及暂时需要留在同一个分子里的项。 |
| ir.py 旧 1256 | `- new_term = FloorDiv(term, divisor)` | 删除对每个加项直接构造独立商的旧路径；该路径没有证明不会进位。 |
| ir.py 旧 1257 | `- new_term = eliminate_zero_term(new_term)` | 删除旧路径中对独立商的后处理；新路径只对安全独立商或合并商执行该处理。 |
| ir.py 旧 1258 | `- split_terms.append(new_term)` | 删除将每项独立商无条件累加的旧行为。 |
| ir.py 新 1291 | `+ if V.graph.sizevars.statically_known_multiple_of(term, divisor):` | 复用静态整除证明；该项余数为 0，可安全移出 FloorDiv。 |
| ir.py 新 1292 | `+ split_terms.append(FloorDiv(term, divisor))` | 记录已证明整除项的商。 |
| ir.py 新 1293 | `+ else:` | 当前项未被静态整除证明接受，进入收集剩余项的分支。 |
| ir.py 新 1294 | `+ remainder_terms.append(term)` | 收集尚未证明可独立拆分的原始项，保留其和式关系。 |
| ir.py 新 1295 | `+ if len(remainder_terms) > 1 and _floor_div_terms_do_not_carry(remainder_terms, divisor):` | 剩余项超过一个时尝试额外无进位证明；单项无需做多项证明。 |
| ir.py 新 1296 | `+ split_terms.extend(eliminate_zero_term(FloorDiv(term, divisor)) for term in remainder_terms)` | 证明成功后生成各项独立商，并尝试删除可证明为 0 的商。 |
| ir.py 新 1297 | `+ else:` | 剩余项不超过一项，或无进位证明未成功时，改为构造一个合并 FloorDiv。 |
| ir.py 新 1298 | `+ remainder = FloorDiv(sympy.Add(*remainder_terms), divisor)` | 未证明成功则将剩余项合并后除一次；空列表构成 0，SymPy 可直接求值。 |
| ir.py 新 1299 | `+ split_terms.append(eliminate_zero_term(remainder))` | 把合并商加入结果；类型保护同时兼容它已经化为常量的情况。 |
| triton.py 旧 4665 | `- for mask_var in variable_mask_vars:` | 删除遍历并清除 CSE 语义 mask 的旧循环；下方普通轴 mask 的过滤循环保留。 |
| triton.py 旧 4666 | `- if save_variable_mask:` | 随整个语义 mask 删除循环移除。旧条件在 3×3→6×6 pooling 中未能保留窗口条件，使窗口外邻居参与求和，见上文实际失败证据。 |
| triton.py 旧 4667 | `- continue` | 删除上述旧循环的跳过逻辑；不再需要靠该条件决定语义 mask 的生死。 |
| triton.py 旧 4668 | `- mask_vars.discard(mask_var)` | 删除按地址轴关系清除语义 mask 的操作；地址未使用某轴不代表该轴的有效性条件无用。 |
| triton.py 新 4665 | `+ # Semantic masks can depend on axes absent from the address (for` | 解释语义 mask 可依赖指针表达式未出现的轴。 |
| triton.py 新 4666 | `+ # example, a pooling window boundary). They must still guard the load.` | 给出 pooling 窗口边界实例，要求该条件继续保护 load。 |


## 与社区上游对齐的准确说法

社区 2.10、2.13 没有这套 NPU `split_expression` / `_floor_div_terms_do_not_carry` helper。
因此无进位证明是 NPU 线性化优化的实现，不能称为摘取上游同名修复。
但拆分必须保持 FloorDiv 语义、语义 mask 不能因地址不包含某轴而被删，属于相同的算子正确性要求。

社区两个版本的 `TritonKernel.filter_masks` 主要删除已证明为常量的轴 mask 和 `None`，
没有 NPU 此处按地址轴关系整体清除 CSE 语义 mask 的逻辑。
本提交保留 NPU 现有轴 mask 筛选，只去掉错误的语义 mask 清除，属于恢复与社区一致的访问语义，
不是把 NPU mask 处理函数替换成社区实现。

[2.10 对照摘录](../../../../artifacts/combo-five-fixes-review-20261107/upstream-v2.10.0.md)
和[2.13 对照摘录](../../../../artifacts/combo-five-fixes-review-20261107/upstream-v2.13.0.md)
保留 FloorDiv 求值、打印及 mask 过滤依据。
当前调用链与完整 NPU mask 函数见[NPU 摘录](../../../../artifacts/combo-five-fixes-review-20261107/npu-call-chain.md)。
对照结论仅针对固定标签，没有对未来 master 或所有后端作同等实现声明。

## 风险与可能的性能变化

| 改动 | 正确性前提 / 风险 | 性能上可能变化 |
| --- | --- | --- |
| FloorDiv 类型保护 | 非 FloorDiv 原样返回，不能继续执行专用逻辑 | 增加编译期类型判断，已求值结果无需重建 |
| 整除项提取 | 依赖静态整除判断准确 | 可能减少复杂除法；无法整除部分保留合并 |
| 余数无进位证明 | 依赖符号范围和有效循环长度上界准确 | 增加编译期求界，证明成功可能暴露更多化简机会 |
| 保留语义 mask | 保证 mask 的定义顺序、广播形状和依赖关系正确 | 可能增加原来错误删除的谓词及临时变量，也可能减少窗口外实际 load |

无法笼统承诺 F4 没有性能回退。
与旧错误拆分相比，安全实现可能保留更复杂的索引表达式；语义 mask 恢复后也可能改变寄存器压力、
广播形状、调度和 autotune 候选可用性。无进位证明缓解的是安全拆分机会损失，不是性能无回退证明。

该实现没有增加运行期 shape 判断或同步，但 `bound_sympy` 及静态关系查询仍消耗编译时间。
对范围分析不支持的表达式保守少优化；如果将来要扩大支持，应先证明范围和商余规则，而不是放宽严格小于条件。
影响范围包括使用这些 NPU Index helper 的普通 kernel，所以 Combo False/True 都验证很重要。

## 已有验证，以及它没有证明什么

| 验证层次 | 已有结果 | 能支持的结论 |
| --- | --- | --- |
| 原始方法定向 | 最终无进位优化后 2/2 PASS | 三组输入、两个输出、两种 Combo 设置及原有检查通过 |
| 符号枚举 | 500 组随机域，13,982 个点，69 个域证明可拆，全部等价 | 已采样范围内的拆分结果与原 FloorDiv 一致 |
| 边界专项 | 安全拆分、恰好进位、三项进位、负数、未知范围、不支持表达式 | 关键保守分支有执行证据 |
| DSL 审计 | 5 次 linear_combo 尝试均通过；语义 mask 定义在 load 前 | 目标路径的修复没有只靠回退获得 PASS |
| 最新全量 | `a8dcca84bf` 下本方法 2/2 PASS；总体 407/409 PASS | 五个修复叠加后的已选社区适配集通过情况 |

证据分别为[定向结果](../../../../artifacts/combo-f4-nocarry-20261106/f4_nocarry_validation/evidence/functional_results.json)、
[符号检查](../../../../artifacts/combo-f4-nocarry-20261106/symbolic_checks.json)、
[代码生成检查](../../../../artifacts/combo-f4-nocarry-20261106/codegen_checks.json)、
[最新全量汇总](../../../../artifacts/combo-full-a8dcca84bf-20261106/result.json)。
真实生成边界仍包含 `(21 + 16*y1) // 6` 和 `(21 + 16*x2) // 6`，说明可能进位的和式没有被强行展开。

随机枚举不是所有符号表达式的形式证明；没有测量运行性能、编译耗时或所有硬件的行为。
当前 full run 也仅是已选 62 个源方法的 409 个实例，并非全部 PyTorch 测试。
本文引用已有验证，未新增生产测试或重复执行远程任务。

## 提交整理与移植注意

最初 F4 为 `a2f7b18da9`，随后将无进位证明合并进同一提交，最终为 `53f8af09f1`。
本篇 diff 是最终 F4 相对父提交的完整变化，包含类型保护、除法拆分、证明 helper、mask 四部分，
不是只展示后来增加的优化。F4 在 PR 中仍然只有一个提交。

移植时需同时保留 helper import、类型保护及 mask 改动；只复制无进位 helper 不能修复原始方法的所有失败层。
最终安装源码使用完整文件而非测试进程函数 overlay，11 项源码运行前后校验一致。
统一版本、补丁校验与运行命令见[总览](inductor-combo-fixes-v2.10-review.md)。
