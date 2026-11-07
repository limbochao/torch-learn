# F8：benchmark 参数发现提前写入未绑定的 unbacked SymInt

本修复处理 `test_combo_kernel_size_hint_failure` 执行生成 wrapper 时的
`UnboundLocalError: ... 'u0' ... not associated with a value`。
benchmark 收集尺寸参数时调用了会写 wrapper 语句的 NPU helper，提前生成依赖 `u0` 的赋值，
但 `u0` 要等 `nonzero` 执行后才有值。修复将空 kernel 名称调用限制为返回参数描述，不输出运行期赋值。

| 项目 | 内容 |
| --- | --- |
| PR 对应项 | 第 3 点 |
| 修复提交 | `fdf73f229eb9a69267252a728f12a2eafcd19ee6` |
| 修改入口 | `torch_npu/_inductor/codegen/wrapper.py` 的 `_NPUKernelCodegenMixin.generate_node_numel_expr` |
| 单提交修改量 | 1 个文件，新增 3 行、删除 0 行 |
| 最新验证 | `a8dcca84bf52e8a3e0c46aa46c2a4f663875f10e`，目标 1/1 PASS |
| 上游关系 | 社区 2.10、2.13 均已有空名称 benchmark 不生成赋值的对应约定 |

## 动态尺寸与 unbacked SymInt 的区别

普通输入动态尺寸可以来自 `x.size(0)`：在运行 wrapper 一开始，输入张量已经提供了这个尺寸。
`unbacked SymInt` 则表示无法仅从输入 shape 得到的符号整数；它可以依赖张量中的实际数据。
此处 `u0 = torch.nonzero(x).size(0)` 表示非零元素个数，要先执行 `nonzero`，再从结果读取。
输入长度是 32，并不意味着 `u0` 一定是 32。

目标方法开启 `capture_dynamic_output_shape_ops`、`combo_kernels`、`benchmark_combo_kernel`，
输入为长度 32 的 FP16 张量，并将第 0 维标为动态。其核心计算是：

```python
def fn(x):
    nz = torch.nonzero(x)
    u0 = nz.size(0)
    t1 = torch.ones(u0, device=device)
    t2 = torch.zeros(u0 + 1, device=device)
    t3 = torch.zeros(u0 * 2, device=device)
    t4 = torch.zeros(u0 - x.size(0), device=device)
    return t1 - 1, t2 + 2, t3 * 3, t4 / 4
```

用例将 `torch.compile(fn, fullgraph=True)` 与 eager 结果比较。
这里 `u0+1`、`u0*2` 这类复合表达式会进入 NPU 的 node numel 生成逻辑，
是区分“参数发现”和“实际执行时赋值”所必需的场景。

## 错误是怎样产生的

NPU 的 `add_numel_to_call_args` 遍历轴，遇到 `sympy.Integer` 或 `sympy.Symbol` 时可以直接传参；
遇到复合表达式时调用 `wrapper.generate_node_numel_expr(name, node, numel_expr)`。
普通 Triton benchmark 的 `kernel_benchmark_extra_args` 用空字符串名称调用
`add_numel_to_call_args("", ...)`，以便收集可用于 benchmark 的尺寸参数。

旧 helper 没有识别空名称，仍执行以下操作：

1. 拼出局部名字，例如 `_x0_numel`。
2. 化简尺寸表达式，处理 precomputed size 符号。
3. 将 `_x0_numel = 1 + u0` 写入模型 wrapper。
4. 返回符号参数对象。

第 3 步的副作用不属于参数发现，而且发生在实际调用的尺寸绑定之前。
运行 wrapper 读到这条语句时，`nonzero` 结果尚未绑定 `u0`，于是报错。
该阶段关系可用以下示意说明；下面不是完整原始 wrapper：

```python
# 错误顺序：benchmark 发现阶段留下的赋值被放到过早的位置
_x0_numel = 1 + u0
# ... 后续才执行 nonzero，并从其结果绑定 u0
```

基线真实异常保存在[原始结果](../../../../artifacts/combo-fixes-85dec1f-20261106/baseline/evidence/functional_results.json)
的 `test_combo_kernel_size_hint_failure` 记录中。
函数调用关系与 helper 全文见[固定 NPU 调用链](../../../../artifacts/combo-five-fixes-review-20261107/npu-call-chain.md)。

## 三行代码为什么足够

新增条件位于 `simplify`、`ensure_size_computed`、`writeline` 之前。
空名称直接返回 `SymbolicCallArg(expr, numel_expr)`，保留原始尺寸表达式，让 benchmark 上层按
`inner_expr` 获取 size hint，并使用已有 `unbacked_symint_fallback` 机制生成评估参数。
本提交不把符号值强行转成 `int`，不把评估 hint 当成真实模型输出尺寸，也不要求在编译时执行 `nonzero`。

`SymbolicCallArg` 是包装“调用参数名与对应符号表达式”的对象。
返回它表示“这个参数描述已经收集到”，不要求此刻在模型运行代码里创建同名 Python 局部变量。
真正 kernel 的非空名称调用仍走原来的化简、预计算符号处理与赋值路径，
在实际调度的运行期位置生成所需尺寸。

## 完整修改与逐行说明

下面的 diff 原样取自本提交相对父提交的变化，包含新增、删除及必要上下文。
表中“旧/新”行号分别属于 diff 两侧的文件；空白行仅用于排版，不逐项计入。
后续提交可能使行号移动，定位时应以本节固定的 commit 为准。

```diff
diff --git a/torch_npu/_inductor/codegen/wrapper.py b/torch_npu/_inductor/codegen/wrapper.py
index c8c8731c5b..5ab57e394c 100644
--- a/torch_npu/_inductor/codegen/wrapper.py
+++ b/torch_npu/_inductor/codegen/wrapper.py
@@ -72,6 +72,9 @@ class _NPUKernelCodegenMixin:
     # generate numel expr for range_tree_node
     def generate_node_numel_expr(self, kernel_name: str, node, numel_expr):
         expr = f"{kernel_name}_{node.name}_numel"
+        # Benchmark argument discovery must not emit runtime size assignments.
+        if not kernel_name:
+            return SymbolicCallArg(expr, numel_expr)
         simplified = V.graph.sizevars.simplify(numel_expr)
         # Ensure all PRECOMPUTED_SIZE symbols in the *simplified* expression
         # are defined before we emit the line that uses them.
```

| 文件 / 位置 | 修改行 | 为什么这样处理 |
| --- | --- | --- |
| wrapper.py 新 75 | `+ # Benchmark argument discovery must not emit runtime size assignments.` | 说明空名称调用只负责发现 benchmark 参数，不应产生模型运行期赋值。 |
| wrapper.py 新 76 | `+ if not kernel_name:` | 识别 benchmark 参数发现调用；实际调用传非空 kernel 名称，因此继续走原路径。 |
| wrapper.py 新 77 | `+ return SymbolicCallArg(expr, numel_expr)` | 返回包含参数名称和原符号表达式的对象；在 simplify、ensure_size_computed、writeline 之前结束本次调用。 |


## 与上游 2.10 / 2.13 的具体对应

两个社区版本的 `WrapperCodeGen.generate_numel_expr` 都采用如下结构：

```python
arg = SymbolicCallArg(sym, tree.numel)
is_benchmark_kernel = kernel_name == ""
if not is_benchmark_kernel:
    self.writeline(SymbolicCallArgLine(self, arg, V.graph))
return arg
```

两个版本的普通 Triton `kernel_benchmark_extra_args` 都以空名称进行参数收集，并单独处理
`SymbolicCallArg.inner_expr`。这说明空字符串并非本次临时约定，而是上游已有的调用契约。
源码见[2.10 摘录](../../../../artifacts/combo-five-fixes-review-20261107/upstream-v2.10.0.md)
及[2.13 摘录](../../../../artifacts/combo-five-fixes-review-20261107/upstream-v2.13.0.md)。

NPU 的函数名为 `generate_node_numel_expr`，参数是拆分后的 node，实际命名及赋值实现也不同。
本提交同步的是“空名称不产生 wrapper 赋值”的行为；没有把 NPU 非空名称路径整体替换成社区的
`SymbolicCallArgLine`，也没有声称 NPU wrapper 的所有符号处理都已与社区完全相同。

## 影响范围与风险

判断条件使用 `if not kernel_name`。按类型约定传入字符串时，它识别空字符串。
若未来新增调用者把 `None` 当作真实 kernel 名称传入，也会命中该分支；
目前接口标注为 `str`，文档不把未约定的调用方式当成已支持用法。

主要风险是某个未来空名称调用者实际依赖生成局部赋值，这将与 benchmark 调用契约冲突。
对照已检查的调用链，空名称用于收集，非空名称用于真实调用，因此当前三行修复与契约一致。
保留 `inner_expr` 也保证上层还能够求 hint，避免用空返回值掩盖错误。

运行期不新增设备工作；修复会移除参数发现误写的 wrapper 语句。
编译期少做这一调用中的尺寸化简和写出，没有引入额外编译分支循环。
未进行编译耗时和运行性能测量；不以代码行数推断所有动态模型都无性能变化。

## 已有验证与可复查范围

[单项结果](../../../../artifacts/combo-fixes-85dec1f-20261106/fix_f8/evidence/functional_results.json)
为 1/1 PASS，检查原完整方法，保持 `nonzero`、复合输出尺寸及 eager 对照。
[最新全量结果](../../../../artifacts/combo-full-a8dcca84bf-20261106/result.json)
在最终安装源码 `a8dcca84bf` 下再次通过该实例；11 项源码前后校验一致。

验证没有枚举全部数据相关尺寸算子，例如所有形式的 unique、布尔索引及符号表达式组合；
本修复恢复的是该 helper 的副作用约定，不能据单个方法声明 unbacked SymInt 全面支持。
本文只引用已有运行，没有新增生产测试，也没有在文档编写阶段重跑环境。
统一环境、运行脚本及精确版本关系见[总览](inductor-combo-fixes-v2.10-review.md)。
