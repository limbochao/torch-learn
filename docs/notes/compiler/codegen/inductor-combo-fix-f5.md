# F5：普通 Index 与 Combo kernel 缺少值为 1 的编译期常量

本修复补齐 Triton metadata 中 `equal-to-one` 参数的真实常量值。
随机数用例中的 `load_seed_offset1` 已在签名里被特化为 `constexpr`，但常量字典没有数值 1，
导致 TTIR 构建读取到 `None` 并失败。修复复用社区 helper，同时覆盖普通 Index kernel 和 Combo kernel。

| 项目 | 内容 |
| --- | --- |
| PR 对应项 | 第 5 点 |
| 修复提交 | `a8dcca84bf52e8a3e0c46aa46c2a4f663875f10e` |
| 修改入口 | `NPUIndexTritonKernel.codegen_kernel`、`NPUComboKernel.jit_line` |
| 单提交修改量 | 2 个文件，新增 7 行、删除 2 行 |
| 最新验证 | 最终源码 `a8dcca84bf52e8a3e0c46aa46c2a4f663875f10e`，Combo False/True 共 2/2 PASS |
| 上游关系 | 社区 2.10、2.13 的普通 Triton 与 Combo 路径均执行同一 helper 的常量登记 |

## 当前问题是什么，为什么没有改 index dtype

问题编号 F5 沿自之前的随机数用例调查。更早环境曾出现 int32/int64 索引相关检查失败，
但本次新基线没有复现那个旧错误，所以生产修复没有修改 index_expr 的 dtype，也没有改 RNG 算法。
旧问题与本次实际错误不能只因用例名称相同就当作同一个根因。

本次原始审计记录显示，linear Combo 的 TTIR 候选在以下位置失败：

```text
tmp3 = tl.load(in_ptr0 + load_seed_offset1)
AttributeError("'NoneType' object has no attribute 'type'")
```

这条候选错误来自[基线路径审计](../../../../artifacts/combo-fixes-85dec1f-20261106/baseline/evidence/audit.test_torchinductor.test_rand_like_deterministic.jsonl)。
pytest 最外层还可能报告随后回退路径中的 `output_shape is not None` 断言；
那不是这条常量丢失错误的直接报错位置。必须结合候选审计与最终阶段看待失败，不能只按最末条异常定根因。

## 为什么偏移 1 必须出现在 constants 中

`signature` 说明参数的类型及特化方式；`triton_meta["constants"]` 则提供被特化参数的实际编译期数值。
`constexpr` 表示编译该 kernel 时已知的常量，而不是设备运行时再从普通参数槽读取的整数。
因此“签名标了 constexpr”和“提供了 constexpr 值”是两个必须同时完成的步骤。

随机数生成先从 seed buffer 读取不同的 seed，`load_seed_offset1` 是其中一个读取偏移，值为 1。
它表示 seed buffer 的元素位置，不是把随机种子统一设为数值 1。
没有这项常量时，`in_ptr0 + load_seed_offset1` 无法构造正确的指针运算。

修复后真实生成 metadata 保留了如下对应关系；这里只抽取需要阅读的字段，完整候选原格式另存：

```python
signature = {"load_seed_offset1": "constexpr"}
constants = {"load_seed_offset1": 1}
```

[生成代码核对](../../../../artifacts/combo-fixes-85dec1f-20261106/codegen_checks.json)
中的 `F5_seed_constants` 记录该值为 1。

## helper 会不会把其他运行期参数错误固定为 1

本提交调用社区 `equal_1_arg_indices(signature)`，没有自定义“看起来像 1 就特化”的规则。
在已对照的社区 2.10、2.13 中，其内部 `_arg_equals_1` 要求：

1. 参数是 `SizeArg`。
2. `arg.expr` 是具体的 Python `int` 或 `sympy.Integer`。
3. `statically_known_equals(arg.expr, 1)` 成立。

所以不是对所有 tensor 参数或任意动态符号进行强行特化；一个仅在当前样例取值为 1、
表达式仍是符号的参数，并不会因为这三个新赋值语句就变成固定常量。
签名检查与常量登记共同沿用上游规则，这比根据参数名字硬编码 `load_seed_offset1 = 1` 更通用。

## 为什么要改两个生成入口

当 `combo_kernels=False`，用例走普通 Index kernel，需在 `gen_numel_args` 完成后，
使用最终的 signature 为 `triton_meta["constants"]` 登记 equal-to-one 参数。

当 `combo_kernels=True`，多个子 kernel 组合后的参数签名由 Combo 自己处理，
其 `jit_line` 也创建独立的 `triton_meta`。只修普通入口不会自动填充 Combo 的字典。
因此第二处把空字典初始化改为根据最终签名构造字典。
两处都使用参数名字作为 key，与 `signature_to_meta` 使用的名字一致。

## 完整修改与逐行说明

下面的 diff 原样取自本提交相对父提交的变化，包含新增、删除及必要上下文。
表中“旧/新”行号分别属于 diff 两侧的文件；空白行仅用于排版，不逐项计入。
后续提交可能使行号移动，定位时应以本节固定的 commit 为准。

```diff
diff --git a/torch_npu/_inductor/codegen/triton.py b/torch_npu/_inductor/codegen/triton.py
index ce1fbf932a..efbc7ebaa9 100644
--- a/torch_npu/_inductor/codegen/triton.py
+++ b/torch_npu/_inductor/codegen/triton.py
@@ -45,6 +45,7 @@ from torch._inductor.codegen.triton import (
     TMACompatibilityChecker, maybe_upcast_float32,
 )
 from torch._inductor.codegen.triton_utils import (
+    equal_1_arg_indices,
     config_of,
     should_unwrap_unspec_arg,
     signature_of,
@@ -3018,6 +3019,8 @@ class NPUIndexTritonKernel(TritonKernel):
             inductor_meta["kernel_num_gb"] = num_gb
 
         self.gen_numel_args(signature, triton_meta, triton_meta_signature, argdefs)
+        for index in equal_1_arg_indices(signature):
+            triton_meta["constants"][signature[index].name] = 1
         if self.is_unified_simt_kernel():
             triton_meta["configs"] = [config_of(signature)]
 
diff --git a/torch_npu/_inductor/codegen/triton_combo_kernel.py b/torch_npu/_inductor/codegen/triton_combo_kernel.py
index f96029bd5a..53cfe69930 100644
--- a/torch_npu/_inductor/codegen/triton_combo_kernel.py
+++ b/torch_npu/_inductor/codegen/triton_combo_kernel.py
@@ -8,7 +8,7 @@ from torch._inductor.codegen.common import ArgName, ConstexprArg, SizeArg
 from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
 from torch._inductor.codegen.triton import TritonKernel
 from torch._inductor.codegen.triton_combo_kernel import ComboKernel
-from torch._inductor.codegen.triton_utils import signature_to_meta, config_of
+from torch._inductor.codegen.triton_utils import signature_to_meta, config_of, equal_1_arg_indices
 from torch._inductor.runtime.hints import DeviceProperties
 from torch._inductor.runtime.runtime_utils import next_power_of_2
 from torch._inductor.utils import IndentedBuffer, Placeholder, triton_version_uses_attrs_dict
@@ -195,7 +195,9 @@ class NPUComboKernel(ComboKernel):
                 signature, size_dtype=size_dtype, argdefs=argdefs
             ),
             "device": DeviceProperties.create(V.graph.get_current_device_or_throw()),
-            "constants": {},
+            "constants": {
+                signature[index].name: 1 for index in equal_1_arg_indices(signature)
+            },
             "mix_mode": "aiv"
         }
 
```

| 文件 / 位置 | 修改行 | 为什么这样处理 |
| --- | --- | --- |
| triton.py 新 48 | `+ equal_1_arg_indices,` | 普通 Index kernel 导入社区已有的 equal-to-one 参数筛选函数。 |
| triton.py 新 3022 | `+ for index in equal_1_arg_indices(signature):` | 遍历最终签名中满足社区 equal-to-one 规则的参数位置。 |
| triton.py 新 3023 | `+ triton_meta["constants"][signature[index].name] = 1` | 以参数名注册编译期数值 1，补齐签名与 constants 之间的对应关系。 |
| triton_combo_kernel.py 旧 11 | `- from torch._inductor.codegen.triton_utils import signature_to_meta, config_of` | 替换 Combo 的原 import，以引入同一个社区筛选函数。 |
| triton_combo_kernel.py 新 11 | `+ from torch._inductor.codegen.triton_utils import signature_to_meta, config_of, equal_1_arg_indices` | Combo 复用社区 helper；没有新增 NPU 自定义的常量判定规则。 |
| triton_combo_kernel.py 旧 198 | `- "constants": {},` | 删除始终为空的 Combo 常量字典初始化。 |
| triton_combo_kernel.py 新 198 | `+ "constants": {` | 初始化包含 equal-to-one 参数的 Combo 编译期常量字典。 |
| triton_combo_kernel.py 新 199 | `+ signature[index].name: 1 for index in equal_1_arg_indices(signature)` | 每个满足规则的参数名对应常量 1，等价于社区逐项赋值循环。 |
| triton_combo_kernel.py 新 200 | `+ },` | 结束字典推导式；后面的 mix_mode 等 NPU metadata 保持原值。 |


## 与上游对齐到什么程度

| 路径 | 社区 v2.10.0 | 社区 v2.13.0 | NPU 修复 |
| --- | --- | --- | --- |
| 普通 Triton codegen | helper 遍历后按参数名写入 constants=1 | 同样登记 | 复用 helper 与相同赋值语义 |
| Combo jit metadata | 初始化后逐项登记 | 同样登记 | 用等价字典推导式初始化 |
| 参数筛选 | SizeArg、具体整数表达式、静态等于 1 | 相同三个条件 | 直接调用已安装社区实现 |

[社区 2.10 摘录](../../../../artifacts/combo-five-fixes-review-20261107/upstream-v2.10.0.md)
和[社区 2.13 摘录](../../../../artifacts/combo-five-fixes-review-20261107/upstream-v2.13.0.md)
均保存了普通与 Combo 的登记语句，以及 helper 的完整判定。
这五个提交里，F5 属于最直接的上游已有行为补齐；NPU 文件结构、额外 metadata 和插入位置仍与社区不同。
不能把这种局部行为对齐等同于整个 Triton 后端逐行一致。

## 正确性和性能风险

主要正确性风险是 signature 与 constants 的参数身份不一致，或后端签名尚未稳定就过早登记。
普通入口将登记放在 `gen_numel_args` 之后；Combo 使用当次 `jit_line` 的签名，
避免以旧的参数下标构造新签名的常量表。最终 key 采用参数名，而不是把不同签名的相同数字下标混用。

运行期不新增随机数调用、不修改 seed 消耗顺序、不更改 offset 算法或数值分布。
修复让编译器拿到本应已知的常量，会改变缺失常量时失败的编译行为，也可能改变相关特化代码与缓存内容。
helper 仅处理具体整数 1，并非为所有动态值 1 新增版本，因此不能简单宣称会发生无界的动态特化膨胀。

编译期多遍历一次签名并填字典，没有新增设备侧分支或同步。
当前没有性能 benchmark；目标用例的确定性与数值断言通过，不代表随机数吞吐量已测得无回退。
其他常量类型、不同 Triton 版本及其他硬件仍需独立验证。

## 已有验证及与历史提交的关系

`test_rand_like_deterministic` 使用长度 1024 的 float32 输入，编译函数返回两次 `rand_like`。
它重置 seed 后比较相同位置结果，检查连续调用结果应不同、同次返回的两个随机张量不同，并保留原范围检查。
Combo False/True 两种实例都运行，所以同时覆盖两个 metadata 入口。

[constants-only 单项验证](../../../../artifacts/combo-fixes-85dec1f-20261106/f5_constants_only/evidence/functional_results.json)
为 2/2 PASS，特意没有叠加此前诊断时尝试的 dtype 修改。
当时提交为 `bc9d3c4ee9`；后因 F4 合并优化而重放为本篇的 `a8dcca84bf`，F5 补丁内容相同。
2026-11-07 的[最新全量](../../../../artifacts/combo-full-a8dcca84bf-20261106/result.json)
已经用最终安装源码重新覆盖这两个实例，均 PASS，因此本篇最新结论不只依赖重放前的历史结果。

统一版本、安装校验及运行方式见[总览](inductor-combo-fixes-v2.10-review.md)。
本文没有重新执行环境测试，没有把旧 dtype 问题描述成当前仍存在的问题。
