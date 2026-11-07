# F11：Combo 中间 buffer 已移出签名，但 store 仍引用它

本修复处理 Combo Triton 编译中的 `out_ptr1 is not defined`：中间 buffer 已被判定不需要实际输出，
所以参数签名中没有对应指针；但最终函数体仍保留对它的 store。
缺失的是删除状态在 Combo 最终渲染时的传播，不应通过给签名重新补一个无用输出指针解决。

| 项目 | 内容 |
| --- | --- |
| PR 对应项 | 第 2 点 |
| 修复提交 | `9b1a894d6500fe76bde4b75c3f31ca389f2a21ae` |
| 所依赖的前一提交 | `64a3e7e8b9`，已补 inplace 删除状态及渲染 handler |
| 修改入口 | `NPUComboKernel.codegen_kernel` |
| 单提交修改量 | 1 个文件，新增 2 行、删除 1 行；其中一处仅更新注释 |
| 最新验证 | `a8dcca84bf52e8a3e0c46aa46c2a4f663875f10e`，目标 3/3 PASS |

## 受影响的三个方法与失败证据

`test_mutated_benchmark`、`test_round_robin_dispatch`、`test_dynamic_shapes_mutated`
都包含多个分支的逐元素计算和输入突变，核心函数为：

```python
def test_mutated(a, b, c, d):
    a.add_(1)
    b.sigmoid_()
    c = torch.add(c, 5)
    d.tanh_()
    return a, b, c, d
```

第一个方法输入 shape 分别为 `(10,10)`、`(20,20)`、`(10,10)`、`(30,8)`；后两个为
`(10,10)`、`(20,5)`、`(10,10)`、`(5,18)`。
`a`、`b`、`d` 原地更新，而 `c` 产生新结果；融合与突变处理会产生可消去的中间 buffer。
方法名与配置覆盖 benchmark、dispatch、动态尺寸相关路径；实际动态行为应以采集到的生成签名为准，
不能只凭方法名假定所有维度都为动态。

基线在 Triton 编译阶段失败，关键错误如下：

```text
NoTritonConfigsError: No valid triton configs for kernel triton_
NameError('out_ptr1 is not defined')
```

其中一个实际报错 store 为：

```python
                tl.store(out_ptr1 + (x1 + 10*y0), tmp5, x1_mask & y0_mask)
```

动态方法中的对应地址包含 `ks0`，但仍是同一个未声明输出指针问题。
各方法的原始失败日志见[基线结果](../../../../artifacts/combo-fixes-85dec1f-20261106/baseline/evidence/functional_results.json)。
这属于源码生成错误，在设备实际执行和数值比较之前就会被编译器拒绝。

## 为什么第一个提交还不能解决 F11

F10/F12 修复聚合的是 `inplaced_to_remove`，表示需要删除的 inplace 逻辑写回。
F11 还涉及另一种集合 `removed_buffers`，表示被整体移除的 buffer。
在社区 `common.py` 中，`remove_buffer` 会把 `args.output_buffers[name]` 标记为 `REMOVED`，
并把名字加入 kernel 的 `removed_buffers`。
签名生成因而跳过该输出；另一方面，延迟 store 必须在渲染时读取同样的删除决定。

这两处必须一致：

| 位置 | 正确行为 | 旧 Combo 状态 |
| --- | --- | --- |
| 参数签名 | 移除已删中间 buffer 的输出指针 | 已移除 |
| 函数体 | 删除对应 buffer 的 DeferredLine store | 未能读取子 kernel 的 removed_buffers，仍保留 |

F10 的 handler 切换解决了“在哪里读取”的问题，却没有自动补齐所有“需要读取的集合”。
因此 F11 需要额外一行集合合并，且可以独立作为一个提交解释。
这也说明 F10 与 F11 不是把相同改动拆成两个编号：它们补齐的是两个不同字段。

## 修复如何恢复签名与函数体一致

在遍历子 kernel 时，执行 `self.removed_buffers |= subkernel.removed_buffers`。
最终渲染继续使用前一提交建立的 Combo handler，于是 `DeferredLine` 的公共删除判断
同时能读到 `self.removed_buffers` 与 `self.inplaced_to_remove`。

不新增 output 参数、不为中间值分配实际设备内存、不保留无意义写回，也不更改突变操作本身。
更新注释只是使文字覆盖两类删除状态，没有新增第二种渲染机制。

## 完整修改与逐行说明

下面的 diff 原样取自本提交相对父提交的变化，包含新增、删除及必要上下文。
表中“旧/新”行号分别属于 diff 两侧的文件；空白行仅用于排版，不逐项计入。
后续提交可能使行号移动，定位时应以本节固定的 commit 为准。

```diff
diff --git a/torch_npu/_inductor/codegen/triton_combo_kernel.py b/torch_npu/_inductor/codegen/triton_combo_kernel.py
index e328cf5460..f96029bd5a 100644
--- a/torch_npu/_inductor/codegen/triton_combo_kernel.py
+++ b/torch_npu/_inductor/codegen/triton_combo_kernel.py
@@ -376,6 +376,7 @@ class NPUComboKernel(ComboKernel):
     def codegen_kernel(self, name: Optional[str] = None) -> str:
         heuristics_list, size_hints_list = [], []
         for subkernel in self.sub_kernels:
+            self.removed_buffers |= subkernel.removed_buffers
             self.inplaced_to_remove |= subkernel.inplaced_to_remove
             h, s = self.select_heuristics(subkernel)
             heuristics_list.append(h)
@@ -454,7 +455,7 @@ class NPUComboKernel(ComboKernel):
         if config.benchmark_combo_kernel:
             code.splice(self.codegen_kernel_benchmark(num_gb=0))
 
-        # Deferred stores must see the subkernels' removed inplace buffers.
+        # Deferred stores must see the subkernels' removed buffers.
         # Keep this state local until every partition passes validation.
         with V.set_kernel_handler(self):
             return code.getvalue()
```

| 文件 / 位置 | 修改行 | 为什么这样处理 |
| --- | --- | --- |
| triton_combo_kernel.py 新 379 | `+ self.removed_buffers \|= subkernel.removed_buffers` | 补齐普通已删除 buffer 的状态传播；这是 F11 所需状态，与上一提交的 inplace 删除集合互补。 |
| triton_combo_kernel.py 旧 457 | `- # Deferred stores must see the subkernels' removed inplace buffers.` | 说明延迟 store 渲染必须能看到子 kernel 的 inplace 删除状态；F11 后将注释范围扩为全部删除状态。 |
| triton_combo_kernel.py 新 458 | `+ # Deferred stores must see the subkernels' removed buffers.` | 更新注释：渲染现同时依赖 removed_buffers 和 inplaced_to_remove。 |


## 是否与上游对齐

社区 `v2.10.0`、`v2.13.0` 的 `is_buffer_removed` 都将 `removed_buffers` 与
`inplaced_to_remove` 分开检查；不能因为其中一个集合可见，就忽略另一个集合。
2.10 调度器在生成 Combo 源码前向 graph 合并子 kernel 的这两个集合；2.13 的 `process_kernel`
也执行这两类合并。

NPU 候选路径为了允许校验失败后回退，延迟 graph 状态提交，所以本提交将状态保留在 Combo 本地。
它与上游的 buffer 删除结果对齐，位置上属于 NPU 专用适配，不是社区同位置的一行原样回合。
源码对照见[2.10 摘录](../../../../artifacts/combo-five-fixes-review-20261107/upstream-v2.10.0.md)、
[2.13 摘录](../../../../artifacts/combo-five-fixes-review-20261107/upstream-v2.13.0.md)，
NPU 时序见[调用链摘录](../../../../artifacts/combo-five-fixes-review-20261107/npu-call-chain.md)。

## 风险和性能影响

该提交只传播已有删除决定，适用范围为 `NPUComboKernel`，不会改变普通单 kernel 的删除判定。
主要正确性前提是子 kernel 删除集合准确，以及不同子 kernel 的 buffer 命名与全图逻辑身份一致。
不能为了避免 NameError 而无条件删掉任意 store；本补丁只删除明确标记的 buffer 对应延迟行。

与 F10 相同，候选验证前不把这些集合写入 graph，因此不提前改变回退所见的全图删除状态。
该提交依赖 F10 的正确 handler；若单独移植到未包含 F10 的旧分支，只加集合并集未必生效。

设备端预期减少应被消去的中间写回，不会增加新算子或新内存分配。
编译期增加一次集合合并；实际性能没有测量，不能把删除无效 store 直接换算成性能收益。
通过三个方法也不能外推到所有别名、突变链和跨分区组合。

## 已有验证

[单项验证](../../../../artifacts/combo-fixes-85dec1f-20261106/fix_f11/evidence/functional_results.json)
记录上述三个方法 3/3 PASS。
审计显示对应目标路径实际走到 `linear_combo`，并非因回退而掩盖同一个 Combo 编译错误。
[代码生成核对](../../../../artifacts/combo-fixes-85dec1f-20261106/codegen_checks.json)
中的 `F11_pointer_bindings` 对所记录候选检查了所有被引用指针都已声明。
它检查的是名字绑定完整性，不是所有指针访问地址都已获得通用形式证明。

2026-11-07 完成的[最新全量](../../../../artifacts/combo-full-a8dcca84bf-20261106/result.json)
在最终源码 `a8dcca84bf` 下再次覆盖三方法，均 PASS。
全量其余结果与版本校验见[文档总览](inductor-combo-fixes-v2.10-review.md)。
这是已有 NPU 运行结果的归档，编写本文未重复运行用例，也没有改变输入、数值断言或阈值。
