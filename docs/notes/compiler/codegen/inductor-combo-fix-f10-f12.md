# F10 / F12：Combo 延迟渲染丢失 inplace 删除状态

本修复处理 Combo 归约结果被重复 store 覆盖的问题：正确的 `mean` 已经写出，随后本应删除的
`sum` 又写入同一地址。计算 mean 的公式没有错，错误发生在最终生成代码保留了多余的写回。
F10 和 F12 的用例不同，但命中同一套删除状态和延迟渲染机制，因此使用一个提交。

| 项目 | 内容 |
| --- | --- |
| PR 对应项 | 第 1 点 |
| 问题编号 | F10、F12 |
| 修复提交 | `64a3e7e8b97282f6051f2d43c0b7150341118a4b` |
| 基线 | `85dec1f1a1`，已包含 F3 |
| 修改入口 | `torch_npu/_inductor/codegen/triton_combo_kernel.py` 的 `NPUComboKernel.codegen_kernel` |
| 单提交修改量 | 1 个文件，新增 5 行、删除 1 行 |
| 最新验证 | 五个修复叠加后的 `a8dcca84bf52e8a3e0c46aa46c2a4f663875f10e`，目标 2/2 PASS |

## 什么场景触发

F10 的 `test_dynamic_shapes_persistent_reduction_mixed_x_dim_cuda` 同时编译三条归约：

```python
def fn(x, y, z):
    return x.sum(1), y.mean(1), z.max(1)
```

实际输入分别为 `(16,128)`、`(32,128)`、`(32,256)`，第一维通过 `mark_dynamic` 标为动态，
范围为 1 到 256。`y.mean(1)` 的归约长度为 128。
这里的 CUDA 后缀来自原社区用例名称，实际验证经过已有 NPU 适配，不表示在 CUDA 上完成了验证。

F12 的 `test_mean` 输入为 `[1,2,4,8]`，返回整张量 mean、末维 mean、保留维度的倒数第二维 mean、
以及第 0、1 维 mean。两者都需要先计算 sum，再除以归约元素数，且存在中间结果的存储复用。

旧代码实际生成的 F10 关键片段如下，保持原生成格式：

```python
            tmp10 = tl.sum(tmp8, 1).reshape(X0BLOCK_SUB, 1)
            tmp11 = 128.0
            tmp12 = (tmp10 / tmp11)
            tl.store(in_out_ptr0 + (x0), tmp12, x0_mask)
            tl.store(in_out_ptr0 + (x0 ), tmp10, x0_mask)
```

第一次写回的是 `sum/128`；第二次写回的是 sum，地址相同，最终值由第二次 store 决定。
这段证据能把问题定位到写回删除逻辑，而不需要推测浮点除法、归约顺序或随机误差。
完整历史描述见[总表中的 F10 生成代码证据](inductor-combo-tests-liulm-v2.10.0-combokernel-dev.md#f10-的历史生成代码证据已修复)。

## 删除状态为什么没有生效

`inplaced_to_remove` 是 kernel 内记录“某个复用存储的逻辑 buffer 已无需再单独写回”的集合。
这里删除的是一条逻辑结果对应的 store；它不意味着将仍存活的最终输出张量整体删除。
例如 mean 与 sum 共享实际输出指针时，sum 的 store 应删除，mean 的 store 必须保留。

`DeferredLine` 是一种延迟决定是否输出的代码行。生成 store 时可以先保留这个对象，
直到 `IndentedBuffer.getvalue()` 真正转成源码字符串时，再检查该逻辑 buffer 是否已删除。
社区 `common.py` 中的判断读取以下四处集合：

```python
V.graph.removed_buffers
V.kernel.removed_buffers
V.graph.inplaced_to_remove
V.kernel.inplaced_to_remove
```

因此，“子 kernel 里已经标记删除”还不够：最终渲染时必须能通过 graph 或当前 `V.kernel` 读到它。
`V` 是 Inductor 的编译上下文入口，`V.kernel` 表示当前正在处理的 kernel，并不会自动等同于某个子 kernel。

NPU Combo 的调度流程先生成所有分区候选源码，再用 `make_ttir_for_check` 验证，验证全部通过后
才 `mark_run`、合并 graph 删除状态并生成实际调用。候选生成传入 `only_gen_src_code=True`，
故此阶段不会把子 kernel 的删除集合提前合入 graph。
旧 `code.getvalue()` 又没有切到持有这些状态的 Combo handler，导致延迟行看不到正确状态。

这些时序由[固定版本的 NPU 调用链摘录](../../../../artifacts/combo-five-fixes-review-20261107/npu-call-chain.md)
中的 `codegen_combo_kernel`、`generate_combo_kernel_code` 直接佐证。

## 为什么采用“局部聚合 + 正确渲染上下文”

第一步将所有子 kernel 的 `inplaced_to_remove` 聚合到 Combo 自身。
第二步仅在最终 `code.getvalue()` 期间用 `V.set_kernel_handler(self)` 使 `V.kernel` 指向 Combo。
这样 `DeferredLine` 就能按现有公共规则删除多余 store。

这两步缺一不可：只聚合集合但仍在错误 handler 下求值，集合不可见；只切 handler 但 Combo 集合为空，
仍缺少子 kernel 的决定。`with` 退出后恢复原 handler，避免改变后续编译任务的上下文。

也可以设想提前向 `V.graph.inplaced_to_remove` 写入集合，但 NPU 具有候选失败后回退到普通 kernel 的流程。
若候选尚未验证就提交全图删除状态，回退时可能错误继承未采用候选的结果。
本修复沿用原来的“验证通过后才提交 graph 状态”的时序，只补齐候选内部渲染所需信息。

## 完整修改与逐行说明

下面的 diff 原样取自本提交相对父提交的变化，包含新增、删除及必要上下文。
表中“旧/新”行号分别属于 diff 两侧的文件；空白行仅用于排版，不逐项计入。
后续提交可能使行号移动，定位时应以本节固定的 commit 为准。

```diff
diff --git a/torch_npu/_inductor/codegen/triton_combo_kernel.py b/torch_npu/_inductor/codegen/triton_combo_kernel.py
index 6939fd54a1..e328cf5460 100644
--- a/torch_npu/_inductor/codegen/triton_combo_kernel.py
+++ b/torch_npu/_inductor/codegen/triton_combo_kernel.py
@@ -376,6 +376,7 @@ class NPUComboKernel(ComboKernel):
     def codegen_kernel(self, name: Optional[str] = None) -> str:
         heuristics_list, size_hints_list = [], []
         for subkernel in self.sub_kernels:
+            self.inplaced_to_remove |= subkernel.inplaced_to_remove
             h, s = self.select_heuristics(subkernel)
             heuristics_list.append(h)
             size_hints_list.append(s)
@@ -453,7 +454,10 @@ class NPUComboKernel(ComboKernel):
         if config.benchmark_combo_kernel:
             code.splice(self.codegen_kernel_benchmark(num_gb=0))
 
-        return code.getvalue()
+        # Deferred stores must see the subkernels' removed inplace buffers.
+        # Keep this state local until every partition passes validation.
+        with V.set_kernel_handler(self):
+            return code.getvalue()
 
     def combo_grid_meta(self) -> dict[str, Any]:
         dynamic_shape = bool(self.dynamic_shape_args)
```

| 文件 / 位置 | 修改行 | 为什么这样处理 |
| --- | --- | --- |
| triton_combo_kernel.py 新 379 | `+ self.inplaced_to_remove \|= subkernel.inplaced_to_remove` | 将当前子 kernel 的 inplace 删除集合并入 Combo 本地集合。集合并集保留所有子 kernel 的决定，重复执行不会产生重复条目。 |
| triton_combo_kernel.py 旧 456 | `- return code.getvalue()` | 删除旧位置的直接渲染；新增的同一句在 Combo handler 生效期间求值，DeferredLine 因而读取正确的删除状态。 |
| triton_combo_kernel.py 新 457 | `+ # Deferred stores must see the subkernels' removed inplace buffers.` | 说明延迟 store 渲染必须能看到子 kernel 的 inplace 删除状态；F11 后将注释范围扩为全部删除状态。 |
| triton_combo_kernel.py 新 458 | `+ # Keep this state local until every partition passes validation.` | 说明时序约束：候选验证前只修改 Combo 局部集合，避免提前污染 graph 的删除状态。 |
| triton_combo_kernel.py 新 459 | `+ with V.set_kernel_handler(self):` | 临时将 V.kernel 指向当前 Combo；退出上下文会恢复之前的 handler。 |
| triton_combo_kernel.py 新 460 | `+ return code.getvalue()` | 删除旧位置的直接渲染；新增的同一句在 Combo handler 生效期间求值，DeferredLine 因而读取正确的删除状态。 |


## 是否与社区上游对齐

对照固定标签 PyTorch `v2.10.0` 与 `v2.13.0`，而不是未固定版本的 master：

| 对照位置 | 社区行为 | 本修复的关系 |
| --- | --- | --- |
| `common.is_buffer_removed` / `DeferredLine` | 两个版本均检查 graph 与当前 kernel 的两类删除集合 | 完全沿用该删除语义 |
| 2.10 `SIMDScheduling.generate_combo_kernel_code` | 子 kernel 状态先合入 graph，再生成 Combo 源码 | NPU 候选验证时序不同，不能照搬状态提交位置 |
| 2.13 `SIMDScheduling.process_kernel` | 调度处理中将 kernel 删除集合合入 graph | 同样说明最终渲染必须能看到删除状态 |
| `ComboKernel.codegen_kernel` | 没有本提交同样的局部聚合与 handler 补丁 | 本提交是 NPU 路径的适配修复，不是直接摘取某个上游提交 |

可逐段核查[社区 2.10 源码摘录](../../../../artifacts/combo-five-fixes-review-20261107/upstream-v2.10.0.md)
与[社区 2.13 源码摘录](../../../../artifacts/combo-five-fixes-review-20261107/upstream-v2.13.0.md)。
“与上游对齐”的准确含义是恢复同一条 buffer 删除契约，NPU 的验证与回退时序继续保留。

## 正确性、性能与回退风险

正确性风险集中在集合的归属：删除集合中的名字必须仍表示对应逻辑 buffer，不能误删存活输出。
本提交不重新判断哪个 buffer 可以删，只传播子 kernel 已作出的决定，使用既有 DeferredLine 判断规则。
集合并集不改变计算图的依赖关系，也不改变 mean 的算法、归约精度或输出 dtype。

运行期没有新增分支、同步或额外算子；生成代码会删除错误的冗余 store。
编译期增加集合合并和一次上下文切换，开销与子 kernel 删除条目数有关。
这给出了“不直接新增设备工作”的代码依据，但没有性能 benchmark，不能据此承诺所有模型零回退。
handler 改变还会影响该次渲染期间其他依赖 `V.kernel` 的延迟行，所以需要关注不同别名组合的生成结果。

失败候选不提前更新 graph 的删除集合，降低了回退状态污染的风险；本提交没有为所有可能编译副作用
建立通用回滚机制，也没有声称覆盖所有候选失败路径。

## 已有验证与证据边界

2026-11-06 的[单项验证](../../../../artifacts/combo-fixes-85dec1f-20261106/fix_f10/evidence/functional_results.json)
中，F10、F12 两个方法均 PASS。
[生成代码核对](../../../../artifacts/combo-fixes-85dec1f-20261106/codegen_checks.json)
记录目标 mean 的 `in_out_ptr0` 最终各只出现一次 store，错误的 sum 覆盖消失。

2026-11-07 完成的[最新全量结果](../../../../artifacts/combo-full-a8dcca84bf-20261106/result.json)
使用五个修复叠加源码 `a8dcca84bf`，这两个实例再次通过。
该全量共 409 个实例、407 PASS、2 FAIL；剩余 F6/F7 与本修复不同，原因见总表。
11 项安装源码运行前后哈希一致。此处引用已有真实运行，本文编写阶段未另启 NPU 测试。

回归命令与环境版本见[五篇文档总览](inductor-combo-fixes-v2.10-review.md)。
目标方法为 `test_mean` 和 `test_dynamic_shapes_persistent_reduction_mixed_x_dim_cuda`，
验证使用已有适配文件及原数值断言；没有为本提交新增生产测试文件。
