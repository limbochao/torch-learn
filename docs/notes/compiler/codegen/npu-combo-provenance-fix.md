---
published: false
---

# NPU Combo 为什么丢失 FX 追踪，以及如何按真实分区修复

## 问题现象

原用例的 relu、sigmoid、tanh 计算和 trace 文件输出都已完成，但文件里的 `cppCodeToPost` 是空字典。
测试本来要看到“这个设备 kernel 对应 relu、sigmoid、tanh”，实际却查不到任何来源，因此失败。
问题发生在调试信息登记阶段，张量数值计算没有因此出错。

## 是否同步上游 CUDA 修复

**本次不是直接同步一份 CUDA bug 修复，而是补齐 NPU 的追踪接入。**
相关上游背景是 [PyTorch PR #163378][upstream-reference]，它把来源映射登记统一放进
`codegen_comment`，并让 Combo 传入真实成员。当前对照的[上游实现][upstream-code]
仍是 `self.codegen_comment(combo_kernel_node.snodes, kernel_name)`。

NPU 调用自己的 Combo 生成函数，却把无实际来源的外壳传给公共追踪入口，漏掉了这项约定。
本修复还按 NPU 实际生成分区收集叶节点，并统一命名和来源，因此不等于直接复制上游那一行。
`#163378` 是调用约定的参考 PR，不是本次 NPU 补丁的一一对应同步 PR。

NPU 原修复提交为 `0dfe37f3ca`，移植到 master 后为 `35f0fabe78`。

## 追踪到底记录什么

FX 图是 PyTorch 记录的一张运算关系图，其中节点可以是 relu、sigmoid 等操作。
kernel 是最终交给设备的小程序；一个 Combo kernel 可以实现图里的好几个操作。
provenance 指记录“这个程序来自图里的哪些操作”，便于从性能分析或生成代码回查模型运算。

这里的 post-grad 图是经过自动求导相关变换后的 FX 图，不只指反向计算。
`cppCodeToPost` 是保存这类关系的字段名，NPU Triton kernel 的来源也登记在里面。
例如应有 `某 kernel -> [relu, sigmoid, tanh]`；空映射表示程序虽然生成了，来源没有登记。

## 外壳不是实际执行的节点

Combo 是把多条独立计算分支放进一个 kernel。外层 `combo_kernel_node` 负责组织成员，
自己的 `node` 是 None，并不直接拥有 FX origins。
IR 是编译器内部表示运算的数据结构，origins 是记录它来自哪些 FX 操作的来源信息。
共同追踪函数遇到 `snode.node is None` 就跳过，所以把外壳交给它，只能得到空映射。
可以把外壳理解为装成员的目录：目录有组织作用，真正的运算和来源还在里面的成员上。

旧代码在生成每个分区时，都把同一个外壳传给两个入口：

```python
self.define_kernel(src_code, [combo_kernel_node], kernel, None)
self.codegen_comment([combo_kernel_node], kernel_name)
```

第一个入口负责 kernel 名称和生成代码的来源注释；第二个入口除了写注释，还调用共同追踪函数登记映射。
只改一边会让名称/注释和实际追踪来源不一致。

## 对照代码看修复逻辑

修复位于
[NPUTritonScheduling.codegen_combo_kernel](https://gitcode.com/luqichao/pytorch_new/blob/5db866a16403644c265f3f43d747a856124de8b1/torch_npu/_inductor/codegen/scheduling.py#L502)：

```python
for src_code, kernel, node_group in kernel_code_list:
    # 原有成员 mark_run、buffer 状态同步等逻辑保持原位置。
    node_schedule = list(OrderedSet(
        snode for node in node_group for snode in node.get_nodes()
    ))
    kernel_name, _ = self.define_kernel(src_code, node_schedule, kernel, None)
    self.codegen_comment(node_schedule, kernel_name)
```

按执行顺序读这几行：

1. `kernel_code_list` 中一项就是实际生成的一个 kernel。“分区”就是原候选被拆开后实际生成的一组成员。
   `node_group` 只包含这一项的成员，
   不是外层所有 Combo 成员。
2. `get_nodes()` 把融合成员展开为真实叶子节点，即拆开组织外壳后实际表示运算的底层节点。
   这些叶子有 IR 和 origins，追踪函数才能读到 FX 来源。
3. `OrderedSet` 去掉重复叶子，同时保留首次出现的调度顺序；最后转成共同追踪函数要求的 list。
4. 同一份节点列表同时交给定义和注释/追踪入口，名称、来源注释和映射使用同一个分区边界。

例如两个实际分区分别执行 relu/sigmoid 和 tanh，第一个 kernel 只登记前两个，
第二个只登记 tanh。若把外层的三个 origins 全复制给每个 kernel，映射虽然非空，却是错误的。
缓存命中时 `define_kernel` 可能复用源码，但它后面的 `codegen_comment` 每次仍执行，
因此当前图的追踪关系不会因为复用源码而被漏掉。

## 验证看什么

原实例：

```bash
python -m pytest -v \
  test_provenance_tracing.py::TestProvenanceTracingArtifact::test_triton_kernel_to_post_grad_tracing_combo_kernel
```

此方法依次执行 Inductor 和 AOTInductor（提前编译并打包模型的路径），两条路径均通过。
映射包含实际 kernel 对应的 `relu`、`sigmoid`、`tanh`，不再为空。
恢复真实来源以后，NPU 的描述性 kernel 名变为 `triton_poi_fused_relu_sigmoid_tanh_0`；
用例分支提交 `3f7bfbe` 只调整 NPU 的预期名称，仍严格比较完整节点列表和 debug handle。
debug handle 是追踪来源时使用的编号，改 kernel 名称不应丢掉这些对应关系。
没有删除映射断言，也没有补一个与真实生成无关的固定映射。

原计算图、输入和融合目标没有改变。仍需通过实际分区来验多分区与缓存复用的映射，
本次原实例的 PASS 不替代所有这些组合的穷举覆盖。

## 验证版本与复现

责任仓库为 `Ascend/pytorch` 的 `torch_npu`，本地仓库名 `pytorch_new`，分支
`bugfix/combo_fix_meta`；包含六项修复的 Python overlay 提交为
`5db866a16403644c265f3f43d747a856124de8b1`。
overlay 指用修复后的 Python 文件覆盖已安装包中的对应文件。
基础包为 `2.13.0+gitefc05dd`，PyTorch 为 `2.13.0+cpu`
（`cf30153c4c131c8164ee7798e5022d810682e2cb`），Triton 为 `3.6.0`。

用例来自 `Community-Use-Case-Testing` 的 `fix/combo_case`，实际提交
`7332db7c9fc5a5d10da2f17ec29e22866d38154a`，目录为 `v2.13.0/inductor_npu`。
在 NPU 工作目录加载上述版本环境，设置 `ASCEND_RT_VISIBLE_DEVICES` 为获准物理卡，
使用独立 Inductor/Triton 缓存后执行上述 node ID。
安装文件和测试源码的校验结果、逐实例报告及完整 PASS 回归见
[最新执行结果](inductor-combo-tests-v2.13.md#validation213)。

2026-11-15 的已通过用例集合回归及必要定向补测覆盖本说明的 1 个方法、1 个实例，均 PASS。
实际版本、全部参数报告和执行来源见[完整回归与合并校验](inductor-combo-tests-v2.13.md#validation213)。

六项修改已迁入 master 基线分支，对应 [PR !49793][npu-pr] 和 [Issue #6157][npu-issue]。
移植后只做了本地静态核对，以上 NPU 结果仍属于原修复分支。

[upstream-reference]: https://github.com/pytorch/pytorch/pull/163378
[upstream-code]: https://github.com/pytorch/pytorch/blob/cf30153c4c13/torch/_inductor/codegen/simd.py#L3734
[npu-pr]: https://gitcode.com/Ascend/pytorch/merge_requests/49793
[npu-issue]: https://gitcode.com/Ascend/pytorch/issues/6157
