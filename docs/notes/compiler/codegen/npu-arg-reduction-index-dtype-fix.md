---
published: false
---

# NPU argmax/argmin 为什么要统一索引 dtype

## 问题现象

原 `test_combo_kernel_filter_cpu` 的计算结果检查已通过，最后却报“预期 2 个 kernel，实际 4 个”。
测试包含两条逐元素计算和两条带索引的 min / max 归约，原目标是分别组成两个 Combo。
生成候选存在类型不一致时，不能只看回退后结果算对了就判定 Combo 正常。
原始的数量断言失败见[失败日志][before-arg]。

进一步检查代码发现，归约用的位置索引实际可能是 int32，编译器保存的临时变量信息却写成 int64。
类型校验据此生成了与实际表达式不符的要求。修复后同一方法通过，实际生成两个 kernel，
归约内部的位置索引与类型检查都使用 `tl.int32`。

## 是否同步上游 CUDA 修复

**是，适配上游 dtype / shape 一致性修复中的 arg 归约部分。** 对应
[PyTorch PR #183938][upstream-pr]，合入提交为 [065e261caa1][upstream-commit]。
该 PR 扩大临时变量类型和形状的校验范围，并让 arg 归约结果、索引表达式和累加器使用同一个
`select_index_dtype()` 结果。

NPU 原修复提交为 `0503095a63`，移植到 master 后为 `54dbc8ce24`。
这里只适配索引类型这一部分，另外按 NPU 的索引轴生成方式补了显式转换；没有同步整个 PR 的其他改动。

## 数值和索引是两条不同的数据

归约是把一组元素汇成更少结果，例如把一行数求和、求最大值。
argmax / argmin 返回最大 / 最小值的位置；`max(dim)`、`min(dim)` 则同时返回数值和位置。
例如 `[3.0, 8.0, 4.0]` 的最大值是 `8.0`，从 0 开始数的位置是 `1`。
数值可以是 FP32（32 位浮点数），位置则是整数，不能把两者的类型混在一起。

dtype 就是数据类型；int32 / int64 分别用 32 / 64 位保存整数。
DSL 是编译器生成的 Triton 程序。CSE 原意是复用相同计算，避免重复计算；
这里还负责记录每个临时值的类型和形状。可以理解为一张登记表：表上写 int64，实际程序却产生 int32，
后面按表做类型检查就会出错。

累加器保存“扫到目前为止最好的数值和位置”；逻辑索引是按张量逻辑顺序数到的位置，
物理轴索引是生成程序在实际遍历轴上的编号。转置等非连续布局中两者不能随意互换，本修复只统一整数类型。

旧代码有的地方直接声明 `torch.int64`，有的地方只广播索引轴、没有先把实际轴转换成相同类型。
不能只改声明，也不能只改一条 `.to()`：最终参与比较的索引、初始极大值、归约结果都要一致。

## 对照代码看修改

[reduction](https://gitcode.com/luqichao/pytorch_new/blob/5db866a16403644c265f3f43d747a856124de8b1/torch_npu/_inductor/codegen/triton.py#L4389) 先从已有功能描述选一次类型：

```python
index_dtype = self.features.select_index_dtype() if reduction_type in arg_types else None
```

之后同一个 `index_dtype` 用在三个地方：

1. 创建 arg 归约的 CSE 结果，索引结果不再写死 int64。
2. 真正生成索引表达式时，逻辑索引和物理轴都显式转成对应 `tl.int32` 或 `tl.int64`，再广播。
3. 循环归约的索引累加器也用该类型，其初值取 `torch.iinfo(index_dtype).max`。

例如选择 int32 时，先把位置转成 int32，再扩展到与待比较数值相同的形状。
这里的广播只是把同一行位置索引配到每个数值，不改变位置的含义。
[索引轴生成](https://gitcode.com/luqichao/pytorch_new/blob/5db866a16403644c265f3f43d747a856124de8b1/torch_npu/_inductor/codegen/triton.py#L4518) 的关键逻辑是：

```python
index_type = self.dtype_to_str(index_dtype)
# 先改变真实 DSL 的类型，再按同一类型记入 CSE。
f"tl.broadcast_to({reduce_axis.symbol()}.to({index_type}).reshape({reshape_str}), "
f"{masked_value}.shape)"
```

[循环累加器](https://gitcode.com/luqichao/pytorch_new/blob/5db866a16403644c265f3f43d747a856124de8b1/torch_npu/_inductor/codegen/triton.py#L4687) 同样用所选类型的最大整数初始化。
这样“用于比较的位置”“保存的位置”和“编译器记录的位置类型”说的是同一件事。
这不改变非连续输入的逻辑索引顺序，也不改变 argmax/argmin 对外输出 tensor 的 dtype；
输出缓冲区仍按 ATen 的结果类型分配。

## 怎么证明修好了

`test_combo_kernel_filter_cpu` 包含 NPU arg 归约，是这次真实触发的回归入口。
[首次实测](../../../../artifacts/inductor-combo-tests-v2.13-meta-three-9f639af716-20261115/validation-summary.json)
记录该方法 PASS、两个生成 kernel，以及实际索引类型 `tl.int32`。
实际生成代码也同时包含 `r1.to(tl.int32)` 和对应的 `tl.static_assert(... == tl.int32)`；
最终输出缓冲区仍是 `torch.int64`，见[生成代码][generated-arg]。
这个实例不能代表大地址空间的 int64 路径已经全部实测；大索引仍由原选择器决定。

## 验证版本

代码属于 `Ascend/pytorch` 的 `torch_npu`；本地仓库名为 `pytorch_new`，修复分支为
`bugfix/combo_fix_meta`。以下代码链接固定到包含这六项修复的提交
`5db866a16403644c265f3f43d747a856124de8b1`。

基础 torch_npu 包提交为 `efc05dd90b5fe31717d5f13e51fb3e49035bea30`，
验证通过 Python overlay 加载修复文件，即用修复后的 Python 文件覆盖已安装包中的对应文件；
不能把包版本字符串当成实际修复版本。
PyTorch 为 `2.13.0+cpu`（`cf30153c4c131c8164ee7798e5022d810682e2cb`），Triton 为 `3.6.0`。
完整运行命令、文件哈希和逐实例结果见[最新验证汇总](inductor-combo-tests-v2.13.md#validation213)。

2026-11-15 的已通过用例集合回归及必要定向补测覆盖本说明的 1 个方法、1 个实例，均 PASS。
实际版本、全部参数报告和执行来源见[完整回归与合并校验](inductor-combo-tests-v2.13.md#validation213)。

六项修改已迁入 master 基线分支，对应 [PR !49793][npu-pr] 和 [Issue #6157][npu-issue]。
移植后只做了本地静态核对，以上 NPU 结果仍属于原修复分支。

[upstream-pr]: https://github.com/pytorch/pytorch/pull/183938
[upstream-commit]: https://github.com/pytorch/pytorch/commit/065e261caa1
[before-arg]: ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/results/shard-04/25-test_combo_kernel_filter_cpu.log
[generated-arg]: ../../../../artifacts/inductor-combo-tests-v2.13-meta-three-9f639af716-20261115/generated/00-c54lxsxukfe3da3sxldyfkfvtrcljcoecbxqturu2jip4l2m7bvl.py
[npu-pr]: https://gitcode.com/Ascend/pytorch/merge_requests/49793
[npu-issue]: https://gitcode.com/Ascend/pytorch/issues/6157
