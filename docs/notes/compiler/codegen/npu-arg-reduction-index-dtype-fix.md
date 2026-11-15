---
published: false
---

# NPU argmax/argmin 为什么要统一索引 dtype

修复提交：`0503095a63116ca0cd815486b15324b9affef044`。问题是 arg 归约产生的索引表达式、
累加器和 CSE 声明没有统一使用 kernel 选定的索引类型。实际 DSL 是一种类型，
上层缓存却按另一种类型记录，可能触发生成代码的 dtype 校验失败。

## 数值和索引是两条不同的数据

argmax/argmin 既要比较数值，也要保存“最大/最小值在第几个位置”。
数值可以是 FP32，位置则是整数。CSE 是复用临时计算的缓存，它还记录临时值的 dtype；
记录必须与实际生成的 Triton 表达式一致。

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

例如选择 int32 时，
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
这个实例不能代表大地址空间的 int64 路径已经全部实测；大索引仍由原选择器决定。

## 验证版本

代码属于 `Ascend/pytorch` 的 `torch_npu`；本地仓库名为 `pytorch_new`，修复分支为
`bugfix/combo_fix_meta`。以下代码链接固定到包含这六项修复的提交
`5db866a16403644c265f3f43d747a856124de8b1`。

基础 torch_npu 包提交为 `efc05dd90b5fe31717d5f13e51fb3e49035bea30`，
验证通过 Python overlay 加载修复文件；不能把包版本字符串当成实际修复版本。
PyTorch 为 `2.13.0+cpu`（`cf30153c4c131c8164ee7798e5022d810682e2cb`），Triton 为 `3.6.0`。
完整运行命令、文件哈希和逐实例结果见[最新验证汇总](inductor-combo-tests-v2.13.md#validation213)。

本轮完整 PASS 回归及必要的定向复测再次覆盖本说明的 1 个方法、1 个实例，均 PASS。
实际版本、全部参数报告和执行来源见[完整回归与合并校验](inductor-combo-tests-v2.13.md#validation213)。
