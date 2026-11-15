---
published: false
---

# NPU Combo 的 optimize_mem 为什么要按当前图重新计算

修复提交：`43c727fb812260f1fe7afa22b336fa2987cf0693`。问题是 Combo 生成的
`inductor_meta` 没有明确传递当前图的 `optimize_mem` 策略，训练前向可能落入默认 True。
这项修复只同步已有策略，没有新增一种内存优化。

## 错在什么地方

`inductor_meta` 是生成代码附带给运行时的设置字典。`optimize_mem` 决定 autotune 保存可变输入时，
是否允许按内存预算把备份放到 CPU。运行时读取不到字段时，`pop("optimize_mem", True)` 默认开启；
`copy_args_to_cpu_if_needed` 在 False 时则直接返回空字典。

普通 kernel 已按当前图区分推理、反向、训练前向。Combo 复用了首成员的 metadata，
却没有明确重建这个图级字段。成员是谁与当前整张图处于哪个阶段是两回事，不能让默认值代替图的状态。

## 对照代码看修改

入口是 [NPUComboKernel.jit_line](https://gitcode.com/luqichao/pytorch_new/blob/5db866a16403644c265f3f43d747a856124de8b1/torch_npu/_inductor/codegen/triton_combo_kernel.py#L284)：

```python
inductor_meta = self.sub_kernels[0].create_inductor_meta()
# 中间保留原有 runtime block、mutation 等字段处理。
inductor_meta["optimize_mem"] = V.graph.is_inference or V.graph.is_backward
```

大白话就是：生成这个 Combo 时，重新问一次“现在是不是推理图或反向图”。
是就写 True；两者都不是，也就是训练前向，就写 False。
这样运行时收到的是当前图明确给出的答案，不再依赖缺字段时的默认值。

| 当前图 | 写入值 |
| --- | --- |
| 推理 | True |
| 反向 | True |
| 训练前向 | False |

运行时消费入口是
[copy_args_to_cpu_if_needed](https://gitcode.com/luqichao/pytorch_new/blob/5db866a16403644c265f3f43d747a856124de8b1/torch_npu/_inductor/runtime/triton_heuristics.py#L697) 和
[缓存 autotune 的字段读取](https://gitcode.com/luqichao/pytorch_new/blob/5db866a16403644c265f3f43d747a856124de8b1/torch_npu/_inductor/runtime/triton_heuristics.py#L3363)。
修复没有改变 kernel 的数学计算、成员分组或 block 参数。

## 怎么证明修好了

`test_combo_inductor_meta_has_optimize_mem` 检查推理生成代码含正确字段；
`test_combo_inductor_meta_optimize_mem_false_in_training_forward` 检查训练前向明确为 False。
首次定向记录是 [2 个方法、2 个实例 PASS](../../../../artifacts/inductor-combo-tests-v2.13-optimize-mem-43c727fb81-20261114/jobs.json)。
这证明对应图的生成字段和执行通过；并不等于实测了所有图的 CPU 备份流量。

## 验证版本

代码属于 `Ascend/pytorch` 的 `torch_npu`；本地仓库名为 `pytorch_new`，修复分支为
`bugfix/combo_fix_meta`。以下代码链接固定到包含这六项修复的提交
`5db866a16403644c265f3f43d747a856124de8b1`。

基础 torch_npu 包提交为 `efc05dd90b5fe31717d5f13e51fb3e49035bea30`，
验证通过 Python overlay 加载修复文件；不能把包版本字符串当成实际修复版本。
PyTorch 为 `2.13.0+cpu`（`cf30153c4c131c8164ee7798e5022d810682e2cb`），Triton 为 `3.6.0`。
完整运行命令、文件哈希和逐实例结果见[最新验证汇总](inductor-combo-tests-v2.13.md#validation213)。
