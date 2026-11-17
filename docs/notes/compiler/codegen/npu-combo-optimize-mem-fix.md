---
published: false
---

# NPU Combo 的 optimize_mem 字段来自哪里，当前有什么作用

## 问题现象

原用例在检查生成代码时失败：推理图找不到 `'optimize_mem': True`，训练前向图找不到
`'optimize_mem': False`。计算结果本身没有因此报错，也没有实测到 NPU 发生额外 CPU 拷贝。
两项原失败见[推理记录][before-inference]和[训练前向记录][before-forward]。

这项修改补齐的是生成代码附带的设置。当前 NPU 调优路径没有接通下面介绍的 CPU 备份调用，
因此不能把它描述成已经修复了 NPU 的内存占用或训练错误。

## 是否同步上游 CUDA 修复

**是，同步上游 Inductor 的 Combo 字段修复。** 对应
[PyTorch PR #180790][upstream-pr]，合入提交为 [0274ad69c3e][upstream-commit]。
上游原来漏传此字段，运行时便按默认 True 处理；该 PR 让 Combo 与上游普通 kernel 使用同一图阶段策略。
这里的“普通 kernel 已有策略”指上游实现，当前 NPU 的普通 kernel 并没有显式生成这个字段。

NPU 原修复提交为 `43c727fb81`，移植到最新 master 后为 `6bcf326768`。
两边同步的是字段取值规则，运行时是否真正使用该规则还要看各自的调用链。

## 这个开关原本解决什么事

kernel 是一次发给设备执行的小程序，Combo 把几条独立计算分支放进同一个程序。
`inductor_meta` 是随程序附带的设置字典，不是参与张量运算的数据。
autotune 指试跑几组分块等配置，选择较快的一组。

如果待试跑的程序会原地修改输入，例如 `a.add_(1)`，每组配置都得从相同输入开始。
上游会准备输入副本；设备内存预算不够时，可以先备份到 CPU，再恢复输入。
`optimize_mem` 控制是否允许走这条节省设备内存的备份路径，不控制模型张量的一般内存分配。

训练前向关闭此策略的原因见[源码注释][upstream-policy]：前向逐层产生中间结果，
内存峰值本来就在增长。只按此前见过的峰值判断备份预算，容易引入大量不必要的 CPU 拷贝。
因此上游只在推理图或反向图打开此策略；训练前向沿用关闭策略。

推理是只计算模型输出；训练前向计算输出并为求梯度准备中间结果；反向则计算梯度。

## 对照代码看修改

入口是 [NPUComboKernel.jit_line](https://gitcode.com/luqichao/pytorch_new/blob/5db866a16403644c265f3f43d747a856124de8b1/torch_npu/_inductor/codegen/triton_combo_kernel.py#L294)：

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

上游的 [bench][upstream-bench] 确实调用 CPU 备份，再通过备份恢复或克隆参数。
当前 NPU 的情况不同：

| NPU 代码 | 实际作用 |
| --- | --- |
| `cached_autotune` 的 `pop("optimize_mem", True)` | 读取字段，缺失时保存默认 True |
| `copy_args_to_cpu_if_needed` | 定义了备份能力，但已检查的调优路径没有调用它 |
| 普通、批量及分组调优的 `clone_args` | 直接克隆试跑参数，没有通过上述 CPU 备份分支 |

代码入口为
[copy_args_to_cpu_if_needed](https://gitcode.com/luqichao/pytorch_new/blob/5db866a16403644c265f3f43d747a856124de8b1/torch_npu/_inductor/runtime/triton_heuristics.py#L697) 和
[缓存 autotune 的字段读取](https://gitcode.com/luqichao/pytorch_new/blob/5db866a16403644c265f3f43d747a856124de8b1/torch_npu/_inductor/runtime/triton_heuristics.py#L3364)。
同一文件的 `bench`、批量调优和分组试跑直接调用 `clone_args`，分别见第 1478、1653、3017 行。

所以，保留此字段的当前价值是对齐上游 metadata 约定，并为将来接通备份路径准备正确策略。
这两行不会让当前 NPU 突然少用内存，也不会改变 kernel 的计算、成员分组或分块参数。

## 怎么证明修好了

`test_combo_inductor_meta_has_optimize_mem` 检查推理生成代码含正确字段；
`test_combo_inductor_meta_optimize_mem_false_in_training_forward` 检查训练前向明确为 False。
首次定向记录是 [2 个方法、2 个实例 PASS](../../../../artifacts/inductor-combo-tests-v2.13-optimize-mem-43c727fb81-20261114/jobs.json)。
这证明对应图的字段和执行通过。两个测试使用 relu / sigmoid，没有原地修改输入，
不能用于证明 CPU 备份、恢复或内存节省已经生效。

## 验证版本

代码属于 `Ascend/pytorch` 的 `torch_npu`；本地仓库名为 `pytorch_new`，修复分支为
`bugfix/combo_fix_meta`。以下代码链接固定到包含这六项修复的提交
`5db866a16403644c265f3f43d747a856124de8b1`。

基础 torch_npu 包提交为 `efc05dd90b5fe31717d5f13e51fb3e49035bea30`，
验证通过 Python overlay 加载修复文件，即用修复后的 Python 文件覆盖已安装包中的对应文件；
不能把包版本字符串当成实际修复版本。
PyTorch 为 `2.13.0+cpu`（`cf30153c4c131c8164ee7798e5022d810682e2cb`），Triton 为 `3.6.0`。
完整运行命令、文件哈希和逐实例结果见[最新验证汇总](inductor-combo-tests-v2.13.md#validation213)。

2026-11-15 的已通过用例集合回归及必要定向补测覆盖本说明的 2 个方法、2 个实例，均 PASS。
实际版本、全部参数报告和执行来源见[完整回归与合并校验](inductor-combo-tests-v2.13.md#validation213)。

六项修改已迁入 master 基线分支，对应 [PR !49793][npu-pr] 和 [Issue #6157][npu-issue]。
移植后只做了本地静态核对，以上 NPU 结果仍属于原修复分支。

[upstream-pr]: https://github.com/pytorch/pytorch/pull/180790
[upstream-commit]: https://github.com/pytorch/pytorch/commit/0274ad69c3e
[upstream-policy]: https://github.com/pytorch/pytorch/blob/cf30153c4c13/torch/_inductor/codegen/triton.py#L6466
[upstream-bench]: https://github.com/pytorch/pytorch/blob/cf30153c4c13/torch/_inductor/runtime/triton_heuristics.py#L1237
[before-inference]: ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/results/shard-01/32-test_combo_inductor_meta_has_optimize_mem.log
[before-forward]: ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/results/shard-05/38-test_combo_inductor_meta_optimize_mem_false_in_training_forward.log
[npu-pr]: https://gitcode.com/Ascend/pytorch/merge_requests/49793
[npu-issue]: https://gitcode.com/Ascend/pytorch/issues/6157
