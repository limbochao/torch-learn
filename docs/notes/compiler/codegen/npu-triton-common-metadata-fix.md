---
published: false
---

# NPU 为什么要复用共同的 Triton 编译 metadata

## 问题现象

原 `test_combo_triton_meta_has_disable_ftz` 在检查生成代码时失败：设置 False 时找不到
`'disable_ftz': False`，设置 True 时也找不到对应字段。也就是用户给了开关，生成代码却没带上它。
原失败见[测试日志][before-ftz]。

这次确认的是设置传递缺失。原测试没有证明 NPU 的极小数计算错误，
修复后也没有证明 NPU 已实现所有开关对应的硬件行为。

## 是否同步上游 CUDA 修复

**是，适配上游 Combo 编译选项传递修复。** 对应
[PyTorch PR #180789][upstream-pr]，合入提交为 [a886dc98482][upstream-commit]。
上游让 Combo 复用 `TritonKernel.triton_meta_common()`，从同一入口取得公共编译设置。

NPU 原修复提交为 `9f639af716`，移植到 master 后为 `0c9b63bdd8`。
NPU 的普通 kernel 和 Combo 都自行构造字典，因此两个入口都补了这个公共调用。

## 两种 metadata 要分清

metadata 就是生成代码附带的信息字典。kernel 是设备执行的小程序，Combo 是把几条计算分支合在一起。
`triton_meta` 告诉底层编译器怎样编译和启动程序；`inductor_meta` 主要放调优、估算成本等信息。
本问题属于前者：NPU 手工列字典时，漏掉了公共函数已经提供的开关。

| 字段 / 术语 | 大白话含义 |
| --- | --- |
| FTZ，flush to zero | 把非常接近 0 的非正规浮点小数直接当成 0 |
| `disable_ftz` | 是否禁止上述清零；True 表示要求保留这些极小数 |
| `enable_fp_fusion` | 是否允许把浮点运算合成指令，例如把乘和加合成一次计算，可能改变舍入过程 |
| PDL，programmatic dependent launch | CUDA 的一种依赖启动机制，用程序中的同步点协调前后 kernel |
| `launch_pdl` | 是否使用上述启动机制；转发此标志不等于 NPU 已支持 PDL |
| helper | 可复用的小函数；这里指统一生成这几个字段的 `triton_meta_common()` |

例如用户把 `disable_ftz` 设为 True，合理的生成结果首先必须带上 True，后端才有机会解释这个要求。
“带上开关”和“硬件按开关执行”是两步。本补丁只补第一步，第二步需要后端实现和极小数专项验证。

## 对照代码看修改

两个入口分别是
[普通 kernel](https://gitcode.com/luqichao/pytorch_new/blob/5db866a16403644c265f3f43d747a856124de8b1/torch_npu/_inductor/codegen/triton.py#L2490) 和
[Combo](https://gitcode.com/luqichao/pytorch_new/blob/5db866a16403644c265f3f43d747a856124de8b1/torch_npu/_inductor/codegen/triton_combo_kernel.py#L279)。修改使用相同写法：

```python
triton_meta = {
    "signature": ...,
    "device": ...,
    "constants": ...,
    **TritonKernel.triton_meta_common(),
    "mix_mode": "aiv",
}
```

`**` 的意思是把 helper 返回的键值加入当前字典。
大白话就是：共同代码已经知道当前应该传哪些开关，NPU 直接取这份答案，
避免手工抄几个字段以后又漏掉。NPU 专属 `mix_mode="aiv"` 仍放在后面，保留其明确设置。

这也让普通 kernel 与 Combo 遵循同一个来源，避免一个路径传了、另一个没传。
没有修改 Triton-Ascend 的编译选项注册，更没有把 PDL 当作 NPU 已支持的启动协议。

## 怎么证明修好了

`test_combo_triton_meta_has_disable_ftz` 的 False、True 两个实例均 PASS。
[首次实测摘要](../../../../artifacts/inductor-combo-tests-v2.13-meta-three-9f639af716-20261115/validation-summary.json)
记录生成字段确实分别为 False、True。
同一记录也明确：该版本 Ascend 后端没有注册有效的 `disable_ftz` 编译选项。
因此这项修复证明 metadata 传递和原用例数值通过，不宣称极小数保留能力已经实现。

## 验证版本

代码属于 `Ascend/pytorch` 的 `torch_npu`；本地仓库名为 `pytorch_new`，修复分支为
`bugfix/combo_fix_meta`。以下代码链接固定到包含这六项修复的提交
`5db866a16403644c265f3f43d747a856124de8b1`。

基础 torch_npu 包提交为 `efc05dd90b5fe31717d5f13e51fb3e49035bea30`，
验证通过 Python overlay 加载修复文件，即用修复后的 Python 文件覆盖已安装包中的对应文件；
不能把包版本字符串当成实际修复版本。
PyTorch 为 `2.13.0+cpu`（`cf30153c4c131c8164ee7798e5022d810682e2cb`），Triton 为 `3.6.0`。
完整运行命令、文件哈希和逐实例结果见[最新验证汇总](inductor-combo-tests-v2.13.md#validation213)。

2026-11-15 的已通过用例集合回归及必要定向补测覆盖本说明的 1 个方法、2 个实例，均 PASS。
实际版本、全部参数报告和执行来源见[完整回归与合并校验](inductor-combo-tests-v2.13.md#validation213)。

六项修改已迁入 master 基线分支，对应 [PR !49793][npu-pr] 和 [Issue #6157][npu-issue]。
移植后只做了本地静态核对，以上 NPU 结果仍属于原修复分支。

[upstream-pr]: https://github.com/pytorch/pytorch/pull/180789
[upstream-commit]: https://github.com/pytorch/pytorch/commit/a886dc98482
[before-ftz]: ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/results/shard-04/28-test_combo_triton_meta_has_disable_ftz.log
[npu-pr]: https://gitcode.com/Ascend/pytorch/merge_requests/49793
[npu-issue]: https://gitcode.com/Ascend/pytorch/issues/6157
