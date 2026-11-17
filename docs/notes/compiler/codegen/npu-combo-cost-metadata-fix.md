---
published: false
---

# NPU Combo 的访存量、FLOP 和 benchmark 带宽如何修复

## 问题现象

原测试直接报 `combo source missing kernel_num_gb`；只开带宽记录时，也找不到
`'kernel_num_gb'`。另一个测试检查生成的 benchmark 代码，发现没有实际访存量。
旧 Combo 代码把 `num_gb` 写死为 0，所以即使程序读写了数据、耗时正常，算出来的带宽仍是 0。
原失败见[成员汇总记录][before-cost]和[带宽记录][before-profile]。

这里错的是性能统计，不是张量计算结果。成本指读写多少数据、做多少次运算，不是费用。
kernel 是设备执行的小程序；Combo 把几条独立计算分支合在一个程序中，每条分支称为成员。

## 是否同步上游 CUDA 修复

**是，适配上游 Inductor 的 Combo 成本统计修复。** 对应
[PyTorch PR #180813][upstream-pr]，合入提交为 [b3fbb25559b][upstream-commit]。
上游把成本估算移到普通 kernel 与 Combo 成员都能调用的入口，再把成员结果相加，
并把 Combo benchmark 的 `num_gb=0` 改为实际总量。

NPU 原修复提交为 `72b1727020`，移植到 master 后为 `cb600ef44d`。
NPU 使用自己的 `_kernel_cost_meta()` 接上相同规则，未直接照搬整份上游方法。

## 成本数字从哪里来

`kernel_num_gb` 是按生成图估算的读写字节数除以十亿，不是硬件实测流量；
`kernel_flop` 是已有估算器给出的浮点运算次数，FLOP 指一次浮点加、乘等运算。
benchmark 是试跑并计时；`profile_bandwidth` 是附带记录带宽的模式。
带宽用估算访存量除以实测耗时计算，所以分子写成 0，最后也只能得到 0。

metadata 是随生成代码附带的字典。这几个字段相当于告诉性能工具“这次运行大约做了多少工作”，
它们不会参与张量数值计算，也不是设备硬件计数器测出的实际内存流量。

## 先让普通 kernel 和成员共用来源

[NPUIndexTritonKernel._kernel_cost_meta](https://gitcode.com/luqichao/pytorch_new/blob/5db866a16403644c265f3f43d747a856124de8b1/torch_npu/_inductor/codegen/triton.py#L1933)
集中决定什么时候估算、导出哪些值：

```python
if config.benchmark_kernel or config.profile_bandwidth or config.benchmark_combo_kernel:
    metadata["kernel_num_gb"] = self.estimate_kernel_num_bytes() / 1e9
if config.benchmark_kernel or config.benchmark_combo_kernel:
    flops = self.estimate_flops()
    if flops is not None:
        metadata["kernel_flop"] = flops
```

普通 kernel 通过 `inductor_meta.update(self._kernel_cost_meta())` 使用这份结果。
Combo 对每个真实成员调用同一函数，再在
[jit_line](https://gitcode.com/luqichao/pytorch_new/blob/5db866a16403644c265f3f43d747a856124de8b1/torch_npu/_inductor/codegen/triton_combo_kernel.py#L301) 中求和：

```python
sub_metas = [sub._kernel_cost_meta() for sub in self.sub_kernels]
self._kernel_num_gb = sum(meta.get("kernel_num_gb") or 0 for meta in sub_metas)
```

例如两个成员各估算读写 8,192 字节，合计就是 16,384 字节，即 `1.6384e-05 GB`。
先用相同估算器算每个成员，再相加，才能与这些成员分别作为普通 kernel 时对得上。

## 再把同一个数交给 benchmark

[Combo benchmark 生成入口](https://gitcode.com/luqichao/pytorch_new/blob/5db866a16403644c265f3f43d747a856124de8b1/torch_npu/_inductor/codegen/triton_combo_kernel.py#L550)
从 `num_gb=0` 改为 `num_gb=self._kernel_num_gb`。
普通 kernel 的 benchmark 也直接读取刚生成的 `inductor_meta["kernel_num_gb"]`，
不再另外维护一份容易不同步的局部变量。

只开 `profile_bandwidth` 时导出 GB；`benchmark_kernel` 模式还导出 FLOP。
只开 `benchmark_combo_kernel` 时也会计算成员访存量，并将总量交给 Combo 的计时代码。
没有打开这些模式时不额外增加统计字段。
这项修复没有新增逐成员 `inductor_meta_i` 字典，也没有实现独立成员调优。
也就是说，它能回答“这个 Combo 合计读写多少”，不负责让每个成员单独选择分块配置。

## 怎么证明修好了

三个方法分别检查成员之和、仅 profile_bandwidth 的字段范围、benchmark 收到的实际 GB：
`test_combo_kernel_num_gb_and_flop_match_standalone_sum`、
`test_combo_inductor_meta_has_kernel_num_gb_under_profile_bandwidth`、
`test_benchmark_combo_kernel_emits_real_num_gb`。

[首次实测摘要](../../../../artifacts/inductor-combo-tests-v2.13-meta-three-9f639af716-20261115/validation-summary.json)
记录普通成员各为 `8.192e-06 GB`，Combo 及 benchmark 均为 `1.6384e-05 GB`。
该图的既有 FLOP 估算为 0；修复忠实导出和相加，没有为 sigmoid 等运算新增 FLOP 公式。

## 验证版本

代码属于 `Ascend/pytorch` 的 `torch_npu`；本地仓库名为 `pytorch_new`，修复分支为
`bugfix/combo_fix_meta`。以下代码链接固定到包含这六项修复的提交
`5db866a16403644c265f3f43d747a856124de8b1`。

基础 torch_npu 包提交为 `efc05dd90b5fe31717d5f13e51fb3e49035bea30`，
验证通过 Python overlay 加载修复文件，即用修复后的 Python 文件覆盖已安装包中的对应文件；
不能把包版本字符串当成实际修复版本。
PyTorch 为 `2.13.0+cpu`（`cf30153c4c131c8164ee7798e5022d810682e2cb`），Triton 为 `3.6.0`。
完整运行命令、文件哈希和逐实例结果见[最新验证汇总](inductor-combo-tests-v2.13.md#validation213)。

2026-11-15 的已通过用例集合回归及必要定向补测覆盖本说明的 3 个方法、3 个实例，均 PASS。
实际版本、全部参数报告和执行来源见[完整回归与合并校验](inductor-combo-tests-v2.13.md#validation213)。

六项修改已迁入 master 基线分支，对应 [PR !49793][npu-pr] 和 [Issue #6157][npu-issue]。
移植后只做了本地静态核对，以上 NPU 结果仍属于原修复分支。

[upstream-pr]: https://github.com/pytorch/pytorch/pull/180813
[upstream-commit]: https://github.com/pytorch/pytorch/commit/b3fbb25559b
[before-cost]: ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/results/shard-02/24-test_combo_kernel_num_gb_and_flop_match_standalone_sum.log
[before-profile]: ../../../../artifacts/inductor-combo-tests-v2.13-full-a3cfe56-20261113/results/shard-05/32-test_combo_inductor_meta_has_kernel_num_gb_under_profile_bandwidth.log
[npu-pr]: https://gitcode.com/Ascend/pytorch/merge_requests/49793
[npu-issue]: https://gitcode.com/Ascend/pytorch/issues/6157
