---
published: false
---

# NPU Combo 的访存量、FLOP 和 benchmark 带宽如何修复

修复提交：`72b172702045fd1bb4f084dd67c5faa65c180694`。问题分为同一条链上的两处：
普通 kernel 与 Combo 的成本字段来源不统一，Combo benchmark 又把访存量写死为 0。
结果可能缺少统计字段，或把有访存的 kernel 显示为零带宽。

## 成本数字从哪里来

`kernel_num_gb` 是按生成图估算的读写字节数除以十亿，不是硬件实测流量；
`kernel_flop` 是已有算术量估算器返回的操作数。带宽用访存量除以耗时计算，
所以把前者写成 0，后面的带宽数字再精确也没有意义。

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

只开带宽 profiling 时仅导出 GB；benchmark 模式还导出 FLOP。没有打开这些模式时不额外增加统计字段。
这项修复没有新增逐成员 `inductor_meta_i` 字典，也没有实现独立成员调优。

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
验证通过 Python overlay 加载修复文件；不能把包版本字符串当成实际修复版本。
PyTorch 为 `2.13.0+cpu`（`cf30153c4c131c8164ee7798e5022d810682e2cb`），Triton 为 `3.6.0`。
完整运行命令、文件哈希和逐实例结果见[最新验证汇总](inductor-combo-tests-v2.13.md#validation213)。

本轮完整 PASS 回归及必要的定向复测再次覆盖本说明的 3 个方法、3 个实例，均 PASS。
实际版本、全部参数报告和执行来源见[完整回归与合并校验](inductor-combo-tests-v2.13.md#validation213)。
