---
published: false
---

# NPU 多轴归约为什么写不回，以及标量地址如何匹配单元素结果

修复提交：`5db866a16403644c265f3f43d747a856124de8b1`。两个多轴 sum 本来应生成一个 Combo，
实际却因为生成代码不合法而退回两个普通 kernel。最终数值能通过，不代表原 Combo 生成成功。

## 用一行 DSL 看清问题

输入是两个 FP32 tensor，shape 都为 `(32, 64, 128)`，对第 0、2 维求和，
每个输入保留中间 64 个输出。当前 Native Index 生成把保留维放在标量循环里，每次计算一个输出：

```python
tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
tl.store(out_ptr0 + (x0 ), tmp4, None)
```

`tl.sum` 得到一个标量，`promote_to_tensor` 再把它提升成 shape 为 `(1,)` 的单元素 block tensor。
但 `x0` 是循环标量，所以 `out_ptr0 + x0` 还是标量地址。
Triton 要求 block 结果配套 block 地址，不能把“一维、一个元素的结果”交给标量指针。
因此 TTIR 校验报 `Value argument cannot be block type if pointer argument is not a block`。
TTIR 是 Triton 编译检查的中间表示，这个错误发生在设备执行之前。

## 为什么不能只删除结果的提升

[reduction_resize](https://gitcode.com/luqichao/pytorch_new/blob/5db866a16403644c265f3f43d747a856124de8b1/torch_npu/_inductor/codegen/triton.py#L4044) 在
`triton_tensor_ndim() == 1` 时明确使用 `promote_to_tensor`，后续融合运算沿用这个 tensor 布局。
问题出在写回地址没有配套跟上，直接改掉结果布局会同时改变后续运算所依赖的约定。

旧写回代码只在“多归约轴且非 persistent”时加 `tl.arange(0, 1)`，
遗漏了同样产生单元素结果的 persistent 路径。persistent 表示一个 program 覆盖完整归约范围，
并不意味着其结果一定是标量类型。

## 对照代码看修复逻辑

新增
[_reduction_store_index](https://gitcode.com/luqichao/pytorch_new/blob/5db866a16403644c265f3f43d747a856124de8b1/torch_npu/_inductor/codegen/triton.py#L3141)：

```python
if self.numof_reduction_axis() > 1 and (
    not self.persistent_reduction or self.triton_tensor_ndim() == 1
):
    return f"{index_str} + tl.arange(0, 1)"
return index_str
```

第一层要求多归约轴，限定在原问题所在的生成路径。
第二层保留原非 persistent 的处理，并补上实际生成 rank 为 1 的 persistent 情况。
这个 rank 判断与 `reduction_resize` 的提升条件一致，不依赖某个固定 shape 或 kernel 名。

`tl.arange(0, 1)` 是只有一个值 0 的一维 block。加上它以后，地址的数值没有变，
但地址从标量变成 `(1,)` block，刚好与结果匹配：

```python
tl.store(out_ptr0 + (x0 + tl.arange(0, 1)), tmp4, None)
```

只有一个 lane，因此不会把一个输出地址广播成多个有效写入。
普通保留维向量地址加这个零也不会改变元素集合；标量 mask 可以按 Triton 原规则广播到单元素 block。

[store_reduction](https://gitcode.com/luqichao/pytorch_new/blob/5db866a16403644c265f3f43d747a856124de8b1/torch_npu/_inductor/codegen/triton.py#L3176) 与
[普通 store](https://gitcode.com/luqichao/pytorch_new/blob/5db866a16403644c265f3f43d747a856124de8b1/torch_npu/_inductor/codegen/triton.py#L3238) 共用这个函数，
覆盖归约直接写回以及归约结果经过融合运算再写回两个入口。
block pointer、atomic 写回、单归约轴及 persistent 的更高 rank 输出继续走原有生成规则；
没有更改累加精度、归约元素范围或延迟写回时机。

## 怎么证明是真修复

先对保存的原失败 DSL 做独立重放：同一组输入、同一配置，原地址编译失败，
只规范化两处写回地址后编译、执行及原 `atol=rtol=1e-4` 数值检查通过。
重放使用 `X0BLOCK=7`，64 个输出还覆盖了最后一个不满 block 的循环边界。

再执行原 pytest：

```bash
python -m pytest -v test_combo_kernels.py::ComboKernelTests::test_combo_kernel_per_config_subkernel_red
```

原数值检查和“一共一个生成 kernel”目标均通过；实际生成是含两个成员的 persistent Combo。
NPU 用例提交 `7332db7` 另将后续 CUDA 参数名断言适配为真实 NPU ABI：共享 `X0BLOCK`，
两个成员各有完整 `R2BLOCK_SUB=32`、`R1BLOCK_SUB=128`。
输入、归约维度、数值容差和单 Combo 目标全部保留，不能用放宽数量目标代替生产修复。

全量回归中的 `test_combo_kernel_per_config_subkernel_r0_r1` 也受到这项修复影响。
其原输入保留维为 16，归约维为 32、64；原先退回两个普通 kernel，旧数量断言因此通过。
现在两条归约已合法合为一个 persistent Combo，旧断言仍要求两个，于是在数量检查失败。
用例提交 `023fd81` 改为检查一个真实 Combo、两个完整成员和共享 `X0BLOCK`；
原非连续输入及 eager/compiled 分别对相同 FP64 基准的数值检查保留。
[定向复测生成代码](../../../../artifacts/inductor-combo-tests-v2.13-case-followup-023fd81-20261115/generated-metadata.json)
确认两个写回地址也都按本修复规范化，复测 PASS。

## 验证版本与复现

责任仓库为 `Ascend/pytorch` 的 `torch_npu`，本地仓库名 `pytorch_new`，分支
`bugfix/combo_fix_meta`；包含两个修复的 Python overlay 提交为
`5db866a16403644c265f3f43d747a856124de8b1`。
基础包为 `2.13.0+gitefc05dd`，PyTorch 为 `2.13.0+cpu`
（`cf30153c4c131c8164ee7798e5022d810682e2cb`），Triton 为 `3.6.0`。

用例来自 `Community-Use-Case-Testing` 的 `fix/combo_case`，实际提交
`7332db7c9fc5a5d10da2f17ec29e22866d38154a`，目录为 `v2.13.0/inductor_npu`。
在 NPU 容器工作目录激活 `pta_213_case`，设置 `ASCEND_RT_VISIBLE_DEVICES` 为获准物理卡，
使用独立 Inductor/Triton 缓存后执行上述 node ID。
安装文件和测试源码的校验结果、逐实例报告及完整 PASS 回归见
[最新执行结果](inductor-combo-tests-v2.13.md#validation213)。

本轮完整 PASS 回归及必要的定向复测再次覆盖本说明的 2 个方法、2 个实例，均 PASS。
实际版本、全部参数报告和执行来源见[完整回归与合并校验](inductor-combo-tests-v2.13.md#validation213)。
