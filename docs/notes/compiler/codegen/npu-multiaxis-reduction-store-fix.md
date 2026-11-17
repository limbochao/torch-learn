---
published: false
---

# NPU 多轴归约为什么写不回，以及标量地址如何匹配单元素结果

## 问题现象

两个多轴 sum 本来应生成一个 Combo，实际却因为生成代码不合法而退回两个普通 kernel。
所以原测试的数值检查可以通过，随后“一共只生成一个 kernel”的断言却失败。
这不是求和结果算错，而是预期的组合程序没有编译成功，被回退路径掩盖了。

把保存的原失败 Triton 程序单独重放，可以直接看到编译错误：
`Value argument cannot be block type if pointer argument is not a block`。
大白话就是“要写的是一组元素，给出的地址却只是一个标量地址”。

## 是否同步上游 CUDA 修复

**不是。本次是 NPU 自有多轴索引与写回生成逻辑的修复，没有对应的上游同步 PR。**
修改入口是 NPU 的 `NPUIndexTritonKernel.store_reduction` 和 `store`，
解决 NPU 将保留维放入标量循环后产生的地址形状问题。
上游同样有[将标量归约结果转成 tensor 的公共规则][upstream-resize]，
但共用这一规则不等于本补丁来自某个 CUDA PR，也不能据此断言 CUDA 同输入一定有或没有此问题。

NPU 原修复提交为 `5db866a164`，移植到 master 后为 `528e87d566`。

## 先把几个词说清楚

归约是把多份数值合成更少结果；多轴归约就是一次沿几个维度求和等。
kernel 是交给设备执行的小程序，Combo 把几条独立分支放进同一个程序。
DSL 是这里生成的 Triton 程序，store 是将结果写回内存，pointer 是要写入的地址。

scalar 是单个值；block tensor 是一组带形状的元素，即使只有一个元素，它仍带有 `(1,)` 这样的形状。
`rank` 表示维数：这里判断的是生成程序内部 tensor 的维数，不是原 PyTorch 输入的维数。
例如输入可以是三维，归约中间值经展平后内部 rank 仍可能为 1。
这里的 block 是 Triton 类型 / 形状概念，与“成员独立块调优”功能不是一回事。

## 用一行 DSL 看清问题

输入是两个 FP32（32 位浮点）tensor，shape 都为 `(32, 64, 128)`，对第 0、2 维求和，
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
遗漏了同样产生单元素结果的 persistent 路径。
persistent reduction 表示一次程序实例覆盖完整归约范围，减少分段归约循环；
它描述如何完成归约，不保证最终结果是标量类型。

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

`tl.arange(0, 1)` 的内容只有 `[0]`，所以它只把地址变成单元素数组，不会多写一个位置。
lane 指 block 中的一个元素位置；mask 是限制哪些元素允许读写的条件。
普通保留维向量地址加这个零也不会改变元素集合；标量 mask 可以按 Triton 原规则广播到单元素 block。

[store_reduction](https://gitcode.com/luqichao/pytorch_new/blob/5db866a16403644c265f3f43d747a856124de8b1/torch_npu/_inductor/codegen/triton.py#L3176) 与
[普通 store](https://gitcode.com/luqichao/pytorch_new/blob/5db866a16403644c265f3f43d747a856124de8b1/torch_npu/_inductor/codegen/triton.py#L3238) 共用这个函数，
覆盖归约直接写回以及归约结果经过融合运算再写回两个入口。
block pointer、atomic 写回、单归约轴及 persistent 的更高 rank 输出继续走原有生成规则；
没有更改累加精度、归约元素范围或延迟写回时机。

这里 block pointer 指用整块描述方式访问内存，atomic 指多个执行者可安全累加同一地址的原子写回。
上述新增判断只用于普通指针写回，范围如下：

| 普通指针写回条件 | 本次行为变化 |
| --- | --- |
| 单归约轴 | 无 |
| 多归约轴、非 persistent | 保留原先增加单元素零偏移的行为 |
| 多归约轴、persistent、内部 rank = 1 | 新增单元素零偏移，使标量地址匹配单元素结果 |
| 多归约轴、persistent、内部 rank > 1 | 无，不属于本次补齐条件 |

最后一行只说明没有改这条路径，不代表已穷举证明所有高 rank 布局都不会发生形状问题。
普通 kernel 和 Combo 成员共用这个生成器，所以影响范围不限于 Combo。
恢复 Combo 后，测试场景从两个回退 kernel 变成一次组合启动，执行安排会变化；
没有做性能 A/B 测量，不能据地址数值不变就保证整体性能不变。

## 怎么证明是真修复

先对保存的原失败 DSL 做独立重放：同一组输入、同一配置，原地址编译失败，
只规范化两处写回地址后编译、执行及原 `atol=rtol=1e-4` 数值检查通过，
其中 `atol`、`rtol` 分别表示绝对和相对误差容差。
重放使用 `X0BLOCK=7`，64 个输出还覆盖了最后一个不满 block 的循环边界。

再执行原 pytest：

```bash
python -m pytest -v test_combo_kernels.py::ComboKernelTests::test_combo_kernel_per_config_subkernel_red
```

原数值检查和“一共一个生成 kernel”目标均通过；实际生成是含两个成员的 persistent Combo。
NPU 用例提交 `7332db7` 另将后续 CUDA 参数名断言适配为实际 NPU 调用参数：共享 `X0BLOCK`，
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

2026-11-15 的已通过用例集合回归及必要定向补测覆盖本说明的 2 个方法、2 个实例，均 PASS。
实际版本、全部参数报告和执行来源见[完整回归与合并校验](inductor-combo-tests-v2.13.md#validation213)。

六项修改已迁入 master 基线分支，对应 [PR !49793][npu-pr] 和 [Issue #6157][npu-issue]。
移植后只做了本地静态核对，以上 NPU 结果仍属于原修复分支。

[upstream-resize]: https://github.com/pytorch/pytorch/blob/cf30153c4c13/torch/_inductor/codegen/triton.py#L4523
[npu-pr]: https://gitcode.com/Ascend/pytorch/merge_requests/49793
[npu-issue]: https://gitcode.com/Ascend/pytorch/issues/6157
