---
published: false
---

# NPU 为什么要复用共同的 Triton 编译 metadata

修复提交：`9f639af716399c15e07dbe341ecb2490a9f36e87`。问题是 NPU 的普通 kernel 和 Combo
各自构造 `triton_meta` 字典，漏掉了共同代码已定义的编译设置。
这里修的是设置传递，不是实现所有设置对应的底层硬件能力。

## 两种 metadata 要分清

`triton_meta` 带有函数参数类型、设备、编译设置，主要交给编译与运行时；
`inductor_meta` 放调优和统计等信息。`disable_ftz`、`enable_fp_fusion`、`launch_pdl`
属于共同 helper `TritonKernel.triton_meta_common()` 给出的前一类设置。

三个字段分别表达是否禁止极小浮点数清零、是否允许浮点运算融合、是否采用 PDL 启动。
“字段传到了 NPU”与“NPU 编译器已经实现对应行为”必须分别验证。

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
验证通过 Python overlay 加载修复文件；不能把包版本字符串当成实际修复版本。
PyTorch 为 `2.13.0+cpu`（`cf30153c4c131c8164ee7798e5022d810682e2cb`），Triton 为 `3.6.0`。
完整运行命令、文件哈希和逐实例结果见[最新验证汇总](inductor-combo-tests-v2.13.md#validation213)。

本轮完整 PASS 回归及必要的定向复测再次覆盖本说明的 1 个方法、2 个实例，均 PASS。
实际版本、全部参数报告和执行来源见[完整回归与合并校验](inductor-combo-tests-v2.13.md#validation213)。
