---
published: false
---

# NPU Combo 为什么丢失 FX 来源，以及如何对齐上游

## 问题现象

原用例已经完成 relu、sigmoid、tanh 的计算和 trace 文件输出，但 `cppCodeToPost` 是空字典。
这个字段记录“设备 kernel 对应哪些 FX 运算”，为空时无法从生成代码或性能分析结果回查模型运算。
问题发生在来源登记阶段，原张量计算没有因此报错。

FX 图是 PyTorch 记录的运算关系图；kernel 是交给设备执行的程序。
provenance 就是记录“这个程序来自图里的哪些操作”。
本用例期望 `triton_poi_fused_0:1 -> [relu, sigmoid, tanh]`，其中 `:1` 是本次调用的 debug handle。

## 上游来源与当前提交

当前实现对齐上游 Combo 的调用方式，相关来源为
[PyTorch PR #163378][upstream-reference]。该 PR 把来源映射登记统一接入 `codegen_comment`。
对照的[上游代码][upstream-code]分别传递：

```python
self.define_kernel(src_code, [combo_kernel_node], kernel)
self.codegen_comment(combo_kernel_node.snodes, kernel_name)
```

上游给定义入口传外壳，给来源追踪入口传内部成员；两个入口的入参不能混为一谈。
这属于补齐 NPU 的上游追踪接入，不是同步一份独立的 CUDA 数值计算 bug 修复。
当前实现提交为原分支 `961b95d9ac`、master PR 分支 `0239de8824`。

## 为什么外壳和成员不等价

`combo_kernel_node` 负责组织一组成员，自己的 `node` 为 None。
IR 是编译器内部表示运算的数据结构，真实运算的 IR 上保存 `origins`，即它来自哪些 FX 节点。
`.snodes` 表示组内成员；`[combo_kernel_node]` 则是只放了外壳的列表。

公共追踪函数遍历传入的列表，读取 `snode.node.origins`；遇到 `snode.node is None` 就跳过，
不会自动打开 Combo 外壳。因此 NPU 原来的这一行无法登记来源：

```python
self.codegen_comment([combo_kernel_node], kernel_name)
```

相对修复前，最终只需要把这一行的参数改为上游使用的成员列表：

```python
self.codegen_comment(combo_kernel_node.snodes, kernel_name)
```

`define_kernel` 继续传 `[combo_kernel_node]`；NPU 额外传入的 `None` 对应本地接口参数，
不改变上述来源处理方式。这个修复没有重新调度、重新分区或改变张量运算。

## 定向复测与证据

2026-11-17 在 Ascend NPU 的 `pta_213_case` 环境执行原用例：

```bash
python -m pytest -v test_provenance_tracing.py::TestProvenanceTracingArtifact::test_triton_kernel_to_post_grad_tracing_combo_kernel
```

用例来自 `Community-Use-Case-Testing:fix/combo_case`，提交
`51830b03317074409a04c0ff167ee0d7ad6c897c`，目录为 `v2.13.0/inductor_npu`。
用例只撤回上一版修复引入的 NPU 专属 kernel 名称预期，恢复上游的 `triton_poi_fused_0`；
完整来源列表和 debug handle 的比较保留，见[用例差异][test-diff]。

该方法依次覆盖 Inductor 与 AOTInductor，两条路径均通过，映射都完整包含
`relu`、`sigmoid`、`tanh`。原生 pytest 报告为 **1 passed**，没有自定义 pytest 插件。
测试使用独立缓存，源码在执行前后保持一致，见[复测报告][validation]及[实际版本与文件哈希][context]。

实际加载的是 PR 提交 `0239de8824465109aec4313f9dd1d50a1fba5116` 的 `scheduling.py`，
基础 torch_npu 为 `2.13.0+gitefc05dd`，其他已安装修复文件保留；这属于 Python overlay 验证，
不代表重新构建并完整测试了 master 包。
PyTorch 为 `2.13.0+cpu`（`cf30153c4c131c8164ee7798e5022d810682e2cb`），Triton 为 `3.6.0`。

本次只复测此方法，其余 104 个方法保留原报告，见[汇总合并校验][merge]。
多分区、融合成员及缓存复用的来源完整性不由这个单分区用例保证；当前保持上游实现的处理范围。

[upstream-reference]: https://github.com/pytorch/pytorch/pull/163378
[upstream-code]: https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/_inductor/codegen/simd.py#L3734
[validation]: ../../../../artifacts/inductor-combo-tests-v2.13-provenance-upstream-0239de8824-20261117/validation-summary.json
[context]: ../../../../artifacts/inductor-combo-tests-v2.13-provenance-upstream-0239de8824-20261117/run-context.json
[test-diff]: ../../../../artifacts/inductor-combo-tests-v2.13-provenance-upstream-0239de8824-20261117/test-change.diff
[merge]: ../../../../artifacts/inductor-combo-tests-v2.13-provenance-upstream-0239de8824-20261117/merge-verification.json
