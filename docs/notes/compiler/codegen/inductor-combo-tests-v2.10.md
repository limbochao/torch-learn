---
published: false
---

# 2.10 Combo/foreach 最新结果与问题分析

<a id="native210"></a>

<!-- combo-section:native210:start -->

## 最新执行结果

最新 foreach 复测完成时间：2026-11-10 16:41:25（北京时间）。

| 范围 | PASS | FAIL | SKIP |
| --- | ---: | ---: | ---: |
| 54 个源测试方法 | 50 | 4 | 0 |
| 402 个参数实例 | 397 | 5 | 0 |

本次更新的 27 个 foreach 方法、373 个实例全部通过；其他 27 个方法沿用相同安装包的有效结果，
并非所有方法在同一测试源码快照上重跑。

### 实际验证版本与口径

- 用例目录：`v2.10.0/test_inductor`，直接使用 2.10 原生用例。
- 包：`torch_npu 2.10.0.post7.dev20261110`，包提交 `2e109cfd759b14f302ee16a6896c767a3a236854`。
- PyTorch `2.10.0+cpu`；Triton 实际导入版本 `3.6.0`。本轮没有叠加生产代码修复，安装源码前后哈希一致。
- foreach 测试源码 `3d9ca5392d6ec9f1d161565ebc14eae8acc03d3b`；其他方法的结果对应 `f998b3ddbbed76955e0df794099285d495df5849`。
- foreach 在 NPU 上保留数值、输入布局、mutation 和梯度检查，不要求 CUDA 的 kernel 数量和专用生成文本；功能 PASS 不证明 FOREACH 分组融合。

### 当前未通过用例

| 测试文件 | 用例 | 最新结果 | 当前分析 |
| --- | --- | --- | --- |
| test_torchinductor.py | test_adaptive_avg_pool2d1 | 失败（FAIL 2；共 2） | 数值比较失败；AssertionError: Tensor-likes are not close! / Mismatched elements: 288 / 288 (100.0%) / Greatest absolute difference: 5.596200466156006 at index (1, 2, 0, 2) (up to 1e-05 allowed) |
| test_combo_kernels.py | test_2d_blocking_benchmark | 失败（FAIL 1；共 1） | 定向复测 generated_kernel_count=6，原断言要求 7–8；Combo 已生成且无回退。计数包含 benchmark 候选，最终分区为 1D 单节点及 2D 双节点；不能按该计数直接判定融合未生效 |
| test_combo_kernels.py | test_dynamic_shapes_2d_blocking | 失败（FAIL 1；共 1） | 定向复测 generated_kernel_count=6，原断言要求 7–8；Combo 已生成且无回退。计数包含 benchmark 候选，最终分区为 1D 单节点及 2D 双节点；不能按该计数直接判定融合未生效 |
| test_provenance_tracing.py | test_triton_kernel_to_post_grad_tracing_combo_kernel | 失败（FAIL 1；共 1） | provenance 映射检查失败；AssertionError: The length of the sequences mismatch: 0 != 1 |

### 处理判断

`test_adaptive_avg_pool2d1` 是实际数值失败，Combo False/True 两个实例都失败；
仅凭当前报告不能将它归因为 Combo 融合特有问题，需要对实际生成索引和语义 mask 定位。

两个 2D blocking 方法失败于计数 6 对原断言 7–8；已确认 Combo 生成成功，没有独立 kernel 回退。
计数包含 benchmark 候选，不能当作实际 launch 数量，也不能据此判为融合未启用。
正式生成分为 1D 单节点及 2D 双节点，社区同样要求相同的 blocking strategy；
当前没有 CUDA 对照证据解释 7–8 的候选来源，先保留为策略/测试预期差异。

provenance 方法失败于映射长度 0 对预期 1，属于追踪信息检查失败，当前未证明数值计算错误。

[完整 54 方法 CSV](inductor-combo-tests-native-v2.10.csv)；
[当前 foreach 结果](../../../../artifacts/combo-native210-foreach-functional-20261110/summary.json)；
[其他方法结果来源](../../../../artifacts/combo-native210-daily20261110/summary.json)；
[计数定位证据](../../../../artifacts/combo-native210-daily20261110/fusion_diagnosis/benchmarks/events.json)；
[执行命令](../../../../scripts/tests/combo_kernel_210/README-native-210.md)。

<!-- combo-section:native210:end -->
