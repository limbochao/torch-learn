# 2.10 原生 Combo/foreach 用例执行说明

结果和当前问题分析只维护 [2.10 固定文档](../../../docs/notes/compiler/codegen/inductor-combo-tests-v2.10.md)。
测试目录为 `Community-Use-Case-Testing/v2.10.0/test_inductor`，
使用该目录原生实现，不搬运 2.13 方法替代缺失测试。

## 清单

- [源方法清单](combo_native_210_selected.json)：54 个方法，包含源类名，供收集实际参数化实例。
- [筛选依据](combo_native_210_selection.json)：来源提交、文件哈希及原清单的逐项选择依据。
- [当前结果 CSV](../../../docs/notes/compiler/codegen/inductor-combo-tests-native-v2.10.csv)。
- [通过实例的实际执行名](combo_native_210_passed_nodeids.txt)：每行一个真实 pytest node ID，
  格式为 `文件名::执行类名::参数化后的用例名`，相对于测试目录。

嵌套函数随外层方法执行，不作为独立方法重复计数。
`select_native_210.py` 从 2.10 兼容流程结果表读取来源信息，结合 2.10 原生源码筛选，不依赖 2.13 进展表。
测试初始化使用 `ut/210_combo` 的对应修改，至少更新到 `9ac2aaf`；具体实际验证提交见固定文档。

## 直接 pytest 执行

NPU 初始化已放入测试文件，不需要自定义 pytest 插件、额外 `PYTHONPATH` 或 manifest 环境变量。
在正确环境中进入 `v2.10.0/test_inductor` 后执行：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 pytest -vs \
  test_combo_kernels.py::ComboKernelTests::test_activation_functions
```

环境已经选卡时沿用其配置；不强制设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD`、
`TORCHINDUCTOR_COMPILE_THREADS` 或 `OMP_NUM_THREADS`。

按通过实例 TXT 选择用例：

```bash
mapfile -t cases < /path/to/torch-learn/scripts/tests/combo_kernel_210/combo_native_210_passed_nodeids.txt
ASCEND_RT_VISIBLE_DEVICES=1 pytest -vs "${cases[@]}"
```

执行前检查物理卡授权和占用情况。源码变更后需要避免复用旧编译代码时，
可以另外指定新的 `TORCHINDUCTOR_CACHE_DIR`，不清理共享缓存。

## 验证口径与发布

NPU foreach 功能适配保留数值、stride、dtype、mutation 和梯度断言，
仅 NPU 不要求 CUDA 的 kernel 数量和专用生成文本；其他后端保留原检查。
功能 PASS 不证明 FOREACH 分组融合。Combo 文件继续保留对应原始断言；
`test_mean` 未主动开启 Combo，不能将其原样通过当作显式 Combo 覆盖。

`publish_native_210.py` 在报告完整且安装源码未变时，更新结果 CSV 和固定文档的 `native210` 章节。
foreach 定向更新只覆盖对应方法，其余方法保留实际来源，不追加旧轮次或历史修复章节。
PASS 的当前分析保持“—”，执行异常和 SKIP 不记为通过。
