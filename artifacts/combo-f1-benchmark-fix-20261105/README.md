# F1 修复复测与 F8 阻塞证据

验证日期：2026-11-05。环境：NPU_A5 Docker，torch 2.10.0+cpu、
torch_npu 2.10.0.post7+gite32cdeb、Triton 3.6.0、CANN 9.1.0。

基线提交：`e32cdebf77e797fc85a1a8c3fe1e0eef050f9c0e`。
修复提交：`6c6454567f795cb580ca07da68884e15bdd16c6a`，
分支 `fix/v2.10.0-combo-benchmark-dynamic-args`，已推送到 pytorch_new 的 origin。

实际通过进程内替换加载 `NPUComboKernel.kernel_benchmark_extra_args()`，
其源文件 SHA-256 与最终修复提交相同，见 `validation_summary.json`。
安装包保持基线版本。提交整理只删除了新增测试，最终生产修复与实测版本一致。

## 结果

六个原 F1 方法定向复测：5 个通过，1 个在执行 wrapper 时因 `u0` 提前使用失败，记录为 F8。
逐方法的 `test_*.jsonl` 保留 setup/call/teardown 结果；五个通过方法三个阶段全部通过。
调试时的 10 项 codegen 检查也全部通过，最终修复提交不包含新增测试。

`validation_summary.json` 同时记录基线与定向复测的合并状态：49/62 个方法、394/409 个实例通过。
这不是全部 409 个实例的一次重新运行；其他方法继续沿用原始基线结果。

当次方法隔离执行的命令形式如下；两个插件分别负责沿用既有测试依赖适配和进程内加载候选方法：

```bash
python -m pytest -p combo_validation_plugin -p f1_validation_patch -q -rA \
    --tb=short --disable-warnings \
    test_unbacked_symints.py::TestUnbackedSymintsPRIVATEUSE1::test_combo_kernel_size_hint_failure_npu
```

## 生成代码与源码对照

- `fixed_dynamic_activations_benchmark.py`：修复后返回 14 个用户参数，含 6 个动态尺寸。
- `unbacked_output_code.py`：完整生成 wrapper，显示 `_x0_numel = 1 + u0` 等三条语句早于 `u0` 绑定。
  内嵌 Combo benchmark 已返回四个动态尺寸，说明 F1 已越过，F8 仍阻塞实际输出比较。
- `source/npu_triton_add_numel_to_call_args.py`：基线 NPU 对复合尺寸调用 wrapper helper。
- `source/npu_wrapper_generate_node_numel_expr.py`：基线 helper 无条件向 wrapper 写入尺寸计算。
- `source/upstream_triton_kernel_benchmark_extra_args.py`：上游 2.10 用空 kernel 名字收集 benchmark 尺寸。
- `source/upstream_wrapper_generate_numel_expr.py`：上游 2.10 对空名字不写 wrapper，提供对应行为依据。

源码片段按原方法保留，用于阅读对照。F8 定位为 NPU benchmark 收集参数时的 wrapper 写入副作用，
与 F1-F7 的既有根因不同；有问题的源码在基线中已存在，不是 F1 提交新增。
本次仅记录分析，未修复 F8，也未新增禁用 Combo 等远程对照运行。

原始全量基线位于 `../combo-ut-210-npu-a5-20261105/`，未覆盖。
本目录产物已泛化临时路径和安装目录，移除完整 backend_hash；保留原生成代码排版及计算逻辑。
