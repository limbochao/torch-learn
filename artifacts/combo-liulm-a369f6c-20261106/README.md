# liulm Combo 能力新基线复测证据

验证日期：2026-11-06。完整复测完成：**62 个方法中 49 个全部通过、13 个失败；
409 个参数化实例中 394 个通过、15 个失败、0 个跳过**。全部方法结束，无超时项。

基线：`liulm/v2.10.0-combokernel-dev`，提交
`a369f6c8da77e78a18ccab65e566d0fd6bb0b005`。
本地 HEAD、liulm 远端分支及 NPU_A5 安装包的 git 版本已核对一致。
环境为 Docker、Ascend950PR，torch 2.10.0+cpu、torch_npu 2.10.0.post7+gita369f6c、
Triton 3.6.0、CANN 9.1.0；11 个相关安装源码文件的 SHA-256 校验一致。

本轮直接运行安装包，不修改共享安装目录、不加载旧 F1/F3 overlay。
生产代码保持新基线原样。本轮测试侧适配与旧基线最终执行版本完全一致。

## 范围和执行方式

重新收集 62 个源方法、409 个参数化实例，实例集合与此前核定清单一致。
来源是 `Community-Use-Case-Testing` 的提交
`be6049a0059677ca408da1360653b694b6d1afb5`，使用 v2.13.0 NPU 测试目录。

沿用 [测试适配与运行脚本](../../scripts/tests/combo_kernel_210/README.md)。
最终完整运行在新的独立验证目录进行；原先少量预运行的日志保留在远端，不计入结果。
最终 runner 仅扩展允许的物理卡选择，并在每个方法启动前核对 4、5 卡可用时段。
未改变测试计算、参数化范围或 benchmark 设置。

```bash
cd <validation-root>
python capture_liulm_environment.py
export ASCEND_RT_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD"
export COMBO_SELECTED_MANIFEST="$PWD/combo_npu_a5_selected.json"
export COMBO_COLLECTION_JSON="$PWD/evidence/collection.json"
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
cd Community-Use-Case-Testing/v2.13.0/inductor_npu
python -m pytest -p combo_validation_plugin --collect-only -q \
    test_torchinductor.py test_combo_kernels.py test_foreach.py \
    test_unbacked_symints.py test_provenance_tracing.py
cd ../../..
unset COMBO_COLLECTION_JSON
COMBO_PHYSICAL_DEVICES=1,3,4,5 python -u run_combo_selected.py
python export_combo_results.py
python capture_combo_evidence.py
```

物理卡 4、5 仅在北京时间 09:00–22:00 允许时段内使用；物理卡 6 未使用。
每个方法独立 pytest 进程，卡内串行、卡间并行。各卡有独立编译缓存。
原输入、stride、dtype、数值/梯度/突变断言及精度阈值保持不变。
测试依赖适配、CUDA 专属 kernel 数量、源码结构和 profiler/grid 检查的处理与旧基线一致。
功能通过不保证最终采用 Combo 融合，也不表示满足 CUDA 的 grid、融合数量或性能要求。

## 结果与版本对应

- `environment.json`：新基线安装版本、设备及 11 项源码校验。安装包版本以此为准，忽略 SSH 欢迎页的旧提示。
- `collection.json`：本轮重新收集、与旧集合核对一致的 409 个实例。
- `validation_summary.json`：最终统计、按文件结果及新旧基线变化。
- `functional_results.json`：62 个方法的返回值、409 个实例的结果及完整失败堆栈。
- `integrity_checks.json`：409 个实例完整上报，setup/call/teardown 检查通过，11 项安装源码前后未变。
- `test_equivalence.json`、`collection_equivalence.json`：五个执行文件与旧适配逐字节一致，409 个实例集合相同。
- `jobs.jsonl`、`run_summary.json` 和 `test_*.jsonl`/`.xml`：各方法开始/结束、逐实例阶段报告。
- `failure_analysis.json`：13 个失败方法与 F2–F8 分类对应；helper 用例被 F3 阻塞，不能判定 F9 已修复。
- `generated_on_failure/`：失败时捕获的生成代码，保持原格式；F8 的 wrapper 可见 `_x0_numel` 计算早于 `u0` 绑定。
- `installed_source/`、`original_tests/`、`executed_tests/`、适配 patch、`scripts/`：关键源码及原始/实际执行测试快照。
- `fix_relationship.json`：新基线与旧 F1/F3/F9 对应方法的 AST 比较。
- `archive_manifest.json`：441 份归档文件的原始/脱敏 SHA-256。个人目录、PID、XML 主机名和 backend_hash 已脱敏。
  生成文件的实际本地路径因脱敏不可直接复用；重新执行应使用上面的公共运行流程。
  `generated_python_manifest.json` 是远端完整缓存快照的清单，归档只保留失败时捕获的生成代码，未复制完整通过缓存。

“原基线最新结果”来自旧基线 `e32cdebf77` 与 F1/F3 定向复测的合并记录，
为 55/62 个方法、400/409 个实例通过，并非一次全量重跑或一个统一安装版本。
本轮新结果来自新基线的完整执行，不将旧结果混入新计数。

对应 [新基线表格](../../docs/notes/compiler/codegen/inductor-combo-tests-liulm-v2.10.0-combokernel-dev.md)。

## 本轮结论

Foreach 的 27 个方法、372 个实例全部通过。其余 13 个失败方法共 15 个失败实例，均能对应此前问题：
F2 1 个、F3 7 个、F4 2 个、F5 2 个、F6/F7/F8 各 1 个实例。
没有识别出新的失败类型。新基线包含与旧 F1 一致的实现，但没有包含旧 F3 分区适配，
因此旧 F3 修复后已通过的 6 个方法本轮重新失败。helper 用例本轮提前被 F3 阻塞，未验证到 F9。

原始证据压缩包的远端与本地 SHA-256 一致：
`0d5fc11f0a49e11fc9bedc497c14aa66fa3326fbae14983986ab1baa254407fe`。
原始证据保留在验证环境；本目录保存可公开引用的脱敏文本。
