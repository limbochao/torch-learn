# v2.13 测试在 v2.10 NPU 上的功能验证

对应文档：`docs/notes/compiler/codegen/inductor-combo-tests-v2.13-vs-v2.10.md`。

本次范围为 62 个源测试方法，核定 409 个参数化实例：54 个 2.10 已有方法、
7 个部分支持方法的旧版分支、1 个目标行为已存在的方法。11 条嵌套函数记录随外层方法执行，
不作为独立测试。已判定缺新增能力或缺场景修复的 42 个方法，以及 XPU 专用待验证方法不运行。

执行环境为 NPU_A5 Docker，工作目录 `/home/l30023782/PTA_210/combo_kernel_ut`。
使用已安装的 torch 2.10.0 和 torch_npu 2.10.0.post7+gite32cdeb；
包版本中的提交与本地 `pytorch_new` 的 `jimmy/v2.10.0-1102-combo` 提交一致。
只允许使用已获授权的物理卡 1、2、3，各子进程通过 `ASCEND_RT_VISIBLE_DEVICES` 映射为逻辑卡 0。

远端 GitHub clone 超时后，使用本地 `Community-Use-Case-Testing` 的提交
`be6049a0059677ca408da1360653b694b6d1afb5` 导出 `v2.13.0/inductor_npu` 目录。
导出的是提交中的文件，未复制本地未跟踪文件。测试树放在上述工作目录的
`Community-Use-Case-Testing/v2.13.0/inductor_npu`。

## 脚本

将本目录的 Python 文件和 `combo_npu_a5_selected.json` 复制到远端工作目录根下使用：

- `adapt_combo_210.py`：备份原测试到 `original_cases/`，应用测试侧适配，输出完整 diff 和改动清单。
- `combo_validation_plugin.py`：初始化允许的 NPU，补测试导入符号，筛选目标方法和设备，记录逐实例结果。
- `run_combo_selected.py`：按源方法隔离 pytest 进程，在物理卡 1、2、3 上执行；保留参数化实例。
- `capture_combo_evidence.py`：保存安装包相关源码快照、校验和，以及已生成的 Python kernel/wrapper。
- `export_combo_results.py`：按核定实例汇总结果，拒绝范围外实例；跳过及 teardown 错误不计为通过。

不修改安装包；保留被测函数、输入 shape/stride/dtype、数值断言、梯度断言及精度阈值。
功能适配移除 CUDA 特定的 kernel 数量和 FileCheck 源码结构断言。
两个 grid 用例还移除依赖 `ProfilerActivity.CUDA` 的采样与 grid 断言，保留原输入和输出比较。
因此功能通过不等于 NPU 已满足 CUDA 的融合分组、launch grid、性能或生成代码形式要求。

依赖适配包括：提供缺失的测试工具符号、移除 2.10 缺少的中性初始化配置、
从算子参数列表移除不存在的 `_foreach_clone`；部分支持的方法仅选择共享块或固定 8 节点上限。
测试文件原有的宽泛 NPU 跳过标记被移除，以实际尝试执行本轮目标方法。
剩余 skip 必须单独统计，不计为通过。所有适配以实际生成的 patch 为准。

## 执行

```bash
cd /home/l30023782/PTA_210/combo_kernel_ut
python adapt_combo_210.py
export ASCEND_RT_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD"
export COMBO_SELECTED_MANIFEST="$PWD/combo_npu_a5_selected.json"
export COMBO_COLLECTION_JSON="$PWD/evidence/collection.json"
export TORCHINDUCTOR_COMPILE_THREADS=1
cd Community-Use-Case-Testing/v2.13.0/inductor_npu
python -m pytest -p combo_validation_plugin --collect-only -q \
    test_torchinductor.py test_combo_kernels.py test_foreach.py \
    test_unbacked_symints.py test_provenance_tracing.py
cd ../../..
python -u run_combo_selected.py
python capture_combo_evidence.py
python export_combo_results.py
```

`run_combo_selected.py` 不带参数运行清单中的所有方法；带方法名可运行指定方法。
`COMBO_EXCLUDE_METHODS` 是逗号分隔的已执行方法名，供分批执行使用。
`COMBO_SUMMARY_NAME` 可区分不同批次汇总文件。
`COMBO_PHYSICAL_DEVICES=2` 可将指定方法的复测限制到已授权的物理卡 2；默认仍使用 1、2、3。
重测前应保留前次对应 `.log/.jsonl/.xml`，避免覆盖证据。
