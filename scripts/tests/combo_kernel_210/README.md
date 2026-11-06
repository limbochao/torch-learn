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

## 后台一键执行与自动收尾

liulm 最新结果见 [基线表](../../../docs/notes/compiler/codegen/inductor-combo-tests-liulm-v2.10.0-combokernel-dev.md)。
原失败项的两轮结果已单独列在正文前部，不需要从 62 行全量表中查找。

`run_validation.py` 在已准备好的独立 phase 目录中完成收集、并行执行、汇总、证据捕获和完整性检查。
目录需包含实际适配的测试树、`combo_npu_a5_selected.json`、现有 runner/plugin/export/capture 脚本。
定向目录的 manifest 选原失败方法，预期为 15 个实例；全量目录选择完整 manifest，预期为 409 个实例。
每次使用新的 phase 目录，避免覆盖历史报告。安装包由调用者提前安装，脚本不自动修改共享安装目录。

```bash
# 远端 Docker 验证目录；先确认物理卡授权和占用，再执行。
python run_validation.py --root <prepared-phase> --devices 1,2,3 \
    --with-f3 --expected-instances 409 --detach
```

F3 模式要求该 phase 已准备本轮使用的 `combo_baseline_audit.py`、`f3_horizontal_partition.py`，
并在 `combo_validation_plugin.py` 中调用 `combo_baseline_audit.install()`。
这些实际插件和 F3 方法保存在 [本轮执行快照](../../../artifacts/combo-liulm-0fd5a10-20261106/scripts/)。
仅新包模式不带 `--with-f3`，原失败项使用 `--expected-instances 15`。
任务只写 `validation.log`、`validation_status.json` 和 `evidence/`，不会反复输出进度。

`publish_validation.py` 可接管已经运行的本轮 full 任务，无需重启或重复执行用例。
它默认每 180 秒静默检查完成标记；结束后一次下载、脱敏归档、校验、更新 Markdown/CSV，并提交推送到 origin/main。
三个阶段的父目录使用 `install/`、`failed_first/`、`with_fixes/`、`full/` 结构。
本地 raw 目录保留对应原始证据；代码分支不是 main 或推送不是快进时会报错退出，不会强制推送。
它通过临时 Git index 只提交本轮文档和证据，保留其他已暂存修改。

```bash
# 本地 torch-learn 仓库中执行，无需持续查看测试输出。
python scripts/tests/combo_kernel_210/publish_validation.py \
    --host NPU_A5 --remote-root <remote-validation-parent> \
    --raw-root <local-raw-evidence> --repo <torch-learn-root> --detach
```

完成状态、统计和推送 commit 写入 `<local-raw-evidence>/publisher_status.json`；异常写入同目录的 `publisher.log`。
如需手动刷新进行中的结果，对同一命令加 `--snapshot-only` 并去掉 `--detach`，随后运行：

```bash
python scripts/tests/combo_kernel_210/update_liulm_results.py --repo <torch-learn-root> --running
```

快照只读取现有报告，不重跑用例。表格固定跟随原版本对照 CSV 的 116 行及顺序，保留原表的未运行原因；
列为测试文件、函数名、存在性/差异、执行范围、旧基线结果、最新结果、当前分析。
最新结果优先使用完整全量报告，其次使用带时间的全量快照；全量尚未结束时，已有定向结果单独注明来源。
执行异常、跳过、未完成均不记为通过。即使最终控制器因采集失败而报错，发布脚本也会保存快照并推送实际结果，
状态记为 `published_with_execution_errors`，不会宣称完整验证通过。

`archive_liulm_run.py` 和 `update_liulm_results.py` 也支持单独执行，路径通过参数指定。
新增且未经分析的全量失败会明确标为“待分析”，不会套用已有问题分类，也不会把定向结果外推为全量结果。
这些新增入口已做语法和 CLI 检查；当前 NPU 全量任务沿用已经启动的 runner，不为验证入口重复运行测试。

## 单独复测失败项的自动收尾

失败项使用新的 phase 目录与新缓存，筛选最新表中“失败”和“执行异常”的方法；测试树与 F3 插件沿用已验证版本。
用 `run_validation.py --with-f3 --expected-instances <实例数> --detach` 启动后，在本地执行：

```bash
python scripts/tests/combo_kernel_210/publish_retest.py \
    --host NPU_A5 --remote-phase <remote-prepared-phase> \
    --raw-root <new-local-raw-directory> --repo <torch-learn-root> --detach
```

发布脚本默认每 120 秒静默等待完成标记，统一下载证据，校验实例清单、源码与 F3 实际加载记录，
保存在原证据目录的 `retests/<phase>/` 中，并提交推送。状态和提交写入 raw 目录的 `publisher_status.json`。
表格更新脚本读取 `retests/index.json`，只用已结束复测覆盖对应方法的“最新结果”和“当前分析”；
未复测方法仍使用全量记录，CSV 保持原表 116 行、7 列及未运行原因。全量原始统计保留在文档中。

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

## 集成 F3 后的逐项修复验证

F3 合入后的包不再使用 `--with-f3`，避免覆盖新分支的分区规则。
`--devices` 接受显式授权的非负物理卡号；启动前按当前会话授权和占用选择，脚本不停止其他进程。
每个问题使用独立 phase，仍由 `run_validation.py` 完成收集、执行和汇总，不需要逐条查看进度。
下载 phase 的 `evidence/` 与 `validation_status.json` 后，可以批量归档：

```bash
python scripts/tests/combo_kernel_210/archive_fix_series.py \
    --raw-root <downloaded-phases> --output <artifact-directory> \
    --phases baseline fix_f10 fix_f11 fix_f8 fix_f4_complete f5_constants_only
```

归档要求每轮已完整结束、实例清单齐全、安装源码在运行前后未变化。
失败轮次也保留真实结果；日志去敏后记录原始和归档文件的 SHA256。
总表更新支持复测目录中的 `validation_context.json`：`label` 说明实际基线，
`fix_commits` 按方法名记录修复提交；通过项的提交写在“最新结果”，当前分析仍为 `—`。

`untested_failure_note` 可注明未纳入最新定向的历史失败，避免将旧失败误判为新基线仍失败。
