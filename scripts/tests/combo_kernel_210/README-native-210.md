# 2.10 目录原生用例筛选

本轮直接使用 `Community-Use-Case-Testing/v2.10.0/test_inductor` 中的测试，
范围仍以原先 116 行 Combo/foreach 用例清单为准，不扩展到整个 test_inductor 目录。
这里的“原生”指用户指定目录中的现有 2.10 版本，保留该目录已有的 NPU 迁移代码。

筛选结果为 **54 个源测试方法**。本轮已在 NPU_A5 启动收集和执行，最终状态见用例表。
旧适配版的 409 个实例和 PASS/FAIL 结果不能作为本轮结果。

| 测试文件 | 源测试方法数 |
| --- | ---: |
| test_torchinductor.py | 4 |
| test_combo_kernels.py | 21 |
| test_foreach.py | 27 |
| test_unbacked_symints.py | 1 |
| test_provenance_tracing.py | 1 |
| 合计 | 54 |

- [用例表](../../../docs/notes/compiler/codegen/inductor-combo-tests-native-v2.10.csv)：54 行，保留原清单顺序，
  记录文件名、函数名、存在性/实现差异、原生执行说明及本轮结果。
- [执行选择清单](combo_native_210_selected.json)：54 个源方法，包含源类名，供后续收集时匹配参数化实例。
- [筛选审计](combo_native_210_selection.json)：原 116 行逐行分类、来源提交和文件 SHA256。
  54 行入选、11 行为嵌套函数（随外层执行）、51 行在指定文件中没有对应独立方法。

表格的实现差异说明沿用原版本对照表，描述 2.13 相对 2.10 的差异；
本轮执行以指定目录的实际 2.10 实现为准，不补齐这些差异。

## 与此前 2.13 适配版的范围变化

此前 62 个方法中有以下 8 个在 2.10 目录没有定义，因此不再搬运或适配运行：

| 测试文件 | 函数名 |
| --- | --- |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_poi |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_per |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_red_per |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_red |
| test_combo_kernels.py | test_combo_kernel_per_config_subkernel_block_size |
| test_combo_kernels.py | test_combo_kernel_max_num_nodes |
| test_combo_kernels.py | test_combo_kernel_dynamic_shapes_grid_changes |
| test_combo_kernels.py | test_combo_inductor_meta_no_kernel_num_gb_without_profile |

`test_mean` 的 2.10 方法没有主动开启 Combo，原样运行只能反映其原始配置下的行为。
`test_adaptive_avg_pool2d1` 和 `test_rand_like_deterministic` 保留原版 Combo=False/True 参数化。
首轮严格执行保留 Combo/foreach 的 kernel 数量和生成代码检查。
后续按用户要求，foreach 增加了下述 NPU 功能验证口径；Combo 文件保持首轮断言。

## foreach 功能复测（2026-11-10）

测试提交 `3d9ca53` 在 2.10 文件的 27 个入选 foreach 方法中，
以 `check_kernel_count = GPU_TYPE != "npu"` 控制原数量断言，其他后端继续检查数量。
NPU 保留数值、突变等功能检查；没有修改安装包、开启全局 Combo 或补 FOREACH 后端能力。
两个 backward 方法同时对齐原 2.13 适配版：参考计算使用当前算子的 `original_op`，
并用 `assertTrue` 检查梯度 `allclose` 的返回值。

本轮只选择原清单中 `file == "test_foreach.py"` 的 27 个方法，
并使用首轮 collection 中同文件的 373 个 node ID 作为 `expected_nodes.json`，防止漏跑或多跑。
采用独立 phase，保留修改前 `test_foreach.py` 作为 `original_cases`，其余初始化和运行脚本沿用首轮。

```bash
python run_validation.py --root "$PWD" \
    --test-root Community-Use-Case-Testing/v2.10.0/test_inductor \
    --devices 1,3,4,5 --expected-instances 373 --detach
```

`validation_context.json` 中设置 `validation_scope: native_foreach_functional`。
同一个 publisher 会按此标记只刷新 foreach 的 27 行，其余结果与历史严格执行证据保留。
通过项的当前分析为 `—`，结果明确标为功能复测；功能 PASS 不表示 foreach 已融合。

```bash
python scripts/tests/combo_kernel_210/publish_native_210.py \
    --remote-phase /home/l30023782/PTA_210/combo-native210-foreach-functional-20261110-run \
    --raw-root /home/qc/Repos/Ascend/tmp/combo-native210-foreach-functional-20261110-results \
    --artifact combo-native210-foreach-functional-20261110 --push \
    --stop-timer combo-native210-foreach-functional-20261110.timer
```

本地开发机用户 systemd timer 首次 10 分钟后检查，此后每 15 分钟检查；
完成后自动下载、核对源码与实例完整性、更新表格、提交推送并停止 timer。

本轮运行中按用户要求释放了物理卡 4，后续仅使用 1、3、5。
`resume_migrated_validation.py` 使用 phase 中的 `migration_plan.json` 接管未停止的 pytest：
等待原进程结束后，核对完整阶段报告和 JUnit XML，再继续该卡的新队列。
被中断方法的部分报告放入 `evidence/interrupted/`，方法在其他卡重新执行；已完成方法不重跑。
最终仍按原来的 373 个 node ID 和安装源码 SHA256 完整性检查收尾，publisher 不变。

## 重新生成

在 torch-learn 仓库根目录运行：

```bash
python scripts/tests/combo_kernel_210/select_native_210.py
```

可用 `--source`、`--table`、`--output`、`--manifest`、`--audit` 覆盖路径。
脚本只用 AST 解析源码，不导入 torch/test 模块，不改测试文件，不启动 NPU 测试。
缺失同名方法不推断为“不支持该能力”，只记录不在本轮原生范围。

`select_native_210.py` 会重新生成待执行表；本轮执行结果已发布后不要覆盖该表。

## 执行及自动收尾

首轮为 `test_combo_kernels.py` 补充与 foreach 相同的 NPU 初始化，提交 `f998b3d`，
推送到测试仓库 `origin/ut/210_combo`。没有删除或放宽原始测试断言。

`native_validation_plugin.py` 在 `transfer_to_npu` 之后恢复 NPU 的 Triton 可用性探测。
社区 `is_gpu()` 也不识别 NPU 设备测试基类；插件确认实际 NPU/Triton 可用且测试设备选择包含 NPU 后，
开启测试层的 `RUN_GPU`，使 `CommonTemplate` 的四个目标方法生成 NPU 实例。
插件保留实际 skip，不伪造缺失工具符号，不叠加生产代码修复。

准备独立 phase：从上述测试提交导出完整 `v2.10.0/test_inductor` 目录，复制
`run_validation.py`、`run_combo_selected.py`、`export_combo_results.py`、`capture_combo_evidence.py`；
将 `native_validation_plugin.py` 复制为 `combo_validation_plugin.py`，
将 `combo_native_210_selected.json` 复制为 `combo_npu_a5_selected.json`。
同时保存 `validation_context.json` 和安装源码的 `expected_source_sha256.json`。

在 NPU_A5 Docker 工作目录中的该 phase 执行：

```bash
python run_validation.py --root "$PWD" \
    --test-root Community-Use-Case-Testing/v2.10.0/test_inductor --devices 1,3,5 --detach
```

入口先收集，要求准确覆盖 54 个源方法，再按方法隔离进程并行执行；完成后核对逐实例
setup/call/teardown 报告以及安装源码 SHA256。参数化实例数量以本轮实际收集为准。

本地的 `publish_native_210.py` 每次只检查一次远端状态。完成后下载报告和日志、脱敏归档，
刷新独立的原生用例 CSV/Markdown，不覆盖旧 2.13 适配版结果。
带 `--push` 时，仅提交本轮表格和证据并推送 `origin/main`，保留其他工作区及暂存修改。

```bash
python scripts/tests/combo_kernel_210/publish_native_210.py \
    --remote-phase /home/l30023782/PTA_210/combo-native210-daily20261110-run \
    --raw-root /home/qc/Repos/Ascend/tmp/combo-native210-daily20261110-run-results --push
```

本次已在本地开发机创建用户 systemd 定时器 `combo-native210-daily20261110-run.timer`，
首次 10 分钟后检查，此后每 15 分钟检查，完成或验证异常后停止。
状态写入 raw-root 的 `publisher_status.json`，无需持续查看 pytest 日志。
