# 2.13 Combo/foreach 执行说明

结果和当前问题分析只更新 [2.13 固定文档](../../../docs/notes/compiler/codegen/inductor-combo-tests-v2.13.md)。
用例适配进展统一更新 [2.13 进展表](../../../docs/notes/compiler/codegen/inductor-combo-tests-v2.13.csv)，
同表保留用例类型、2.10 源码存在性和直接运行兼容性，结果与分析以实际 2.13 报告为准。
运行批次使用独立证据目录，表格和文档文件名不随分支或提交变化。

## 环境与执行范围

使用 NPU_A5 的 `pta_213_case` 环境，在用户指定的 `PTA_213` 工作目录中执行。
测试源为 `Community-Use-Case-Testing/v2.13.0/inductor_npu`。
执行前确认实际安装代码；采用 Python overlay 时记录所依据的提交及覆盖文件哈希，
基础 wheel 与 C++ 版本不因 overlay 自动改变。具体当前版本见固定文档。

`combo_213_all_selected.json` 是源方法清单；`combo_213_table_rows.json` 维护原表行与外层方法的对应关系。
嵌套函数随外层方法验证，不作为独立 pytest 方法计数。
后续只关注最新逐实例报告中的 FAIL、SKIP，执行 ERROR 也按失败处理，不进行全量测试。
已 PASS 的方法和参数实例保留历史结果；不直接把完整源方法清单交给 runner 执行。
硬件或平台协议 SKIP 保留明确原因，不能计入 PASS；软件 SKIP 继续定位和适配。

## 生成定向清单

在 `torch-learn` 仓库根目录，使用进展表对应的最新完整结果快照生成清单：

```bash
python scripts/tests/combo_kernel_210/select_213_table.py \
  --source /path/to/cases/v2.13.0/inductor_npu \
  --reference artifacts/<latest-results> \
  --manifest /path/to/nonpassing-manifest.json --layout /path/to/table-rows.json
```

脚本核对进展表与快照的方法状态，输出完整表格行映射，并仅为 FAIL、SKIP、ERROR 实例设置 `only_nodes`。
一个方法同时有 PASS 和 FAIL 参数时，只选失败参数。生成后可以按本次修复涉及的方法进一步缩小清单。

## 自动执行

在远端激活 conda 并进入准备好的工作目录后执行：

```bash
conda activate pta_213_case
python /path/to/torch-learn/scripts/tests/combo_kernel_210/run_plain_pytest.py \
  --test-root /path/to/cases/v2.13.0/inductor_npu \
  --manifest /path/to/nonpassing-manifest.json --layout /path/to/table-rows.json \
  --output /path/to/new-results --cache-root /path/to/new-cache \
  --test-commit <actual-test-commit> --npu-fix-commit <actual-overlay-commit> \
  --devices 1,2,3 --foreach-functional
```

不使用 overlay 时省略 `--npu-fix-commit`。选卡必须遵循本轮授权，示例不授予卡使用权限。
`--foreach-functional` 只记录所用测试的功能适配口径，不给测试注入 monkeypatch。
子进程直接运行 `python -m pytest -vs`，无自定义 pytest 插件。

`--collect-only` 只收集；复测通过 `only_nodes` 精确选取，不使用代表参数抽样替代失败实例。
输出目录必须不存在；默认每方法超时 3600 秒。
设备异常可以换到获授权的卡重试，功能失败保留真实断言和原报告。

多个进程共用获授权卡时，各进程使用独立的 `TORCHINDUCTOR_CACHE_DIR`、`TRITON_CACHE_DIR` 和输出目录。
单个失败或跳过实例也可在测试目录直接运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=3 TORCHINDUCTOR_CACHE_DIR=/path/to/new-cache \
python -m pytest -vs \
  test_combo_kernels.py::ComboKernelMetadataTests::test_combo_inductor_meta_optimize_mem_false_in_training_forward
```

## 合并定向复测并发布

`merge_failed_results.py` 按实例更新参考快照，拒绝未完成报告、重复实例和历史 PASS 覆盖。
合并前验证 PyTorch/Triton 基础版本不变、各复测阶段安装源码一致；
torch_npu 可以更新为修复后的包，每条报告保留实际包提交、测试提交和 Python overlay 提交。
未重跑的实例保留原结果与验证来源，不能称为最新代码全量验证。

```bash
python scripts/tests/combo_kernel_210/merge_failed_results.py \
  --reference artifacts/<reference-run> --rerun artifacts/<phase-1> artifacts/<phase-2> \
  --output artifacts/<new-merged-run>

python scripts/tests/combo_kernel_210/publish_plain_pytest.py \
  --local-artifact artifacts/<new-merged-run> --name <new-merged-run>
```

发布器只更新 `inductor-combo-tests-v2.13.csv`，保留表格行顺序和源码对照列；
定向结果必须先合并为完整快照，避免覆盖未重跑用例的进展。
Markdown 始终更新 `inductor-combo-tests-v2.13.md` 的当前结果章节。
对应问题分析同步在该文档维护，PASS 不保留旧失败原因，正文不累积旧轮次结果。
`--push` 可提交推送，`--stop-timer <unit>.timer` 可在收尾时停止本地定时检查。

[2.13 PASS 清单](combo_native_213_passed_nodeids.txt) 仅保留为已有通过范围的查询依据。
2.10 原生清单及结果按 [2.10 说明](README-native-210.md) 单独维护。
