# 2.13 Combo/foreach 执行说明

结果和当前问题分析只更新 [2.13 固定文档](../../../docs/notes/compiler/codegen/inductor-combo-tests-v2.13.md)。
运行批次使用独立证据目录，文档文件名不随分支或提交变化。

## 环境与执行范围

使用 NPU_A5 的 `pta_213_case` 环境，在用户指定的 `PTA_213` 工作目录中执行。
测试源为 `Community-Use-Case-Testing/v2.13.0/inductor_npu`。
执行前确认实际安装代码；采用 Python overlay 时记录所依据的提交及覆盖文件哈希，
基础 wheel 与 C++ 版本不因 overlay 自动改变。具体当前版本见固定文档。

`combo_213_all_selected.json` 是源方法清单；`combo_213_table_rows.json` 维护原表行与外层方法的对应关系。
嵌套函数随外层方法验证，不作为独立 pytest 方法计数。
问题修复阶段优先从最新 `jobs.json` 中选择报告状态为 FAIL 的 node ID，
在清单项的 `only_nodes` 中写入准确实例，不重跑同方法已经 PASS 的参数。

## 自动执行

在远端激活 conda 并进入准备好的工作目录后执行：

```bash
conda activate pta_213_case
ASCEND_RT_VISIBLE_DEVICES=3 python /path/to/torch-learn/scripts/tests/combo_kernel_210/run_plain_pytest.py \
  --test-root /path/to/cases/v2.13.0/inductor_npu \
  --manifest /path/to/failed-only-manifest.json \
  --output /path/to/new-results --cache-root /path/to/new-cache \
  --test-commit <actual-test-commit> --npu-fix-commit <actual-overlay-commit> \
  --devices 1,2,3,6 --foreach-functional
```

不使用 overlay 时省略 `--npu-fix-commit`。选卡必须遵循本轮授权，示例不授予卡使用权限。
`--foreach-functional` 只记录所用测试的功能适配口径，不给测试注入 monkeypatch。
子进程直接运行 `python -m pytest -vs`，无自定义 pytest 插件。

`--collect-only` 只收集；`--max-instances-per-method 2` 每个方法最多选两个代表参数。
抽样 PASS 不代表完整泛化范围通过。输出目录必须不存在；默认每方法超时 3600 秒。
设备异常可以换到获授权的卡重试，功能失败保留真实断言和原报告。

单例可在测试目录直接运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=3 TORCHINDUCTOR_CACHE_DIR=/path/to/new-cache \
python -m pytest -vs \
  test_combo_kernels.py::ComboKernelMetadataTests::test_combo_inductor_meta_optimize_mem_false_in_training_forward
```

## 合并失败项复测并发布

`merge_failed_results.py` 按实例更新参考快照，拒绝未完成报告、重复实例和历史 PASS 覆盖。
合并前验证基础环境与各复测阶段的安装源码一致；每条报告保留实际验证提交。
历史 PASS 未重跑时，必须在当前文档说明，不能称为最新代码全量验证。

```bash
python scripts/tests/combo_kernel_210/merge_failed_results.py \
  --reference artifacts/<reference-run> --rerun artifacts/<phase-1> artifacts/<phase-2> \
  --output artifacts/<new-merged-run>

python scripts/tests/combo_kernel_210/publish_plain_pytest.py \
  --local-artifact artifacts/<new-merged-run> --name <new-merged-run> \
  --table-name inductor-combo-tests-native-v2.13-sync-64dc85c
```

CSV 保持固定名称和原有行顺序；Markdown 始终更新 `inductor-combo-tests-v2.13.md` 的当前结果章节。
对应问题分析同步在该文档维护，PASS 不保留旧失败原因，正文不累积旧轮次结果。
`--push` 可提交推送，`--stop-timer <unit>.timer` 可在收尾时停止本地定时检查。
