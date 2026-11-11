# 2.13 ComboKernel 同步分支验证

2026-11-11 已基于 `sync/combo-kernel-master:d12230b556` 完成此前失败项复验。

| 范围 | PASS | FAIL | SKIP |
| --- | ---: | ---: | ---: |
| 本次 33 个失败方法 | 1 | 32 | 0 |
| 本次 35 个失败参数实例 | 1 | 34 | 0 |
| 保留历史结果后的 105 个方法 | 69 | 32 | 4 |
| 保留历史结果后的 314 个已选实例 | 275 | 34 | 5 |
| 2.10 基线 PASS 的 60 个重合方法 | 60 | 0 | 0 |

**本轮只重新执行失败实例，没有重跑历史 PASS、SKIP 或 foreach 全参数。**
cumsum/helper 本次通过，其余 59 个重合方法沿用历史 PASS；总计 69 个 PASS 中有 14 个抽样 PASS。
原清单 464 个泛化实例中有 150 个未选择，不记为 SKIP 或 PASS。

[用例 CSV](inductor-combo-tests-native-v2.13-sync-64dc85c.csv) 保留 116 行原顺序和五列，
嵌套函数随其外层方法验证。实例列保留实际复验版本，PASS 的当前分析为“—”。
当前最值得处理的是 optimize_mem、provenance 和真实统计/聚合，
详见 [最新问题分类、优先级与源码依据](inductor-combo-tests-v2.13-latest-failures.md)。

运行环境是 NPU_A5/Ascend950PR 的 `pta_213_case`；PyTorch `2.13.0+cpu`、Triton `3.6.0`。
基础 wheel 仍为 `torch_npu 2.13.0+git64dc85c`，C++ 未重新构建；
实际将 d12230b556 的全部 191 个 Inductor Python 文件覆盖到安装目录，并核对前后哈希一致。
测试源码为 `c1ce0688898b9ca1edebc2a4ff4dbffd8b287f28`，189 个测试 Python 文件已核对哈希。
本轮使用物理卡 3、6 和独立缓存，直接执行原生 `python -m pytest -vs`，没有自定义插件。

foreach 沿用 2.10 的 NPU 功能适配：保留数值、mutation 和梯度断言，
不检查 CUDA/foreach 专用融合数量和生成文本。抽样 PASS 不证明完整泛化范围通过，也不证明 FOREACH 分组融合。
4 个历史 SKIP 不作为功能缺失证据，本轮没有重试。

[执行脚本及命令](../../../../scripts/tests/combo_kernel_210/README-213.md)；
[本轮日志、JUnit 和哈希证据](../../../../artifacts/inductor-combo-tests-native-v2.13-sync-d12230b-failure-rerun/)；
[重合方法对照](../../../../artifacts/inductor-combo-tests-native-v2.13-sync-d12230b-failure-rerun/baseline-pass-comparison.json)。
逐实例实际版本保存在 `jobs.json` 的 `validation`，旧验证过程保存在历史 artifact 与 `rerun_history`。
