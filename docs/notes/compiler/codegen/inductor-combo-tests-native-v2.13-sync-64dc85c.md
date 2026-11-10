# 2.13 ComboKernel 同步分支验证

使用 NPU_A5 的 `pta_213_case` 环境及本地 `sync/combo-kernel-master` 对应包。
torch_npu：`2.13.0+git64dc85c`，提交：`64dc85c99c1398a0bd2355b72cc2851d0688b964`。
NPU Python 修复叠加提交：`bb5e1d0a84`。
PyTorch：`2.13.0+cpu`，Triton：`3.6.0`。
测试快照：`c1ce068`，目录为 `v2.13.0/inductor_npu`。

状态：`complete`；116 行表格、105 个独立源方法，314 个参数化实例。方法结果：`{'PASS': 68, 'FAIL': 33, 'SKIP': 4}`。
已报告实例结果：`{'PASS': 274, 'FAIL': 35, 'SKIP': 5}`。

[用例表](inductor-combo-tests-native-v2.13-sync-64dc85c.csv)；[执行脚本说明](../../../../scripts/tests/combo_kernel_210/README-213.md)；[原始证据](../../../../artifacts/inductor-combo-tests-native-v2.13-sync-64dc85c-failure-fixes/)。

使用原生 pytest，无自定义 pytest 插件；设置选卡变量。foreach 已按 2.10 的 NPU 功能验证口径调整：取消本轮 27 个方法的主动 NPU skip，NPU 上不检查 kernel 数量及 foreach 专用生成代码；数值、原地修改、梯度断言保留。其他后端保留原融合检查。功能 PASS 不证明 FOREACH 分组融合生效。

表格中的异常摘要来自 JUnit 报告；尚未逐项定位的错误不能直接据此归因于 ComboKernel。PASS 项不保留历史失败分析。

本次仅复验原失败实例及取消旧 NPU skip 后的实例；未重跑历史 PASS。当前表为逐实例更新后的汇总，不能视为全部方法在最新修复上重新执行。每条报告的验证版本与来源保存在 `jobs.json` 的 `validation` 字段中，本轮运行统计：`[{'artifact': 'priority', 'test_commit': 'c1ce068', 'npu_fix_commit': '257e31bde7', 'method_results': {'PASS': 17, 'FAIL': 4}, 'instance_results': {'PASS': 17, 'FAIL': 4}}, {'artifact': 'remaining', 'test_commit': 'c1ce068', 'npu_fix_commit': '257e31bde7', 'method_results': {'FAIL': 35, 'PASS': 1}, 'instance_results': {'FAIL': 37, 'PASS': 1}}, {'artifact': 'cleanup', 'test_commit': 'c1ce068', 'npu_fix_commit': 'bb5e1d0a84', 'method_results': {'PASS': 6}, 'instance_results': {'PASS': 6}}]`。

当前为问题修复阶段：剩余 foreach 每个源方法最多执行 2 个代表实例，按 node ID 排序选择两端以覆盖不同算子。此前完成的方法保留原报告。表格中的“抽样 PASS”只证明所选实例通过，不代表整个参数化方法全部通过。原清单 464 个实例，本轮保留已执行结果并缩减后计划 314 个实例；不是完整参数化全量。实际抽样 node ID 见证据目录的 `foreach-sample-nodeids.txt`。

表格行顺序与 `inductor-combo-tests-liulm-v2.10.0-combokernel-dev.csv` 一致。嵌套函数随外层方法验证，不作为独立 pytest 方法重复计数；不沿用 2.10 的排除范围。
已复用相同包、相同测试文件及相同 node ID 的既有结果：`0` 个方法。

优先验收 2.10 基线 PASS 的 60 个源方法，当前结果：`{'PASS': 59, 'FAIL': 1}`。未完成、SKIP 和仅完成数值比较的方法不能提前作为完整 PASS。2.10 表来自适配后的 2.13 功能用例：部分 kernel 数量、CUDA DSL、profiler/grid 检查已移除；对照时须区分功能退化、测试适配差异与融合策略变化。当前分析和后续定时发布均优先标记这组方法。[重合项同步差异分析](inductor-combo-tests-v2.10-pass-v2.13-regressions.md)。

上述 PASS 中有 14 个方法为抽样 PASS，完整参数范围未全部执行。

执行前后安装的 Inductor Python 源码哈希一致：`True`。
