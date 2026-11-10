# 2.13 ComboKernel 同步分支验证

使用 NPU_A5 的 `pta_213_case` 环境及本地 `sync/combo-kernel-master` 对应包。
torch_npu：`2.13.0+git64dc85c`，提交：`64dc85c99c1398a0bd2355b72cc2851d0688b964`。
NPU Python 修复叠加提交：`5a5bda0ae36a30a865ea52f7cd1df2a5a2ea532c`。
PyTorch：`2.13.0+cpu`，Triton：`3.6.0`。
测试快照：`a889082`，目录为 `v2.13.0/inductor_npu`。

状态：`testing`；116 行表格、105 个独立源方法，464 个参数化实例。方法结果：`{'PASS': 20, 'SKIP': 5, 'FAIL': 32, 'RUNNING': 1, 'PENDING': 47}`。
已报告实例结果：`{'PASS': 212, 'SKIP': 6, 'FAIL': 34}`。

[用例表](inductor-combo-tests-native-v2.13-sync-64dc85c.csv)；[执行脚本说明](../../../../scripts/tests/combo_kernel_210/README-213.md)；[原始证据](../../../../artifacts/inductor-combo-tests-native-v2.13-sync-64dc85c-interface-fix/)。

使用原生 pytest，无自定义 pytest 插件；设置选卡变量。另设独立 TORCHINDUCTOR_CACHE_DIR：`/home/l30023782/PTA_213/combo-sync-64dc85c-interface-fix-20261110/cache-full-retry1`，避免 Python 修复后复用旧编译代码；未清理共享缓存。foreach 已按 2.10 的 NPU 功能验证口径调整：取消本轮 27 个方法的主动 NPU skip，NPU 上不检查 kernel 数量及 foreach 专用生成代码；数值、原地修改、梯度断言保留。其他后端保留原融合检查。功能 PASS 不证明 FOREACH 分组融合生效。

表格中的异常摘要来自 JUnit 报告；尚未逐项定位的错误不能直接据此归因于 ComboKernel。PASS 项不保留历史失败分析。

foreach 全量前置验证：`[{'name': 'test_single_list__foreach_abs', 'status': 'PASS'}, {'name': 'test_foreach_map_backward_unary_foreach_map_abs', 'status': 'PASS'}]`。两项均通过才恢复完整任务：`True`。前置实例不额外计入完整清单统计。

执行顺序已调整：卡 1、6 当时正在执行的 foreach 方法继续到结束，卡 2、3 先执行其他文件的 78 个源方法（88 个实例）；这些方法完成后，剩余 foreach 才在卡 1、2、3、6 执行。已完成结果保留，未重跑；各阶段使用独立新缓存。当前阶段：`non_foreach_first`，阶段状态：`testing`。

阶段 `non-foreach-first` 缓存：`/home/l30023782/PTA_213/combo-sync-64dc85c-interface-fix-20261110/cache-non-foreach-first`。

阶段 `foreach-last` 缓存：`/home/l30023782/PTA_213/combo-sync-64dc85c-interface-fix-20261110/cache-foreach-last`。

表格行顺序与 `inductor-combo-tests-liulm-v2.10.0-combokernel-dev.csv` 一致。嵌套函数随外层方法验证，不作为独立 pytest 方法重复计数；不沿用 2.10 的排除范围。
已复用相同包、相同测试文件及相同 node ID 的既有结果：`0` 个方法。

优先验收 2.10 基线 PASS 的 60 个源方法，当前结果：`{'PASS': 17, 'SKIP': 1, 'FAIL': 10, 'PENDING': 32}`。未完成、SKIP 和仅完成数值比较的方法不能提前作为完整 PASS。2.10 表来自适配后的 2.13 功能用例：部分 kernel 数量、CUDA DSL、profiler/grid 检查已移除；对照时须区分功能退化、测试适配差异与融合策略变化。当前分析和后续定时发布均优先标记这组方法。[重合项同步差异分析](inductor-combo-tests-v2.10-pass-v2.13-regressions.md)。
