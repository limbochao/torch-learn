# 2.13 ComboKernel 同步分支验证

使用 NPU_A5 的 `pta_213_case` 环境及本地 `sync/combo-kernel-master` 对应包。
torch_npu：`2.13.0+git64dc85c`，提交：`64dc85c99c1398a0bd2355b72cc2851d0688b964`。
PyTorch：`2.13.0+cpu`，Triton：`3.6.0`。
测试快照：`a2931d1`，目录为 `v2.13.0/inductor_npu`。

状态：`testing`；62 个源方法，415 个参数化实例。方法结果：`{'PENDING': 52, 'SKIP': 8, 'RUNNING': 2}`。
实例结果：`{}`。

[用例表](inductor-combo-tests-native-v2.13-sync-64dc85c.csv)；[执行脚本说明](../../../../scripts/tests/combo_kernel_210/README-213.md)；[原始证据](../../../../artifacts/inductor-combo-tests-native-v2.13-sync-64dc85c/)。

使用原生 pytest，无自定义 pytest 插件；仅设置选卡变量。保留指定目录已有的 NPU 适配和全部原始断言，没有沿用 2.10 foreach 放宽数量检查的修改。

表格中的异常摘要来自 JUnit 报告；尚未逐项定位的错误不能直接据此归因于 ComboKernel。PASS 项不保留历史失败分析。
