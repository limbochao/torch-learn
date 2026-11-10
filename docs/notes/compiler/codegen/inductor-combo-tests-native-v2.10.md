# 2.10 原生 Combo/foreach 用例执行结果

2026-11-10：本轮执行中，最终结果待自动汇总。

本次已收集 54 个源测试方法、402 个参数化实例，使用 NPU_A5 Docker 的物理卡 1、3、5 并行执行。
安装包为 `torch_npu 2.10.0.post7.dev20261110`（源码 `2e109cfd759b`），
`torch 2.10.0+cpu`，实际导入 `triton 3.6.0`。

测试使用 `Community-Use-Case-Testing/v2.10.0/test_inductor`，测试提交 `f998b3d`。
仅为 Combo 文件补充 NPU 初始化，测试插件处理 NPU 设备选择及结果记录；
保留原版输入、参数化、配置、数值及 kernel 数量/生成代码断言。
本轮没有叠加生产代码修复。

用户预期为 51 个方法 PASS；实际通过数将在全部执行结束后记录，跳过和执行异常不计为 PASS。

[54 个方法的 CSV 表](inductor-combo-tests-native-v2.10.csv)。

本地用户 systemd 定时器 `combo-native210-daily20261110-run.timer` 首次 10 分钟后检查，
之后每 15 分钟检查。完成后自动核对逐实例报告和安装源码 SHA256，
将本页与 CSV 更新为实际结果，归档日志并推送 torch-learn，随后停止定时器。
