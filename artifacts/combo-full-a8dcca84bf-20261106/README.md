# 当前 Combo 修复分支的全量任务

PR：[liujm2001/pytorch !7](https://gitcode.com/liujm2001/pytorch/merge_requests/7)，目标 `v2.10.0-combokernel-dev`。
安装源码为 `a8dcca84bf52e8a3e0c46aa46c2a4f663875f10e`，五个修复提交，F4 仍为单个提交 `53f8af09f1`。
源码和 PR 的核验信息见 [pr.json](pr.json)、[启动记录](launch.json)。

任务于 2026-11-06 23:57:45（北京时间）后台启动，范围为原核定的 62 个方法、409 个参数实例，
使用获授权的物理卡 4、5、6 并行；不加载函数修复或旧 F3 overlay。
这份记录保存启动配置，不将进行中的用例计为通过。
精确实例清单见 [expected_nodes.json](expected_nodes.json)，安装源码约束见
[expected_source_sha256.json](expected_source_sha256.json)，修复提交映射见
[validation_context.json](validation_context.json)。

## 全量执行结果

2026-11-07 01:31:59（北京时间）完成，**60/62 个方法、407/409 个实例通过**，2 个实例失败，0 跳过、0 执行异常。
11 项安装源码运行前后校验一致；六个修复问题涉及的 8 个方法、10 个实例均通过。
结果见 [result.json](result.json)，原始报告与完整性检查见
[functional_results.json](../combo-liulm-0fd5a10-20261106/retests/full_a8dcca84bf_20261106/functional_results.json)和
[integrity_checks.json](../combo-liulm-0fd5a10-20261106/retests/full_a8dcca84bf_20261106/integrity_checks.json)。

- F6：NPU eager 的 contiguous 不支持 channels_last memory format，失败发生在 torch.compile 之前。
- F7：provenance 的 cppCodeToPost 映射为空（预期 1 项），Inductor 追溯断言失败，AOTI 未执行。

定时任务已于 2026-11-07 01:35:08 自动发布结果，提交 `a25bfbf3a9`，随后停止。
以下保留本轮启动及定时配置。

## 自动执行与定时收尾

使用已有 run_validation.py / run_combo_selected.py 执行，prepare_validation.py 创建独立 phase，
在运行前校验安装源码，并对实际收集的节点集合逐项比对核定的 409 个节点。
沿用同一适配测试树，保持输入、stride、dtype、数值/梯度断言和精度阈值。
脚本没有安装或切换共享 Python 包，也不停止其他任务。

本机用户 systemd 定时器 `combo-full-a8dcca84bf-20261106.timer` 已创建：
首次检查为 2026-11-07 01:03（北京时间，调度精度 1 分钟），之后每 30 分钟一次。
每次以 `publish_retest.py --once` 读取一次任务状态；若仍在运行则退出，不在进程内循环等待。
结束后统一下载、脱敏、校验实例清单及运行前后的源码，刷新 Markdown/CSV，
通过临时 Git index 仅提交本轮文档和证据并推送，成功后停止自己的定时器。
执行异常、跳过和未完成实例单独记录；新失败依据本轮日志标为待分析，避免套用历史原因。
控制器或源码完整性检查失败时保存证据，状态记为 `validation_error`，保留旧表并停止定时器。

完成后的证据写入原结果目录的 `retests/full_a8dcca84bf_20261106/`，并更新
[总表](../../docs/notes/compiler/codegen/inductor-combo-tests-liulm-v2.10.0-combokernel-dev.md)。
脚本与重用方法见 [执行说明](../../scripts/tests/combo_kernel_210/README.md)。
本地最终状态写入 `tmp/combo-full-a8dcca84bf-20261106/publisher_status.json`（工作区根目录下）。
