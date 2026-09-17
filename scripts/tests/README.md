# tests

本目录存放功能验证、实验验证或回归测试脚本。

建议区分：

- 单点验证脚本。
- 批量扫描脚本。
- 性能或行为对比脚本。

需要远程 NPU 环境验证的脚本，应在文档中标明运行前提和环境变量。

## 当前脚本

- `elementwise_dynamic_perf/`: 配套提供 elementwise 性能采集和宽表 CSV 对比脚本；支持 CUDA/NPU，按
  `EXECUTION=eager|static|dynamic|group` 分进程采集，再将相同场景的 execution 和设备结果横向合并；
  `group` 为启用 Ascend symbolic group autotune 的 dynamic 编译路径。
- `dynamic_ub_tiling/`: 从 `group_kernel/all_case/` 挑选并改造的动态 UB tiling 性能 case，覆盖
  pointwise、broadcast、transcendental、reduction 和多输入 live-value 场景；使用
  `scripts/tools/compile_mode_perf.py --dynamic-ub [WATERMARK]` 开启并设置 UB 水位；传入该参数时会在同一
  次运行中并列采集普通 dynamic 和 `dynamic_ub`，结果中的 `dynamic_ub_*` 列用于直接比较；不传该参数则
  只采集普通 dynamic 并使用 Legacy 路径。
- `run_model_profile_rounds.sh`: 重复执行完整模型测试命令，从命令和日志定位 compile、profile 目录，
  汇总 kernel 耗时并保存每轮产物。测试命令在脚本内配置，用法：`bash run_model_profile_rounds.sh [rounds]`。
