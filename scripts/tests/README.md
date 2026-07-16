# tests

本目录存放功能验证、实验验证或回归测试脚本。

建议区分：

- 单点验证脚本。
- 批量扫描脚本。
- 性能或行为对比脚本。

需要远程 NPU 环境验证的脚本，应在文档中标明运行前提和环境变量。

## 当前脚本

- `elementwise_dynamic_perf/`: 配套提供 elementwise 性能采集和宽表 CSV 对比脚本；支持 CUDA/NPU，按
  `EXECUTION=eager|static|dynamic` 分进程采集，再将相同场景的 execution 和设备结果横向合并。
