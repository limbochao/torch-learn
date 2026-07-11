# tests

本目录存放功能验证、实验验证或回归测试脚本。

建议区分：

- 单点验证脚本。
- 批量扫描脚本。
- 性能或行为对比脚本。

需要远程 NPU 环境验证的脚本，应在文档中标明运行前提和环境变量。

## 当前脚本

- `pointwise_op_cost_cases.py`: 分类验证 elementwise Aten 算子，并通过 NPU profiler 对比 generated pointwise
  kernel 的执行时间。
