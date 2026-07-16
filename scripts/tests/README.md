# tests

本目录存放功能验证、实验验证或回归测试脚本。

建议区分：

- 单点验证脚本。
- 批量扫描脚本。
- 性能或行为对比脚本。

需要远程 NPU 环境验证的脚本，应在文档中标明运行前提和环境变量。

## 当前脚本

- `pointwise_op_cost_cases.py`: 按 P0/P1 和 memory/compute bound 分类，对比 static 与指定维度符号化的
  generated pointwise kernel，并增加 eager 基线；支持 CUDA/NPU，通过 profiler 文件解析一次完整 case 调用的
  device-side kernel 总开销。运行时必须设置 `RUN_ID`；同 ID 结果自动去重汇总到
  `<PROFILE_ROOT>/<RUN_ID>/summary.csv`，汇总仅使用 `execution=eager|static|dynamic` 区分执行路径；共享
  `PROFILE_ROOT` 时支持跨设备追加。
