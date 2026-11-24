# 需求设计

Pages 入口见 [index.md](index.md)。

## 目录分类

- `symbolic-group/`：symbolic grouped autotune 相关设计和实现说明。
- `dynamic-shape/`：通用 dynamic shape 测试和能力设计。
- `cooperative-reduction/`：NPU cooperative reduction 需求分析、方案草案及流程图。
- `split-cooperative-reduction-comparison/`：相同 eager forward 的 GPU/NPU split 与 cooperative profiler 对照。

Cooperative reduction 文档：

- [需求分析](cooperative-reduction/requirements.md)。
- [初步方案](cooperative-reduction/design.md)（草案，待评审）。
- [Split/cooperative 对比结果](split-cooperative-reduction-comparison/results.md)及[测试说明](../../scripts/tests/reduction_strategy_pk/README.md)。
