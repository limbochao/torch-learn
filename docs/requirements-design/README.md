# 需求设计

Pages 入口见 [index.md](index.md)。

## 目录分类

- `symbolic-group/`：symbolic grouped autotune 相关设计和实现说明。
- `dynamic-shape/`：通用 dynamic shape 测试和能力设计。
- `cooperative-reduction/`：NPU cooperative reduction 需求分析、方案草案及流程图。
- `split-cooperative-reduction-comparison/`：NPU split reduction 需求规格、当前实现方案，以及相同 eager forward 的 GPU/NPU split 与 cooperative profiler 对照。

Split reduction 文档：

- [NPU Split Reduction 当前实现方案](split-cooperative-reduction-comparison/npu-split-reduction-design.md)（选择器已实现，性能阈值待验收）。
- [NPU Split Reduction 适配需求规格（PyTorch 2.13）](split-cooperative-reduction-comparison/npu-split-reduction-requirements.md)（草案，待评审）。
- [数据测试脚本说明](../../scripts/tests/npu_split_calibration/README.md)。

Cooperative reduction 文档：

- [需求分析](cooperative-reduction/requirements.md)。
- [初步方案](cooperative-reduction/design.md)（草案，待评审）。
- [Split/cooperative 对比结果](split-cooperative-reduction-comparison/results.md)及[测试说明](../../scripts/tests/reduction_strategy_pk/README.md)。
