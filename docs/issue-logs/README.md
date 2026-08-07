# 问题定位日志

本目录记录 PyTorch 相关问题的定位过程，重点保留可复现信息和结论依据。

## 目录分类

- `codegen/`：generated DSL、lowering、reduction 和 codegen 问题。
- `dynamic-shape/`：static/dynamic graph 与 kernel 行为差异。
- `symbolic-group/`：symbolic group 分组、config 和性能现象。

## 建议记录内容

- 问题背景和现象。
- 复现方式和环境信息。
- 关键日志、报错栈或中间产物。
- 代码分析路径和证据。
- 结论、修复方案、验证结果。
- 后续风险或待确认问题。

新增日志时可以参考仓库中的 [issue log 模板](../../templates/issue-log.md)。
