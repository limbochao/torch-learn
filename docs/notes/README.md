# 学习笔记

本目录存放 PyTorch 相关学习笔记、源码阅读记录和机制分析。

## 建议分类

- `frontend/`: Dynamo、FX、export 等前端机制。
- `compiler/codegen/`: Inductor codegen 概念和 kernel 分类。
- `compiler/symbolic-group/`: symbolic grouped autotune 线上机制。
- `compiler/debugging/`: runnable 适配和通用调试方法。
- `runtime/`: dispatcher、executor、device runtime 等运行时内容。
- `ops/`: 算子语义、shape、dtype、layout 等专题记录。

新增笔记时可以参考仓库中的 [学习笔记模板](../../templates/note.md)。
