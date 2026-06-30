---
title: 问题定位日志
---

# 问题定位日志

本目录记录 PyTorch 相关问题的定位过程，重点保留可复现信息和结论依据。

## 建议记录内容

- 问题背景和现象。
- 复现方式和环境信息。
- 关键日志、报错栈或中间产物。
- 代码分析路径和证据。
- 结论、修复方案、验证结果。
- 后续风险或待确认问题。

新增日志时可以参考 `templates/issue-log.md`。

## 日志列表

- [RMSNorm SIMD 多 reduction 轴 codegen 精度问题定位](rms-norm-simd-multi-reduction-codegen.md)
- [RMSNorm SIMD 多 reduction 轴 codegen 修改详解](rms-norm-simd-multi-reduction-codegen-change-detail.md)
