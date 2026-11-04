---
title: CICD 问题修复
---

# CICD 问题修复

记录 CICD 模型问题的修复方案、代码改动、修复前后 kernel 和验证结果。
模型的性能差距与待优化项见 [CICD 性能分析](../performance/index.md)。

- [NPU Index codegen：seed constexpr 常量遗漏修复与性能验证](index-triton-seed-constexpr-fix.md)：
  修复 Index checker 失败引起的 no-linear 回退；Electra Train 整步实测减少 14.983 ms（9.55%）。
  对应 [ElectraForCausalLM 模型分析](../performance/models/ElectraForCausalLM.md#attention-constexpr-validation)。
