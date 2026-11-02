---
title: CICD 性能分析
---

# CICD 性能分析

记录 CICD 模型在 NPU/GPU 上的 Eager、Inductor 性能对比。
每篇分析明确具体 kernel、劣化幅度、比较边界、责任模块和验证范围。

提升比统一采用 `(Eager 单步耗时 - Inductor 单步耗时) / Eager 单步耗时`，
目标是 NPU 提升比持平或超过 GPU。问题排序直接比较 NPU Inductor 与 GPU Inductor 的对应计算，
Eager 用于整步提升比目标计算，不作为性能问题的唯一筛选依据。

| 日期 | 场景 | 当前结论 |
| --- | --- | --- |
| 2026-11-02 | [两个 Train 对 GPU Inductor 的差距][gpu-gap] | 分模型汇总首层、FMA、后续 attention 和矩阵乘的优化模块 |
| 2026-10-31 | [Electra / LayoutLM Train 首层 attention][attention] | NPU Inductor 漏填 seed 常量，触发慢路径回退 |

[gpu-gap]: electra-layoutlm-train-gpu-inductor-gap-20261102.md
[attention]: electra-layoutlm-train-first-attention-20261031.md
