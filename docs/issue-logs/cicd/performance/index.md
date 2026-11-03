---
title: CICD 性能分析
---

# CICD 性能分析

提升比采用 `(Eager 单步耗时 - Inductor 单步耗时) / Eager 单步耗时`。
优化目标为 NPU 的编译提升比持平或超过 GPU。
逐项比较同一语义计算的局部提升比，不用 NPU/GPU Inductor 绝对耗时差给问题排序，
也不用整网 Eager 比例缩放单个 kernel。

- [全量分析与共性问题：40 个模型、78 个场景](full-survey-20261102.md)：按模型分别记录 Infer / Train，
  具体列出 kernel、代码边界、GPU 做法、优化模块、耗时占比与整网贡献预算。
- [ElectraForCausalLM](models/ElectraForCausalLM.md)、[LayoutLMForMaskedLM](models/LayoutLMForMaskedLM.md)：
  保留首层 attention 和 FMA 已定位点，补充符合新口径的具体问题。
- [Train 首层 attention 的 constexpr/checker 根因](electra-layoutlm-train-first-attention-20261031.md)：
  历史源码与 checker 定位证据。
- [两模型绝对耗时差历史分析](electra-layoutlm-train-gpu-inductor-gap-20261102.md)：
  已标记口径被替代，MM/AddMM 70.049 ms 不再作为编译优化依据。

- [MobileBERT Train 参数梯度归约深挖](mobilebert-parameter-reduction-20261103.md)：
  IR 分段归约策略与 tiling 候选漏搜，附指定分支单 kernel 实测、GPU 对照和收益估算。
- [MobileBERT 其余归约逐 kernel 分析](mobilebert-remaining-reductions-20261103.md)：
  `_9`、`_7`、`_13` 新增确认，`_20` 保留 GPU 收益核验边界，`_0` 分段无收益；含五组实测和整网外推。
