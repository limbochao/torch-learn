---
title: 问题定位日志
---

# 问题定位日志

<section class="tl-note-band" markdown="1">
本分区记录 PyTorch、torch_npu 和 Inductor 相关问题的定位过程。重点不是只保存结论，而是保留复现方式、证据链、关键代码路径和验证结果。
</section>

<div class="tl-section-heading" markdown="1">
## 日志列表

按问题域归档，优先展示能复用排查方法的记录。
</div>

<div class="tl-card-grid tl-card-grid-two" markdown="1">
<article class="tl-feature tl-card-accent-rose" markdown="1">
<div class="tl-meta">
  <span class="tl-badge">Codegen</span>
  <span class="tl-badge">SIMD</span>
  <span class="tl-badge">Accuracy</span>
</div>

### [RMSNorm SIMD 多 reduction 轴 codegen 问题复盘](codegen/rms-norm-simd-multi-reduction-codegen.html)

记录 RMSNorm weight grad 在 `torch.compile` 后出现 NPU Inductor 精度错误的定位过程，包含错误 DSL、axis flatten、mask/value 坐标和 store 位置分析。
</article>

<article class="tl-feature tl-card-accent-blue" markdown="1">
<div class="tl-meta">
  <span class="tl-badge">Codegen</span>
  <span class="tl-badge">Dynamic Shape</span>
  <span class="tl-badge">DLRM</span>
</div>

### [DLRM dynamic 与 static codegen 对比分析](dynamic-shape/dlrm-dynamic-vs-static-codegen-analysis.html)

对比 `dynamic=True` 与 `dynamic=False` 下的 FX graph 和 generated code，解释 dynamic shape、kernel type、
SIMT 切分及后续 concat lowering 的差异。
</article>

<article class="tl-feature tl-card-accent-amber" markdown="1">
<div class="tl-meta">
  <span class="tl-badge">Symbolic Group</span>
  <span class="tl-badge">Performance</span>
  <span class="tl-badge">Observation</span>
</div>

### [relu_40 / relu_42 性能差异记录](symbolic-group/inductor-symbolic-group-relu-40-42-performance.html)

仅保存非线上代码产物中的性能现象和 kernel 信息，不把待验证推测写成根因。
</article>
</div>

<div class="tl-section-heading" markdown="1">
## 记录结构

新增日志建议尽量覆盖这些信息，便于后续复现、审阅和复用。
</div>

<ul class="tl-index-list">
  <li>
    <div><strong>问题背景</strong></div>
    <div>
      <p>说明触发场景、用户可见现象、影响范围，以及为什么需要定位。</p>
    </div>
  </li>
  <li>
    <div><strong>复现信息</strong></div>
    <div>
      <p>记录环境、输入规模、运行命令、最小复现脚本和必要的日志开关。</p>
    </div>
  </li>
  <li>
    <div><strong>证据链</strong></div>
    <div>
      <p>保留报错栈、中间产物、关键生成代码、源码分析路径和对照实验。</p>
    </div>
  </li>
  <li>
    <div><strong>结论验证</strong></div>
    <div>
      <p>说明根因、修复方向、验证命令、验证结果，以及仍需关注的风险。</p>
    </div>
  </li>
</ul>
