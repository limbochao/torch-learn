---
title: 首页
---

<section class="tl-hero" markdown="1">
<div class="tl-hero-main" markdown="1">
<p class="tl-eyebrow">PyTorch / torch_npu / Inductor</p>

# torch-learn

`torch-learn` 是面向 PyTorch 学习笔记、需求设计和问题定位复盘的知识库。这里优先保留能复用的分析路径、验证证据和工程结论，逐步承接 `torch_scripts` 中仍有价值的文字内容。

<div class="tl-actions">
  <a class="tl-button tl-button-primary" href="issue-logs/">查看问题定位</a>
  <a class="tl-button" href="notes/">阅读学习笔记</a>
  <a class="tl-button" href="requirements-design/">需求设计</a>
</div>
</div>

<aside class="tl-panel" markdown="1">
## 当前重点

<ul class="tl-link-list">
  <li>
    <a href="issue-logs/rms-norm-simd-multi-reduction-codegen.html">RMSNorm SIMD 多 reduction 轴问题复盘</a>
    <span>从现象、错误 DSL、代码路径到修复语义的完整定位记录。</span>
  </li>
  <li>
    <a href="notes/compiler/inductor-codegen-terms.html">Inductor Codegen 术语说明</a>
    <span>整理 DSL、reduction lane、tile、stride、tiling 等常用概念。</span>
  </li>
</ul>
</aside>
</section>

<section markdown="1">
<div class="tl-section-heading" markdown="1">
## 内容入口

从读源码、写设计到归档问题，按文字内容的阅读路径组织。
</div>

<div class="tl-card-grid" markdown="1">
<article class="tl-card tl-card-accent-teal" markdown="1">
### [学习笔记](notes/)

源码阅读、机制分析和术语沉淀。适合记录 Dynamo、FX、Inductor、codegen、scheduler、runtime 等主题。
</article>

<article class="tl-card tl-card-accent-blue" markdown="1">
### [需求设计](requirements-design/)

需求背景、方案设计、接口约束、验证计划和风险记录。用于把零散讨论收敛成可评审材料。
</article>

<article class="tl-card tl-card-accent-rose" markdown="1">
### [问题定位日志](issue-logs/)

复现方式、关键日志、代码证据、根因结论和验证结果。重点保留以后能复用的 debug 路径。
</article>

</div>
</section>

<section markdown="1">
<div class="tl-section-heading" markdown="1">
## 推荐阅读

优先放置当前最能代表仓库价值的内容，后续可按主题继续扩展。
</div>

<div class="tl-card-grid tl-card-grid-two" markdown="1">
<article class="tl-feature tl-card-accent-rose" markdown="1">
<div class="tl-meta">
  <span class="tl-badge">Issue log</span>
  <span class="tl-badge">Inductor</span>
  <span class="tl-badge">SIMD reduction</span>
</div>

### [RMSNorm SIMD 多 reduction 轴 codegen 问题复盘](issue-logs/rms-norm-simd-multi-reduction-codegen.html)

围绕 RMSNorm weight grad 的 `torch.compile` 精度问题，记录错误 DSL、axis layout、accumulator 生命周期和 store 位置的分析证据。
</article>

<article class="tl-feature tl-card-accent-teal" markdown="1">
<div class="tl-meta">
  <span class="tl-badge">Notes</span>
  <span class="tl-badge">Inductor</span>
  <span class="tl-badge">Codegen</span>
</div>

### [Inductor Codegen 术语说明](notes/compiler/inductor-codegen-terms.html)

整理阅读 generated DSL 时经常遇到的基础术语，为后续 codegen 问题复盘提供统一上下文。
</article>
</div>
</section>
