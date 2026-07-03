---
title: 学习笔记
---

# 学习笔记

<section class="tl-note-band" markdown="1">
本分区存放 PyTorch 相关学习笔记、源码阅读记录和机制分析。内容优先沉淀可复用概念、关键代码路径、实验结论和与 NPU/Inductor 相关的差异点。
</section>

<div class="tl-section-heading" markdown="1">
## 已整理内容

当前以编译和 codegen 方向为主，后续可继续扩展到前端捕获、runtime 和算子语义。
</div>

<ul class="tl-index-list">
  <li>
    <div>
      <strong>Compiler / Inductor</strong>
      <p>codegen 基础概念</p>
    </div>
    <div>
      <a href="compiler/inductor-codegen-terms.html">Inductor Codegen 术语说明</a>
      <p>解释 DSL、reduction lane、tile、stride、tiling 等阅读生成 kernel 时常见的术语。</p>
    </div>
  </li>
</ul>

<div class="tl-section-heading" markdown="1">
## 建议分类

新增笔记可以先按主题落到下列方向，等内容变多后再拆更细的导航。
</div>

<div class="tl-card-grid" markdown="1">
<article class="tl-card tl-card-accent-blue" markdown="1">
### Frontend

Dynamo、FX、export、graph break、guard 和 shape capture 等前端机制。
</article>

<article class="tl-card tl-card-accent-teal" markdown="1">
### Compiler

Inductor、codegen、scheduler、autotune、generated DSL 和后端 lowering。
</article>

<article class="tl-card tl-card-accent-amber" markdown="1">
### Runtime

dispatcher、executor、device runtime、stream、event 和 profiling 相关机制。
</article>

<article class="tl-card tl-card-accent-green" markdown="1">
### Ops

算子语义、shape、dtype、layout、stride 和精度行为专题记录。
</article>
</div>
