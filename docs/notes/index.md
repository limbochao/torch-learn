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
      <a href="compiler/codegen/inductor-codegen-terms.html">Inductor Codegen 术语说明</a>
      <p>解释 DSL、reduction lane、tile、stride、tiling 等阅读生成 kernel 时常见的术语。</p>
    </div>
    <div>
      <a href="compiler/codegen/inductor-pointwise-vs-elementwise.html">Inductor Pointwise 与 Elementwise 范围区别</a>
      <p>说明 Triton pointwise 为什么大于狭义 elementwise，并列出 broadcast、view/reindex、mask、copy、fill 等子类。</p>
    </div>
    <div>
      <a href="compiler/codegen/inductor-combo-tests-v2.13-vs-v2.10.html">v2.13.0 Combo/Foreach 用例在 v2.10.0 的存在性核对</a>
      <p>逐项核对 116 条记录及 2.10 支持范围，汇总 NPU_A5 的 409 个实例结果，附 CSV 表格。</p>
    </div>
    <div>
      <a href="compiler/codegen/inductor-combo-tests-liulm-v2.10.0-combokernel-dev.html">liulm v2.10.0-combokernel-dev 的 Combo/Foreach 功能复测</a>
      <p>单独记录 liulm 新能力基线的 62 个方法、409 个实例复测情况，并与旧基线结果对照。</p>
    </div>
    <div>
      <a href="compiler/symbolic-group/inductor-symbolic-group-v2.10.html">torch_npu v2.10.0 Symbolic Group 实现</a>
      <p>仅依据 v2.10.0 线上代码说明准入、代表 shape、candidate plan、autotune 和 dispatch。</p>
    </div>
    <div>
      <a href="compiler/debugging/npu-fx-graph-runnable-adaptation.html">NPU FX Graph Runnable 通用适配</a>
      <p>说明输入符号化、自定义算子占位、static/dynamic/group 模式和产物收集的通用边界。</p>
    </div>
    <div>
      <a href="compiler/debugging/torchbench-213-setup-and-test.html">TorchBench 2.13 环境准备与模型测试</a>
      <p>直接安装配套软件，测试时自动准备模型数据，默认使用 Inductor + Triton 检查精度。</p>
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
