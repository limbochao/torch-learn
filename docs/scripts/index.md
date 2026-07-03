---
title: 脚本索引
---

# 脚本索引

<section class="tl-note-band" markdown="1">
本分区为 `scripts/` 下的复现脚本、实验脚本和公共工具提供阅读入口。脚本说明优先补充用途、运行命令、输入输出、验证结论和适用环境。
</section>

<div class="tl-section-heading" markdown="1">
## 公共工具

可被多个复现或验证流程复用的工具说明。
</div>

<div class="tl-card-grid tl-card-grid-two" markdown="1">
<article class="tl-feature tl-card-accent-amber" markdown="1">
<div class="tl-meta">
  <span class="tl-badge">Profiler</span>
  <span class="tl-badge">NPU</span>
  <span class="tl-badge">CSV</span>
</div>

### [NPU profiler 工具](npu-profiler.html)

说明 `TorchNpuProfiler` 和 `ProfileResultParser` 的采集、解析和按 shape 对比方式。
</article>
</div>

<div class="tl-section-heading" markdown="1">
## 复现脚本

脚本文件位于仓库根目录的 `scripts/repro/`，Pages 页面中优先保留和问题日志关联的阅读入口。
</div>

<ul class="tl-index-list">
  <li>
    <div>
      <strong>RMSNorm codegen</strong>
      <p>精度问题复现</p>
    </div>
    <div>
      <p>`rms_norm_simd_multi_reduction_repro.py` 用于复现 RMSNorm weight grad 在 NPU Inductor SIMD 多 reduction 轴场景下的精度问题。</p>
      <p>关联文档：<a href="../issue-logs/rms-norm-simd-multi-reduction-codegen.html">问题复盘</a>。</p>
    </div>
  </li>
  <li>
    <div>
      <strong>RMSNorm pass case</strong>
      <p>对照验证</p>
    </div>
    <div>
      <p>`rms_norm_simd_multi_reduction_pass_case.py` 和 `rms_norm_simd_multi_reduction_manual_tiling.py` 用于构造对照场景和手工 tiling 验证。</p>
    </div>
  </li>
</ul>

<div class="tl-section-heading" markdown="1">
## 目录对应关系

仓库中的脚本目录按用途保持轻量拆分。
</div>

<div class="tl-card-grid tl-card-grid-two" markdown="1">
<article class="tl-card tl-card-accent-rose" markdown="1">
### `scripts/repro/`

最小复现、对照实验和手工验证脚本。
</article>

<article class="tl-card tl-card-accent-blue" markdown="1">
### `scripts/tests/`

功能测试、实验测试或验证脚本。
</article>
</div>
