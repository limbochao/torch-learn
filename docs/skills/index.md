---
title: Torch skills
---

# Torch skills

<section class="tl-note-band" markdown="1">
本分区为仓库中的 Torch 相关技能说明提供展示入口。skill 源文件仍以仓库根目录下的 `skills/` 为准，Pages 页面只保留用途说明和源码链接。
</section>

<div class="tl-section-heading" markdown="1">
## 技能列表

这些流程适合沉淀重复出现的分析、归档或同步任务。
</div>

<div class="tl-card-grid" markdown="1">
<article class="tl-card tl-card-accent-rose" markdown="1">
<div class="tl-meta">
  <span class="tl-badge">Codegen</span>
  <span class="tl-badge">Debug</span>
</div>

### [torch-codegen-debugging](https://github.com/limbochao/torch-learn/blob/main/skills/torch-codegen-debugging/SKILL.md)

用于 PyTorch、torch_npu、Inductor codegen 失败、生成 kernel 错误、scheduler/tiling 问题和 `torch.compile` 后精度异常定位。
</article>

<article class="tl-card tl-card-accent-blue" markdown="1">
<div class="tl-meta">
  <span class="tl-badge">Archive</span>
  <span class="tl-badge">Issue log</span>
</div>

### [debug-issue-archive](https://github.com/limbochao/torch-learn/blob/main/skills/debug-issue-archive/SKILL.md)

用于把 debug 会话、问题复盘、最小复现、日志去敏、root cause、fix 和 validation 归档成稳定文档。
</article>

<article class="tl-card tl-card-accent-teal" markdown="1">
<div class="tl-meta">
  <span class="tl-badge">Sync</span>
  <span class="tl-badge">Skills</span>
</div>

### [torch-learn-skill-sync](https://github.com/limbochao/torch-learn/blob/main/skills/torch-learn-skill-sync/SKILL.md)

用于编辑 `torch-learn/skills` 下的 skill 后，对照并同步本地 Codex 安装目录中的同名 skill。
</article>
</div>

<div class="tl-section-heading" markdown="1">
## 维护建议

新增 skill 时保持说明、触发场景和参考资料边界清楚。
</div>

<ul class="tl-index-list">
  <li>
    <div><strong>独立目录</strong></div>
    <div><p>一个完整 skill 使用独立目录，并包含 `SKILL.md`。</p></div>
  </li>
  <li>
    <div><strong>参考资料</strong></div>
    <div><p>与 skill 配套的参考资料可放在该 skill 目录下的 `references/`。</p></div>
  </li>
  <li>
    <div><strong>临时沉淀</strong></div>
    <div><p>尚未整理成 skill 的经验记录，可以先放在 `skills/README.md` 或独立 Markdown 中。</p></div>
  </li>
</ul>
