# skills

存放 Torch 相关技能说明，以及待整理成标准 skill 的分析流程。

## 推荐结构

完整 skill 建议使用如下结构：

```text
skills/<skill-name>/
  SKILL.md
  references/
  scripts/
  assets/
  evals/
```

从 `torch_scripts/skills` 迁移 skill 时，先确认是否仍然通用，再补充使用边界和参考资料。

## 已整理技能

- `torch-codegen-debugging/`: PyTorch/torch_npu/Inductor codegen 问题定位流程。
- `debug-issue-archive/`: debug 类问题归档和去敏写作流程。
- `torch-learn-skill-sync/`: source skill 与本地 installed skill 的同步流程。
- `openai-image-generation/`: 使用 Codex 凭证或自定义 OpenAI-compatible 凭证调用 Responses API 生图、修图和局部重绘。
