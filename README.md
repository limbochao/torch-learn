# torch-learn

`torch-learn` 用于沉淀 PyTorch 相关学习笔记、需求设计、测试脚本、问题定位日志和 Torch 相关 skills。

这个仓库的长期目标是逐步替代 `torch_scripts`，并通过 GitHub Pages 提供更方便的浏览入口。

## 目录结构

- `docs/`: GitHub Pages 展示内容，面向阅读和检索。
- `scripts/`: 可直接运行或复用的测试脚本、复现脚本。
- `skills/`: Torch 相关 Codex skills 或技能说明。
- `templates/`: 学习笔记、问题定位日志等模板。

## GitHub Pages

建议在 GitHub 仓库设置中启用 Pages：

- Source: `Deploy from a branch`
- Branch: `main`
- Folder: `/docs`

启用后，`docs/index.md` 会作为展示入口。

## 内容约定

- 学习笔记优先放到 `docs/notes/`。
- 需求设计优先放到 `docs/requirements-design/`。
- 问题定位日志优先放到 `docs/issue-logs/`。
- 可运行脚本放到 `scripts/`，并在 `docs/scripts/` 建立说明或索引。
- skills 放到 `skills/`，并在 `docs/skills/` 建立说明或索引。
