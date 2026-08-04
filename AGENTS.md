# torch-learn Agent Notes

本文件用于 `torch-learn` 仓库的项目级协作约定。

## GitHub Pages 内容边界

- `docs/` 是 GitHub Pages 展示入口，页面上主要展示文字类内容。
- Pages 主入口优先展示学习笔记、需求设计和问题定位日志。
- 脚本、工具、skills 和其他非文字类内容可以保留在仓库对应目录中，但不作为 Pages 主入口展示。
- 维护约定、协作规范和 agent 行为要求应写入 `AGENTS.md`，不要作为仓库正文内容展示。
- 每篇归档文档尽量回答三个问题：问题是什么，证据在哪里，结论如何验证。

## 测试脚本约定

- 本仓库不承载业务代码，不设置独立的 `tests/` 单元测试目录。
- 公共工具统一放在 `scripts/tools/`，用于支持测试、复现和分析脚本；工具类脚本禁止编写单元测试，
  也禁止为其创建 `tests/`、`scripts/ut/` 等单元测试目录。
- 功能测试、实验验证和回归测试脚本放在 `scripts/tests/`，问题复现脚本放在 `scripts/repro/`。
- 脚本说明统一放在 `scripts/` 内，其它目录只保留引用入口，不重复维护说明。
- Skill 自带的 `scripts/` 属于需要随 Skill 安装的 bundled resources，不纳入仓库级 `scripts/`。

## 分析产物约定

- 需要被文档长期引用的 generated graph、generated code 等原始产物放在 `artifacts/<topic>/`。
- `artifacts/` 中的文件是分析证据，不作为可执行脚本维护；临时日志和无长期引用价值的产物不提交。
