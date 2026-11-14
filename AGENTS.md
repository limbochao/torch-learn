# torch-learn Agent Notes

本文件用于 `torch-learn` 仓库的项目级协作约定。

## GitHub Pages 内容边界

- `docs/` 是 GitHub Pages 展示入口，页面上主要展示文字类内容。
- Pages 主入口优先展示学习笔记、需求设计和问题定位日志。
- 脚本、工具、skills 和其他非文字类内容可以保留在仓库对应目录中，但不作为 Pages 主入口展示。
- 维护约定、协作规范和 agent 行为要求应写入 `AGENTS.md`，不要作为仓库正文内容展示。
- 每篇归档文档尽量回答三个问题：问题是什么，证据在哪里，结论如何验证。

## 目录结构变更约束

- 默认保持仓库现有顶层目录和 `docs/` 下的既有分类结构；新增内容优先放入最接近的现有目录，并只做必要的局部索引更新。
- 新增顶层分类目录、跨既有分类批量移动文件、合并或拆分目录、重写站点导航及大范围修改文档链接，均属于结构性变更。
- 结构性变更实施前，必须向用户说明拟调整的目录、迁移范围和主要影响，并取得明确确认；不能仅凭“整理”“归类”“避免混乱”等要求推定已获得目录重组授权。
- 未取得确认时，可以提出目录调整建议，但不得实际移动已有文件或改变主要入口结构。

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
- 文档引用 generated DSL、`output_code.py` 或 kernel 片段时，默认保持原始产物的格式，不主动
  pretty-print、重排或拆分参数列表。
- 只有用户明确要求 pretty JSON 等展示形式时才增加格式化换行；generated artifact 中的原始长行
  不受普通文档或代码行宽约束。

## Combo 用例文档维护

- Combo/foreach 测试与问题分析仅维护 `docs/notes/compiler/codegen/inductor-combo-tests-v2.10.md`
  和 `docs/notes/compiler/codegen/inductor-combo-tests-v2.13.md` 两份版本文档。
- 文档只保留最新结果、实际验证版本和对应的当前问题分析；不再按分支、提交或执行批次新增文档。
- 不在正文累积旧轮次结果、已解决问题或历史修复说明，历史内容通过 Git 和原始证据追溯。
- 已 PASS 的用例不保留旧失败分析；脚本 README 仅说明运行方式，不重复维护结果和问题分析。
- CSV 结果表和 `artifacts/` 原始证据独立保留；自动发布只更新固定版本文档的当前结果章节。
- 2.13 用例适配进展统一维护 `docs/notes/compiler/codegen/inductor-combo-tests-v2.13.csv`，
  同表保留 2.10 源码对照信息，不再按分支或比较范围维护另一份 2.13 进展表。
- 后续 2.13 验证只选择最新逐实例报告中的 FAIL、SKIP（含执行 ERROR），不重跑历史 PASS，
  不启动全量测试；参数化方法也必须通过 `only_nodes` 排除已 PASS 的实例。
- 定向复测结果按实例合并回当前快照，保留未重跑实例的结果和验证来源，再更新唯一进展表。
