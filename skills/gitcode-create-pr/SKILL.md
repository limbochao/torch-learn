---
name: gitcode-create-pr
description: >-
  Use this skill whenever the user asks to draft, review, create, or update a Pull Request on GitCode, especially
  a contribution to Ascend/pytorch. 适用于检查本地 branch/diff、区分同仓与 fork PR、生成简洁 PR 描述、
  创建或更新 PR、建立并验证结构化 Issue 关联；默认 target 为 Ascend/pytorch:master。
---

# GitCode Create PR

这个 skill 用来检查待提交变更、生成聚焦修改内容和原因的 Pull Request 描述，并通过 bundled script 调用 GitCode API。
默认 target 是 `https://gitcode.com/Ascend/pytorch` 的 `master` 分支；用户指定 release branch 或其他仓库时覆盖默认值。

## Runtime

- 使用 Python 3 运行 `scripts/create_pr.py`，只依赖 standard library 和本机 Git credential helper。
- 优先读取 `GITCODE_TOKEN`；变量不可见时安全回退到 `git credential fill`，不接受命令行中的明文 token。
- 默认 endpoint 为 `https://api.gitcode.com/api/v5`。
- 创建接口为 `POST /repos/:owner/:repo/pulls`，更新接口为 `PATCH /repos/:owner/:repo/pulls/:number`。
- `--issue` 会在创建或更新后调用 linked-issues API，并通过 GET 验证结构化关联结果。
- 脚本支持 `--dry-run`，只显示去敏后的 URL 和 payload，不发起创建请求。

官方依据：

- [GitCode OpenAPI 使用入门](https://docs.gitcode.com/docs/apis/)
- [创建 Pull Request](https://docs.gitcode.com/docs/apis/post-api-v-5-repos-owner-repo-pulls/)
- [更新 Pull Request](https://docs.gitcode.com/docs/apis/patch-api-v-5-repos-owner-repo-pulls-number/)
- [Pull Request 关联 Issue](https://docs.gitcode.com/docs/apis/post-api-v-5-repos-owner-repo-pulls-number-linked-issues/)
- [获取项目 Pull Request 列表](https://docs.gitcode.com/docs/apis/get-api-v-5-repos-owner-repo-pulls/)
- [Ascend/pytorch PR 模板](https://gitcode.com/Ascend/pytorch/blob/master/.gitcode/PULL_REQUEST_TEMPLATE.md)

## 默认目标

除非用户明确指定其他 target，使用：

```text
owner: Ascend
repo: pytorch
base: master
web: https://gitcode.com/Ascend/pytorch
```

不要从目录名直接推断 source fork。当前工作区中的 `pytorch` 可能跟踪 upstream GitHub，而 GitCode PR 的 source branch 必须已经存在于
GitCode 同仓或用户 fork 中。

## 工作流

1. 在 source repository 中读取 `git status --short`、当前 branch、`git remote -v`、相对 target branch 的 commits 和 diff stat。
   不要把 uncommitted changes 描述成已经进入 PR。
2. 确认 target owner/repo/base 和 source branch。`Ascend/pytorch` 默认 base 是 `master`，release 修复通常需要用户明确指定
   `vX.Y.Z` target。
3. 确认 source branch 已推送到 GitCode：
   - 同仓 PR：`head` 使用 branch 名，不传 `fork_path`。
   - 跨仓 PR：`head` 使用 `username:branch`，并传 `fork_path=owner/repo`。
   - 如果 branch 尚未推送，不要静默 push。展示 remote、branch 和将执行的 push，取得用户授权后再推送。
4. 搜索相同 source branch/base 或相同 title 的 open PR，避免重复创建。GitCode 项目 PR 列表接口不支持 `head` 参数时，使用
   `author`、`base` 获取候选列表，再按返回的 `source_branch` 和 source repo 精确过滤。
5. 从 commits 和 diff 生成 title/body。用户没有指定格式时，正文使用 `1.`、`2.` 编号列出实际修改，每项同时说明“改了什么”和
   “为什么改”。生成 body 时，不要在自然语言段落或单个列表项中插入手动换行符，交由网页根据显示宽度自动换行；只有代码块中的
   代码、命令或其他必须保持格式的内容才手动换行。不要写未纳入 PR 的方案、排查历史、无关验证过程或空模板章节。只有用户明确
   要求或仓库强制要求时才使用 `assets/ascend-pytorch-pr-template.md`。
6. 只描述有代码或验证证据支持的收益。不要编造 CI、UT、device validation 或 review 结论。
   默认不在 PR 正文中写验证结果、测试命令、CI 状态或未执行的测试；只有用户明确要求包含验证信息时，才加入真实且可追溯的验证结果。
7. 做 sanitization：不得包含内部需求链接、内部问题单、token、个人路径、客户数据、内部 URL 或机器信息。社区 Issue 使用公开
   `https://gitcode.com/Ascend/pytorch/issues/<number>` 链接。
8. 向用户展示最终 target/source、title、body、labels、reviewers 和 branch cleanup/squash 选项。创建 PR 是外部可见写操作；只有用户
   已明确批准这份内容，或明确要求按当前内容立即创建时，才执行非 `--dry-run` 命令。
9. 先运行 `--dry-run` 检查 request shape，再使用相同参数执行实际创建或更新。
10. 指定 `--issue` 时，不能只把 Issue URL 写进正文。脚本会调用 linked-issues API 并 GET 验证；最终报告必须确认结构化关联列表
    包含目标 Issue。

## 创建命令

同仓 PR：

```bash
python3 <skill-dir>/scripts/create_pr.py \
  --title "fix: handle dynamic reduction shape" \
  --head fix-dynamic-reduction \
  --base master \
  --body-file /path/to/pr-body.md \
  --issue 1234 \
  --dry-run
```

跨 fork PR：

```bash
python3 <skill-dir>/scripts/create_pr.py \
  --title "fix: handle dynamic reduction shape" \
  --head contributor:fix-dynamic-reduction \
  --fork-path contributor/pytorch \
  --base master \
  --body-file /path/to/pr-body.md \
  --issue 1234 \
  --dry-run
```

用户确认后，保持参数不变并移除 `--dry-run`。脚本也允许 `--head fix-dynamic-reduction --fork-path contributor/pytorch`，此时会自动
规范为 `contributor:fix-dynamic-reduction`。

更新已有 PR 的正文并关联 Issue：

```bash
python3 <skill-dir>/scripts/create_pr.py \
  --update-number 1234 \
  --body-file /path/to/pr-body.md \
  --issue 5678 \
  --dry-run
```

确认 dry-run 后移除 `--dry-run`。脚本会依次 PATCH PR、POST Issue 关联，并 GET 验证关联结果。

## Supported Options

- `--owner`、`--repo`：target repository，默认 `Ascend`、`pytorch`。
- `--title`、`--head`：创建 PR 时必填；更新已有 PR 时可省略。
- `--base`：默认 `master`。
- `--update-number`：更新已有 PR 的序号；使用后不能再传 `--head` 或 `--fork-path`。
- `--body` 或 `--body-file`：可选；实际向 `Ascend/pytorch` 创建时应提供完整模板正文。
- `--fork-path`：跨仓 PR 的 source repository，格式 `owner/repo`。
- `--issue`：一个或多个 Issue 编号，逗号分隔。脚本通过 linked-issues API 建立并验证结构化关联。
- `--labels`：逗号分隔。
- `--assignees`、`--testers`：username，多个使用半角逗号分隔；仓库自动指派规则可能覆盖这些字段。
- `--milestone-number`：里程碑序号。
- `--draft`、`--squash`、`--prune-source-branch`、`--close-related-issue`：显式启用对应布尔选项。
- `--squash-commit-message`：仅与 `--squash` 一起使用。
- `--token-env`、`--no-git-credential`、`--base-url`、`--timeout`、`--dry-run`、`--json`：认证、endpoint、
  超时、预览和输出控制。

## Credential Safety

优先让用户在启动 Codex 的环境中设置 token：

```bash
export GITCODE_TOKEN='<personal-access-token>'
```

如果变量是在另一个 shell 中后来设置的，已经运行的 Codex 进程通常不可见。脚本默认尝试 `git credential fill` 复用已配置的
GitCode HTTPS 凭据；凭据只在进程内使用，不打印、不落盘。使用 `--no-git-credential` 可以禁用回退。不要要求用户把 token 发到
聊天里，不要读取、打印或提交 shell profile。脚本通过 `Authorization: Bearer` header 认证，不使用 `access_token` query 参数。

## Failure Handling

- `400`：检查 title/head/base、fork `head` 格式和 body 字段。
- `401`：检查 token 是否存在、有效；不要输出 token 值。
- `403`：账号无创建权限，或 source/target 分支不可访问。
- `404`：检查 target owner/repo、source fork 和 branch，也可能是权限不足。
- `409`/`422`：通常是同一 source/target 已有 PR、分支无差异，或字段约束不满足。
- `429`：遵守限流，等待后串行重试。

请求超时或响应丢失后，先按 author/base 查询候选 PR，并根据 `source_branch`、source repo 和 title 确认是否已创建，再决定是否
串行重试。GitCode 可能先创建成功、后返回超时；不要立即并发重试。
