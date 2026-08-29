---
name: gitcode-create-issue
description: >-
  Use this skill whenever the user asks to draft, review, or create an Issue on GitCode, especially for
  Ascend/pytorch. 适用于通过 GitCode API 创建 Issue、提交 bug/repro/需求、检查重复 Issue、整理环境和
  validation 证据；默认仓库为 Ascend/pytorch，并保护 Personal Access Token 和内部信息。
---

# GitCode Create Issue

这个 skill 用来把问题材料整理成可维护的 GitCode Issue，并通过 bundled script 调用 GitCode API。默认目标是
`https://gitcode.com/Ascend/pytorch`，用户指定其他 GitCode 仓库时覆盖默认值。

## Runtime

- 使用 Python 3 运行 `scripts/create_issue.py`，只依赖 standard library 和本机 Git credential helper。
- 优先读取 `GITCODE_TOKEN`；变量不可见时安全回退到 `git credential fill`，不接受命令行中的明文 token。
- 默认 endpoint 为 `https://api.gitcode.com/api/v5`。
- 创建接口为 `POST /repos/:owner/issues`，仓库名通过 JSON body 的 `repo` 字段传递。
- 脚本支持 `--dry-run`，只显示去敏后的 URL 和 payload，不发起创建请求。

官方依据：

- [GitCode OpenAPI 使用入门](https://docs.gitcode.com/docs/apis/)
- [创建 Issue](https://docs.gitcode.com/docs/apis/post-api-v-5-repos-owner-issues/)
- [获取仓库所有 Issues](https://docs.gitcode.com/docs/apis/get-api-v-5-repos-owner-repo-issues/)

## 默认目标

除非用户明确指定其他仓库，使用：

```text
owner: Ascend
repo: pytorch
web: https://gitcode.com/Ascend/pytorch
```

不要仅凭当前本地 git remote 推断目标仓库。Ascend 工作区中的 `pytorch` 目录可能跟踪 upstream GitHub，用户提到
TorchNPU、Ascend PyTorch 或 `Ascend/pytorch` 时仍采用上面的 GitCode 默认值。

## 工作流

1. 收集 Issue 的 title、problem statement、环境、最小 `repro`、actual behavior、expected behavior、关键日志、影响范围和
   已做的 validation。只保留支撑问题的材料。
2. 对日志和正文做 sanitization，移除 token、个人路径、内网 URL、内部问题单、客户数据、容器名和无关环境 dump。
3. 在创建前查重。对 `Ascend/pytorch` 使用标题中的 2-4 个区分度高的关键词搜索 open 和 closed Issues：

   ```bash
   curl --fail-with-body --silent --show-error --get \
     'https://api.gitcode.com/api/v5/repos/Ascend/pytorch/issues' \
     --data-urlencode 'state=all' \
     --data-urlencode 'search=<keywords>' \
     --data-urlencode 'per_page=20'
   ```

4. 根据问题复杂度起草正文。bug/repro 默认参考 `assets/ascend-pytorch-issue-template.md`；用户明确要求空白或最简 Issue 时，直接使用
   用户给定的 title/body，不擅自补充环境、root cause 或测试结果。
5. 检查 title 是否具体，`repro` 是否可执行，actual/expected 是否可比较，代码和日志是否使用 fenced code block。
6. 向用户展示最终 title、target、labels 和 body。创建 Issue 是外部可见写操作；只有用户已经明确批准这份内容，或明确要求按当前
   内容立即创建时，才执行非 `--dry-run` 命令。
7. 先运行 `--dry-run` 检查 request shape，再使用相同参数执行实际创建。不要因为一次请求响应慢而并发重试。
8. 成功后报告 Issue number 和 `html_url`。失败时报告 HTTP status 和去敏后的 API message，不打印 request headers 或 token。

## 创建命令

把正文写入临时或工作区 Markdown 文件后调用脚本，避免多行内容的 shell quoting 问题：

```bash
python3 <skill-dir>/scripts/create_issue.py \
  --title "[Bug] torch.compile fails for ..." \
  --body-file /path/to/issue-body.md \
  --labels "bug,torch.compile" \
  --dry-run
```

用户确认后移除 `--dry-run`：

```bash
python3 <skill-dir>/scripts/create_issue.py \
  --title "[Bug] torch.compile fails for ..." \
  --body-file /path/to/issue-body.md \
  --labels "bug,torch.compile"
```

其他仓库：

```bash
python3 <skill-dir>/scripts/create_issue.py \
  --owner Ascend \
  --repo other-repo \
  --title "Issue title" \
  --body-file /path/to/issue-body.md
```

## Supported Options

- `--owner`、`--repo`：默认 `Ascend`、`pytorch`。
- `--title`：必填。
- `--body` 或 `--body-file`：二选一，必填。
- `--labels`：逗号分隔；API 文档要求单个 label 名称长度为 2-20 且不含特殊字符。
- `--assignee`：负责人 username，多个使用半角逗号分隔。
- `--milestone`：里程碑序号。
- `--security-hole true|false`：是否创建 private Issue。
- `--template-path`、`--issue-type`、`--issue-severity`：GitCode 可选字段；后两项主要用于企业版。
- `--token-env`：token 环境变量名，默认 `GITCODE_TOKEN`。
- `--no-git-credential`：禁用本机 Git credential fallback。
- `--base-url`：覆盖 API base URL，默认官方 `/api/v5` endpoint。
- `--timeout`：API 超时秒数，默认 120 秒。
- `--dry-run`：不访问网络，不要求 token。
- `--json`：输出 machine-readable JSON summary。

## Credential Safety

优先让用户在启动 Codex 的环境中设置 token，例如：

```bash
export GITCODE_TOKEN='<personal-access-token>'
```

如果变量是在另一个 shell 中后来设置的，已经运行的 Codex 进程通常不可见。脚本默认尝试 `git credential fill` 复用已配置的
GitCode HTTPS 凭据；凭据只在进程内使用，不打印、不落盘。使用 `--no-git-credential` 可以禁用回退。不要要求用户把 token 发到
聊天里，不要读取、打印或提交 shell profile。脚本使用 `Authorization: Bearer`，不使用 `access_token` query 参数。

## Failure Handling

- `400`：检查 title/body/repo 和 labels 格式。
- `401`：检查 token 是否存在、有效；不要输出 token 值。
- `403`：当前账号没有目标仓库的 Issue 创建权限，或 private Issue 等字段受限。
- `404`：检查 owner/repo 拼写，也可能是权限不足导致资源不可见。
- `409`/`422`：检查重复资源或字段约束。
- `429`：遵守服务端限流，等待后串行重试。

当失败结果不确定是否已经创建时，先按 title 搜索目标仓库，确认没有成功落单再重试，避免重复 Issue。
