---
name: gitcode-comment-pr
description: >-
  Use this skill whenever Codex must post a comment to a GitCode Pull Request, especially the exact compile
  comment used to trigger Ascend/pytorch CI gates. 适用于通过 GitCode API 对 PR 发表评论、dry-run 请求检查、
  token 去敏和重复评论保护，也可作为 gitcode-sync-pr 的 helper skill。
---

# GitCode Comment PR

通过 GitCode API 向指定 Pull Request 发表评论。默认场景是向新建的 `Ascend/pytorch` PR 评论 `compile` 触发门禁。

## Runtime

- 使用 Python 3 运行 `scripts/comment_pr.py`，只依赖 standard library。
- 默认读取 `GITCODE_TOKEN`，不接受命令行明文 token。
- 默认 endpoint 为 `https://api.gitcode.com/api/v5`。
- 评论接口为 `POST /repos/:owner/:repo/pulls/:number/comments`。
- 重复检查接口为 `GET /repos/:owner/:repo/pulls/:number/comments`。
- 默认拒绝重复提交内容完全相同的评论；只有用户明确要求时才使用 `--allow-duplicate`。

官方依据：

- [提交 Pull Request 评论](https://docs.gitcode.com/docs/apis/post-api-v-5-repos-owner-repo-pulls-number-comments/)
- [获取 Pull Request 评论](https://docs.gitcode.com/docs/apis/get-api-v-5-repos-owner-repo-pulls-number-comments/)

## Workflow

1. 确认 PR URL、评论文本和 repository。不要从当前 git remote 猜测 PR。
2. 评论是外部可见写操作。只有用户明确要求评论，或上层已批准工作流明确包含该评论时才继续。
3. 先 dry-run，确认 URL 和 payload 已去敏且目标正确：

   ```bash
   python3 <skill-dir>/scripts/comment_pr.py \
     --pr-url <gitcode-pr-url> \
     --body compile \
     --dry-run
   ```

4. 保持参数不变移除 `--dry-run`：

   ```bash
   python3 <skill-dir>/scripts/comment_pr.py \
     --pr-url <gitcode-pr-url> \
     --body compile
   ```

5. 脚本会先获取现有评论。存在完全相同的评论时返回 `skipped_duplicate=true`，不再次 POST。
6. 成功后报告 PR URL、comment ID 和是新建还是因重复跳过。

## Supported Options

- `--pr-url`：必填，接受 `https://gitcode.com/<owner>/<repo>/merge_requests/<number>` 和 `/pulls/<number>`。
- `--body`：必填；触发 Ascend/pytorch 编译门禁时必须精确使用小写 `compile`。
- `--allow-duplicate`：显式允许重复评论，默认禁用。
- `--token-env`：token 环境变量名，默认 `GITCODE_TOKEN`。
- `--base-url`：覆盖 GitCode API base URL。
- `--timeout`：HTTP timeout seconds，默认 30。
- `--dry-run`：仅展示去敏后的 GET/POST 计划，不访问 API。
- `--json`：输出 machine-readable JSON。

## Credential Safety

让用户在 shell 中设置 token：

```bash
export GITCODE_TOKEN='<personal-access-token>'
```

不要要求用户把 token 发到聊天里，不要读取、打印或提交 shell profile。脚本使用 `Authorization: Bearer` header，错误消息会
替换可能出现的 token。

## Failure Handling

- `400`：检查 comment body 和 PR number。
- `401`：检查 token 是否存在、有效，不输出 token。
- `403`：账号没有评论权限或 PR 不可访问。
- `404`：检查 PR URL，也可能是权限不足。
- `429`：遵守限流，等待后串行重试。

请求超时或响应丢失后，重新运行默认重复检测。不要绕过检查并盲目再次评论。
