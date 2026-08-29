---
name: gitcode-sync-pr
description: >-
  Use this skill whenever a commit from a GitCode Pull Request must be synchronized to one or more
  Ascend/pytorch maintenance branches. 适用于根据来源 PR、来源分支和 commit 创建目标分支同步分支，执行
  cherry-pick 与语义化冲突处理，推送到 origin，复用来源 Issue 创建 upstream PR，并评论 compile 触发门禁。
---

# GitCode Sync PR

将一个已提交到来源 PR 分支的 commit 同步到一个或多个 `Ascend/pytorch` 目标分支。每个目标分支独立处理，最终返回新建
同步 PR 的链接。

## Inputs

开始前必须取得：

- GitCode 来源 PR 链接，例如 `https://gitcode.com/Ascend/pytorch/merge_requests/1234`。
- 本地来源分支中的 commit SHA，优先使用完整 SHA。
- 一个或多个目标分支名，例如 `master`、`v2.7.1`。
- 本地 source repository 路径。默认可使用 Ascend 工作区中的 `pytorch_new`，但必须通过 remotes 验证身份。

默认 remotes：

```text
origin: 用户的 GitCode fork，用于推送同步分支
upstream: https://gitcode.com/Ascend/pytorch.git
```

不要仅凭目录名推断 remote 角色。

## Dependencies

- 使用 `scripts/inspect_source_pr.py` 读取来源 PR 和关联 Issue。
- 创建 PR 前必须读取并遵循 `$gitcode-create-pr`。
- 创建 PR 后必须读取并遵循 `$gitcode-comment-pr`，向新 PR 评论 `compile`。
- GitCode API 默认从 `GITCODE_TOKEN` 环境变量读取 token。不得读取、打印或提交 token 值。

如果依赖 skill 未安装或脚本不可用，停止外部写操作并报告缺失项。

## Workflow

### 1. Preflight

1. 阅读工作区和 repository 中适用的 `AGENTS.md`。
2. 检查 `git status --short --branch`、当前 branch 和 `git remote -v`。
3. 验证 `origin` 是可推送的 fork，`upstream` 指向目标 GitCode repository。
4. 验证 commit 存在且是 commit object：

   ```bash
   git cat-file -e <commit>^{commit}
   ```

5. 解析来源 PR 并强制检查关联 Issue：

   ```bash
   python3 <skill-dir>/scripts/inspect_source_pr.py \
     --pr-url <source-pr-url> \
     --require-single-issue
   ```

6. 验证本地来源分支名与 PR head branch 一致，并确认来源分支包含指定 commit。来源 PR head SHA 不必等于指定 commit，
   但指定 commit 必须属于该 PR 的提交历史或由用户明确确认。

### 2. Issue Gate

只在来源 PR 恰好关联一个 Issue 时自动继续，并记录 Issue number 和公开链接。

- 0 个 Issue：暂停，要求用户提供要关联的 Issue，或明确批准新 PR 不关联 Issue。
- 多个 Issue：暂停，列出全部 Issue，要求用户明确选择一个 Issue。
- 不得从 PR 正文中的普通 `#123` 文本猜测关联关系；以 GitCode PR linked-issues API 为准。
- 用户确认的 override 只对当前同步任务生效，必须记录在最终结果中。

### 3. Create One Isolated Branch Per Target

目标分支必须串行处理。对每个 `<target>`：

1. 拉取目标分支最新代码：

   ```bash
   git fetch upstream <target>
   ```

2. 从来源 PR 的 head branch 获取 `<source-branch>`，同步分支名固定为：

   ```text
   <source-branch>_<target>
   ```

   例如 `bugfix/group_runtime_args` 同步到 `master` 时使用
   `bugfix/group_runtime_args_master`。

3. 使用 `git check-ref-format --branch` 验证名称。目标分支名含 `/` 时保留原名，不擅自改写。
4. 检查本地和 `origin` 是否已有同名 branch。存在时不要 reset、覆盖或 force push；先让用户决定复用、改名或停止。
5. 从最新 `upstream/<target>` 创建临时 worktree 和新 branch。使用 `mktemp -d` 创建精确临时目录，避免切换或污染用户
   当前 worktree。

### 4. Cherry-pick

在目标 worktree 中执行：

```bash
git cherry-pick <commit>
```

如果成功，继续 validation。如果失败，先检查：

```bash
git status --short
git diff --cc
git show --stat --summary <commit>
git show <commit>^!
```

冲突解决遵循这些规则：

- 只移植来源 commit 实际引入的行为变化，不把来源分支的其他基线差异带入目标分支。
- 保留目标分支中与来源 commit 无关的新增、删除和重构。
- 不要整文件选择 `ours` 或 `theirs`，除非完整 diff 证明这是正确语义。
- 对照来源 commit 的 parent、commit 后版本和目标版本，逐个 conflict hunk 判断。
- 例如目标分支已经删除旧 helper，而来源 commit 只在相邻位置增加新 helper 时，只增加新 helper，不恢复旧 helper。
- 无法确认双方逻辑如何组合时，保留 conflict 状态并让用户确认；不要提交猜测性结果。

解决后执行 `git add` 和 `git cherry-pick --continue`，保留原 commit message 和 author。

### 5. Validate

至少执行：

```bash
git diff HEAD^ HEAD --check
git status --short --branch
git show --stat --oneline HEAD
```

同时检查：

- 不存在 conflict markers。
- 新 commit 的行为变化覆盖来源 commit 的实际改动。
- 运行受影响文件可执行的本地语法检查或 targeted tests。
- 使用 `git range-diff <source>^! HEAD^!` 辅助审查基线差异；不要因为 target 基线不同而机械要求 patch-id 一致。

需要 NPU、Docker 或远程环境验证时，遵循工作区约定先取得用户确认。未运行的验证必须如实写入 PR。

### 6. Push

确认 worktree clean 后推送新 branch：

```bash
git push --set-upstream origin <sync-branch>
```

禁止默认 force push。推送后比较本地 HEAD、`origin/<sync-branch>` 和 `git ls-remote` 的 SHA。

### 7. Create Sync PR

使用 `$gitcode-create-pr` 创建 PR：

- target repository：`Ascend/pytorch`。
- base：当前 `<target>`。
- head：`origin` fork 中的 `<sync-branch>`。
- issue：Issue Gate 得到或用户明确指定的唯一 Issue。
- title：基于来源 PR title，明确这是向 `<target>` 的同步。

PR body 必须在模板正文前部写明：

```markdown
## 同步说明

- 类型：同步 PR
- 来源 PR：<source-pr-url>
- 来源提交：<source-commit>
- 目标分支：<target>
- 关联 Issue：<issue-url-or-approved-none>
```

只陈述实际执行的 validation。创建前执行 `$gitcode-create-pr` 的 duplicate check 和 dry-run。用户已经明确要求完成整个同步
工作流时，视为已经批准创建这些同步 PR；如果 target、Issue、diff 或正文与用户输入不一致，仍需暂停确认。

### 8. Trigger Gate

PR 创建成功后，使用 `$gitcode-comment-pr` 对新 PR 评论精确文本：

```text
compile
```

先 dry-run，再实际评论。helper 的重复检测命中时视为已经触发，不重复提交。

如果评论失败，不要重新创建 PR。保留已创建 PR，并仅重试或报告评论步骤。

### 9. Cleanup And Report

只有在确认临时 worktree clean 且不再需要时，才移除本流程创建的临时 worktree。不要删除本地或远端同步 branch。

最终按目标分支报告：

```text
target | sync branch | commit | issue | PR URL | compile comment
```

同时报告未运行的验证、用户确认的 Issue override 和任何部分失败。

## Recovery Rules

- cherry-pick 冲突未解决：保留 worktree 和冲突状态，报告路径，不 push。
- push 成功但 PR 创建失败：检查相同 head/base 是否已有 PR，只重试 PR 创建。
- PR 创建成功但响应丢失：按 head/base/title 查询，不直接重复创建。
- PR 已创建但 `compile` 评论失败：只重试评论。
- 多 target 中一个失败：停止该 target；其他 target 是否继续由用户输入和失败影响范围决定，不掩盖部分成功。
