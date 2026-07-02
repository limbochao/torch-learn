---
name: torch-learn-skill-sync
description: Use this skill whenever editing, creating, reviewing, or moving skills under torch-learn/skills. It enforces that repository skill changes are checked against the local Codex install directory and, if the same skill is already installed, the installed copy is updated as well. 适用于修改 torch-learn 仓库内 skill 后同步 ~/.codex/skills 的场景。
---

# Torch Learn Skill Sync

这个 skill 用于维护 `torch-learn/skills` 中的 skill source 和本地 Codex installed copy 之间的一致性。修改仓库内 skill 时，如果本地已经安装同名 skill，需要同步修改安装目录，避免后续 Codex 仍加载旧版本。

写作和操作记录采用中文为主、关键技术词保留英文的形式。保留 `source skill`、`installed skill`、`CODEX_HOME`、`SKILL.md`、`frontmatter`、`sync`、`diff` 等英文术语，说明用自然中文承接。

## 触发场景

当任务涉及以下路径或行为时使用：

- 新建或修改 `torch-learn/skills/<skill-name>/SKILL.md`。
- 调整 `torch-learn/skills/<skill-name>/references`、`scripts`、`assets` 等 bundled resources。
- 重命名、移动或删除 `torch-learn/skills` 下的 skill。
- 用户要求“安装 skill”“同步 skill”“更新本地已安装 skill”。
- 用户反馈某个 skill 行为没有生效，怀疑 installed copy 不是最新。

## 基本规则

`torch-learn/skills/<skill-name>` 是 source skill，默认作为修改入口。

本地安装目录默认是：

```text
$CODEX_HOME/skills/<skill-name>
```

如果 `CODEX_HOME` 未设置，按默认路径处理：

```text
/home/qc/.codex/skills/<skill-name>
```

修改 source skill 后必须检查 installed skill 是否存在：

```bash
test -d "${CODEX_HOME:-/home/qc/.codex}/skills/<skill-name>"
```

如果 installed skill 存在，需要同步同一份修改；如果不存在，只在结果里说明该 skill 尚未安装。

## 修改流程

1. 先确认本次涉及的 skill name。可以从路径 `torch-learn/skills/<skill-name>` 推导。
2. 修改 source skill，保持 `name` frontmatter 和目录名一致。
3. 检查 installed skill 是否存在。
4. 如果 installed skill 存在，先比较 source 和 installed：

```bash
diff -ru <source-skill-dir> <installed-skill-dir> | sed -n '1,160p'
```

5. 如果 installed skill 只是旧版本，使用 source 覆盖 installed：

```bash
rsync -a --delete <source-skill-dir>/ <installed-skill-dir>/
```

6. 同步后再次运行 `diff -ru`，确认没有差异。
7. 最终回复中说明 source skill 是否修改、installed skill 是否同步、是否需要重启 Codex。

## 冲突处理

如果 installed skill 中存在 source skill 没有的改动，不要直接覆盖。先判断差异来源：

- 如果差异只是 source 新版本尚未同步，按 source 覆盖。
- 如果 installed skill 有人工本地修改、未归档资源或不明来源文件，暂停同步并向用户说明差异。
- 如果用户明确要求 source authoritative，再用 `rsync -a --delete` 覆盖。

不要把 unrelated installed skills 一起同步。每次只处理本次修改涉及的 skill。

## 新建 skill

新建 `torch-learn/skills/<skill-name>` 后，本地默认还没有 installed copy。除非用户同时要求安装，否则只创建 source skill，并在结果中说明未安装。

如果用户要求安装，复制整个 skill directory 到 `$CODEX_HOME/skills/<skill-name>`。安装后提示需要重启 Codex 才会加载新 skill。

## 删除或重命名 skill

删除或重命名 source skill 前，需要先确认用户意图。installed skill 已存在时，不自动删除本地安装副本，除非用户明确要求。

重命名时要同时更新：

- source directory name。
- `SKILL.md` frontmatter 中的 `name`。
- installed directory name，如果用户要求同步 installed copy。

## 检查清单

完成前跑这些检查：

```bash
find torch-learn/skills/<skill-name> -maxdepth 2 -type f -print | sort
sed -n '1,40p' torch-learn/skills/<skill-name>/SKILL.md
test -d "${CODEX_HOME:-/home/qc/.codex}/skills/<skill-name>" && \
  diff -ru torch-learn/skills/<skill-name> "${CODEX_HOME:-/home/qc/.codex}/skills/<skill-name>"
```

最终回复保持简洁：

- 改了哪个 source skill。
- 是否发现 installed skill。
- 如果存在，是否已同步到 installed copy。
- 是否需要重启 Codex。
