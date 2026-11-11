# Combo 五个修复的固定源码评审证据

[2.10 当前结果](../../docs/notes/compiler/codegen/inductor-combo-tests-v2.10.md#native210)。
本目录在 2026-11-07 从固定 Git 对象提取，不重新运行 NPU 用例。
既有测试结果继续引用原证据目录，避免复制后混淆来源。

| 文件 | 内容 |
| --- | --- |
| f10-f12.patch | 64a3e7e8b9 相对父提交的完整原始 diff |
| f11.patch | 9b1a894d65 相对父提交的完整原始 diff |
| f8.patch | fdf73f229e 相对父提交的完整原始 diff |
| f4.patch | 53f8af09f1 相对父提交的完整原始 diff，含无进位优化 |
| f5.patch | a8dcca84bf 相对父提交的完整原始 diff |
| upstream-v2.10.0.md / upstream-v2.13.0.md | 对应固定标签的源码摘录；片段不改写格式 |
| npu-call-chain.md | 已验证 a8dcca84bf 的调度、尺寸生成及 mask 调用链摘录 |
| target-nodeids.json | 从最新全量 409 节点中筛出的八方法、十实例精确清单 |
| manifest.json | 提交、父提交、修改量、源码及片段 SHA256、来源行区间 |

补丁只包含代码 diff，不包含作者邮箱和本地目录。
源码摘录的行区间是原文件位置，非 Markdown 自身行号。

在对应生产仓中可复核任意补丁，下面以 F4 为例：

```bash
git show --format= --no-ext-diff --no-renames 53f8af09f144aaaec685562acd4c3a3ffaffb71a
git show --format= --numstat 53f8af09f144aaaec685562acd4c3a3ffaffb71a
```

社区内容由 `git show v2.10.0:<source-path>` / `git show v2.13.0:<source-path>` 提取。
标签对应 commit 固定在 manifest，避免把其他分支 HEAD 的行为混入上游对照。
