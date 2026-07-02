---
name: debug-issue-archive
description: Use this skill when turning a debugging session, bug investigation, codegen analysis, traceback diagnosis, or repro workflow into a durable issue-log document. 适用于问题复盘、debug 过程归档、最小复现整理、日志去敏、root cause / fix / validation 记录。
---

# Debug Issue Archive

这个 skill 用来把一次 debug 过程整理成可以长期保存的 issue log。目标不是复述聊天记录，而是沉淀一份其他工程师能直接阅读、复现、验证的技术文档。

写作风格采用中文为主、关键技术词保留英文的形式。例如使用 `repro`、`traceback`、`generated DSL`、`root cause`、`fix strategy`、`validation`、`risk` 等英文术语，正文解释用自然中文承接。避免元叙述、过程性说明和模板化表达。

## 准备信息

归档前先收集能支撑结论的最小证据：

- 问题现象和失败阶段，例如 compile-time crash、runtime error、accuracy mismatch。
- 环境类型，例如目标 device、框架版本、是否 `torch.compile`，不要记录个人主机细节。
- 最小复现 `repro`，包括脚本路径、运行命令、输入 shape / stride / dtype / flags。
- 关键日志、`traceback`、generated kernel、FX readable graph 或其他中间产物。
- 支撑 root cause 的源码位置、分支判断、生成逻辑或运行结果。
- 修复思路、影响范围、validation 结果和残余 risk。

如果复现脚本不在目标文档仓库内，优先把最小版本落到仓库中再引用，避免文档链接到本地临时路径。

## 去敏要求

归档内容需要先做 sanitization：

- 移除用户名、home 目录、私有挂载路径、容器名、trace run id、PID。
- 移除完整 `backend_hash`，除非它是识别构建的必要信息。
- 移除内部 URL、token、key、客户数据、私有模型名。
- 不贴无关的环境 dump 或超长原始日志。

需要保留上下文时，用泛化占位：

- `<workspace>`
- `<repo>`
- `<debug-trace>`
- `<container>`
- `<target-device>`

不要引用仓库中不存在的文件路径。若必须描述外部路径，应写成明确的泛化路径。

## 文档结构

使用自然的技术小标题，不需要机械套模板。内容通常覆盖这些点：

- 问题背景和 symptom。
- 最小 `repro` 及运行方式。
- 失败 artifact，例如 generated DSL、traceback、FX graph。
- artifact 行为分析。
- 期望行为。
- root cause 或 fix strategy。
- 修改影响和 risk。
- 修复后的 validation 或 artifact evidence。

相关内容可以合并到同一节，优先保证阅读顺序顺畅。不要把过程性迭代细节全部写进去；复盘文档关注最终问题、证据、结论和修复。

## Repro 写法

一个好的 `repro` 部分应包含：

- 仓库内脚本路径。
- 目标 function / operator / kernel。
- 输入 shape、stride、dtype、dynamic/static 选项。
- 运行命令。
- 预期 pass/fail 行为。
- 脚本依赖的环境变量，例如 `CHECK=1`、`DYNAMIC=1`。

精度问题建议提供 optional eager comparison，例如用 `CHECK=1` 打开 `torch.testing.assert_close`。脚本输出要能帮助后续定位，但不要把个人路径写入输出。

## Artifact 写法

当 artifact 是判断依据时，尽量保留完整关键内容：

- codegen 问题保留完整 generated kernel body。
- exception 问题保留关键 `traceback`。
- graph 结构问题保留最小 readable graph。

可以删掉无关 metadata，但需要说明删掉的是环境相关信息。generated DSL 的格式尽量贴近真实生成结果，只有在影响阅读时才做轻微排版。

## 分析方式

面向没有参与定位过程的技术读者写：

- 对本地术语做一次解释，例如 DSL、reduction lane、tile、mask。
- 先说明 artifact 怎么读，再指出问题。
- 每个结论都连接到代码、生成语句、日志、命令输出或复现行为。
- 区分 fact 和 inference；没有证据时不要写成结论。

表达要直接、工程化：

- 推荐：`tl.store` 位于 `loop_r2` 内部，因此每个 `r2` tile 都会写出 partial result。
- 避免：这里好像 store 有点问题。

## Fix 和 Risk

修复部分需要说清楚：

- 哪个 source component 发生变化。
- generated behavior 如何变化。
- 哪些 case 应被影响。
- 哪些 case 应保持不变。
- 还剩哪些 risk。

描述范围时使用实现条件，例如 `numof_reduction_axis() > 1 and is_contiguous_reduction()`，不要只写宽泛的“multi-reduction”。

## Validation

记录 functional validation 和 artifact validation：

- 运行命令。
- 输出标记，例如 `check=passed`。
- 本地检查，例如 `py_compile`、`git diff --check`。
- 修复后 generated artifact 中对应错误行为已经消失的证据。

如果没有验证，需要明确说明未运行的项和原因。

## 仓库卫生

归档到文档仓库时：

- repro 脚本放到 `scripts` 或 `repro` 目录。
- 需要公开展示的 issue log 才加入 index。
- 内部详解、review 辅助材料可以设置为 `published: false`，并且不要从公开 index 链接。
- 文件名保持描述性和稳定。
- 不覆盖无关用户修改。
- 提交前扫描个人路径和机器信息。

推荐扫描：

```bash
grep -RInE "/home/[^ ]+|/data/[^ ]+|PID|run_[0-9]|root@|localhost|token|secret" docs scripts || true
```

有条件时运行：

```bash
git diff --check
python -m py_compile <repro-script>
```
