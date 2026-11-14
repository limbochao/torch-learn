# Combo/foreach 验证脚本

执行结果和当前问题分析只维护两个版本入口：

- [2.10 最新结果与分析](../../../docs/notes/compiler/codegen/inductor-combo-tests-v2.10.md)
- [2.13 最新结果与分析](../../../docs/notes/compiler/codegen/inductor-combo-tests-v2.13.md)

本目录的 README 仅说明脚本使用，不重复记录测试结果或修复历史。

## 当前执行入口

- [2.10 原生用例筛选与直接 pytest 命令](README-native-210.md)
- [2.13 原生 pytest 调度与结果发布](README-213.md)

`run_plain_pytest.py` 按源方法匹配实际参数化 node ID，以独立 pytest 子进程在指定卡上运行。
`only_nodes` 可将一个方法限制为指定失败实例，避免重跑同方法的已通过参数。
输出包含命令、日志、JUnit、逐实例结果及执行前后源码哈希；每次使用新目录和独立编译缓存。

`publish_native_210.py` 更新 2.10 原生结果 CSV 和固定文档的 `native210` 章节。
`publish_plain_pytest.py` 只更新 `inductor-combo-tests-v2.13.csv` 和固定文档的 `validation213` 章节。
2.13 后续通过 `select_213_table.py` 从最新报告生成 FAIL/SKIP 实例清单，不重跑历史 PASS 或全量。
`combo_documents.py` 限定文档路径和可替换章节，不按执行批次生成 Markdown，
也不覆盖同一文档内的其他当前分析。

CSV 逐方法记录最新状态，PASS 的当前分析保持“—”；正文只保留最新结果与对应分析。
历史内容通过 Git 和原始证据追溯。

## 已适配用例的兼容执行入口

`prepare_validation.py`、`run_validation.py` 用于此前将 2.13 用例适配后在 2.10 执行的流程。
它们与直接运行 2.10 原生用例的口径不同，不混算结果。

```bash
python prepare_validation.py --template <prepared-template> --root <fresh-phase> \
  --manifest <selected-methods.json> --expected-nodes <expected-nodeids.json> \
  --source-hashes <expected-source-sha256.json> --context <validation-context.json> \
  --devices <authorized-physical-devices> --detach
```

`publish_retest.py`、`publish_validation.py` 负责这一兼容流程的收尾，
`update_liulm_results.py` 仅更新其 CSV 和原始证据，不再创建独立分析文档，
也不覆盖当前原生 2.10 文档。

## 后台检查

测试任务在远端工作目录独立运行；定时检查可以在本地用 `systemd-run --user` 创建。
检查程序每次只读取阶段状态，完成后统一读取结果，避免反复打印测试日志。
选卡以用户当前授权及工作区规则为准，完成或取消后停止对应定时器。
