# NPU_A5 复测：上次新增通过的 98 行

本轮只重跑 `npu-inductor-213-retry-20261011/results.csv` 中上轮非 PASS、重跑后变为 PASS 的 98 行，共 120 个实际 pytest 节点。

执行环境为 NPU_A5 的 `pta_213_case` conda 环境，使用 NPU 0-6 七卡并发；每张卡使用独立的 `TORCHINDUCTOR_CACHE_DIR`。

| 结果 | 行数 | 节点数 |
| --- | ---: | ---: |
| PASS | 92 | 114 |
| SKIP | 6 | 6 |
| FAIL | 0 | 0 |
| ERROR | 0 | 0 |

结果已合并到上轮总表的 `NPU_A5复测结果`、`NPU_A5复测卡`、`NPU_A5复测节点数`、`NPU_A5复测通过数`、`NPU_A5复测跳过数` 和 `NPU_A5复测说明` 列。

6 行 SKIP 的原因已在总表逐行注明：4 行大张量用例未同步 NPU 内存能力门控，1 行因 NPU 能力检测未达到测试要求，1 行仍保留动态形状用例自身的 skip 装饰器。静态 Triton 文件中的 5 条用例已补充 `_dynamo.patch_has_triton()` 并在 NPU_A5 复测通过。

本轮 adapter 测试使用的代码提交为 `Community-Use-Case-Testing` 分支 `ut/adapter_to_npu` 的 `2774f8b`。
