# 2.13 ComboKernel 分支验证

本轮使用 NPU_A5 的 `pta_213_case` conda 环境，工作目录为 `/home/l30023782/PTA_213`。
目标安装包为 `torch_npu 2.13.0+git64dc85c`，对应 `sync/combo-kernel-master` 提交
`64dc85c99c1398a0bd2355b72cc2851d0688b964`。PyTorch 为 `2.13.0+cpu`，Triton 为 `3.6.0`。

用例源为测试仓库 `v2.13.0/inductor_npu`，完整范围快照提交为 `3f5e543`。
本次只为相关测试文件补充缺失或过晚的 Triton 探测初始化，保留该目录已有的 NPU 适配代码，
不放宽 kernel 数量、源码或数值断言，不改安装包。
该目录的 foreach 方法存在主动 NPU skip，理由为不支持 NPU FOREACH 分组融合；
本轮保留并如实报告 SKIP，不把这些结果计为 PASS。

执行范围以 `inductor-combo-tests-liulm-v2.10.0-combokernel-dev.csv` 的全部 116 行为准，
不沿用 2.10 的排除范围。`select_213_table.py` 按 2.13 源码解析为 105 个独立方法和 11 行嵌套函数；
嵌套函数随其外层方法验证，结果表保留原顺序和全部 116 行。

`run_plain_pytest.py` 使用 `combo_213_all_selected.json` 中的 105 个源方法，
`combo_213_table_rows.json` 记录全部表格行与外层测试的对应关系。
逐文件执行原生 pytest 收集，从输出中解析实际参数化 node ID，按照源类名和最长方法名前缀精确匹配。
随后按源方法分组，以独立 pytest 子进程在指定物理卡上执行。
每个子进程的入口均为 `python -m pytest -vs`，不加载自定义插件，不注入 Python monkeypatch。
除选卡外，不强制设置编译线程、OMP 线程或 pytest 自动加载开关。

在装有上述环境的机器中执行：

```bash
source /home/l30023782/miniforge3/etc/profile.d/conda.sh
conda activate pta_213_case
cd /home/l30023782/PTA_213

ASCEND_RT_VISIBLE_DEVICES=6 python /path/to/torch-learn/scripts/tests/combo_kernel_210/run_plain_pytest.py \
  --test-root /path/to/Community-Use-Case-Testing/v2.13.0/inductor_npu \
  --manifest /path/to/torch-learn/scripts/tests/combo_kernel_210/combo_213_all_selected.json \
  --layout /path/to/torch-learn/scripts/tests/combo_kernel_210/combo_213_table_rows.json \
  --output /path/to/fresh-output-directory \
  --test-commit 3f5e543 --devices 6,1
```

`--representatives` 每个文件只选一个实际参数化实例；`--collect-only` 仅收集、不执行。
默认每个方法超时 3600 秒，可通过 `--timeout` 调整。输出目录必须不存在，以免覆盖历史证据。
选卡须遵守当前任务授权和环境约束，不应直接沿用示例中的物理卡号。

输出包括 `status.json`、`collection.json`、`nodeids.txt`、`jobs.json`、`results.csv`、逐方法日志和 JUnit XML。
`PASS` 要求该方法的全部预期实例都有报告且没有失败或跳过；报告缺失或异常退出记为 `ERROR`。
脚本还核对执行前后的 torch/torch_npu Inductor Python 源码哈希，防止混入安装包变动。
表格的“当前分析”保留实际异常摘要，不能把异常摘要直接当作根因结论。

本轮首批 62 个方法已在同一安装包上执行完成。扩展范围时使用
`--reuse /home/l30023782/PTA_213/combo-sync-64dc85c-20261110/results` 复用这些结果，只补跑 43 个方法。
复用前核对包提交、Triton 版本、安装源码哈希、对应测试文件内容和实际 node ID 集合；
任一项不一致即停止复用。原日志、XML 和命令一并保存，已知失败和跳过也如实保留。

## 本轮执行与自动收尾

完整范围快照及运行目录：`/home/l30023782/PTA_213/combo-sync-64dc85c-all116-20261110`。
`run.log` 记录调度结果，`results/status.json` 记录阶段，`results/` 保存逐方法证据。
本轮使用物理卡 6 和 1；卡号沿用当前任务授权范围，未使用 4、5 卡或告警的 2 卡。

本地开发机的用户级 systemd timer `combo-sync-213-all116-20261110.timer` 检查远端状态，
首次 10 分钟后检查，随后每 15 分钟检查一次。测试本身始终在 NPU_A5 上执行。
完成后自动下载证据、刷新独立的 2.13 表格并推送 torch-learn，最后停止 timer。
状态为 error 时同样保存证据和失败状态，不能把收集失败视为用例通过。

手动检查一次：

```bash
python scripts/tests/combo_kernel_210/publish_plain_pytest.py \
  --remote-root /home/l30023782/PTA_213/combo-sync-64dc85c-all116-20261110 \
  --name inductor-combo-tests-native-v2.13-sync-64dc85c-all116 \
  --table-name inductor-combo-tests-native-v2.13-sync-64dc85c \
  --push --stop-timer combo-sync-213-all116-20261110.timer
```

`--initial` 可以提前发布正在运行的状态表；日常定时检查不下载中间日志。
结果表为 `docs/notes/compiler/codegen/inductor-combo-tests-native-v2.13-sync-64dc85c.csv`，
完整范围证据保存在带 `-all116` 后缀的 `artifacts/` 子目录，首批证据和 2.10 结果表保持原样。
