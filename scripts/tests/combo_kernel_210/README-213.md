# 2.13 ComboKernel 分支验证

本轮使用 NPU_A5 的 `pta_213_case` conda 环境，工作目录为 `/home/l30023782/PTA_213`。
目标安装包为 `torch_npu 2.13.0+git64dc85c`，对应 `sync/combo-kernel-master` 提交
`64dc85c99c1398a0bd2355b72cc2851d0688b964`。PyTorch 为 `2.13.0+cpu`，Triton 为 `3.6.0`。

用例源为测试仓库 `v2.13.0/inductor_npu`，初次完整范围快照提交为 `3f5e543`，结果已归档：
105 个独立方法为 1 PASS、70 FAIL、34 SKIP；464 个实例为 3 PASS、77 FAIL、384 SKIP。
安装包的 Inductor Python 源码在执行前后保持一致。

最新 foreach 功能验证快照为 `a889082`，按 2.10 的 NPU 验证规则修改：
取消本轮 27 个 foreach 方法的主动 NPU skip；NPU 上不检查 kernel 数量及 foreach 专用生成代码，
保留数值、原地修改和梯度断言，以及其他后端原有的融合检查。
功能 PASS 不证明 FOREACH 分组融合生效。其余不在本轮选择范围内的 skip 保持不变。
后续按用户要求先修复同步接口，安装包叠加修复的范围见下文。

执行范围以 `inductor-combo-tests-liulm-v2.10.0-combokernel-dev.csv` 的全部 116 行为准，
不沿用 2.10 的排除范围。`select_213_table.py` 按 2.13 源码解析为 105 个独立方法和 11 行嵌套函数；
嵌套函数随其外层方法验证，结果表保留原顺序和全部 116 行。

`run_plain_pytest.py` 使用 `combo_213_all_selected.json` 中的 105 个源方法，
`combo_213_table_rows.json` 记录全部表格行与外层测试的对应关系。
逐文件执行原生 pytest 收集，从输出中解析实际参数化 node ID，按照源类名和最长方法名前缀精确匹配。
随后按源方法分组，以独立 pytest 子进程在指定物理卡上执行。
每个子进程的入口均为 `python -m pytest -vs`，不加载自定义插件，不注入 Python monkeypatch。
不强制设置编译线程、OMP 线程或 pytest 自动加载开关。Python 修复后的复测另设独立编译缓存目录。

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
  --test-commit a889082 --devices 6,1 --foreach-functional \
  --npu-fix-commit 5a5bda0ae36a30a865ea52f7cd1df2a5a2ea532c \
  --cache-root /path/to/fresh-cache-directory
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

## 同步接口修复后重跑

用户要求先修复同步问题，原 foreach 单独复测已取消，对应
`combo-sync-213-foreach-functional-20261110.timer` 已停止，不发布为最终结果。
新运行目录为 `/home/l30023782/PTA_213/combo-sync-64dc85c-interface-fix-20261110`。

修复分支为 NPU 仓库的 `fix/combo-sync-interfaces-v2.13`，基于安装包对应的 `64dc85c99c`：

- `19845c38a3`：移除 Combo 入口对已不存在的 `node_can_linear()` 的调用条件，
  与当前 2.13 普通 kernel 调度一致，先尝试 index codegen，保留已有失败回退。
  2.10 的旧方法读取 `inductor_ascend_linear_mode`；当前 2.13 同样没有该配置，因此不能直接复制旧方法。
- `a79fb5eda3`：`generate_combo_kernel_code` 从四元 tuple 改为上游 `NodeInfo`。
  2.10 分区代码以 `[1]` 取 tiling，2.13 则读取 `.tiling` 和 `.features`，旧数据结构无法满足新接口。
  新 `NodeInfo` 复用创建 NPU 子 kernel 的 features、tiling 及实际 persistent-reduction 状态；
  取 node_schedule 也同步改为字段访问。
- `4b4176acbf`：补齐 2.13 单节点分区协议。正式 codegen 生成普通 NPU kernel，
  benchmark 返回占位值并由社区调用者复用节点计时，避免生成仅含一个子 kernel 的 Combo 和重复计数。
  所有分区验证 TTIR 通过后才统一 mark_run、更新图状态，保留现有失败回退的前置验证条件。
- `5a5bda0ae3`：单节点分区会丢弃预建的潜在 Combo 子 kernel，再创建普通 kernel；
  `Kernel.__init__` 已为前者增加计数，丢弃时需要对应减去一次。
  这与社区 `ComboKernel.create_sub_kernel` 移除子 kernel 独立计数的原则一致，避免把未生成的 kernel 计入指标。

只在 `pta_213_case` 安装目录覆盖 `npu_combined_scheduling.py` 和 `scheduling.py` 两个 Python 文件，
没有重新构建 wheel。覆盖前核对文件哈希与原完整测试快照一致，原文件保存在新运行目录的 `original/`，
`overlay.json` 记录安装位置、备份位置及修改前后哈希。
包版本仍显示 `2.13.0+git64dc85c`，不能仅凭版本字符串区分本次 Python 修复；结果额外记录叠加提交。

先运行原失败路径和 foreach 数值、梯度代表实例，再启动完整清单。
用户补充要求 foreach 全量前先验证 1–2 个实例。本轮暂停收集后的调度，使用最新叠加修复和
独立缓存，先执行普通 `_foreach_abs` 与 `foreach_map(abs)` 反向传播；两项都 PASS 才恢复完整任务。
前置验证证据为结果目录中的 `foreach-preflight.json`、`foreach-preflight.xml` 和日志，
这两个前置实例不额外加入完整清单的统计。可直接执行：

```bash
# 在 pta_213_case 中进入本轮 cases/v2.13.0/inductor_npu 目录
ASCEND_RT_VISIBLE_DEVICES=6 TORCHINDUCTOR_CACHE_DIR=/path/to/fresh-foreach-cache \
python -m pytest -vs \
  test_foreach.py::ForeachTests::test_single_list__foreach_abs \
  test_foreach.py::ForeachTests::test_foreach_map_backward_unary_foreach_map_abs
```

本轮两项前置验证均 PASS，耗时约 145 秒，安装源码哈希一致；随后已自动恢复完整任务。
恢复后，原收集进程报告物理卡 6 的设备子进程启动超时（E39007/507033），未执行任何测试实例。
该次证据保留在远端 `results-collection-failed/`。已核对安装源码不变，重新启动完整任务，
改在卡 1 收集清单，在卡 1、6 执行，使用新缓存 `cache-full-retry1`；继续沿用已通过的两项前置验证。
前置失败时保留日志并分析原因，不能直接放行全部 foreach。
全新缓存下，普通激活 Combo 与 benchmark 激活用例已通过，确认上述两条接口路径均可执行。
foreach 的带 mask 用例与 abs 梯度代表实例也已通过数值验证。
单节点两个实例已在新缓存下通过数值与数量断言：benchmark 关闭为 2、开启为 4；
生成的是普通 `GridNpu` kernel，源码中不含 `combo_grid_meta`。
用例仍因要求 CUDA 的 `Grid1D` 字面值而 FAIL，未为得到 PASS 修改该断言。
其余融合数量、内存峰值和后端专用代码断言保持原样，由完整复测如实记录。
完整清单不复用原包结果，仍使用原生 pytest。设置独立 `TORCHINDUCTOR_CACHE_DIR`：
Python 侧覆盖不改变包版本，旧 FX/代码缓存可能仍被命中，已在早期探测中观察到旧 Combo 代码；
不隔离缓存就不能把运行结果归到修复提交。不会清理其他任务的共享缓存。额外参数为：

```bash
--test-commit a889082 --npu-fix-commit 5a5bda0ae36a30a865ea52f7cd1df2a5a2ea532c \
--foreach-functional \
--cache-root /home/l30023782/PTA_213/combo-sync-64dc85c-interface-fix-20261110/cache-full-retry1
```

新本地 timer 为 `combo-sync-213-interface-fix-retry1-20261110.timer`，首次 10 分钟后检查，
随后每 15 分钟检查；测试在 NPU_A5 执行，最新分配顺序见下文。
完成后下载证据、更新同一份 116 行表并推送，停止 timer。手动检查命令：

```bash
python scripts/tests/combo_kernel_210/publish_plain_pytest.py \
  --remote-root /home/l30023782/PTA_213/combo-sync-64dc85c-interface-fix-20261110 \
  --name inductor-combo-tests-native-v2.13-sync-64dc85c-interface-fix \
  --table-name inductor-combo-tests-native-v2.13-sync-64dc85c \
  --push --stop-timer combo-sync-213-interface-fix-retry1-20261110.timer
```

## 运行中调整：先其他文件，最后 foreach

用户要求优先执行其他文件，并立即使用另外两张卡。已冻结原调度器发放任务，
保留卡 1、6 正在执行的两个 foreach 方法和此前完成的五个方法。
卡 2、3 单独执行其余文件的 78 个源方法、88 个实例，使用新缓存 `cache-non-foreach-first`。
正在执行的 foreach 结束后不再接续；其他文件全部结束后，再将剩余 foreach 分到卡 1、2、3、6，
使用新缓存 `cache-foreach-last`。所有阶段使用相同测试快照和安装源码，不重复计算前置探测。

`reorder_live_run.py` 负责这次运行的交接和聚合，原始结果仍写入同一 `results/`，
发布脚本、116 行表格和定时任务保持原入口。交接记录是运行根目录的 `handoff.json`，
原状态备份在 `reorder-checkpoint/`，独立阶段的清单、进程号、命令和日志在 `phases/`。
当前命令、阶段状态和清单也同步到总结果，随证据发布。

交接前只暂停原调度器父进程，不暂停 pytest 子进程；通过原 JUnit 参数定位正在执行的进程组。
交接器校验 PID、启动时间、工作目录及安装源码哈希，从仍由原父进程持有的 Linux zombie
进程读取真实退出状态，再结合 JUnit 判定结果。全部活动任务退出后才终止旧调度器，
防止旧队列恢复并继续发放 foreach。缺失报告和异常退出仍记为 ERROR。

该交接器用于上述已经准备好的运行目录，不是任意运行的通用恢复命令，不能重复启动。
在 `pta_213_case` 中进入该运行根目录后，由后台进程执行：

```bash
python -u reorder_live_run.py \
  --run-root /home/l30023782/PTA_213/combo-sync-64dc85c-interface-fix-20261110
```

## 优先验收 2.10 的 PASS 方法

以 `inductor-combo-tests-liulm-v2.10.0-combokernel-dev.csv` 中最新结果为“通过”的 60 个源方法为优先范围，
均已包含在当前完整清单。发布脚本自动生成 `baseline-pass-comparison.json`，并在当前 116 行表格的
“当前分析”标记非 PASS 的重合项，不增加列；PASS 项仍不保留历史失败说明。
旧基线去掉了部分 kernel 数量、CUDA DSL 和 profiler/grid 检查，不能把同名方法的 FAIL 一律视作实现回退。

已对两个重合的 reduction 方法做独立新缓存的原生 pytest 定向定位，安装源码未变。
日志确认均先分成一个包含两节点的 Combo，再因静态 BLOCK_SUB 未定义而回退；
其中 red 方法回退后又重复释放输入。接口同步遗漏、回退生命周期和测试适配差异分别记录在
[优先项分析](../../../docs/notes/compiler/codegen/inductor-combo-tests-v2.10-pass-v2.13-regressions.md)。
该定位运行不覆盖完整测试结果。

## 首次完整范围执行记录（已完成）

完整范围快照及运行目录：`/home/l30023782/PTA_213/combo-sync-64dc85c-all116-20261110`。
`run.log` 记录调度结果，`results/status.json` 记录阶段，`results/` 保存逐方法证据。
首轮使用物理卡 6 和 1，未使用 4、5 卡。
用户随后明确允许物理卡 1、2、3、6，也允许共享；查询状态异常时先实际运行一个小用例判断，
不能仅凭 Alarm 或状态查询结果排除该卡。当前两路长任务保留进度，2、3 卡另做小用例验证，
其结果记录在最新运行目录的 `results/device-probes/`，供后续任务分配参考。
实测卡 2 的 `_foreach_abs` 单例 PASS（约 134 秒），虽然查询显示 Alarm，仍可以执行。
卡 3 同一单例在 pytest 收集阶段达到 240 秒超时，尚未进入测试；不能据此归因于 ComboKernel。
随后卡 3 直接 NPU 小算例 PASS（约 28 秒，含设备初始化、张量创建、加法和数值比较），
说明基础设备运算可用，不能仅因前一次收集超时排除该卡。

此前本地开发机的用户级 systemd timer `combo-sync-213-all116-20261110.timer` 检查远端状态，
首次 10 分钟后检查，随后每 15 分钟检查一次。测试本身始终在 NPU_A5 上执行。
完成后已下载证据、刷新独立的 2.13 表格并推送 torch-learn，旧 timer 已停止。
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
