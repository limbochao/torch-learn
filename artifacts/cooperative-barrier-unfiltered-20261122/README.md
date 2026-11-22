# Cooperative reduction：解除两项过滤的设备实测

2026-11-22（北京时间）按用户要求，实际运行此前因核数预算跳过的 original 配置，
并对数值错误的 native/debug 配置继续计时。正文及曲线见
[同步验证第 6.4 节](../../docs/requirements-design/cooperative-reduction/barrier-comparison.md#64-非规约轴未满核与满核)。

## 数据入口与解释

- `npu/output-axis.json`、`cuda/output-axis.json` 是逐配置的最终结果；不是只包含通过点的性能表。
  NPU：100 点，64 PASS、24 WRONG、12 TIMEOUT，88 点计时；CUDA：54 点，40 PASS、14 启动失败。
  两端均无 SKIP，24 个 WRONG 全部保留计时。
- NPU 原始完整矩阵为 `npu/output-axis-initial-100.json`，最初为 62 PASS、24 WRONG、14 TIMEOUT。
  X=56/S=16 的两个 original 点已完成 sum 计时，但在后续整数交换阶段达到 60 秒总时限；
  `npu-original-s16-recheck-213/` 将时限改为 180 秒，两点均通过，sum 仍约 2.53 秒。
  最终结果只替换这两点，通过 `merge-provenance.json`、`rerun_source`、`previous_attempt_source`
  保留完整来源；该目录为复测原始记录，不额外增加矩阵配置数。
- `accuracy` 是 NaN workspace 初始化后的首次 sum 检查，`post_accuracy` 是计时后的检查。
  固定输入重复运行后通过，不能覆盖首次失败。`exchange` 是变化 phase/epoch 的严格整数检查。
- `median_us` 及 `samples_us` 测完整 sum kernel。数值错误或后续检查未完成时，已采集的耗时仍保留。
  `FAIL_TIMEOUT` 的墙钟时限包含子进程初始化、编译、执行及检查，不能换算为 kernel 耗时。
- 主进程结果是最终状态。被终止的 `cases/*/active-case.json` 保留最后执行阶段，可能仍为 `RUNNING`，
  应结合主结果的 `FAIL_TIMEOUT`、`worker_exit_code` 阅读。
- CUDA 规模超限在 `sum_launch` 抛出 `too many blocks in cooperative launch`，没有完成耗时；
  不将启动失败或未完成调用补成 0 µs。
- `manifest.json`、NPU 的 `manifest-run-*.json` 记录实际命令、软件栈、脚本及 helper 摘要。
  NPU 恢复执行前的记录通过 `run_manifest` 保留原版本；隔离点另有 `worker_manifest` 和 `worker_log`。
- 同一主进程中的 helper 复用地址；隔离子进程独立分配地址，沿用相同随机种子、shape 与算法。
- `ir/` 为运行时实际编译对象导出的 TTIR、adapter 或 GPU IR；不包含缓存二进制。
  `installed-ta-*.py` 是当前 TA 分核、编译与 launcher 的只读快照，供核对 atomic 黑名单与映射。

## 执行环境

NPU 使用 NPU_A5 的 Docker、`pta_213_case`、物理卡 5；device/stream 有效限额均为
28 Cube / 56 Vector，未设置 `NPU_DEVICE_LIMIT`。工作树位于
`/home/l30023782/PTA_213/torch-learn-cooperative-20261122`。
首段源代码为 `90d2475`，恢复执行为 `62512a8`，恢复时保留已完成的 19 条记录。
前一个超时点为 300 秒，后续每个隔离点为 60 秒；没有修改安装包或重置整卡。
`npu-timing-control-213/` 是矩阵结束后的两个 plain 复核点，关闭预测计时，仍为 32/10/7；
X=7 的短/长轴分别为 81.096/3025.420 µs。物理卡 5 与前轮卡 4 的绝对耗时差异尚未定位，
正文及 CSV 只用同轮相同 shape 的 plain 计算倍率，不混用历史基线。
本次引用的设备测试均在 `pta_213_case` 中完成，实际包版本以各 manifest 为准。
复核结束后物理卡 5 为 Health OK、无运行进程，见 `npu/physical-5-final-*.txt`。

GPU 仅使用 GPU_4090 的 `pytorch-dev-gpu67`，物理卡 6，UUID
`GPU-528f4e79-58c0-e843-bf5e-0b40c2671cd7`，容器逻辑卡 0。工作树位于
`/home/gaojg/dev/torch-learn-cooperative-20261122`，源代码为 `4cf4918`。
容器出现 NVML 初始化失败，按此前授权仅重启该容器，18:13:26 恢复；重启后实际 CUDA 计算通过。
采集于 18:17～18:20；进程退出码 1 来自已记录的 14 个启动失败，不表示 54 点矩阵中断。

## 复现命令

激活对应环境、按工作区规定检查设备占用并设置可见设备后，在容器工作树内执行：

```bash
# NPU；初始命令 case-timeout 为 300，恢复执行时缩短为 60
python -u scripts/tests/cooperative_reductions/output_axis_cases.py \
  --backend npu --physical-device 5 --rows auto \
  --allow-over-core --time-wrong-results --op-timeout-ms 5000 \
  --case-timeout 60 --max-sample-ms 500 --resume \
  --out artifacts/cooperative-barrier-unfiltered-20261122/npu

# NPU；延长两个已完成 sum、但未完成后续校验的点的时限
python -u scripts/tests/cooperative_reductions/output_axis_cases.py \
  --backend npu --physical-device 5 --rows 56 --splits 16 --only-kind original \
  --allow-over-core --time-wrong-results --op-timeout-ms 5000 \
  --case-timeout 180 --max-sample-ms 500 \
  --out artifacts/cooperative-barrier-unfiltered-20261122/npu-original-s16-recheck-213

# GPU；通过 UUID 选择物理卡 6 后运行
python -u scripts/tests/cooperative_reductions/output_axis_cases.py \
  --backend cuda --physical-device 6 --rows auto \
  --allow-over-core --time-wrong-results --case-timeout 120 \
  --out artifacts/cooperative-barrier-unfiltered-20261122/cuda

# 更新文档使用的同名 SVG、PNG、CSV
python scripts/tests/cooperative_reductions/plot_output_axis_cases.py \
  --npu artifacts/cooperative-barrier-unfiltered-20261122/npu/output-axis.json \
  --cuda artifacts/cooperative-barrier-unfiltered-20261122/cuda/output-axis.json \
  --out docs/requirements-design/cooperative-reduction/assets/barrier-comparison
```

`sha256.json` 对归档文件作摘要索引；`verification-summary.json` 保留矩阵完整性、
计时/状态、源码版本与 CSV 一致性检查结果。它们用于核对记录，不额外启动设备计算。
编译器 `.ttadapter` 和 npu-smi 输出保留原始行尾空格、尾部空行，不为通过格式检查改写证据；
手写文档和脚本单独通过空白检查。
