# 一键测试 split / cooperative reduction

`one_click.sh` 在同一台 Linux GPU 或 NPU 机器上完成环境检查、源码冻结、测试、CSV/JSON 汇总、绘图、证据打包与离线 profiler 审计。无需连接原来的远端机器。相同的 199 组 eager forward、精度标准与 profiler device duration 口径随脚本一起携带。

## 环境准备

在已有 PyTorch/Triton 环境或容器中运行，Python 需要 3.10 及以上。本实验验证过 PyTorch 2.13；脚本会检查实际 Inductor cooperative/split、配置记录器、Graph 与 profiler 接口，随后用 smoke 验证真实编译和执行。NPU 环境还需 torch_npu、Ascend Triton 的 SIMD 编译与 `extension.sync_block_all("all_vector", 0)`，并能执行 `npu-smi`。

在 `torch-learn` 根目录安装绘图依赖：

```bash
python -m pip install -r scripts/tests/reduction_strategy_pk/requirements-plot.txt
```

也可使用 `--no-plot` 先导出 CSV/JSON，再在有 matplotlib/numpy 的机器绘图。用 `PYTHON=/绝对路径/python bash ...` 可指定 Python；已激活 conda 时直接使用下面的命令。

## GPU

CUDA 卡号是**当前可见设备的逻辑索引**，受已有 `CUDA_VISIBLE_DEVICES` 影响；完整 `GPU-...` UUID 也可使用。程序会解析 UUID 并在实际 worker 中再次核验，所有策略按同一个 case 在同卡测量。

```bash
# 查看当前可见 GPU；不会执行测试 forward
bash scripts/tests/reduction_strategy_pk/one_click.sh --backend cuda --list-devices

# 首次使用：四个代表 forward，保留完整三会话 profiler 确认
bash scripts/tests/reduction_strategy_pk/one_click.sh \
  --backend cuda --devices 0 --smoke --out "$HOME/torch-learn-runs/reduction-strategy-pk/my-gpu-smoke"

# 完整 199 组，先自动执行独立的四用例兼容检查，再运行正式矩阵
bash scripts/tests/reduction_strategy_pk/one_click.sh \
  --backend cuda --devices 0,1 --out "$HOME/torch-learn-runs/reduction-strategy-pk/my-gpu-full"
```

默认各 GPU 都参与完整队列，allocator 上限为每卡总显存的 50%。某张卡剩余显存较少时，将它列为小用例卡：

```bash
bash scripts/tests/reduction_strategy_pk/one_click.sh \
  --backend cuda --devices 6,7 --small-devices 7 \
  --gpu-memory-fraction 0.20 --small-memory-fraction 0.035 \
  --out "$HOME/torch-learn-runs/reduction-strategy-pk/my-gpu-shared"
```

这样卡 7 只承担 `X*R<=1048576` 的用例，allocator 上限为 3.5%；卡 6 承担大用例。这些索引仍以当前可见设备为准。预检检查输入、Graph 保留输出和 scratch 的估算；显存上限和估算都不等于资源预留，实际 OOM/执行失败会记录在对应 case 中。

## NPU

NPU `--devices` 使用**物理卡号**，并通过 ACL 检查逻辑 0 到该物理卡的映射。已有 `ASCEND_RT_VISIBLE_DEVICES` 时，选择器必须在其范围内。先激活包含 torch_npu/Ascend Triton 的环境并加载对应 CANN 环境。

```bash
# 检查指定物理卡的环境和有效核数
bash scripts/tests/reduction_strategy_pk/one_click.sh \
  --backend npu --devices 2,3 --check-only --no-plot --out "$HOME/torch-learn-runs/reduction-strategy-pk/my-npu-check"

# 四用例 smoke
bash scripts/tests/reduction_strategy_pk/one_click.sh \
  --backend npu --devices 2 --smoke --out "$HOME/torch-learn-runs/reduction-strategy-pk/my-npu-smoke"

# 完整矩阵，多卡各自串行执行 case
bash scripts/tests/reduction_strategy_pk/one_click.sh \
  --backend npu --devices 2,3 --out "$HOME/torch-learn-runs/reduction-strategy-pk/my-npu-full"
```

有效 Vector 预算默认取所选卡的最小有效核数，也可显式限制：

```bash
bash scripts/tests/reduction_strategy_pk/one_click.sh \
  --backend npu --devices 2 --npu-vector-cores 56 \
  --out "$HOME/torch-learn-runs/reduction-strategy-pk/my-npu-56core"
```

预算须为大于等于 2 的偶数，且不得超过任何所选卡的实际有效核数。torch_npu 要求 `Vector = 2 * Cube`；例如 `--npu-vector-cores 14` 会仅在子进程环境中设置 `NPU_DEVICE_LIMIT=7,14`，重新检查预算是否生效。cooperative 的 S≤预算且 X*S≤预算；split 允许至 2 倍预算的分片，并搜索交错分片和独立输出并行度。56 核时搜索集合与原实验相同。X*2 超出预算的 cooperative 记录为 NOT_APPLICABLE，plain/split 仍会运行。

NPU cooperative 保持 `tl.debug_barrier()` → `extension.sync_block_all("all_vector", 0)` → `tl.debug_barrier()`。支持该接口、编译通过、功能检查通过和生产同步契约属于不同验证范围；新 SoC 或编译模式必须先通过实际 smoke。脚本中的 split/cooperative 是 SIMD DSL 原型，plain 是实际 `torch.compile` 基线。

每个新任务取队列前查询 HBM 占用；超过 95% 时暂缓，默认最长等待 900 秒。可用 `--max-hbm-percent` 和 `--resource-wait-timeout` 调整。`--timeout` 是每个 case 的无进展期限，默认 900 秒，不是整个矩阵的总耗时。

## 结果与复现参数

指定 `--out` 后，结果统一保存在该目录：

| 文件/目录 | 用途 |
| --- | --- |
| `RUN_SUMMARY.md` | 结果入口、胜负统计、设备、CSV 与图表链接 |
| `figures/results.csv`、`results.json` | 每个 case 的所有策略性能、状态、配置及来源 |
| `figures/summary.json` | 严格配对、plain 对照、natural 对照与 noisy 计数 |
| `figures/*.svg`、`*.png` | 策略散点、R 扫描、矩阵图；使用 `--no-plot` 时不生成 |
| `run/cases/<id>/` | 生成代码、IR、精度检查和原始 profiler |
| `source/`、`source-sha256.json` | 本次冻结的运行源码及配置记录器 |
| `invocation.json`、`preflight/` | 参数、实际 UUID/物理卡、有效核数、版本与安装源码摘要 |
| `input-fingerprints.json` | 当前用例对应的 CPU 输入字节 SHA256，覆盖 213/214/215 三个种子 |
| `evidence.tar.gz`、`.sha256` | 去掉可重建缓存与重复二进制后的证据包及摘要 |
| `evidence-verification.json` | 从 trace/CSV 独立重算确认和 search/refine 的审计 |
| `cuda-launch-verification.json` | CUDA 实际 kernel 名称、thread block 与 cooperative grid 检查 |
| `completion.json` | 是否完整、预期用例数、各阶段退出码 |

未指定 `--out` 时，仓库入口与独立包均默认写入 `$HOME/torch-learn-runs/reduction-strategy-pk/<backend>-<时间>/`。`--out` 可以指定其他项目外的目录，`~` 会展开为用户主目录；指定 torch-learn 内的目录会在初始化设备前报错。新输出目录必须为空。脚本保留原始失败，不将无效策略计入胜负；退出码 0 表示完整测试与审计通过，2 表示测试或核验不完整，1 表示入口/环境错误，130 表示中断。

torch-learn 只发布最终结果、图表、测试脚本及说明。原始 profiler、生成代码/IR、日志、冻结源码与证据包均留在项目外；需要发布结果时，只复制 `figures/` 中的最终 CSV/JSON 和 PNG/SVG。

指标仅为 profiler 中完整 forward 的必要设备操作 duration 之和，含 CUDA semaphore 初始化；编译、warmup、CPU duration 和 kernel 间空隙不计入性能。默认保持 3 个独立确认会话，每会话 3 批，每批 5 次 forward；`X*R>33554432` 时为 3 次。smoke 只减少用例和 NPU 搜索，不能作为完整矩阵结果。

R 扫描只显示实际采集的 X，图例覆盖已有策略；未采集的算子明确标注没有 PASS 样本。仅含 ordinary reduction 的定向运行会在严格对照图中标注没有严格配对，实际时间仍保留在 CSV 和 R 扫描中。

默认**编译、预热、Graph capture/replay 与校验全程 no_grad**，减少 grad context 改变造成的再编译差异。旧实验使用初始 no_grad 编译和后续默认 context；要按该口径复跑可加入 `--legacy-grad-context`。CSV 中 `initial_compile` 与 `dsl_launch` 标注配置来源阶段；启动核验无法证明所有内部 R tile。

## 中断续跑、定向重跑与离线绘图

使用同一目录、卡、核数和测量参数，再加入 `--resume`。冻结源码及安装模块摘要须保持一致。已完成的 PASS 或 ERROR 终态不会重测；被中断/未完成的 case 会先将旧目录保存在 `run/interruptions/`，再从头执行。目录锁和实际进程检查防止两个协调器同时操作同一输出。

```bash
bash scripts/tests/reduction_strategy_pk/one_click.sh \
  --backend cuda --devices 0,1 --out "$HOME/torch-learn-runs/reduction-strategy-pk/my-gpu-full" --resume

# 新目录重新执行明确失败的用例；保持原完整矩阵的核数和测量参数
bash scripts/tests/reduction_strategy_pk/one_click.sh \
  --backend npu --devices 2 --case-ids sum-x1-r1048576-float32-contiguous \
  --skip-smoke --out "$HOME/torch-learn-runs/reduction-strategy-pk/my-npu-retry"

python scripts/tests/reduction_strategy_pk/summarize.py \
  --input "$HOME/torch-learn-runs/reduction-strategy-pk/my-npu-full/run" \
  --replace-failed "$HOME/torch-learn-runs/reduction-strategy-pk/my-npu-retry/run" \
  --out "$HOME/torch-learn-runs/reduction-strategy-pk/my-npu-merged"
```

可以用 `--max-elements 1048576` 明确限制工作集；该模式会记录减少后的用例数。限制也适用于兼容 smoke，必要时缩小其 X/R。`--plan-only` 离线导出将要执行的 case 与兼容 smoke 清单，不初始化 PyTorch 或设备。`--skip-smoke` 跳过完整矩阵前的兼容检查，适合已经验证过相同源码与环境时使用。

在另一台有绘图依赖的机器重算汇总，或合并一组 CUDA 与一组 NPU 的结果：

```bash
python scripts/tests/reduction_strategy_pk/summarize.py \
  --input /data/my-gpu-full/run --input /data/my-npu-full/run --out /data/comparison
```

不同 GPU 机器的同 backend 矩阵应分别汇总，保留各自设备与来源。比较前对齐源码摘要、case 集、有效核数、grad context 和 Graph 提交参数，并核对 `input-fingerprints.json` 中 `records` 的实际输入摘要；不把不同机器的同一 case 当成一条记录覆盖。

## 独立包搬到其他机器

在仓库里构建一个只含测试源码、配置记录器、说明和绘图依赖清单的包：

```bash
python scripts/tests/reduction_strategy_pk/pack.py \
  --out scripts/tests/reduction_strategy_pk/dist/reduction-strategy-pk.tar.gz
```

将 `.tar.gz` 和 `.tar.gz.sha256` 复制到目标 Linux 机器，在同一目录执行：

```bash
sha256sum -c reduction-strategy-pk.tar.gz.sha256
tar -xzf reduction-strategy-pk.tar.gz
cd reduction-strategy-pk
python -m pip install -r requirements-plot.txt

bash one_click.sh --backend cuda --devices 0 --smoke
# 或
bash one_click.sh --backend npu --devices 2 --smoke

# 正式测试
bash one_click.sh --backend cuda --devices 0 --out "$HOME/torch-learn-runs/reduction-strategy-pk/gpu-full"
# 或
bash one_click.sh --backend npu --devices 2 --out "$HOME/torch-learn-runs/reduction-strategy-pk/npu-full"
```

独立包不依赖完整仓库、历史证据或原 SSH 别名。`bundle_manifest.json` 记录每个实际源文件的 SHA256；未提交修改也包含在包里。`dist/` 中的包仅含测试源码与说明。最终结果可从 `figures/` 回收；完整运行目录及 `evidence.tar.gz` 应回收到项目外的数据目录。
