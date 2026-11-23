# Split 与 cooperative 的整图对照证据

2026-11-23，结论与生成代码解释见
[同步与策略报告](../../docs/requirements-design/cooperative-reduction/barrier-comparison.md#45-split-与-cooperative整图对照与分层组合)。
这组记录区分实际 CUDA Inductor 和 A5 手写匹配原型，不与其他目录的同步微基准混合。

## 环境与执行版本

| 项目 | CUDA | NPU |
| --- | --- | --- |
| 执行位置 | GPU_4090 的 `pytorch-dev-gpu67` | NPU_A5 Docker，`pta_213_case` |
| worktree | `/home/gaojg/dev/torch-learn-strategy-20261123` | `/home/l30023782/PTA_213/torch-learn-strategy-20261123` |
| 原始输出根目录 | `/mnt/raid1/gaojg/data/dev/cooperative-strategy-20261123` | `/home/l30023782/PTA_213/cooperative-results/strategy-20261123` |
| 实际设备 | RTX 4090 物理卡 6，128 SM | Ascend950PR_9579 物理卡 1，56 Vector 核 |
| 包版本 | torch `2.13.0+cu126`，Triton `3.7.1`，CUDA `12.6` | torch `2.13.0+cpu`，torch_npu `2.13.0+git93d7d01`，Triton `3.6.0` |
| PyTorch commit | `cf30153c4c131c8164ee7798e5022d810682e2cb` | 同版本 torch，沿用原有 TA/NPUIR 安装 |
| 设备筛选 | `CUDA_VISIBLE_DEVICES=GPU-528f4e79-58c0-e843-bf5e-0b40c2671cd7` | `ASCEND_RT_VISIBLE_DEVICES=1`，`NPU_DEVICE_LIMIT=28,56` |
| 主矩阵执行脚本 | `6925163` 的 `strategy_compare.py` | `e5356bd` 的 `npu_strategy_compare.py` |
| 组合实验执行脚本 | `2712491` 的 `strategy_compare.py` | 未做 NPU 分层组合 |

两卡均共享使用。NPU 卡 1 开始时 Health OK、利用率采样为 0%、一个约 516 MB 的驻留进程，
沿用本会话对 1/2/3 卡的共享授权。GPU 经 UUID 确认为物理卡 6，启动前利用率 0%、约 10.2 GiB 空闲。
不改安装包，不重置设备，不操作其他进程。

`2712491` 相对主矩阵脚本增加实际选中配置、worker 源码摘要和多 callable 的计算 kernel 计数修正，
不改变主矩阵计算或计时方法。主矩阵每个 case 只有一个 callable，原计数不受该修正影响。
NPU 主矩阵包含之前同脚本的 X=1/R=32768/S=8 三种计算的小范围检查结果；`--resume` 保留原值。
各 suite 的 `manifest.json` 记录命令、UTC 时间和实际脚本 SHA256；组合 case 另外保存 worker 摘要。
`results.json` 的 X/R 已转为整数；各 CUDA case 的原始 `result.json` 可能保留字符串，数值相同。

## 数据及方法

| 目录 | 内容 |
| --- | --- |
| `cuda/` | 四种计算 × X=1/4 × R=32768/1048576 × plain/split/cooperative，48 组全部 PASS |
| `cuda-mixed/` | `(a+b).sum(),(a-b).sum(-1)`，X=32/R=32768，三种策略全部 PASS |
| `cuda-hybrid/` | X=1，R=1048576/8388608，split/cooperative/显式 G=4 分层，六组全部 PASS |
| `npu/` | X=1，R=32768/1048576，sum/RMSNorm/softmax，S=1/2/4/8/16/32/56 的匹配原型，78 条全部 PASS |
| `npu-recheck-short/`、`npu-recheck-long/` | 针对两个噪声点的定向复测，每点 split/cooperative 共 4 条全部 PASS |

CUDA 开关为 plain 两者均关、split 只开 `split_reductions`、cooperative 两者均开但 force 为 False。
两种缓存和 `triton.multi_kernel` 关闭；保留 Inductor 正常的 kernel 候选选择。
开启 cooperative 不代表所有规约实际命中，以原始 decorator、barrier、wrapper 和选中配置为准。
NPU 原型是 SIMD、4 warps、BLOCK=1024，split 两计算 kernel，cooperative 一计算 kernel加 all_vector。
RMSNorm/softmax 两种 NPU 策略都会重新读输入；没有模拟 CUDA persistent 的数据保留。

各 CUDA case 独立进程、180 秒时限；预热 5 次，CUDA Graph 捕获 16 次完整调用，
再预热 3 次 replay，7 组 Event 样本，每样本 replay 20 次。NPU 每样本 replay 10 次，其余计时结构相同。
编译、eager 校验、CPU 回传、Python 调度和 CPU 内存分配不计时；CUDA compiled wrapper 的
semaphore 清零等捕获设备工作保留。它是完整图的设备序列耗时；非 Graph 的端到端延迟另待测量。
`generated_kernel_count` / `source_summary.run_calls` 指计算 kernel，不能据此断言只有一次设备发射。
Profiler 设备事件在正式计时后单独采集，不用于计算表中中位数。

输入连续 fp32、seed=1731。三组变化输入加计时后检查，参考为同设备 eager fp32，
非旧 sum 探针使用的 CPU fp64。sum/mixed 的 `atol=0.02,rtol=3e-4`；
RMSNorm/chain 为 `3e-5,3e-4`；softmax 为 `1e-8,3e-4`。
没有覆盖 NaN/Inf、尾块、动态、训练和其他 dtype；这些仍按需求单独验收。
CUDA 计时后另作一次 compiled 调用并检查，未单独保存最后一次 Graph replay 的输出；
NPU 使用预分配输出，计时后检查最后一次 replay 的输出。

每个 case 保留 `result.json`、实际 generated wrapper/DSL 或 TA TTIR/adapter；
`results.json` 汇总耗时、7 个原始样本及数值误差，NPU 同一 case 的文件含两个策略。
初始化时的 `active.json` 和重复打印 result 的终端日志不归档，编译缓存也不归档。
NPU metadata 记录实际 AutoBlockify 与编译参数，不能只用 `launch_cooperative_grid=True` 证明前进性。

两个 sum 点的原次样本波动较大，随后按相同脚本、设备、缓存和参数定向复测：

| R / S / 策略 | 原次中位数 µs | 原次 min～max µs | 复测中位数 µs | 复测 min～max µs |
| --- | ---: | --- | ---: | --- |
| 32768 / 56 / cooperative | 54.063 | 20.511～118.147 | 5.921 | 5.867～5.936 |
| 1048576 / 4 / split | 68.326 | 27.921～131.408 | 28.070 | 28.049～28.104 |

每次复测同时运行另一种策略，共 4 条，全部精度通过。原始数据不修改；绘图命令传入两个
`--npu-recheck`，在图和 CSV 对应位置使用最新记录，CSV 的 source 标明实际来源。
135 个独立配置由 57 个 CUDA 与 78 个 NPU 点组成；4 条复测原样另存。
最优配置和正文六行汇总不受这两个点影响。干扰来源未定位，不以原次高值解释固定屏障成本。

这是热数据、共享设备的选点实验；同图的 split 才是 cooperative 额外价值的主要基线。
普通 plain 很慢所形成的高倍率不等同于优于已有 split，也不能作为模型或生产阈值。
没有将 CUDA 的融合收益当作 NPU 集成收益。

## 重跑与刷新

命令与脚本参数统一维护在[脚本说明](../../scripts/tests/cooperative_reductions/README.md#split-与-cooperative-整图对照)。
新 GPU 仅需 `strategy_compare.py` 与匹配 CUDA PyTorch/Triton；离线绘图另需
`plot_strategy_compare.py`、`plot_barrier_suite.py` 与 Matplotlib。
图、CSV 位于 `docs/requirements-design/cooperative-reduction/assets/strategy-comparison/`。
使用新的结果路径运行绘图命令可覆盖同名 SVG/PNG/CSV，原始 JSON 不被修改。
