# NPU cooperative reduction 调研证据

对应[需求调研](../../docs/requirements-design/npu-cooperative-reductions-research.md)，
执行入口为 [probe.py](../../scripts/tests/cooperative_reductions/probe.py)。
以下是实验原始产物和派生的版本核对信息，不作为独立可执行脚本维护。
generated Python、TTIR 和 adapter 未做格式重排。

环境为 NPU_A5 Docker 内 `pta_213_case`，日期 2026-11-18，北京时间。
使用经用户授权共享的物理卡 2；`device-map.json` 通过 CANN API 核对 user device 0 → physical 2。
未替换安装包，未修改业务仓库，未运行 CUDA semaphore 自旋协议。

| 目录/文件 | 含义 |
| --- | --- |
| `inspect/stack.json` | package metadata、version.py、安装源码 SHA256；PATH 编译器也记录在内 |
| `inspect/actual-npuir-compiler.json` | Triton 实际使用的包内 BiShengIR 2.0.0，不是 PATH 上的 1.2.0 |
| `inspect/installed-source-excerpts.txt` | 安装代码片段及原始行号 |
| `inspect/installed-backend-contracts.txt` | 实际后端的编译器选择、launch 与锁相关片段及 SHA256 |
| `inspect/source-comparison.json` | 安装文件与本地源码基线的 SHA256 对比，不只依赖版本字符串 |
| `inspect/runtime-symbols.json` | helper 可见性：继承上游 x_grid_barrier，缺少 cooperative heuristic |
| `inspect/effective-cores.json` | 无 NPU_DEVICE_LIMIT；Inductor/Triton 均查询到 56 Vector core |
| `compile-fixed/` | 最终固定 4 warps 的编译矩阵、完整错误、成功/失败前的中间 IR；均未 launch |
| `compile-simt-exact/` | 按实际型号 Ascend950PR_9579、4 warps 复核 pure SIMT；三项错误类型相同 |
| `compile/` | 初次默认 32 warps 的编译记录；不能据默认线程数错误判定同步能力 |
| `inductor/` | 静态 `(a+b).sum(-1)` 的 8 个配置结果、实际 selected config、generated code 与错误 |
| `dynamic-reuse/` | 动态 R 同一 callable 复用；复用采集不 reset Dynamo；2 个普通路径通过，force 失败 |
| `index-fallback/`、`index-fallback.log` | 定向诊断：Index 的 KeyError R → no-linear decorator 缺失 |
| `bench-warm/` | 最终原型计时；额外预热 graph replay；21 条正确性与计时记录 |
| `bench/` | 首轮原型记录；用于观察跨轮波动；不能和最终数据混成独占卡统计 |
| `validation-provenance.json` | 各阶段实际脚本提交；每个子目录的 command.json 保留命令参数 |

最终编译矩阵 9 项中 6 项通过、3 项失败。两项 native sync 在 pure SIMT 明确报
`hivm.hir.sync_block` legalization 失败；semaphore 在 pure SIMT 报
`MaxThreads out of range`，未进一步归因。
静态 Inductor 8 项中 5 项通过、3 项失败；最终动态 3 项中 2 项通过、1 项失败。
这些是受测配置的结果，不是完整功能通过率。

手写原型固定 fp32、SIMD、B=1024，没有 autotune。性能测试在共享卡与热数据下进行，
保留五组样本，不能据此确定生产阈值或 HBM 峰值。
native visibility 的 1/2/8 program 各覆盖 8 phase；sum 原型覆盖多组、尾块及 NPU Graph。
没有在运行时故意发射非法 Grid 或可能无限等待的 semaphore kernel。
