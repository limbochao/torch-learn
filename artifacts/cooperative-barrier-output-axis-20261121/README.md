# 输出并行度与 TA 同步接口：原始证据

分析、曲线和关键日志已内嵌在
[验证报告](../../docs/requirements-design/cooperative-reduction/barrier-comparison.md#64-非规约轴未满核与满核)。
本目录保存用于复核的完整数据，不归档整个编译缓存或设备二进制。

2026-11-21 在 NPU_A5 Docker 的 `pta_213_case` 环境执行，工作目录为
`/home/l30023782/PTA_213/torch-learn`。使用用户授权共享的物理卡 4；
起止检查 Health OK、AIV/Cube 利用率 0%，有 3 个其他驻留进程。
设备代码提交为 `1783f4534323269bd8e80444faaec538501520d5`。

同日 18:34～18:35 在 GPU_4090 的 `pytorch-dev-gpu67` 中补测，使用共享物理卡 6、RTX 4090
的 128 个 SM。PyTorch `2.13.0+cu126`、Triton `3.7.1`、CUDA `12.6`、驱动 `535.183.01`，
设备代码提交为 `94e7b6459e015476351ef1a98c0ad30c37d51ac3`。前后检查空闲显存约 10.5 GiB、
采样利用率 0%；UUID 与所选逻辑卡 0 一致。

| 数据 | 内容 |
| --- | --- |
| [output-axis.json](npu/output-axis.json) | 100 个配置：54 PASS 并计时、24 错误结果、22 original 超预算跳过 |
| [manifest](npu/manifest.json) | 命令、版本、设备、56 核、代码/helper SHA256、编译选项 |
| [运行日志](npu/output-axis.log) | 每项状态、计时、编译器告警；进程退出状态为 0 |
| [实际执行 IR](npu/ir/exchange-x56-s16-native.ttadapter) | 选定配置的 TTIR、adapter 与 metadata，包含通过点和错误点 |
| [S=2 最终 IR](npu/final-x56-s2/module.hivm.opt.mlir) | X=56 的通过配置，从实际 adapter 仅编译导出，循环步长 56 |
| [S=16 最终 IR](npu/final-x56-s16/module.hivm.opt.mlir) | X=56 的失败配置，同样保留循环内同步与步长 56 |
| [访问与选卡记录](environment-access.json) | NPU 选卡依据、共享授权和 GPU 无法访问的检查结果 |
| [GPU output-axis.json](cuda/output-axis.json) | 54 个配置：22 PASS 并计时、32 超 SM 预算跳过，无数值错误 |
| [GPU manifest](cuda/manifest.json) | 实际代码、128 SM、X=7/56/128、helper 摘要和 CUDA 编译选项 |
| [GPU 运行日志](cuda/output-axis.log) | 每项状态与计时；退出状态为 0 |
| [GPU 恢复后的设备检查](cuda/device-access-after-restart.json) | 设备可打开、cuInit=0、CUDA 可用及小计算结果 |

测试固定每行一个输出任务，以 X=7/56 区分未满核/满核，R=32768/1048593，
S=1/2/4/8/16/32/48/56。三种 helper 复用相同地址，另有不交换 partial 的普通归约对照。
54 个通过点均保留计时前后精度与 7 个 device Event 样本。
24 个失败点的 S 为 16/32/48，sum 出现 NaN、整数交换出错；未采集这些错误点的性能。
整个矩阵正常结束，不代表其中全部配置正确；不能只看进程退出状态。

GPU 复用相同 R/S/BLOCK、all-read 算法和 helper 版本，补充自身满核的 X=128。
22 个通过点保留计时前后精度及 154 个 device Event 样本；16 个 original 配置通过 4 phase
整数交换，共检查 4104 个值，误差为 0；sum 最大绝对误差为 `0.00249652`。
`physical-6-before/after.csv` 为容器内 NVML 快照，其 index=0，物理卡身份按 UUID 核对为宿主机卡 6。

两份最终 IR 导出 manifest 均为 `launched=false`，adapter/metadata 摘要与 `npu/ir/` 一致。
`npu/helper-*.py` 为实际调用函数的源码片段；`physical-4-before/after-*` 是设备状态快照。
完整复现命令集中维护在
[脚本 README](../../scripts/tests/cooperative_reductions/README.md#非规约轴未满核与满核)。

`ta-main-dev/` 保存官方 al 目录的全部 10 篇文档及同步 API 的 `core.py` 原始字节。
[版本索引](ta-main-dev/manifest.json)记录分支 `main-dev`、提交
`20ebbba9c695d5353488d3516d54f1d5f3c43310` 和各文件 SHA256。
[安装包 core.py](npu/installed-ta-core.py)用于核对实际环境也拒绝同类型 set/wait 端点。

GPU 补测前出现 NVML/CUDA 不可用，
[前次只读检查](gpu-access-final.txt)为 `Failed to initialize NVML: Unknown Error`。
18:26～18:28 的[进一步诊断](gpu-container-diagnostics.json)确认宿主机 GPU 正常，容器内普通用户
和 root 打开设备均返回 `EPERM`。目标 cgroup 的严格设备白名单未放行 NVIDIA 设备；对应 BPF
程序加载于当天 06:08:44，但触发操作未确认。经用户授权，18:33 仅重启指定容器后恢复，
见[重启记录](gpu-container-restart.json)及上表设备检查；没有调整宿主机配置。
此前 GPU 性能数据保留在相邻的 `cooperative-barrier-comparison-20261121/`，不与本轮数据混合。

`verification-summary.json` 汇总本地一致性核验；`sha256.json` 为本目录其余文件的摘要。
