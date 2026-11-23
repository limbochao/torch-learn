# fp32 sum / amax 全形状扫描与图表刷新

`reduction_matrix.py` 测量手写 `(a+b).sum(-1)` / `(a+b).amax(-1)`，
默认 CUDA，不需要 torch_npu、NPU 设备或 PyTorch 源码仓库。
默认同时测普通归约 `plain` 和 cooperative：后者使用固定的 PyTorch v2.13.0
`x_grid_barrier` 快照和 cooperative launch，与 NPU origin 保持相同协议。
plain 每个 program 完成一行，关闭 cooperative launch，不读写 partial workspace 或 semaphore，
不调用跨核屏障。两者使用相同的 fp32 加法、循环累加、BLOCK 和 warps。
本脚本不是实际 Inductor 策略选择或 split 对照测试。

实际 `torch.compile` 与 NPU DSL 联合搜索 RS/BLOCK 的新入口为 `run_matrix.sh`，支持多卡一键执行，
见[多卡一键测试](COMPILE_MATRIX_README.md)。本文命令及旧便携包继续对应固定 BLOCK 的机制实验。
当前文档主矩阵已用用户回传的 L20 替换 4090；采集方法保持本文所述口径。

## 准备

需要 Python 3.10+、CUDA PyTorch，以及支持 `launch_cooperative_grid=True` 的 Triton。
已测 RTX 4090 环境为 PyTorch `2.13.0+cu126`、Triton `3.7.1`、CUDA 12.6。
其他版本先运行下面的小矩阵；不应删除 cooperative launch 选项来绕过启动失败。
绘图可在另一台无 GPU 的机器完成，需要 Matplotlib 3.8+。

便携包包含采集/绘图脚本、上游 helper 和许可证，以及已测 NPU plain/origin/debug 的 `npu/` 数据。
解压后在包目录执行以下命令：

```bash
tar -xzf cooperative-reduction-matrix-gpu.tar.gz
cd cooperative-reduction-matrix-gpu
```

先检查设备 UUID、空闲显存和占用：

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.free,utilization.gpu --format=csv
CUDA_VISIBLE_DEVICES=GPU-实际UUID python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
print(torch.cuda.get_device_properties(0))
PY
```

在 GPU_4090 环境只能进入 `pytorch-dev-gpu67`，工作目录为 `/home/gaojg/dev`，优先物理卡 6；
其他机器根据该机器的使用约定选择设备。`--physical-device` 仅记录 CUDA 物理编号，
实际选卡由 `CUDA_VISIBLE_DEVICES` 控制；脚本只使用重排后的逻辑卡 0，manifest 保留 UUID。

## 采集

先做独立小矩阵，避免其目录与正式数据混用：

```bash
CUDA_VISIBLE_DEVICES=GPU-实际UUID python -u reduction_matrix.py \
  --rows 1,56 --rnumels 4096 --splits 1,8,64 --out cuda-smoke
```

正式运行：

```bash
CUDA_VISIBLE_DEVICES=GPU-实际UUID python -u reduction_matrix.py --out cuda
```

默认范围如下：

| 项目 | 默认值 |
| --- | --- |
| 计算与类型 | fp32 `(a+b).sum(-1)`、`(a+b).amax(-1)` |
| 输出行数 X | 1、7、56 |
| 每行元素数 R | 4096、8192、16384、32768、65536、131072、262144、524288、1048576 |
| 每行分片数 RS | 1、2、4、8、16、32、48、56、64 |
| kernel 配置 | BLOCK=1024、4 warps，all-read 合并，仅 shard 0 写输出 |

GPU 共 54 组形状，54 个 plain 加 486 个 cooperative 候选；不会按 SM 数预先删除候选。
NPU origin/debug 各使用 RS=1/2/4/8/16/32/48/56，共 864 个候选；
加 54 个 plain，合计 918 个。plain 固定 RS=1，每行一个 program，实际 Grid 为 `(X,)`。
NPU RS=64 已在代表点发生同步超时，不纳入全形状测速，相关诊断另存。
过大的 cooperative launch 可能失败，失败点保留原因，没有完成调用时不填设备耗时。
origin/debug 的 RS=1 仍保留屏障；收益基线必须读取独立的 plain 记录。

已有 cooperative 数据时，可以只补测普通归约，输出到新目录：

```bash
CUDA_VISIBLE_DEVICES=GPU-实际UUID python -u reduction_matrix.py --kinds plain --out cuda-plain
```

只测原有 cooperative 则使用 `--kinds original`。NPU 可选 `plain,original,debug`，
例如 `--kinds plain` 只补测 NPU 基线；选卡、限额和容器环境仍需按工作区约定设置。

中断后使用同一设备和配置续跑：

```bash
CUDA_VISIBLE_DEVICES=GPU-实际UUID python -u reduction_matrix.py --out cuda --resume
```

`--resume` 跳过已有候选，包括失败和超时点；重测必须换输出目录。
每个形状/helper 放在隔离子进程中，异常后重新建 context；只终止脚本自己创建的进程组。
默认无进展执行时限为 60 秒，冷编译/初始化为 180 秒，可按机器调整
`--point-timeout`、`--init-timeout`。超时是看护结果，不是 kernel 延迟。
进程退出码 0 表示扫描结束，是否通过需看逐点状态和最终汇总。

## 测量口径与校验

计时包含输入加载、加法、局部归约、partial 写读、同步和最终输出，排除编译、CPU 参考计算和结果回传。
使用 CUDA Graph + device Event，默认每图 16 次调用、每样本 10 次 replay、7 个样本取中位数。
单次 pilot 估计每样本工作量，目标不超过 50 ms，慢点自动降低 batch/replays，实际值保留在 JSON。
首次预热 5 次、Graph 预热 3 次；pilot 达到 5 ms 时分别降为 1 次。
至少仍测一次，单次调用超过目标时不会截断该调用。

输入为固定种子 CPU 生成的 fp32 随机数；CPU 先做 fp32 加法，再做 fp64 归约作为参考。
每点检查首次结果、最后一次计时输出及两组变化输入。
cooperative 额外做 4 phase 的严格整数交换；plain 无跨 program 通信，不执行该项。
sum 容限为 `atol=0.02,rtol=2e-4`，amax 要求精确一致。
变化输入检查保留已有 workspace/semaphore，整数交换还引入不同分片工作量。
这些检查通过仅代表本矩阵验证通过，不代替特殊值、动态 shape 或并发验收。

## 重建 CSV / SVG

在绘图机器安装依赖，再读取 GPU 输出和包内 NPU 数据：

```bash
python -m pip install 'matplotlib>=3.8'
python plot_reduction_matrix.py \
  --npu npu/candidates.json --cuda cuda/candidates.json --out figures
```

生成下列文件，每个形状/实现只选择全部检查通过且中位耗时最小的 RS：

- `best-rs.csv`：NPU plain/origin/debug、GPU plain/original 共 270 行，保留耗时、最佳 RS、物理卡、
  候选状态计数和 `speedup_vs_plain`。plain 的 RS=1 表示不分片。
- `best-rs-latency.svg` / `.png`：六个子图，五条延迟曲线；虚线为 plain。
- `best-rs-choice.svg` / `.png`：三种 cooperative 实现的最佳 RS 分布，省略不分片的 plain。
- `cooperative-speedup.svg` / `.png`：plain 耗时 / 最佳正确 cooperative 耗时，大于 1 表示受益。

只看 GPU 可省略 `--npu`，得到 108 行；比较多台 GPU 可重复 `--cuda`：

```bash
python plot_reduction_matrix.py --npu npu/candidates.json \
  --cuda cuda-4090/candidates.json --cuda cuda-other/candidates.json --out figures
```

如果只补测了 plain，可将新旧数据分别传入，无需手动合并 JSON：

```bash
python plot_reduction_matrix.py --npu npu/candidates.json \
  --cuda cuda-old/candidates.json --cuda-plain cuda-plain/candidates.json --out figures
```

NPU 对应参数为 `--npu-plain`；多台 GPU 的 `--cuda-plain` 顺序必须与 `--cuda` 一致。
完整扫描已经含 plain 时不要重复添加。加速比只在同一物理卡和相同 UUID 的对应结果间生成；
来自不同采集时段的共享卡结果仍可能受占用、频率和缓存状态影响。

不要将不同 BLOCK 的扫描混合比较。缺候选标为 `INCOMPLETE`，没有正确候选标为 `NO_VALID_RS`，
均不伪造时间。完整扫描且有正确候选才标为 `PASS`。
CSV 中的 `over_core` 记录最佳候选的 `X*RS` 是否超过设备属性核数，不代表合法性保证。
各形状的物理卡以 CSV 的 `physical_device` 及原始 manifest 为准，不将不同卡的绝对耗时配成收益。

在完整 torch-learn 仓库内刷新已有文档图表，将脚本路径加上
`scripts/tests/cooperative_reductions/`，输出改为
`docs/requirements-design/cooperative-reduction/assets/reduction-matrix`；文档会读取同名 SVG。
Markdown 中的代表性数值与结论需随新设备数据另行核对。

## 保留与回传

保留整个 `cuda/` 目录中的 JSON、helper 源码；`candidates.json` 与相邻的 `run.json`
足以重建图表，逐形状 `manifest.json` 提供设备、版本、参数和 helper SHA256。
出错时保留对应 `worker-*.log`。`triton_cache/` 不需要回传。
原始候选不裁剪，`best-rs.csv` 仅保留获选结果，便于复核“最佳”的来源。

固定 helper 的上游提交为 `cf30153c4c131c8164ee7798e5022d810682e2cb`，
函数 SHA256 为 `c9fa8ee90b54adc67a11ba5406b5fe4be55178b974cc26a2bde8c2c9f19f5bb3`。
`bundle_manifest.json` 记录便携包的来源提交与文件摘要。
