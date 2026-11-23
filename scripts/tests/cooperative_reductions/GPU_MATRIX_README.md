# fp32 sum / amax 全形状扫描与图表刷新

`reduction_matrix.py` 测量手写 `(a+b).sum(-1)` / `(a+b).amax(-1)`，
默认 CUDA，不需要 torch_npu、NPU 设备或 PyTorch 源码仓库。
它使用固定的 PyTorch v2.13.0 `x_grid_barrier` 快照和 cooperative launch，
与 NPU origin 保持相同协议。它不是实际 Inductor 策略选择或 split 对照测试。

## 准备

需要 Python 3.10+、CUDA PyTorch，以及支持 `launch_cooperative_grid=True` 的 Triton。
已测 RTX 4090 环境为 PyTorch `2.13.0+cu126`、Triton `3.7.1`、CUDA 12.6。
其他版本先运行下面的小矩阵；不应删除 cooperative launch 选项来绕过启动失败。
绘图可在另一台无 GPU 的机器完成，需要 Matplotlib 3.8+。

便携包包含采集/绘图脚本、上游 helper 和许可证，以及已测 NPU 的 `npu/` 数据。
解压后在包目录执行以下命令。先检查设备 UUID、空闲显存和占用：

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

GPU 共 54 组形状、486 个候选；不会按 SM 数预先删除候选。
NPU origin/debug 各使用 RS=1/2/4/8/16/32/48/56，共 864 个候选；
NPU RS=64 已在代表点发生同步超时，不纳入全形状测速，相关诊断另存。
过大的 cooperative launch 可能失败，失败点保留原因，没有完成调用时不填设备耗时。
RS=1 也保留屏障，不能当作普通无屏障归约基线。

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
每点检查首次结果、最后一次计时输出及两组变化输入，再做 4 phase 的严格整数交换。
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

- `best-rs.csv`：NPU origin、NPU debug、当前 GPU 共 162 行，保留最佳 RS、耗时范围与候选状态计数。
- `best-rs-latency.svg` / `.png`：六个子图，分别比较两种归约、三个 X 下的延迟曲线。
- `best-rs-choice.svg` / `.png`：最佳 RS 分布，格中数字是实际分片数。

只看 GPU 可省略 `--npu`，得到 54 行；比较多台 GPU 可重复 `--cuda`：

```bash
python plot_reduction_matrix.py --npu npu/candidates.json \
  --cuda cuda-4090/candidates.json --cuda cuda-other/candidates.json --out figures
```

不要将不同 BLOCK 的扫描混合比较。缺候选标为 `INCOMPLETE`，没有正确候选标为 `NO_VALID_RS`，
均不伪造时间。完整扫描且有正确候选才标为 `PASS`。
CSV 中的 `over_core` 记录最佳候选的 `X*RS` 是否超过设备属性核数，不代表合法性保证。

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
