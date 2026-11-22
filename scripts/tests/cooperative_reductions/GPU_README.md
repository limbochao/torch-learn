# GPU 输出并行度补测与 SVG 更新

入口为 `output_axis_cases.py`，默认 CUDA；与 NPU 共用 `barrier_suite.py` 的 kernel 和计时算法。
需要可用的 CUDA PyTorch、支持 `launch_cooperative_grid=True` 的 Triton，以及 Python 3.10+。
此前 CUDA 对照实际使用 PyTorch `2.13.0+cu126`、Triton `3.7.1`；其他版本的设备运行尚待验证。
绘图另需 Matplotlib 3.8+，可在没有 GPU 的机器上运行。

## GPU_4090 指定环境运行

使用工作区约定的 `pytorch-dev-gpu67`，优先物理卡 6；其 UUID 为
`GPU-528f4e79-58c0-e843-bf5e-0b40c2671cd7`，启动前以实际查询结果复核。
先按工作区流程同步代码，确认版本包含 `90d2475` 的解除过滤与诊断计时入口，再进入容器工作目录：

```bash
ssh GPU_4090
docker exec -it \
  -e CUDA_VISIBLE_DEVICES=GPU-528f4e79-58c0-e843-bf5e-0b40c2671cd7 \
  -w /home/gaojg/dev/torch-learn pytorch-dev-gpu67 bash
```

下面命令均在该容器中执行。先检查物理编号、UUID、空闲显存和利用率，并确认 CUDA 能识别卡：

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.free,utilization.gpu --format=csv
python - <<'PY'
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
print(torch.cuda.get_device_properties(0))
PY
git rev-parse HEAD
```

CUDA 可用后运行完整补测；`--rows auto` 会在 RTX 4090 上选择 X=7/56/128：

```bash
python -u scripts/tests/cooperative_reductions/output_axis_cases.py \
  --backend cuda --physical-device 6 --rows auto \
  --rnumels 32768,1048593 --splits 1,2,4,8,16,32,48,56 \
  --allow-over-core --time-wrong-results --case-timeout 120 \
  --out artifacts/cooperative-barrier-unfiltered-20261122/cuda
```

将该 `cuda/` 目录的 JSON 和 helper 源码取回本地同一目录后，在 torch-learn 根目录刷新现有图：

```bash
python scripts/tests/cooperative_reductions/plot_output_axis_cases.py \
  --npu artifacts/cooperative-barrier-unfiltered-20261122/npu/output-axis.json \
  --cuda artifacts/cooperative-barrier-unfiltered-20261122/cuda/output-axis.json \
  --out docs/requirements-design/cooperative-reduction/assets/barrier-comparison
```

该命令覆盖文档引用的同名 SVG/PNG/CSV；完整环境、通过/跳过数量与性能结论按实际结果更新报告。
若 NVML/CUDA 初始化失败，则只记录访问错误，不能将其记为已经完成 kernel 测试或生成性能值。

## 便携包运行

解压后在包目录内执行；GPU 测试不需要 torch_npu、TA 或 PyTorch 源码仓库。
先查询目标容器内可见设备，选择有空闲资源的 UUID：

```bash
tar -xzf cooperative-gpu-output-axis.tar.gz
cd cooperative-gpu-output-axis
nvidia-smi --query-gpu=index,uuid,name,memory.free,utilization.gpu --format=csv
```

将下面 `GPU-实际UUID` 替换为目标卡 UUID。脚本使用筛选后的逻辑卡 0，记录实际设备名、UUID、SM 数、
版本及源码 SHA256；`--physical-device` 在 CUDA 上只是可选记录，不负责选卡。
在已有的 GPU_4090 上仍须遵循工作区约定，仅用 `pytorch-dev-gpu67`、优先物理卡 6。

```bash
CUDA_VISIBLE_DEVICES=GPU-实际UUID python output_axis_cases.py \
  --allow-over-core --time-wrong-results --out cuda
```

默认包含以下场景，重复的 X 会去重：

| X | 作用 |
| --- | --- |
| 7 | 与现有 NPU 的未满核 shape 直接对照 |
| 56 | 与现有 NPU 的满核 shape 直接对照；在 GPU 上是否满核按实际 SM 数标注 |
| 当前 GPU 的 SM 数 | GPU 自身的满核对照，例如 128 SM 的 GPU 使用 X=128 |

每种 X 均测试 fp32 `(a+b).sum(-1)`，R=32768/1048593，S=1/2/4/8/16/32/48/56，BLOCK=1024。
仅运行原始协议 `original` 与无 workspace/屏障的 `plain`。S=1 的 original 仍调用 barrier，
应使用 plain 作为普通归约基线。`--allow-over-core` 放开 `X*S<=SM数` 的保守过滤；
超预算 original 逐点使用独立进程，保留启动异常、执行阶段和超时。超 SM 数不等于必然无法启动；
实际 cooperative launch 还受该 kernel 可同时驻留的 block 数约束。
未传该开关时，超预算配置仍记为 `SKIP_CORE_BUDGET`，不启动。
这些为手写 kernel 对照，并不表示上游 Inductor 会自动选择所有 shape/S。

每点先校验 fp32 sum，再用 CUDA Graph + device Event 计时，最后做 4 phase 整数交换检查：
每图 32 次调用、每样本 replay 10 次、7 样本取中位数；计时后再次检查输出。
`--time-wrong-results` 允许 sum 错误后继续计时，计时后保留再次校验的结果。
同一进程内的 helper 使用相同输入与输出地址；隔离进程单独分配地址，输入仍由相同 seed 生成。
CPU 会为长轴建立 fp64 参考；128 SM 的满核长轴约需 1 GiB 输入显存和数 GiB 主机内存。

只对照现有 NPU 的相同 shape 可指定 `--rows 7,56`。首次检查环境可以运行小矩阵：

```bash
CUDA_VISIBLE_DEVICES=GPU-实际UUID python output_axis_cases.py \
  --rows 7 --rnumels 32768 --splits 1,2 --out cuda-smoke
```

缺少 cooperative launch 支持时保留编译异常；不要删掉该选项后把结果混入当前协议对照。
CUDA 检查或执行失败会保存失败记录，并以状态 1 退出；全部通过时为 0。
状态 1 不代表整组未完成，应检查记录数与逐点状态。
数值错误但正常返回的点可以有 `median_us`；编译/启动失败或尚未完成计时的点没有耗时。
预热与重复调用使用相同输入和 workspace，可能掩盖第一次调用的旧值读取错误，因此保留首次和计时后两次校验。

## 更新图表

便携包已包含现有 NPU 的原始结果和 manifest。GPU 跑完后执行：
绘图机缺少 Matplotlib 时，可先执行 `python -m pip install 'matplotlib>=3.8'`。

```bash
python plot_output_axis_cases.py \
  --npu npu/output-axis.json --cuda cuda/output-axis.json --out figures
```

生成 `figures/output-axis-occupancy.svg`、同名 PNG 和 `output-axis.csv`。
相同 X/R 画在同一子图；GPU 自身满核的额外 X 单独成行，不会与 NPU 的 X=56 错配。
图中分别标出设备核数、未满核/满核状态、普通归约基线及各自 Grid 预算；缺失结果不补值。
实线为检查通过，空心虚线为错误或未完成检查但取得的诊断耗时，底部叉号为无耗时的执行失败。
不传 `--npu` 可只画 GPU；不传 `--cuda` 时明确显示 CUDA 数据待补。

在完整 torch-learn 仓库根目录更新文档现有 SVG 时，使用下面命令，将 CUDA 路径改成实际结果文件：

```bash
python scripts/tests/cooperative_reductions/plot_output_axis_cases.py \
  --npu artifacts/cooperative-barrier-unfiltered-20261122/npu/output-axis.json \
  --cuda /实际路径/cuda/output-axis.json \
  --out docs/requirements-design/cooperative-reduction/assets/barrier-comparison
```

会更新文档已经引用的 SVG/PNG，并输出包含 backend、设备名和核数的 CSV。
绘图读取输入 JSON，不修改原始结果；已有 NPU-only 的 `--data` 参数继续作为 `--npu` 别名。
这个绘图入口只接收 `output_axis_cases.py` 的结果，不接收旧的 `barrier_suite.py bench.json`。

## 需要保留的结果

回传 `cuda/output-axis.json`、`cuda/manifest.json`、`cuda/helper-original.py` 即可绘图与复核协议版本。
若发生编译/启动错误，另保留终端异常。`triton_cache/` 不必传回。
便携包的 `bundle_manifest.json` 记录源仓库提交和包内文件摘要；输出 manifest 也记录该提交。

默认 `--cuda-barrier snapshot` 使用 `cuda_triton_helpers.py` 中 PyTorch v2.13.0 的原始函数，
对应提交 `cf30153c4c131c8164ee7798e5022d810682e2cb`，函数源码 SHA256 为
`c9fa8ee90b54adc67a11ba5406b5fe4be55178b974cc26a2bde8c2c9f19f5bb3`，与现有 NPU 原始协议一致。
因此 GPU 环境不必提供同版本的 Inductor 私有函数；Triton 仍需具备编译和 cooperative launch 能力。
若特意对照当前安装的 helper，可传 `--cuda-barrier installed`，实际源码和 SHA256 会单独记录，
应先核对其与当前 NPU 对照的差异。
