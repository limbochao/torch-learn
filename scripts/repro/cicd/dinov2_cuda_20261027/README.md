# DINOv2 CUDA 对照包

用于快速比较 `vit_base_patch14_dinov2.lvd142m` 的 FP32、training、dynamic-shapes、batch size 16、
5/50 轮精度。无需编译 PyTorch，无需准备 ImageNet；首次运行会下载 timm 的预训练权重。

## 环境准备

以下命令面向 Linux x86_64、Python 3.11、支持 CUDA 13 的 NVIDIA GPU。
先执行 `nvidia-smi` 检查驱动；CUDA 13.x 要求驱动至少 580，具体 GPU/驱动组合仍需兼容。
机器需有可用的 C++ 编译器（如 g++）。不在 CUDA 环境安装 torch_npu。

```bash
conda create -n dinov2_cuda python=3.11 -y
conda activate dinov2_cuda
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
# 将本包复制到 CUDA 机器，解压后执行：
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.version.git_version, torch.version.cuda); print(torch.cuda.is_available())'
```

下载模型受网络限制时，可根据所在网络设置 HF_ENDPOINT，例如 `https://hf-mirror.com`。
驱动低于要求时应先选择兼容的环境，不能直接套用上述 cu130 命令。

版本依据：
- https://download.pytorch.org/whl/cu130/torch/
- https://download.pytorch.org/whl/cu130/torchvision/
- https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html

## 运行同一 benchmark

在本包目录执行，CUDA_VISIBLE_DEVICES 按机器实际空闲卡选择：

```bash
CUDA_VISIBLE_DEVICES=0 bash run_cuda.sh benchmark 5
CUDA_VISIBLE_DEVICES=0 bash run_cuda.sh benchmark 50
```

相当于运行原 runner 的以下命令；使用本包的 common.py 对齐 SGD foreach 设置：

```bash
python timm_models.py --device cuda --accuracy --training --float32 --backend inductor \
  --disable-aclgraph --only vit_base_patch14_dinov2.lvd142m --dynamic-shapes \
  --batch-size 16 --iterations 50 --accu-summary --output benchmark.csv
```

这个 runner 参数叫 `--disable-aclgraph`，在 CUDA 上也会关闭 cudagraphs/cudagraph_trees。
CUDA 不需要 `--npu-backend triton`。`benchmark` 保留原 accuracy 流程：尝试 FP64 golden，
检查两次 eager 的一致性，再进行 eager/Inductor 比较。最终状态以日志和 CSV 为准；
FP64 golden 失败后可能切换 cosine 比较，需要核对日志。

## 逐步收集误差

```bash
CUDA_VISIBLE_DEVICES=0 bash run_cuda.sh trajectory 5
CUDA_VISIBLE_DEVICES=0 bash run_cuda.sh trajectory 50
# 仅执行原 runner 的 eager 检查，可选：
CUDA_VISIBLE_DEVICES=0 bash run_cuda.sh eager 50
```

`trajectory` 调用同一 TimmRunner 的训练函数、loss 和 optimizer，但使用与本次 NPU 定位一致的
FP32 eager 标杆，不生成 FP64 golden。它维护两个 eager 副本和一个 Inductor 副本，
从同一模型初始状态开始；每轮独立更新各自的参数，并使用固定输入。
每轮检查 eager 两次结果完全一致，再比较 prediction、loss、梯度、更新后参数及完整结果树。
`torch._dynamo.testing.same(tol=0.02)` 作为完整结果判定，同时记录逐元素 1e-5 和 0.02 阈值统计。
任意一轮失败则 trajectory 最终记 fail_accuracy；原 benchmark 保留自己的最终判定流程。
因此两种模式的结果不能直接混为同一种精度结论。

逐轮保存：
- `results/<运行名>/trajectory.jsonl`：逐步误差、失败参数名；不保存完整 tensor dump。
- `trajectory_summary.json`：首次失败步骤及完整指标。
- `environment.json`：torch/torchvision/timm/Triton、torch commit、CUDA、GPU、TF32、确定性等。
- `benchmark.csv`、`run.log`：结果与完整日志。
- `debug/`、`inductor_cache/`、`triton_cache/`：编译调试输出和缓存。
- `nvidia-smi.txt`、`pip-freeze.txt`：环境记录，三种模式都会保存。

## 对照范围

- runner 基线快照提交：`848e9feaa86f1cefe82ea1307c62c2d8b89bfc6f`。
- `timm_models.py` 保留基线模型路径；`common.py` 只做两类 CUDA 兼容修改：
  为四处 torch.npu 可用性检查加保护；SGD foreach=False，与 NPU 保持一致。
  具体修改见 `cuda_compatibility.diff`，文件校验见 `manifest.json`。
- 模型 pretrained=True、随机种子 1337、输入 16x3x518x518、SGD lr=0.01，
  无 momentum/weight decay，loss 使用原 runner 的 reduce_to_scalar_loss。
- accuracy 配置关闭 TF32、开启确定性；逐轮模式还记录实际生效值。
  默认 TORCHINDUCTOR_EMULATE_PRECISION_CASTS=0，可显式设置该环境变量做另外一组对照。
- 开启 dynamic 编译模式，但输入尺寸固定；未覆盖多个尺寸之间切换。
- CUDA 与 NPU 使用相同输入生成过程，设备 RNG 和归约可能不同；输入并非跨设备逐 bit 相同。
- 官方 torch 2.13.0 wheel 与 NPU 使用的源码构建不保证同一 commit。该包用于快速 CUDA 定界，
  不能据此证明严格同源码、同输入的跨设备数值等价。
- CUDA 使用自己的 codegen；未移植 NPU 的 layernorm fallback 或手动 GELU tiling。
- 本包仅完成 Python/shell 语法检查、屏蔽 torch_npu 后的 CPU 导入与指标 smoke 检查。
  尚未在 CUDA GPU 上执行；OOM、依赖或编译失败不属于已完成的精度验证。
