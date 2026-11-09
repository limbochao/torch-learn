# NPU_A5：TorchBench 2.13 已知配置

此文件供工作区内复用环境；不是对外发布材料。当前任务指令和工作区 AGENTS.md 优先。
这些路径在 2026-10 的模型验证中使用过，重连后核实实际状态，不根据登录 banner 判断激活后的版本。

## 入口

- SSH：`ssh NPU_A5`，确认进入目标 Docker 容器。
- 工作目录：`/home/l30023782/PTA_213/torch_bench`。
- conda：`pta_213_cicd`。
- 环境脚本：工作目录下 `pta_213_cicd/env.sh`，使用前检查内容与路径是否仍匹配。
- runner：`pytorch_new/benchmarks/torchbench` 下的 `torchbench.py`、`huggingface.py`、`timm_models.py`。
- 数据：工作目录下 `pta_213_cicd/data`，经 `TORCHBENCH_DATA_PATH` 配置。
- HuggingFace 端点：`HF_ENDPOINT=https://hf-mirror.com`。

环境脚本应加载 conda 和 CANN；若脚本缺失，按当前安装位置激活环境，不猜测工具链路径。
检查 `sys.executable`、包版本与 `__file__`，特别注意 `PYTHONPATH` 中已有 overlay。
环境名称包含 2.13 也不能代替实际版本核对。

## HuggingFace 动态训练 accuracy 示例

在容器工作目录激活环境后执行；先检查当前 runner 的 `--help` 和参数实现。
输出目录应为本次任务独有。以下不自动加入历史修复 overlay。

```bash
source pta_213_cicd/env.sh
mkdir -p bert_dynamic_accuracy
python pytorch_new/benchmarks/torchbench/huggingface.py \
  --only BertForMaskedLM --accuracy --training --float32 \
  --backend inductor --npu-backend triton --disable-aclgraph \
  --dynamic-shapes --iterations 5 --accu-summary \
  --output "$PWD/bert_dynamic_accuracy/benchmark.csv"
```

这是参数示例，不提供后台监控或超时功能。正式长任务另保存 stdout/stderr、退出码和运行状态。
未显式指定 batch size 时，记录 runner 解析后的值，不从 performance 的默认值推测 accuracy 值。

eager 对照保留相同场景，删除两个 backend 选项并增加 `--execution-mode eager`。
对比默认 batch size 问题时，原命令若为 performance，对照也应保留 performance。

## backend 与修复版本

`triton`、`triton_experimental`、`default` 的映射由当前 runner 和 torch_npu 注册代码决定。
检查 CLI 如何设置 `TORCHINDUCTOR_NPU_BACKEND`，以及注册函数最后选择的实现；
不能仅凭名称、目录或某次日志把它们永久视为同义项。

使用历史 Python overlay 时，核对预期提交和实际加载文件校验值，记录依赖的原安装库。
Python 文件替换不代表 native library 已重编译。不要因为环境里有 overlay 就默认启用它。

## 已验证实例的解释

2026-10-22 的 `BertForMaskedLM` 测试使用八类修复整合提交 `5b18fac653`，
实际版本为 torch `2.13.0+cpu`、torch_npu `2.13.0.dev20261019`、
Triton `3.6.0`、transformers `4.36.0`。
FP32、batch 1、sequence 512、5 步训练、动态编译模式，结果 `pass_accuracy`。
该结果只用于说明如何记录验证范围，不代表未修复基线、其他版本或其他 shape 已通过。
当时编译/预编译线程均为 1，单候选超时约定为 3600 秒；这些不是后续任务的强制默认值。
