# TorchBench 模型矩阵测试

[tests/run_torchbench_matrix.sh](tests/run_torchbench_matrix.sh) 读取模型清单，串行运行八种测试组合，
按模型生成结果宽表。脚本使用 Bash 和 Python 标准库；模型测试依赖当前环境中的
PyTorch、torch_npu、Triton、CANN 及各 suite 的配套依赖。

## 模型清单

CSV 必须使用 `model_name,suit` 表头，`suit` 仅支持 `torchbench`、`timm_models`、`huggingface`。
将下面的占位模型名替换为对应 suite 中的实际模型名：

```csv
model_name,suit
xxx,torchbench
xxx,timm_models
xxx,huggingface
```

支持 UTF-8 BOM、Windows 换行、空行和 CSV 引号。每一行独立测试；同名模型在不同 suite 中不会覆盖日志。
启动前检查完整 CSV 和所需 runner 是否存在，数据和依赖由调用方提前准备。

## NPU_A5 运行

在 `NPU_A5` 的专用工作目录 `/home/l30023782/PTA_213/torch_bench` 中，
激活 conda 环境 `pta_213_cicd` 并加载相应 CANN 环境。脚本不自动激活或安装环境。

```bash
conda activate pta_213_cicd
cd /home/l30023782/PTA_213/torch_bench

# 全量测试：静态/动态 × 推理/训练 × 性能/精度。
bash /path/to/torch-learn/scripts/tests/run_torchbench_matrix.sh \
  --models ./models.csv \
  --output-dir ./torchbench_results

# 仅静态 shape、推理、精度。
bash /path/to/torch-learn/scripts/tests/run_torchbench_matrix.sh \
  --models ./models.csv \
  --shape static --mode inference --metric accuracy \
  --output-dir ./torchbench_results
```

三个筛选参数独立生效，默认均为 `all`。例如仅传 `--shape dynamic`，则每个模型运行四项动态测试。

| 参数或环境变量 | 默认值与含义 |
| --- | --- |
| `--models` | 必填，模型 CSV 路径 |
| `--shape` | `all`；可选 `static`、`dynamic` |
| `--mode` | `all`；可选 `inference`、`training` |
| `--metric` | `all`；可选 `performance`、`accuracy` |
| `--output-dir` | 必填，产物根目录；每次运行创建独立子目录 |
| `--bench-dir` / `TORCHBENCH_DIR` | `/home/l30023782/PTA_213/torch_bench/pytorch_new/benchmarks/torchbench` |
| `--data-path` / `TORCHBENCH_DATA_PATH` | `/home/l30023782/PTA_213/torch_bench/pta_213_cicd/data` |
| `--python` | 当前环境的 `python`；可指定目标环境的 Python 可执行文件 |
| `HF_ENDPOINT` | `https://hf-mirror.com`；已有环境变量优先 |

目录参数优先于同名环境变量。切换机器或目录时可显式覆盖：

```bash
bash /path/to/torch-learn/scripts/tests/run_torchbench_matrix.sh \
  --models ./models.csv \
  --bench-dir /home/qc/PTA_213/pytorch_new/benchmarks/torchbench \
  --data-path /home/qc/PTA_213/pytorch_new/benchmarks/torchbench/dataset \
  --python /path/to/conda/envs/pta_213_cicd/bin/python \
  --output-dir ./results
```

## 测试命令与结果判定

每项均在 `--bench-dir` 中运行对应 `${suit}.py`，公共参数为：

```text
--float32 --backend inductor --npu-backend triton --disable-aclgraph --only MODEL
```

性能使用 `--performance --repeat 1 --iterations 1`，精度使用 `--accuracy --iterations 5`，
分别添加 `--inference` 或 `--training`；动态测试额外添加 `--dynamic-shapes`。
每项还传入独立的 `--output <case_dir>/benchmark.csv`，保存 runner 生成的结果及其配套文件。

汇总依据 stdout/stderr 日志和进程退出码：

- 退出码非零：填 `FAIL`，即使日志中已有通过标志。
- 性能退出码为零：读取日志中最后一条独立的 `1.234x` 格式结果行；缺失则填 `FAIL`。
- 精度退出码为零：存在独立的 `pass_accuracy` 结果行时填 `PASS`；缺失则填 `FAIL`。
- 未选择的场景填 `-`。不支持的训练模式或模型若未产生通过结果，也计为 `FAIL`。

解析时移除终端颜色转义码，保留原始日志。失败项不会中断后续测试。
全部已选项通过时脚本退出 `0`；存在失败时退出 `1`；参数或预检查错误退出 `2`。
Ctrl+C 中断时退出 `130`，保留日志和已经写入的 `results_running.csv`。

## 产物与结果宽表

每次运行创建一个独立目录，模型目录包含行号、suite 和经过处理的模型名。
CSV 的 `model` 列保留原始模型名，`log_dir` 为模型日志目录的绝对路径。

```text
torchbench_results/
  torchbench_20261019_153000_XXXXXX/
    models.csv
    models.nul
    results_20261019_160000.csv
    0001_torchbench_xxx/
      s_infer_perf/
        run.log
        benchmark.csv
      s_infer_acc/
        run.log
        benchmark.csv
      ...
```

`run.log` 包含工作目录、数据路径、HF 端点、完整命令、开始/结束时间、原始 stdout/stderr 和退出码。
`benchmark.csv` 及配套文件由 runner 生成，失败时可能不存在。
`models.csv` 保存本次规范化后的模型清单，`models.nul` 是内部循环使用的模型记录。

每项结束后更新 `results_running.csv`；正常跑完全部已选测试后，重命名为
`results_YYYYMMDD_HHMMSS.csv`，时间取完成时运行机器的本地时间。
最终表头严格为：

```csv
model,s_infer_perf,s_infer_acc,s_train_perf,s_train_acc,d_infer_perf,d_infer_acc,d_train_perf,d_train_acc,log_dir
```

脚本只指定日志及 runner CSV 的目录；编译缓存和额外 dump 的位置继续由原测试环境控制。
