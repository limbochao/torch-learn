# TorchBench 2.13 配套软件、数据与模型测试

提供两个脚本，命令均从 `torch-learn` 仓库根目录执行：

| 脚本 | 功能 |
| --- | --- |
| [tools/torchbench_setup.py](tools/torchbench_setup.py) | 直接运行，安装配套软件和固定版本源码 |
| [tests/run_torchbench.py](tests/run_torchbench.py) | 自动补齐所选模型的数据，运行 eager 或 Inductor + Triton 测试 |

运行前自行激活目标 Python 环境并加载 CANN 环境变量。脚本要求 Python 3.11.8 或更新版本。
`torch`、`torch_npu`、`triton`、`triton_ascend` 由调用方提前准备，安装脚本不会安装或替换它们。
测试使用支持 `--execution-mode`、`--npu-backend` 的 Ascend `pytorch_new/benchmarks/torchbench` runner。

## 配套软件安装

```bash
python scripts/tools/torchbench_setup.py
```

不需要指定目录。Python 包安装到当前环境，配套源码和安装记录自动存放在
`<当前 Python 环境>/share/torchbench-213/`，测试脚本使用同一个 Python 环境时会自动找到它们。
同样支持 `python scripts/tools/torchbench_setup.py install`。

安装脚本先固定已有基础软件的版本，再预解析完整依赖；解析结果若要求安装上述四个软件包中的任何一个，
就会在安装前报错。随后仅安装解析出的配套包，不再自动解析依赖。已有 torchvision、torchaudio、attrs
也会保留版本约束；若它们与目标依赖不兼容，会报出冲突。

| 用途 | 配套依赖 |
| --- | --- |
| 公共依赖 | NumPy 1.26.4、pandas、SciPy、PyYAML、psutil、tabulate、tqdm、requests、Pillow |
| 图像模型 | torchvision 0.28.0、opencv-python-headless 4.11.0.86、tensorboardX |
| HuggingFace | transformers 4.36.0 及其依赖 |
| DLRM | future、ONNX、ml_dtypes 0.5.4、pydot、scikit-learn |
| 强化学习 | gym 0.26.2、gymnasium、pygame、kornia、scikit-image |
| 语音及 Llama | kaldi_io、sentencepiece |
| GNN | 固定提交构建的 torch_geometric 2.4.0 |
| CLIP | 固定提交的 multimodal 源码，以及 ftfy、regex、iopath 0.1.9 |

安装阶段同时下载并准备三份源码：

| 源码 | 提交 |
| --- | --- |
| pytorch/benchmark | `fb14629994956c16d27c7a9940c5f081afbf68dc` |
| pyg-team/pytorch_geometric | `cabcd4097442ba60aa1efa11e1619dd9bb8fb527` |
| facebookresearch/multimodal | `e54f602d1cb33f8a5deb3540955271c4223f7c9c` |

multimodal 通过源码路径导入，避免完整 wheel 的 `attrs==23.1.0` 约束影响基础软件；
CLIP 使用的配套依赖单独安装，不安装 DALL-E。`timm`、`fastNLP` 不在这 50 个模型的必要依赖中。
安装脚本只准备配套软件和源码，模型输入与权重由测试脚本在运行时准备。

## 自动准备数据

无需单独运行下载命令。`run_torchbench.py` 在启动每个模型前检查所需输入、词表和权重，
缺失时自动下载；已通过校验的文件直接复用。只准备本次所选模型需要的数据。

| 模型 | 下载内容 |
| --- | --- |
| Super_SloMo | `Super_SloMo_inputs.tar.gz` 和 VGG16 权重 |
| pytorch_stargan | `pytorch_stargan_inputs.tar.gz` |
| speech_transformer | `speech_transformer_inputs.tar.gz` |
| drq | `obs.pkl` |
| torch_multimodal_clip | `pizza.jpg` 和 `clip_merges.bpe`；不下载预训练模型权重 |
| 9 个 torchvision 模型 | 对应的 ImageNet V1 权重；输入张量由模型生成 |

torchvision 权重对应 `alexnet`、`densenet121`、`mobilenet_v2`、`resnet18`、`resnet50`、
`resnext50_32x4d`、`shufflenet_v2_x1_0`、`squeezenet1_1`、`vgg16`。
VGG16 与 Super_SloMo 共用同一权重，只下载一次。

其余目标模型使用随机输入、源码自带样例或运行时生成数据。21 个 HuggingFace 条目无需预训练权重、
tokenizer 或语料；`basic_gnn_*` 不需要 Reddit 数据；LearningToPaint 不需要额外输入包。

模型数据的下载来源与校验值维护在测试脚本中。已有文件通过校验则复用，输入压缩包自动解压；
源码和输入文件检查完整 SHA-256，torchvision 权重检查官方文件名中的 SHA-256 前缀。
输入已解压时不重复解压；保留压缩包的情况下会对照包内文件检查，发现解压文件缺失则从缓存恢复。
单个模型的数据准备失败时，记录该模型失败并继续其它模型。

安装目录由脚本自动选择，跟随当前 Python 环境：

```text
<sys.prefix>/share/torchbench-213/
  sources/benchmark/           # TorchBench 源码
  sources/pyg/                 # 用于构建 PyG
  sources/multimodal/          # CLIP 源码
  downloads/                  # 源码归档
  constraints.txt
  requirements.txt
  install-plan.json
  install-report.json         # 有实际安装动作时生成
  packages.txt
```

## 模型测试

默认执行 **Inductor + Triton、精度检查、FP32、训练模式**。模型清单不指定时使用原定的 50 个模型。
不支持训练的模型由 runner 按其元数据切换到推理。

```bash
python scripts/tests/run_torchbench.py \
  --runner-dir /path/to/pytorch_new/benchmarks/torchbench \
  --models resnet18
```

**batch size 不需要指定**，默认使用 runner 为每个模型配置的值，包括训练、推理和精度模式的调整。
仅在确实需要覆盖模型默认值时使用可选参数 `--batch-size`。
当前 runner 在精度模式下先为 TorchBench 设置 4、HuggingFace 设置 1，再应用模型的特殊配置；
例如 `resnet18` 的默认精度测试使用 batch size 4。性能模式使用模型配置，不统一指定 batch size。
这些默认规则由 runner 的 `common.py`、`torchbench.py` 和 `huggingface.py` 决定。

运行目录默认是 `RUNNER_DIR/torchbench_workspace`；也可用 `--work-dir` 更改数据、缓存与结果的存放位置，
这不影响配套软件的安装位置。测试脚本自动设置数据、缓存和配套源码路径，无需额外加载路径配置：

```text
torchbench_workspace/
  downloads/                  # 输入压缩包
  data/                       # 已解压输入、obs.pkl、pizza.jpg
  cache/torch/hub/checkpoints/ # torchvision 权重
  cache/iopath/models/text/    # CLIP BPE 词表
  cache/inductor/              # 编译缓存
  cache/triton/
  results/                    # 每次测试的日志与 CSV
```

`--models` 接收空格或逗号分隔的名称，也可用 `--models-file models.txt` 读取清单；
文件支持空行和 `#` 注释。两种方式同时指定时取并集、保持顺序并去重，拼错模型名会直接报错。
脚本自动将 21 个 HuggingFace 模型交给 `huggingface.py`，其余 29 个交给 `torchbench.py`。

| 参数组合 | 实际测试 |
| --- | --- |
| 默认，或 `--inductor --accuracy` | Inductor + Triton 与 eager 的精度比较 |
| `--no-inductor` | 仅 eager：检查两次 eager 运行的一致性，成功状态为 `pass_eager` |
| `--no-accuracy` | Inductor + Triton 与 eager 的性能比较 |
| `--no-inductor --no-accuracy` | 仅 eager 性能测试 |

开启 Inductor 时固定传入 `--npu-backend triton`，不提供独立更换 NPU backend 的选项。
关闭时使用 `--execution-mode eager`，不传编译后端参数；runner 的 eager 分支不会编译。
精度开关使用 runner 的 `--accuracy`，不依赖 msprobe 的 `--precision-checker`。

精度检查默认 `--iterations 2`，性能测试默认 `--repeat 3`，可按需要增大。
`--inference` 改为推理模式；不传 `--batch-size` 时使用模型配置。默认使用可见设备 0，
需要选择其它设备时在命令前设置 `ASCEND_RT_VISIBLE_DEVICES`。

每个模型在独立子进程中顺序运行，结果位于 `WORK_DIR/results/<运行时间>/`，包含每模型的 `.log` 和 `.csv`。
也可使用 `--output-dir` 指向一个尚不存在的结果目录，避免混入旧结果。
脚本同时检查进程退出码和 CSV；精度失败、缺少结果或运行异常均返回非零退出码，并继续处理其余所选模型。

先查看将要执行的命令：

```bash
python scripts/tests/run_torchbench.py \
  --runner-dir /path/to/pytorch_new/benchmarks/torchbench \
  --models resnet18 T5Small --dry-run
```

`--dry-run` 不下载数据、不导入 Torch、不运行模型、不创建结果目录。

## 单模型执行示例

在目标 Python/CANN 环境中，从 `torch-learn` 仓库根目录执行以下两条命令。
下面使用 runner 路径 `/home/l30023782/PTA_213/torch_bench/pytorch_new/benchmarks/torchbench`：

```bash
python scripts/tools/torchbench_setup.py

python scripts/tests/run_torchbench.py \
  --runner-dir /home/l30023782/PTA_213/torch_bench/pytorch_new/benchmarks/torchbench \
  --models resnet18
```

配套软件只需安装一次。测试会自动补齐 ResNet18 权重，并按默认配置执行 Inductor + Triton 精度检查。

## 默认模型清单

```text
AlbertForMaskedLM BartForCausalLM BertForMaskedLM BlenderbotForCausalLM DebertaV2ForMaskedLM
DistilBertForMaskedLM ElectraForCausalLM GPT2ForSequenceClassification LayoutLMForMaskedLM MBartForCausalLM
MT5ForConditionalGeneration MegatronBertForCausalLM MobileBertForMaskedLM OPTForCausalLM PLBartForCausalLM
PegasusForCausalLM RobertaForCausalLM T5ForConditionalGeneration T5Small TrOCRForCausalLM XGLMForCausalLM
BERT_pytorch LearningToPaint Super_SloMo alexnet basic_gnn_edgecnn basic_gnn_gcn basic_gnn_sage dcgan
densenet121 dlrm drq lennard_jones llama mobilenet_v2 nanogpt nvidia_deeprecommender phlippe_densenet
phlippe_resnet pytorch_stargan pytorch_unet resnet18 resnet50 resnext50_32x4d shufflenet_v2_x1_0
soft_actor_critic speech_transformer squeezenet1_1 torch_multimodal_clip vgg16
```
