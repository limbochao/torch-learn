# 四模型八场景与单 kernel 复现

需已配置目标设备上的 torch、torch_npu、Triton-Ascend、CANN 和各模型依赖。
本文脚本均位于 `scripts/repro/cicd/four_model_matrix_20261020/`，从 torch-learn 根目录执行。

## 原始矩阵

```bash
export TORCHBENCH_RUNNER_DIR=/path/to/pytorch_new/benchmarks/torchbench
export MATRIX_DIR="$PWD/four_model_matrix_20261020"
export MATRIX_DEVICES=0,1
python scripts/repro/cicd/four_model_matrix_20261020/run_matrix.py
```

共 32 项，FP32，默认 batch size，禁用 ACL graph，compile threads=1。
设备值表示当前环境中可见设备的编号，应选择空闲设备。
每个场景保存命令、CSV、debug trace 和失败编译配置。单次编译 60 分钟超时，保存复现后终止该场景进程组。
不要在已有队列仍存活时重复启动；脚本只跳过已有终态，不接管 RUNNING 进程。

## 编译失败

Albert 原始文件直接调用一次 `triton.compile(ASTSource(...), target=..., options=...)`，不启动模型或 autotune：

```bash
python scripts/repro/cicd/four_model_matrix_20261020/AlbertForMaskedLM/s_train_perf_original.py
python scripts/repro/cicd/four_model_matrix_20261020/AlbertForMaskedLM/s_train_perf_deferred.py
```

原始用例预期 UB overflow。`*_tile1.py` 仅缩小 X tile；`*_deferred.py` 仅把 load 延后至首次使用之前。
静态训练性能的 tile 已经为 1，`tile1` 仍失败；动态性能的大 tile 仅延后 load 仍失败，参见分析中的对照表。
文件较长的部分是完整原始 kernel 的 81 个输入与计算，没有携带通用 launcher、版本兼容或模型运行模板。

验证四个 Albert 诊断配置与 eager 参数梯度：

```bash
python scripts/repro/cicd/four_model_matrix_20261020/validate_albert.py
```

## 前向精度

```bash
python scripts/repro/cicd/four_model_matrix_20261020/gpt2_original.py
python scripts/repro/cicd/four_model_matrix_20261020/gpt2_variance_mask.py --compile-mode simt_only
python scripts/repro/cicd/four_model_matrix_20261020/gpt2_both_masks.py

python scripts/repro/cicd/four_model_matrix_20261020/bert_original.py
python scripts/repro/cicd/four_model_matrix_20261020/bert_variance_mask.py --compile-mode simt_only
python scripts/repro/cicd/four_model_matrix_20261020/bert_both_masks.py
```

每个文件是一条原始或诊断 Triton kernel、固定输入和一次固定配置 launch，输出 `allclose` 与最大绝对误差。
`original` 预期失败；默认路径仅补方差 mask 仍失败，纯 SIMT 加方差 mask 通过；两处显式 mask 后均通过。
脚本返回 0 表示执行成功，精度结果应读取打印出的 `allclose`，不能仅检查退出码。
这里用固定 seed 和非零均值输入放大误差，embedding 只使用 token 0，缩小词表并为 padded 读取预留有效存储。

TA masked indirect load 的更小复现：

```bash
python scripts/repro/cicd/four_model_matrix_20261020/masked_indirect_load.py
```

有效 lane 应为 3，padded lane 应为 0；默认路径错误返回 9，纯 SIMT 正确。
脚本同时将编译 IR 保存到脚本所在目录，用于核对 mask 在哪个阶段丢失。

## BERT backward 精度

```bash
python scripts/repro/cicd/four_model_matrix_20261020/bert_backward_original.py
python scripts/repro/cicd/four_model_matrix_20261020/bert_backward_fixed.py
```

分别验证 tile=1 和 tile=8。原始 kernel 的 transpose 输出正确，tile=8 的 sum 错误；修正 reshape 后两者均正确。
诊断修改仅用于定界，没有安装到框架。

## 模型级诊断

`patch_accuracy_and_backward.py` 在 `AsyncCompile.triton` 接收生成源码时，只注入已定位的前向 mask 和 BERT
backward reshape 修改。它不修改安装目录，可使用与原始失败场景相同的 runner 参数：

```bash
python scripts/repro/cicd/four_model_matrix_20261020/patch_accuracy_and_backward.py \
  "$TORCHBENCH_RUNNER_DIR/torchbench.py" \
  --only BERT_pytorch --accuracy --training --float32 --iterations 5 \
  --backend inductor --npu-backend triton --disable-aclgraph --output bert_diagnostic.csv
```

加 `--dynamic-shapes` 验证动态场景。GPT2 使用 `huggingface.py --only GPT2ForSequenceClassification --inference`。
检查日志中的 `DIAGNOSTIC_MASKS` 和最终 CSV，确认修改实际生效。该脚本仅用于定界，不是通用 codegen 修复。

## NFNet 动态 pooling 编译失败

```bash
python scripts/repro/cicd/four_model_matrix_20261020/nfnet_pool_original.py
python scripts/repro/cicd/four_model_matrix_20261020/nfnet_pool_tile2048.py
python scripts/repro/cicd/four_model_matrix_20261020/nfnet_train_pool_original.py
python scripts/repro/cicd/four_model_matrix_20261020/nfnet_train_pool_tile2048.py
```

`original` 固定 XBLOCK=4096，预期 UB overflow；`tile2048` 保留相同源码和编译选项，
仅把 XBLOCK 改为 2048，再用固定 seed 输入执行一次，与 eager 2×2 average-pool 比较。
用例将安装版本的 `div_floor_integer` helper 原样内联，不依赖 Inductor heuristics。
直接 runner 的参数保留 signature 中的 constexpr 占位，和运行时整数参数一起传入。

模型级诊断只增加缺失的小 tile，保留原始候选：

```bash
python scripts/repro/cicd/four_model_matrix_20261020/patch_nfnet_candidates.py \
  "$TORCHBENCH_RUNNER_DIR/timm_models.py" \
  --only nfnet_l0 --inference --dynamic-shapes --accuracy --float32 --iterations 5 \
  --backend inductor --npu-backend triton --disable-aclgraph --output nfnet_diagnostic.csv
```

检查 `DIAGNOSTIC_POOL_TILE` 和最终 CSV。这个脚本用于验证 tiling 定界，不是通用框架修复。

问题分析见 [CICD 四模型记录](../docs/issue-logs/cicd/four-model-matrix-20261020.md)。
