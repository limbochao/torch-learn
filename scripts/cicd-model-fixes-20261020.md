# 四模型 Inductor 修复验证

分析见 [修复审查与复测](../docs/issue-logs/cicd/model-fixes-20261020.md)。
脚本目录为 `scripts/repro/cicd/model_fixes_20261020/`。

## 环境

使用已能执行 torch_npu Inductor 的 NPU 环境及 TorchBench 模型依赖。
测试模型的 source runner 由 `TORCHBENCH_RUNNER_DIR` 指定。
本轮使用 FP32、关闭 ACLgraph，单候选编译超时 3600 秒。
将修复分支中的 Python 文件部署到专用包目录，并在 `PYTHONPATH` 中优先使用该目录。
应保留原环境的其它补丁，并记录实际加载的文件及其 SHA256。

## 数值回归

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export TORCHINDUCTOR_NPU_BACKEND=triton
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHNPU_PRECOMPILE_THREADS=4
export REGRESSION_DIR="$PWD/regression_results"
python scripts/repro/cicd/model_fixes_20261020/run_regressions.py
```

此入口分别运行 mask、axes、negative 组，覆盖静态/动态 shape。
negative 组另测 dump/check 两个环境变量。每个子进程的退出码和数值结果分别保存，失败不会阻止后续组。
也可直接使用 `validate_codegen.py --group mask --output mask.json`；动态 shape 额外添加 `--dynamic`。
prod 使用 `value + 0.99`，保证无效 lane 上的 fused 值不是单位元 1，避免无效的 padding 对照。

## 完整模型

```bash
export TORCHBENCH_RUNNER_DIR="$PWD/pytorch_new/benchmarks/torchbench"
export MATRIX_DIR="$PWD/model_results"
export MATRIX_DEVICES=0,1
python scripts/repro/cicd/model_fixes_20261020/run_matrix.py
```

默认运行四个模型的 32 个场景。
通过 `MATRIX_MODELS=AlbertForMaskedLM` 选择模型；
通过 `MATRIX_CASES=d_train_perf` 选择动态训练 performance。
其它 case 名如 `s_infer_acc`、`d_infer_perf`、`s_train_acc`。
重复运行时会跳过已有 PASS/FAIL，因此验证新修复应使用新的 `MATRIX_DIR`，保留此前结果。
模型性能维持默认 batch size，不设置 `--batch-size` 缩小工作负载。

`run_case.py` 记录失败的实际编译 source、signature、constants、options，并对单次 `triton.compile`
设置 60 分钟超时。模型可能编译很多候选，总耗时超过 60 分钟不等于单候选超时。
独立 kernel 的捕获、重放和 compile-only 导出使用 `inductor-kernel-compile-debug` skill；
矩阵 wrapper 的失败记录用于找到目标，不代替独立导出验证。

## TA 独立定界

```bash
python scripts/repro/cicd/model_fixes_20261020/masked_indirect_load.py \
  --output-dir ta_evidence
```

该脚本只依赖 torch、torch_npu、Triton，不运行模型或 Inductor。
一个独立 kernel 验证 template/pure SIMT 两种模式，以及 `other=0`/`other=-7`。
输出数值记录及 TTIR/ttadapter，便于对照 mask 丢失的位置。
它默认记录缺陷，不因已知缺陷退出失败；需要验证已修复的 TA 版本时可以添加 `--expect-fixed`。
本任务不修改 TA 源码或安装包。

## 新暴露的 softmax 编译失败

```bash
# 原候选：预期 UB overflow
python scripts/repro/cicd/model_fixes_20261020/albert_softmax_compile.py

# 同一 DSL，仅缩小 X2 tile：编译并与 eager 比较
python scripts/repro/cicd/model_fixes_20261020/albert_softmax_compile.py --xblock 8 --check
```

目标架构固定为捕获时的 `Ascend950PR_9579`。此文件来自 skill 的单候选导出，
保留原 kernel body、signature 和编译选项，仅移除了本例用不到的公共导出 helper。
默认用例只编译，不创建模型和输入；`--check` 才分配输入并 launch 一次。
框架修复通过通用 tiling 规则搜索可编译候选，不固定该对照 tile。
