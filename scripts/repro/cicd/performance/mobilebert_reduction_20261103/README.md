# MobileBERT add/mul/sum 参数梯度归约诊断

在 NPU_A5 Docker 的 TorchBench 工作目录、`pta_213_cicd` 中运行。
目标框架源码为 `test/cicd_benchmark`（`b3bd74035`），执行前确认关键安装文件与该快照一致。
本脚本固定使用物理卡 2，禁止改成卡 4、5、6；启动前检查卡 2 空闲。

```bash
source pta_213_cicd/env.sh
export ASCEND_RT_VISIBLE_DEVICES=2
export TORCHINDUCTOR_NPU_BACKEND=triton
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/mobilebert_reduction_20261103/cache/inductor"
export TRITON_CACHE_DIR="$PWD/mobilebert_reduction_20261103/cache/triton"
python <torch-learn>/scripts/repro/cicd/performance/mobilebert_reduction_20261103/run.py \
  --output "$PWD/mobilebert_reduction_20261103/final" --regenerate --experiments --profile
```

输入为三个 FP32 contiguous `[128,128,128]` 张量，输出 `(a+b).sum((0,1))` 和
`((a+b)*c).sum((0,1))`，对应原 kernel 的两个参数梯度；输入不依赖未初始化的 wrapper 临时缓冲区。
记录真实 selected config；使用同一组输入和 CPU FP64 结果验证。
64 次调用捕获至 NPUGraph，Event 计时除以调用数，7 轮取中位数。
这个测量包含 graph 内调度开销；不能当作原整网 profile 的逐 kernel duration 或实测整网修复收益。

`--experiments` 重放固定 tile 和两阶段诊断 DSL，并编译显式两阶段 ATen 小图。
原始 fixed-output kernel 和两阶段诊断版在 capture 后先将输出填为 NaN，再 replay、校验，
防止把空 capture 误当作快速 kernel；原始 launcher 每次读取当前 stream，不能提前缓存 stream 指针。
`--profile` 为关键对照额外记录 10 次调用的设备 kernel duration，分别汇总阶段耗时和实际 launch 次数。
设备 profile 与 Event 两种口径分列；`result.json` 同时包含真实 selected config 和全部生成候选。

两阶段 ATen 版返回两个输出，当前自动生成 1 次 partial + 2 次 final，即 3 次 launch；
两阶段诊断版将两路 final 放在同一个 kernel，合计 2 次 launch。
诊断版没有修改安装的 Inductor、Triton 或 npuir，也不是已落地的通用修复。

四方原始 profile 的保守核验可在无卡机器复算：

```bash
python <torch-learn>/scripts/repro/cicd/performance/mobilebert_reduction_20261103/analyze_profile.py \
  --input-root <perf>/MobileBertForMaskedLM_Train --output-root <output>
```

该脚本将 NPU Eager 的全部 168 个同型窗口作为目标 96 次的费用上界，
GPU Eager 按必需操作数量乘各操作最小耗时作为下界；不把聚合平均值当作目标 shape 耗时。
边界含义与源码结论见[专项文档](../../../../../docs/issue-logs/cicd/performance/mobilebert-parameter-reduction-20261103.md)。
