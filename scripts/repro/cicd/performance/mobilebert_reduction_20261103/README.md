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
  --output "$PWD/mobilebert_reduction_20261103/baseline" --regenerate
```

输入为三个 FP32 contiguous `[128,128,128]` 张量，输出 `(a+b).sum((0,1))` 和
`((a+b)*c).sum((0,1))`，对应原 kernel 的两个参数梯度；输入不依赖未初始化的 wrapper 临时缓冲区。
记录真实 selected config；使用同一组输入和 CPU FP64 结果验证。
64 次调用捕获至 NPUGraph，Event 计时除以调用数，7 轮取中位数。
这个测量包含 graph 内调度开销；不能当作原整网 profile 的逐 kernel duration 或实测整网修复收益。
