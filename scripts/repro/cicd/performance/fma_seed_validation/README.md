# FMA fallback 与 attention constexpr 临时对照

`run.py` 在原 TorchBench runner 进程内安装一个临时改动，不修改安装目录。
结论直接维护在对应模型分析文档中。

- `baseline`：保持原实现。
- `no_fma_fallback`：在注册 NPU lowering 前，仅从 `FALLBACK_LIST` 删除 `prims.fma` 和其 default overload。
- `bind_constexpr`：仅在 `NPUIndexTritonKernel.codegen_kernel` 中补齐 `equal_1_arg_indices` 对应的常量值 1。

在指定 TorchBench 容器工作目录激活环境，选择允许且空闲的物理卡后运行：

```bash
python <torch-learn>/scripts/repro/cicd/performance/fma_seed_validation/run.py \
  --runner <runner-dir>/huggingface.py --variant baseline --output-dir <output>/metadata -- \
  --only OPTForCausalLM --device npu --training --performance --float32 --batch-size 2 \
  --backend inductor --npu-backend triton --disable-aclgraph --iterations 20 --repeat 3 \
  --enable-profiler 0 --prof-output-path <output>/profile --output <output>/benchmark.csv
```

为各 variant 使用不同的 Inductor cache 和 debug 输出目录，关闭 FX graph cache，保留全部生成代码。
性能使用稳定 step 的设备 kernel duration 汇总；同时核对生成 wrapper 中的 FMA 外部调用、
首层 attention 的 Grid1D/GridNpu 及 checker 日志。精度使用原 runner 的 `--accuracy --accu-summary`，
保持同一模型、batch 和 FP32，不以编译成功代替精度通过。
`environment.json` 保存版本、关键源码 SHA256、实际 lowering 入口以及安装文件未改动的检查。

完成后运行 `summarize.py --job-root <job-root> --output <summary.json>`，
按 CSV 的真实 Step Id 统计整步和逐 kernel duration，并单独统计 softmax backward 后的
Mul/Add/Mul/Add 窗口；不使用主机 wall time 代替设备时间。
