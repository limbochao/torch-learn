# Elementwise dynamic performance

本目录包含配套使用的两个脚本：

- `elementwise_op_cost_cases.py`：通过现有 CUDA/NPU profiler 工具采集单次 elementwise case 调用的
  device-side 平均耗时，并按 `RUN_ID` 合并到原始 `summary.csv`。
- `elementwise_op_cost_compare.py`：将原始 summary 浓缩为同场景、跨 execution、跨设备的宽表 CSV。

每次采集只运行 `EXECUTION` 指定的路径。不同 execution 和设备使用相同 `RUN_ID`，并共享
`PROFILE_ROOT`：

```bash
RUN_ID=baseline_001 DEVICE=cuda EXECUTION=eager \
    python scripts/tests/elementwise_dynamic_perf/elementwise_op_cost_cases.py
RUN_ID=baseline_001 DEVICE=cuda EXECUTION=static \
    python scripts/tests/elementwise_dynamic_perf/elementwise_op_cost_cases.py
RUN_ID=baseline_001 DEVICE=cuda EXECUTION=dynamic COMPILE_SHAPE=8192 SYMBOLIC_DIMS=0 \
    python scripts/tests/elementwise_dynamic_perf/elementwise_op_cost_cases.py
```

生成浓缩 CSV：

```bash
python scripts/tests/elementwise_dynamic_perf/elementwise_op_cost_compare.py \
    prof_log/elementwise_dynamic_perf/baseline_001/summary.csv
```

产物固定写入 summary 同级目录的 `elementwise_op_cost_comparison.csv`。所有数值保留三位小数；原始 summary
不存在的 execution 或设备数据保持为空。输出唯一键为
`scalar_ops + dtype + first_shape + runtime_shape`；若不同 case 产生相同键，脚本会打印冲突 warning。

输出字段：

```text
bound,scalar_ops,dtype,first_shape,runtime_shape,
cuda_eager_us,cuda_static_us,cuda_dynamic_us,cuda_dynamic_static_ratio,
npu_eager_us,npu_static_us,npu_dynamic_us,npu_dynamic_static_ratio,
npu_cuda_ratio_of_lift
```

其中 `npu_cuda_ratio_of_lift = npu_dynamic_static_ratio / cuda_dynamic_static_ratio`。
