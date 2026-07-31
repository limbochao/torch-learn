# Elementwise dynamic performance

本目录包含配套使用的两个脚本：

- `elementwise_op_cost_cases.py`：通过现有 CUDA/NPU profiler 工具采集单次 elementwise case 调用的
  device-side 平均耗时，并按 `RUN_ID` 合并到原始 `summary.csv`。
- `elementwise_op_cost_compare.py`：将原始 summary 浓缩为同场景、跨 execution、跨设备的宽表 CSV 和 XLSX。

每次采集只运行 `EXECUTION` 指定的路径。不同 execution 和设备使用相同 `RUN_ID`，并共享
`PROFILE_ROOT`：

```bash
RUN_ID=baseline_001 DEVICE=cuda EXECUTION=eager \
    python scripts/tests/elementwise_dynamic_perf/elementwise_op_cost_cases.py
RUN_ID=baseline_001 DEVICE=cuda EXECUTION=static \
    python scripts/tests/elementwise_dynamic_perf/elementwise_op_cost_cases.py
RUN_ID=baseline_001 DEVICE=cuda EXECUTION=dynamic COMPILE_SHAPE=8192 SYMBOLIC_DIMS=0 \
    python scripts/tests/elementwise_dynamic_perf/elementwise_op_cost_cases.py
RUN_ID=baseline_001 DEVICE=npu EXECUTION=dynamic COMPILE_SHAPE=8192 SYMBOLIC_DIMS=0 \
    python scripts/tests/elementwise_dynamic_perf/elementwise_op_cost_cases.py
RUN_ID=baseline_001 DEVICE=npu EXECUTION=custom COMPILE_SHAPE=8192 SYMBOLIC_DIMS=0 \
    python scripts/tests/elementwise_dynamic_perf/elementwise_op_cost_cases.py
RUN_ID=baseline_001 DEVICE=npu EXECUTION=group COMPILE_SHAPE=8192 SYMBOLIC_DIMS=0 \
    python scripts/tests/elementwise_dynamic_perf/elementwise_op_cost_cases.py
```

`EXECUTION=custom` 和 `EXECUTION=group` 仅支持 NPU。`custom` 与 `dynamic` 使用完全相同的
符号化和 `torch.compile(dynamic=None)` 路径，用于比较人为修改后的 torch_npu 行为。`group` 会在导入
`torch` 和 `torch_npu` 前设置
`INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE=1`，并使用与 `dynamic` 相同的符号化和
`torch.compile(dynamic=None)` 路径；普通 `dynamic` 会将该开关设置为 `0`。

`RECORD_RESULTS=0` 可用于临时试验：不要求设置 `RUN_ID`，只将结果打印到 stdout，不写 `summary.csv`，
临时 profiler 产物会在进程退出前删除。默认值为 `1`，保持现有结果记录行为。

记录结果时，脚本会在导入 `torch` 前启用 `TORCH_COMPILE_DEBUG=1`。每次编译后从实际 compile debug 根目录
选择包含最新 `output_code.py` 的完整 trace 目录，并复制到对应场景目录下，与 `profiles/` 并列的
`torch_compile_debug/`。static 保存每个 runtime shape 各自的编译产物；
dynamic/custom/group 将首次编译的同一份产物复制到所有 runtime shape 场景。eager 不生成 compile debug 产物。

profile 和 compile debug 产物按以下层级保存，便于一次查看单个 case 的完整信息：

```text
<PROFILE_ROOT>/<RUN_ID>/<device>/<execution>/[compile_<shape>/dims_<dims>/]<case>/
    <runtime>/
        profiles/
        torch_compile_debug/
```

每条 `summary.csv` 结果还包含一个 autotune tiling 字段：

- `autotune_tiling_configs`：紧凑 JSON 数组，保存每个 kernel 最终选中的 best config；每项包含 kernel、selected
  config、runtime blocks，NPU group 还包含
  `group_id` 和 `feature_inputs`。

static 的每个 runtime shape 都独立编译，因此各行记录各自编译时的 tiling。dynamic/custom 只在使用
`COMPILE_SHAPE` 首次编译时 autotune，因此记录写入该 case 的第一条 runtime shape 行，后续复用 kernel 的行为空。
group 会在每个 runtime shape 的调用窗口捕获该 shape 实际使用的 `group_id` 和 best config。eager 或编译、调用时
未触发 Inductor autotuner 时，该字段为空。

生成浓缩 CSV：

```bash
python scripts/tests/elementwise_dynamic_perf/elementwise_op_cost_compare.py \
    prof_log/elementwise_dynamic_perf/baseline_001/summary.csv
```

只进行 NPU 对比时，不需要补跑 CUDA。使用相同 `RUN_ID` 分别采集需要的 NPU execution，再执行同一个 compare
命令即可，例如：

```bash
RUN_ID=npu_only_001 DEVICE=npu EXECUTION=static \
    python scripts/tests/elementwise_dynamic_perf/elementwise_op_cost_cases.py
RUN_ID=npu_only_001 DEVICE=npu EXECUTION=dynamic COMPILE_SHAPE=8192 SYMBOLIC_DIMS=0 \
    python scripts/tests/elementwise_dynamic_perf/elementwise_op_cost_cases.py
RUN_ID=npu_only_001 DEVICE=npu EXECUTION=custom COMPILE_SHAPE=8192 SYMBOLIC_DIMS=0 \
    python scripts/tests/elementwise_dynamic_perf/elementwise_op_cost_cases.py
RUN_ID=npu_only_001 DEVICE=npu EXECUTION=group COMPILE_SHAPE=8192 SYMBOLIC_DIMS=0 \
    python scripts/tests/elementwise_dynamic_perf/elementwise_op_cost_cases.py

python scripts/tests/elementwise_dynamic_perf/elementwise_op_cost_compare.py \
    prof_log/elementwise_dynamic_perf/npu_only_001/summary.csv
```

纯 NPU 结果只生成已有 execution 对应的 `npu_*` 列，以及它们相对 NPU static 的 ratio；不会生成 `cuda_*`、
`npu_cuda_ratio_of_lift` 或 `npu_<execution>_cuda_ratio_of_lift` 列。

产物固定写入 summary 同级目录的 `elementwise_op_cost_comparison.csv` 和
`elementwise_op_cost_comparison.xlsx`。CSV 保持机器可读的完整重复值；XLSX 合并相邻的场景维度单元格，并增加
冻结表头、筛选、列宽和分组底色，tiling 使用 pretty JSON 展示；ratio 大于 `1.15` 或小于 `0.85` 时标红。
两种产物的数值和动态列完全一致，所有数值保留三位小数；
原始 summary 未运行的 device/execution 不生成对应列；已生成列中不存在的组合保持为空。输出唯一键为
`scalar_ops + dtype + first_shape + runtime_shape`；若不同 case 产生相同键，脚本会打印冲突 warning。

compare 会为已运行的 static/dynamic/custom/group 增加紧跟耗时列的 `*_tiling` 列。CSV 中 dynamic/custom 将同一
`first_shape` 轮次的首次编译 tiling 复制到该轮所有 runtime shape 行；group 保留各 runtime shape 对应
`group_id` 的 tiling。XLSX 中 dynamic/custom 的 tiling 按整个 `first_shape` 轮次合并单元格，group 则按相同
`group_id` 合并连续单元格；static 仍逐 runtime shape 展示。

下列为所有可能字段，实际输出会根据 summary 中已有的 device/execution 选择其子集：

```text
bound,scalar_ops,dtype,first_shape,runtime_shape,
cuda_eager_us,cuda_static_us,cuda_static_tiling,cuda_dynamic_us,cuda_dynamic_tiling,
cuda_dynamic_static_ratio,
npu_eager_us,npu_static_us,npu_static_tiling,npu_dynamic_us,npu_dynamic_tiling,
npu_dynamic_static_ratio,npu_cuda_ratio_of_lift,
npu_custom_us,npu_custom_tiling,npu_custom_static_ratio,npu_custom_cuda_ratio_of_lift,
npu_group_us,npu_group_tiling,npu_group_static_ratio,npu_group_cuda_ratio_of_lift
```

其中：

- `npu_custom_static_ratio = npu_custom_us / npu_static_us`。
- `npu_group_static_ratio = npu_group_us / npu_static_us`。
- `npu_cuda_ratio_of_lift = npu_dynamic_static_ratio / cuda_dynamic_static_ratio`。
- `npu_custom_cuda_ratio_of_lift = npu_custom_static_ratio / cuda_dynamic_static_ratio`。
- `npu_group_cuda_ratio_of_lift = npu_group_static_ratio / cuda_dynamic_static_ratio`。
