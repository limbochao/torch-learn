# tools

本目录存放可被测试、复现和分析脚本复用的辅助工具。

## torch_npu 分支日构建包

`download_torch_npu_daily.py` 通过公开的华为云 OBS 对象列表查询指定 torch_npu 分支的 PyTorch
日构建包。默认选择包含当前 Python 版本的最新日构建；使用 `--resolve-only` 可以只查看下载地址：

```bash
python scripts/tools/download_torch_npu_daily.py master --python-version 3.11 --resolve-only
```

指定历史构建，或下载滚动更新的分支缓存：

```bash
python scripts/tools/download_torch_npu_daily.py v2.7.1 --build 20260804.2 --python-version 3.11
python scripts/tools/download_torch_npu_daily.py v2.7.1 \
  --source cache --python-version 3.11 --arch x86_64
```

使用 `--list-branches` 查看分支，使用 `BRANCH --list-builds N` 查看最近 N 个日构建。

## Triton kernel extraction

`extract_triton_kernel.py` 从 Inductor `output_code.py` 抽取一个独立 Triton kernel 文件。传入生成文件和 kernel 变量名；同一 kernel 有多次 `.run(...)` 调用时，只使用第一次调用的参数：

```bash
python scripts/tools/extract_triton_kernel.py \
  /path/to/output_code.py \
  triton_poi_fused_add_0 \
  -o /path/to/triton_poi_fused_add_0.py
```

输出包括公共导入、Triton DSL、`async_compile.wait(...)`、从 `benchmark_compiled_module` 提取的输入，以及首次调用依赖的局部变量和 launch 语句。未指定 `-o` 时写到当前目录的 `<kernel_name>.py`。

使用 `--include-eager` 可以根据 kernel 前的 `Graph fragment` metadata，在结果文件中额外生成
`eager_forward(...)`：

```bash
python scripts/tools/extract_triton_kernel.py \
  /path/to/output_code.py \
  triton_poi_fused_add_0 \
  --include-eager \
  -o /path/to/triton_poi_fused_add_0.py
```

如果只需要把指定 kernel 对应的 Graph fragment 生成通用性能工具使用的 eager case，可以使用
`--only-eager`。该模式不输出 Triton 定义和 launch，只输出 `eager_forward(...)`、`make_inputs(...)`、
`SAMPLE_BINDINGS`、`COMPILE_BINDINGS`、`DYNAMIC_DIMS` 和 `CASE`：

```bash
python scripts/tools/extract_triton_kernel.py \
  /path/to/output_code.py \
  triton_poi_fused_add_0 \
  --only-eager \
  -o /path/to/triton_poi_fused_add_0_case.py
```

`SAMPLE_BINDINGS` 和 `COMPILE_BINDINGS` 使用符号维度的字典列表。工具从首次 launch 的整数赋值链推导
符号基准值，并为每个符号生成基准值前后各一个示例；用户只需修改字典中的数值，不需要同步维护样本名。
包含表达式的维度（例如 `2*s0 + 1`）也按基础符号绑定，生成输入后再由工具记录实际 shape。无法安全推导
符号值时，`--only-eager` 会报错并要求输入文件中存在可解析的整数赋值。

生成的函数保留 Graph fragment 中的 placeholder 作为形参，并按 FX node 顺序调用对应的
`torch.ops`。工具根据 placeholder metadata 中的 shape、stride、dtype 和 device，通过 `rand_strided(...)`
构造独立的 eager 输入，不复用 kernel launch buffer。Graph fragment 中原本含符号表达式的 tensor 维度会先
通过 `torch._dynamo.mark_dynamic(...)` 标记，再调用 `torch.compile(eager_forward, dynamic=None)`。生成的
`eager_forward(...)` 和 `run_compiled_eager(...)` 可独立用于 eager 编译性能对比。

## CUDA profiler

`cuda_profiler.py` 提供 `TorchCudaProfiler` 采集 PyTorch CUDA profiler Chrome Trace，并通过
`CudaProfileParser` 解析 JSON。parser 只提取 device 侧满足 `ph="X"` 且 `cat="kernel"` 的完整 kernel
事件，并输出以下 CSV 字段：

- `kernel_name`: kernel 完整名称。
- `duration`: device kernel 执行时间，单位为微秒。
- `grid`: launch grid，例如 `[1,1,1]`。
- `block`: launch block，例如 `[128,1,1]`。

直接通过命令行转换：

```bash
python scripts/tools/cuda_profiler.py trace.json -o cuda_kernels.csv
```

不指定 `-o` 时，默认输出到输入文件同目录下的 `<trace_name>_kernels.csv`。

也可以在其它脚本中调用：

```python
from scripts.tools.cuda_profiler import CudaProfileParser

parser = CudaProfileParser("trace.json")
for record in parser.kernel_records():
    print(record.kernel_name, record.duration, record.grid, record.block)

parser.export_kernel_csv("cuda_kernels.csv")
```

`cuda_runtime`、`ac2g`、CPU operator 等非 kernel 事件不会写入 CSV。`grid` 或 `block` 缺失时对应字段为空。
`cuda_kernel_label` 可为汇总表生成短名称；它不修改 Chrome Trace 或 parser 导出的原始 kernel 名称。

## Autotune best tiling

`autotune_tiling.py` 提供 `BestTilingRecorder`，通过运行时 hook 记录 CUDA、NPU 普通 autotune 和 NPU
symbolic group autotune 最终选中的 tiling。工具只返回结构化记录，不落盘、不汇总、不定义业务字段：

```python
from scripts.tools.autotune_tiling import BestTilingRecorder

recorder = BestTilingRecorder("npu")
recorder.install()
try:
    recorder.start_capture()
    try:
        compiled_fn(*args)
    finally:
        tiling_records = recorder.stop_capture()
finally:
    recorder.uninstall()
```

每条记录包含 `device`、`kernel_name`、`selected_config` 和 `runtime_blocks`。NPU group 记录还包含
`group_id` 和 `feature_inputs`。调用方自行决定如何补充场景元数据和保存结果。recorder 使用 Inductor autotuner
的内部接口，升级 PyTorch 或 torch_npu 后需要通过目标设备测试确认 hook 仍然有效。

## NPU profiler

`npu_profiler.py` 提供两个公共工具：

- `TorchNpuProfiler`: 封装 `torch_npu.profiler.profile`，默认 CPU + NPU activity，`profiler_level=1`。
- `ProfileResultParser`: 解析 profiler 结果中的 `kernel_details.csv` 和 `step_trace_time.csv`。

`ProfileResultParser` 只使用 Python 标准库 `csv` 直接扫描结果文件，不依赖 pandas。

## 采集 profile

```python
import torch

from scripts.tools.npu_profiler import TorchNpuProfiler

profiler = TorchNpuProfiler("./prof_log")

with profiler.profile() as prof:
    for _ in range(10):
        compiled_fn(*args)
        torch.npu.synchronize()
        prof.step()
```

也可以使用 `run_steps`：

```python
profiler.run_steps(lambda: compiled_fn(*args), steps=10)
```

默认配置：

- `activities=("CPU", "NPU")`
- `profiler_level=1`
- `record_shapes=True`
- `with_stack=True`
- `wait=2`
- `warmup=1`
- `active=3`
- `repeat=1`

## 解析结果

```python
from scripts.tools.npu_profiler import ProfileResultParser

parser = ProfileResultParser("./prof_log")

for item in parser.kernel_time_by_name(name_prefix="triton"):
    print(item.key, item.count, item.mean_us)

print(parser.average_step_time_us())
```

按 shape 对比同名 kernel：

```python
for item in parser.kernel_time_by_shape(name_prefix="triton"):
    print(item.key, item.mean_us)
```

`kernel_time_by_shape` 会优先读取 `Input Shapes`、`Shape` 等常见 shape 列；如果结果中没有 shape 列，
则使用 profile 根目录下的一级子目录名作为 shape 标签，例如：

```text
prof_log/
  shape_1x32/ASCEND_PROFILER_OUTPUT/kernel_details.csv
  shape_2x32/ASCEND_PROFILER_OUTPUT/kernel_details.csv
```

如果 shape 信息来自其它位置，可以传入回调：

```python
parser.kernel_time_by_shape(shape_key=lambda row: row["__profile_label__"].split("_", 1)[1])
```
