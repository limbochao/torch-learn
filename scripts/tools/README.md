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

输出包括公共导入、Triton DSL、`async_compile.wait(...)`、从 `benchmark_compiled_module` 提取的输入，以及首次调用依赖的局部变量和 launch 语句。输入按变量赋值关系提取，支持 `arg*`、`primals_*`、`where_*` 等从 `call(args)` 解包出的名称。未指定 `-o` 时写到当前目录的 `<kernel_name>.py`。

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

如果符号只出现在输入 shape、但首次 launch 中没有可解析的整数赋值，可以通过 JSON 显式提供基准值：

```bash
python scripts/tools/extract_triton_kernel.py \
  /path/to/output_code.py \
  triton_poi_fused_add_0 \
  --only-eager \
  --symbol-values '{"s11": 64}'
```

生成的函数保留 Graph fragment 中的 placeholder 作为形参，并按 FX node 顺序调用对应的
`torch.ops`。工具根据 placeholder metadata 中的 shape、stride、dtype 和 device，通过 `rand_strided(...)`
构造独立的 eager 输入，不复用 kernel launch buffer。Graph fragment 中原本含符号表达式的 tensor 维度会先
通过 `torch._dynamo.mark_dynamic(...)` 标记，再调用 `torch.compile(eager_forward, dynamic=None)`。生成的
`eager_forward(...)` 和 `run_compiled_eager(...)` 可独立用于 eager 编译性能对比。

Graph fragment 中没有 tensor metadata 的上游 placeholder 会被忽略。对于常见的 metadata 缺失，工具会从
动态 tensor 的首维恢复 shape 标量和 view/logical mask 别名；无 tensor 输入的 `full`/`zeros` 节点则从输出
metadata 反推 shape 表达式，并把基础符号生成为整数形参。无法可靠恢复的节点仍会直接报出缺失名称，不会生成
包含悬空变量的 eager case。生成文件同时补齐 `inf`、`nan`、`nanj` 常量导入。

部分捕获图会把 kernel 之后执行的外部自定义算子一起列入 Graph fragment。例如当前 Qianchuan 用例中的
`qianchuan_triton.softcap` 实际不在对应 Triton kernel 内，工具会把它作为 identity 边界处理，避免 eager case
依赖运行环境中未注册的外部算子。

## Compile mode performance

`compile_mode_perf.py` 接收 `--only-eager` 生成的 case，用一条命令顺序采集编译模式，并自动生成长表、宽表和
XLSX。NPU 默认采集 static、普通 dynamic 和 symbolic group；CUDA 只采集 static 和普通 dynamic，不执行
group，也不与 NPU 结果做对比：

```bash
python scripts/tools/compile_mode_perf.py \
  /path/to/triton_poi_fused_add_0_case.py \
  --device npu:0 \
  --run-id add_relu_001 \
  --output prof_log/compile_mode_perf
```

CUDA 用法：

```bash
python scripts/tools/compile_mode_perf.py \
  /path/to/triton_poi_fused_add_0_case.py \
  --device cuda:0 \
  --run-id add_relu_cuda_001 \
  --output prof_log/compile_mode_perf
```

也可以在同一次运行中传入多个 case 文件、case 目录或 glob 通配符。传入目录时会按字典序收集目录下的
`*_case.py` 文件；glob 支持 `*`、`?`、`[]` 和递归匹配 `**`，结果同样按字典序收集并去重。各 case 仍按顺序
独立执行，最终合并写入同一个
`summary.csv`、`comparison.csv` 和 `comparison.xlsx`；批量模式下每个 case 的 artifacts 放在
`cases/<index>_<case>/` 下。每个 case 完成后会立即打印该 case 的 static/group 摘要，全部完成后再打印
一张包含所有成功 case 的 batch 摘要。执行过程中会显示 `[当前序号/总数]` 进度；某个 case 失败时会记录到
`run.json` 和终端失败清单，并继续执行后续 case。批量执行默认在首轮全部 case 结束后，对失败 case
统一延迟重试 3 轮；不会在失败后立即打断当前批次。可通过 `--retries N` 修改重试轮数，传入 `0` 可关闭重试。

也可以传入一个 case 清单文件，文件内容按一行一个路径填写。空行会被忽略；相对路径按启动命令时的当前
目录解析，也支持绝对路径：

```text
scripts/tests/group_kernel/all_case/triton_poi_fused_add_213_case.py
/data/benchmarks/triton_poi_fused_add_217_case.py
```

```bash
python scripts/tools/compile_mode_perf.py cases.txt --device npu:0
```

```bash
python scripts/tools/compile_mode_perf.py \
  scripts/tests/group_kernel \
  --device npu:0 \
  --run-id selected_cases_001 \
  --output prof_log/compile_mode_perf
```

glob 参数建议使用引号，确保由脚本统一展开，而不是由 shell 提前展开：

```bash
python scripts/tools/compile_mode_perf.py \
  'scripts/tests/group_kernel/pointwise_*_case.py' \
  --run-id pointwise_cases_001
```

执行顺序固定为：一轮 static、按 `COMPILE_BINDINGS` 顺序执行的多轮 dynamic；NPU 最后再执行一轮 group，CUDA
不会启动 group worker。所有 worker 严格串行，前一个进程退出并清理 cache 后才启动下一个。group 只使用第一个
compile binding 编译一次，然后运行全部 `SAMPLE_BINDINGS`；汇总时将同一份 group 结果复用到各个 dynamic first
shape。CUDA 的 comparison 仅填写 static、dynamic 和 `dynamic_static_ratio`，group 列保持为空。

每个 worker 使用本次 run 目录下独立的临时 `TORCHINDUCTOR_CACHE_DIR`。kernel 定义会先复制到对应的
`torch_compile_debug/` 产物目录，worker 退出后立即删除 cache；无论运行成功、失败或中断，控制进程最后都会
再次清理整个临时 cache 根目录，不操作全局 `/tmp/torchinductor_root`。

结果写入 `<output>/<run-id>/`：

```text
run.json
raw_results.jsonl
summary.csv
comparison.csv
comparison.xlsx
artifacts/
  s/sNNN/
  d/fNNN/sNNN/
  g/sNNN/
```

为避免 Windows 下载或解压时触发路径长度限制，artifact 使用紧凑目录名：`s`、`d`、`g` 分别表示 static、
dynamic、group，`fNNN` 表示首次编译 binding 序号，`sNNN` 表示运行样本序号。完整 shape 仍保存在
`raw_results.jsonl`、`summary.csv` 和 `comparison.*` 中，不再重复写入目录名。

`raw_results.jsonl` 保留 binding 和每个 tensor 的 shape、stride、dtype、device；`summary.csv` 是按执行模式
展开的长表。`comparison.csv` 和 `comparison.xlsx` 使用以下固定列顺序：

```text
case,first_shape,shape,static_us,static_tiling,dynamic_us,dynamic_static_ratio,
dynamic_tiling,group_us,group_static_ratio,group_buckets,group_tiling
```

`first_shape` 和 `shape` 来自 `make_inputs(...)` 构造出的实际 tensor；多输入使用
`args[0]=...;args[1]=...;kwargs.name=...` 展示。XLSX 合并 `case`、`first_shape`、`dynamic_tiling` 和
`group_buckets`、`group_tiling`，冻结首行、启用筛选并 pretty-print JSON；除表头外的数据行固定为 40 行高。
`group_buckets` 按 kernel 记录 symbolic group feature 的名称、来源、轴、bucket 边界和
`bucket_factor`，用于结合 `group_tiling` 中的 `feature_inputs` 与 `group_id` 分析当前运行 shape 的分档。
`dynamic_static_ratio` 或
`group_static_ratio` 大于 `1.15` 时标红，小于或等于该值不设置条件格式。

如需单独测量 group 编译和首次 autotune 耗时，可使用 `--group-compile-time`。该模式只运行 group worker，
从 grouped plan 生成结束开始观察并行 binary 编译窗口，以及随后所有 group benchmark 和最优 tiling 选择，
直接在命令行打印阶段耗时和数量，不写入 comparison 表：

编译产物会保留在本次运行目录的 `artifacts/g/torch_compile_debug/` 下。
计时模式会打印 `grouped_kernel_count`、`candidate_count`、`compiled_kernel_count`、
`binary_compile_ms` 和 `group_benchmark_ms`。其中 `binary_compile_ms` 是并行 binary 编译窗口中从第一个
binary 开始到最后一个 binary 完成的 wall-clock 时间，不是各 binary 耗时累加；`group_benchmark_ms` 是从最后
一个 binary 完成到所有 group benchmark 完成并选出各组最优 tiling 的时间。

```bash
python scripts/tools/compile_mode_perf.py /path/to/case.py \
  --device npu:0 --group-compile-time
```

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

### CUDA op statistic

`cuda_op_statistic.py` 将 CUDA Chrome Trace 中的 device kernel 按关联的 CPU operator 聚合，生成与
torch_npu `op_statistic.csv` 相同的列。CUDA trace 的 kernel 和 CPU operator 通过 `External id` 关联；
`Total/Min/Avg/Max Time(us)` 使用 kernel device duration，`Ratio(%)` 以所有 kernel duration 之和为分母。
默认去掉 `aten::` 前缀，并使用 1-based `Device_id` 和 `CUDA_CORE` 作为 `Core Type`：

```bash
python scripts/tools/cuda_op_statistic.py \
  /path/to/trace.json \
  -o /path/to/op_statistic.csv
```

省略 `-o` 时输出到输入 trace 所在目录的 `op_statistic.csv`。若需要按 CUDA kernel 符号而不是 CPU
operator 聚合，可传入 `--name-by kernel`；无法通过 `External id` 找到 CPU operator 的 kernel 也会
自动回退为 kernel 名称。

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
`group_id`、`feature_inputs` 和 `group_features`。调用方自行决定如何补充场景元数据和保存结果。
recorder 使用 Inductor autotuner
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
