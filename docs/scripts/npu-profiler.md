# NPU profiler 工具

`torch_learn.profiler` 提供两个公共工具：

- `TorchNpuProfiler`: 封装 `torch_npu.profiler.profile`，默认 CPU + NPU activity，`profiler_level=1`。
- `ProfileResultParser`: 解析 profiler 结果中的 `kernel_details.csv` 和 `step_trace_time.csv`。

`ProfileResultParser` 只使用 Python 标准库 `csv` 直接扫描结果文件，不依赖 pandas。

## 采集 profile

```python
import torch

from torch_learn.profiler import TorchNpuProfiler

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
from torch_learn.profiler import ProfileResultParser

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
