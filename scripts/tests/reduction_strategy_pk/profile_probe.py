"""Check profiler device events and label attribution before collecting the matrix."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=("cuda", "npu"), required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
import torch

if args.backend == "npu":
    import torch_npu

api = getattr(torch, args.backend)
api.set_device(0)
x = torch.ones((7, 32768), device=args.backend)
for _ in range(5):
    x.sum(-1)
api.synchronize()
if args.backend == "cuda":
    context = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA])
else:
    context = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.out)),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
        record_shapes=False, with_stack=False)
with context as prof:
    for label in ("pk:sum:0", "pk:amax:0", "pk:sum:1"):
        with torch.profiler.record_function(label):
            for _ in range(4):
                x.amax(-1) if "amax" in label else x.sum(-1)
            api.synchronize()
        prof.step()
if args.backend == "cuda":
    prof.export_chrome_trace(str(args.out / "trace.json"))
print(json.dumps({"torch": torch.__version__, "device": str(api.get_device_properties(0)),
                  "outputs": [str(p.relative_to(args.out)) for p in args.out.rglob("*") if p.is_file()]}))
