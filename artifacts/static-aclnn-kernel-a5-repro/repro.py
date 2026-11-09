"""Reproduce the static ACLNN wrapper lifecycle issue on NPU_A5."""

from pathlib import Path

import torch
import torch_npu
from torch_npu._inductor.npu_static_kernel import StaticKernelCompiler


counts = {"enter": 0, "exit": 0}
original_enter = StaticKernelCompiler.__enter__
original_exit = StaticKernelCompiler.__exit__


def traced_enter(self):
    counts["enter"] += 1
    print(f"TRACE enter #{counts['enter']}: {self.result_root}", flush=True)
    return original_enter(self)


def traced_exit(self, exc_type, exc_val, exc_tb):
    counts["exit"] += 1
    print(f"TRACE exit #{counts['exit']}: exc_type={exc_type}", flush=True)
    return original_exit(self, exc_type, exc_val, exc_tb)


StaticKernelCompiler.__enter__ = traced_enter
StaticKernelCompiler.__exit__ = traced_exit

torch.manual_seed(0)
torch._inductor.config.triton.cudagraph_trees = False
torch_npu.npu.aclnn._use_static_aclnn_kernel = True


def fn(x):
    return torch.neg(x)


x = torch.randn((1024, 1024), dtype=torch.float16, device="npu")
reference = fn(x)
compiled = torch.compile(fn, backend="inductor", dynamic=False)

for index in range(4):
    actual = compiled(x)
    torch.npu.synchronize()
    max_diff = (reference - actual).abs().max().item()
    print(f"CALL {index + 1}: max_diff={max_diff}, counts={counts}", flush=True)

build_dirs = Path.cwd().glob("*_kernel_aot_optimization_build_outputs")
output_codes = Path.cwd().glob("torch_compile_debug/**/output_code.py")
print("BUILD_DIRS", [str(path) for path in build_dirs])
print("OUTPUT_CODES", [str(path) for path in output_codes])
