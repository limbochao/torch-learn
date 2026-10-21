"""Reduce MobileNet's failing backward candidate to one scalar conversion."""
import os

import torch
import torch_npu
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource


@triton.jit
def scalar_cast(out, value):
    offsets = tl.arange(0, 64)
    tl.store(out + offsets, value.to(tl.float32))


torch.npu.set_device(0)
dtype = os.environ.get("SCALAR_TYPE", "fp64")
source = ASTSource(
    scalar_cast, signature={"out": "*fp32", "value": dtype},
    constexprs={}, attrs={(0,): [["tt.divisibility", 16]]},
)
options = dict(num_warps=8, num_stages=2, debug=False, compile_mode="simt_only", enable_auto_blockify=True)
triton.compile(source, target=GPUTarget("npu", "Ascend950PR_9579", 32), options=options)
print("COMPILE_PASS", dtype)
