"""Inductor CPU FP64 scalar-tensor signature on NPU, without compilation."""
from types import SimpleNamespace

import torch
import torch_npu
from torch._inductor.codegen.common import TensorArg
from torch._inductor.codegen.triton_utils import signature_of
from torch._inductor.virtualized import V

npu = torch.device("npu")
graph = SimpleNamespace(current_device=npu, is_unspec_arg=lambda name: True,
                        get_current_device_or_throw=lambda: npu)
with V.set_graph_handler(graph):
    dtype = signature_of(TensorArg("bound", "bound", torch.float64), size_dtype="tl.int64")
print("UNWRAPPED_CPU_SCALAR_SIGNATURE", dtype)
assert dtype == "fp32", "CPU FP64 scalar is passed unchanged to an NPU kernel"
