"""Reproduce the pure extern/ACLNN NameError on NPU_A5."""

import torch
import torch_npu


torch.manual_seed(0)
torch._inductor.config.triton.cudagraph_trees = False
torch_npu.npu.aclnn._use_static_aclnn_kernel = True


def fn(a, b):
    return torch.mm(a, b)


a = torch.randn((128, 256), dtype=torch.float16, device="npu")
b = torch.randn((256, 64), dtype=torch.float16, device="npu")
compiled = torch.compile(fn, backend="inductor", dynamic=False)
actual = compiled(a, b)
torch.npu.synchronize()
torch.testing.assert_close(actual, fn(a, b))
