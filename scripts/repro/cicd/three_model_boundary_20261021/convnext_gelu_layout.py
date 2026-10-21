"""Native GELU rejects the channels-last layout generated for ConvNeXt training."""
import os

import torch
import torch_npu  # noqa: F401

x = torch.empty_strided((2, 8, 4, 4), (128, 1, 32, 8), device="npu").fill_(0.5)
if os.environ.get("CONTIGUOUS") == "1":
    x = x.contiguous()
print("INPUT_STRIDE", x.stride(), flush=True)
y = torch.ops.aten.gelu.default(x)
torch.npu.synchronize()
print("PASS", y.stride(), y.flatten()[0].item())
