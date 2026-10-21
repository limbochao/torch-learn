"""ConvNeXt FP32 inference accuracy; HF32=0 is a diagnostic control."""
import os

import timm
import torch
import torch_npu  # noqa: F401

torch.use_deterministic_algorithms(True)
torch.npu.conv.allow_hf32 = os.environ.get("HF32", "1") == "1"
torch.npu.matmul.allow_hf32 = False
model = timm.create_model("convnextv2_nano.fcmae_ft_in22k_in1k", pretrained=True).npu().eval()
torch.manual_seed(1337)
x = torch.randint(256, (8, 3, 288, 288), device="npu").float()
x = (x - x.mean()) / x.std()
with torch.no_grad():
    expected = model(x)
    actual = torch.compile(model, backend="inductor", fullgraph=True)(x)
torch.npu.synchronize()
delta = (actual - expected).abs()
bad = delta > (0.001 + 0.001 * expected.abs())
print("HF32", torch.npu.conv.allow_hf32, "MAX_ABS", delta.max().item(), "BAD", bad.sum().item())
torch.testing.assert_close(actual, expected, rtol=0.001, atol=0.001)
