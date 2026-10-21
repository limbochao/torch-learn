"""Compare equivalent FP32 layer norms followed by the same native convolution."""
import torch
import torch.nn.functional as F
import torch_npu


def compare(label, left, right):
    delta = (left - right).abs()
    bad = delta > (0.001 + 0.001 * right.abs())
    print(label, "max_abs", delta.max().item(), "bad", bad.sum().item(), flush=True)


torch.manual_seed(1337)
x_cpu = torch.randn(8, 72, 72, 80)
x = x_cpu.npu()
a = F.layer_norm(x, (80,), eps=1e-6)
centered = x - x.mean(-1, keepdim=True)
b = centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + 1e-6)
golden = F.layer_norm(x_cpu.double(), (80,), eps=1e-6).float().npu()
compare("LN_native_vs_CPU64", a, golden)
compare("LN_formula_vs_CPU64", b, golden)
compare("LN_inputs", a, b)
w = torch.randn(320, 80, 1, 1, device="npu")
for flag in (True, False):
    torch.npu.conv.allow_hf32 = flag
    ya = F.conv2d(a.permute(0, 3, 1, 2).contiguous(), w)
    yb = F.conv2d(b.permute(0, 3, 1, 2).contiguous(), w)
    compare(f"conv_HF32={flag}", ya, yb)
