"""
RMSNorm weight-gradient repro for SIMD multi-reduction codegen.

Target eager expression:

    grad_weight = (grad_out * q * rsqrt(q_square_sum / head_dim + 1e-6)).sum((0, 1, 2))

The input grad_out is produced by permuting a [B, H, S, D] tensor to logical
[B, S, H, D]. The compiled kernel therefore sees a non-contiguous input whose
stride matches the failing in_ptr0 pattern.

Run:

    CHECK=1 python scripts/repro/rms_norm_simd_multi_reduction_repro.py

Environment variables:

    BATCH=2 SEQ=4096 HEADS=64 HEAD_DIM=128 CHECK=0 DYNAMIC=0
"""

import os

import torch
import torch_npu


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def rms_norm_weight_grad(grad_out_base, q, q_square_sum):
    grad_out = grad_out_base.permute(0, 2, 1, 3)
    inv_rms = torch.rsqrt(q_square_sum.unsqueeze(-1) / q.shape[-1] + 1e-6)
    grad_weight = (grad_out * q.float() * inv_rms).sum(dim=(0, 1, 2))
    return grad_weight.to(torch.bfloat16)


def main():
    batch = env_int("BATCH", 2)
    seq = env_int("SEQ", 4096)
    heads = env_int("HEADS", 64)
    head_dim = env_int("HEAD_DIM", 128)
    check = env_bool("CHECK", False)
    dynamic = env_bool("DYNAMIC", False)

    torch.manual_seed(0)
    device = "npu"

    grad_out_base = torch.randn(
        (batch, heads, seq, head_dim), device=device, dtype=torch.float32
    )
    q = torch.randn((batch, seq, heads, head_dim), device=device, dtype=torch.bfloat16)
    q_square_sum = torch.rand((batch, seq, heads), device=device, dtype=torch.float32) + 1.0

    grad_out = grad_out_base.permute(0, 2, 1, 3)
    print("case=rms_norm_simd_multi_reduction_repro")
    print(f"shape=batch={batch}, seq={seq}, heads={heads}, head_dim={head_dim}")
    print(f"dynamic={int(dynamic)}")
    print(f"check={int(check)}")
    print(f"grad_out_base.shape={tuple(grad_out_base.shape)}")
    print(f"grad_out.shape={tuple(grad_out.shape)}, stride={grad_out.stride()}")
    print(f"q.shape={tuple(q.shape)}, stride={q.stride()}")
    print(f"q_square_sum.shape={tuple(q_square_sum.shape)}, stride={q_square_sum.stride()}")

    compiled = torch.compile(rms_norm_weight_grad, backend="inductor", dynamic=dynamic)
    out = compiled(grad_out_base, q, q_square_sum)
    torch_npu.npu.synchronize()
    print(f"compiled_out.shape={tuple(out.shape)}, dtype={out.dtype}")
    print(f"compiled_out[:8]={out[:8]}")

    if check:
        expected = rms_norm_weight_grad(grad_out_base, q, q_square_sum)
        torch_npu.npu.synchronize()
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
        print("check=passed")


if __name__ == "__main__":
    main()
