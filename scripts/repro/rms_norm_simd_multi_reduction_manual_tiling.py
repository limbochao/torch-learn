"""
Manual-tiling RMSNorm weight-gradient case for SIMD multi-reduction codegen.

This script bypasses torch.compile autotune and launches a single Triton kernel
with explicit tiling. The default tiling intentionally uses R2BLOCK_SUB=7 for
SEQ=128, so the r2 reduction loop has a tail tile.

Run:

    python scripts/repro/rms_norm_simd_multi_reduction_manual_tiling.py
"""

import os

import torch
import torch_npu
import triton
import triton.language as tl
from torch_npu._inductor.utils import get_current_raw_stream


def env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


@triton.jit
def triton_(
    in_ptr0,
    in_ptr1,
    in_ptr2,
    out_ptr1,
    x0_numel,
    r3_numel,
    r2_numel,
    r1_numel,
    X0BLOCK: tl.constexpr,
    X0BLOCK_SUB: tl.constexpr,
    R3BLOCK_SUB: tl.constexpr,
    R2BLOCK_SUB: tl.constexpr,
    R1BLOCK_SUB: tl.constexpr,
):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0 = tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r3 = tl.arange(0, R3BLOCK_SUB)
    loops_r3 = (r3_numel + R3BLOCK_SUB - 1) // R3BLOCK_SUB
    base_r2 = tl.arange(0, R2BLOCK_SUB)
    loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
    base_r1 = tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None, None, None, :]
        x0_mask = x0 < min(X0BLOCK + x0_offset, x0_numel)
        _tmp13 = tl.full(
            [X0BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB * R1BLOCK_SUB],
            0,
            tl.float32,
        )
        for loop_r3 in range(loops_r3):
            r3 = (loop_r3 * R3BLOCK_SUB) + base_r3[:, None, None, None]
            r3_mask = r3 < r3_numel
            for loop_r2 in range(loops_r2):
                r2_1 = (loop_r2 * R2BLOCK_SUB) + base_r2[None, :, None, None]
                r2 = (loop_r2 * R2BLOCK_SUB) + base_r2[None, None, :, None]
                r2_mask = r2 < r2_numel
                r2_1_mask = r2_1 < r2_numel
                for loop_r1 in range(loops_r1):
                    r1_2 = (loop_r1 * R1BLOCK_SUB) + base_r1[None, None, :, None]
                    r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[None, :, None, None]
                    r1_mask = r1 < r1_numel
                    r1_2_mask = r1_2 < r1_numel
                    tmp0 = tl.load(
                        in_ptr0
                        + (
                            x0
                            + 128 * r2
                            + 128 * 128 * r1
                            + 128 * 128 * 8 * r3
                        ),
                        r3_mask & r2_mask & x0_mask & r1_mask,
                        other=0.0,
                    )
                    tmp1 = tl.load(
                        in_ptr1
                        + (
                            x0
                            + 128 * r1_2
                            + 128 * 8 * r2_1
                            + 128 * 8 * 128 * r3
                        ),
                        r1_2_mask & r3_mask & r2_1_mask & x0_mask,
                        other=0.0,
                    ).to(tl.float32)
                    tmp2 = tmp1.permute([0, 2, 1, 3])
                    tmp5 = tl.load(
                        in_ptr2 + (r1 + 8 * r2 + 8 * 128 * r3),
                        r3_mask & r2_mask & r1_mask,
                        other=0.0,
                    )
                    tmp3 = tmp2.to(tl.float32)
                    tmp4 = tmp0 * tmp3
                    tmp6 = 1.0 / 128
                    tmp7 = tmp5 * tmp6
                    tmp8 = 1e-06
                    tmp9 = tmp7 + tmp8
                    tmp10 = tl.rsqrt(tmp9)
                    tmp11 = tmp4 * tmp10
                    tmp12 = tl.reshape(
                        tmp11.permute([3, 0, 1, 2]),
                        [X0BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB * R1BLOCK_SUB],
                    )
                    tmp14 = _tmp13 + tmp12
                    _tmp13 = tl.where(
                        (r1_mask & r2_mask & r3_mask & x0_mask)
                        .permute([3, 0, 1, 2])
                        .reshape(
                            [X0BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB * R1BLOCK_SUB]
                        ),
                        tmp14,
                        _tmp13,
                    )
        tmp13 = tl.sum(_tmp13, 1).reshape(1, 1, 1, X0BLOCK_SUB)
        tmp15 = tmp13.to(tl.float32)
        tl.store(out_ptr1 + (x0 + tl.arange(0, 1)), tmp15, x0_mask)


def rms_norm_weight_grad(grad_out_base, q, q_square_sum):
    grad_out = grad_out_base.permute(0, 2, 1, 3)
    inv_rms = torch.rsqrt(q_square_sum.unsqueeze(-1) / q.shape[-1] + 1e-6)
    grad_weight = (grad_out * q.float() * inv_rms).sum(dim=(0, 1, 2))
    return grad_weight.to(torch.bfloat16)


def main():
    batch = env_int("BATCH", 2)
    seq = env_int("SEQ", 128)
    heads = env_int("HEADS", 8)
    head_dim = env_int("HEAD_DIM", 128)
    if (batch, seq, heads, head_dim) != (2, 128, 8, 128):
        raise ValueError(
            "manual kernel constants currently require BATCH=2 SEQ=128 HEADS=8 HEAD_DIM=128"
        )

    x0block = env_int("X0BLOCK", 16)
    x0block_sub = env_int("X0BLOCK_SUB", 16)
    r3block_sub = env_int("R3BLOCK_SUB", 2)
    r2block_sub = env_int("R2BLOCK_SUB", 7)
    r1block_sub = env_int("R1BLOCK_SUB", 4)

    torch.manual_seed(0)
    device = "npu"
    grad_out_base = torch.randn(
        (batch, heads, seq, head_dim), device=device, dtype=torch.float32
    )
    q = torch.randn((batch, seq, heads, head_dim), device=device, dtype=torch.bfloat16)
    q_square_sum = (
        torch.rand((batch, seq, heads), device=device, dtype=torch.float32) + 1.0
    )
    out = torch.empty(head_dim, device=device, dtype=torch.bfloat16)

    grid = (triton.cdiv(head_dim, x0block),)
    print("case=rms_norm_simd_multi_reduction_manual_tiling")
    print(f"shape=batch={batch}, seq={seq}, heads={heads}, head_dim={head_dim}")
    print(
        "tiling="
        f"X0BLOCK={x0block}, X0BLOCK_SUB={x0block_sub}, "
        f"R3BLOCK_SUB={r3block_sub}, R2BLOCK_SUB={r2block_sub}, "
        f"R1BLOCK_SUB={r1block_sub}"
    )
    print(f"tail_r2={seq % r2block_sub}")

    triton_[grid](
        grad_out_base,
        q,
        q_square_sum,
        out,
        head_dim,
        batch,
        seq,
        heads,
        X0BLOCK=x0block,
        X0BLOCK_SUB=x0block_sub,
        R3BLOCK_SUB=r3block_sub,
        R2BLOCK_SUB=r2block_sub,
        R1BLOCK_SUB=r1block_sub,
        stream=get_current_raw_stream(0),
    )
    torch_npu.npu.synchronize()

    expected = rms_norm_weight_grad(grad_out_base, q, q_square_sum)
    torch_npu.npu.synchronize()
    print(f"out[:8]={out[:8]}")
    print(f"expected[:8]={expected[:8]}")
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    print("check=passed")


if __name__ == "__main__":
    main()
