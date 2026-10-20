"""Albert dynamic training softmax: original tile fails UB allocation; --xblock 8 --check is a control.

Kernel and compiler arguments come from skill trace_compile/repro_compile export, record 1.
Only the export's unused compatibility/launch helpers were removed. Generated body is unchanged.
"""
import argparse

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource


@triton.jit
def triton_per_fused__softmax_all_amax_div_matmul__5(in_out_ptr0, in_ptr0, ks0, z0_numel, y1_numel, x2_numel, r3_numel, Z0BLOCK, Y1BLOCK, Z0BLOCK_SUB: tl.constexpr, Y1BLOCK_SUB: tl.constexpr, X2BLOCK_SUB: tl.constexpr, R3BLOCK_SUB: tl.constexpr):
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0 = tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(1) * Y1BLOCK
    base_y1 = tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2 = tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (x2_numel + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    base_r3 = tl.arange(0, R3BLOCK_SUB)
    loops_r3 = (r3_numel + R3BLOCK_SUB - 1) // R3BLOCK_SUB
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + loop_z0 * Z0BLOCK_SUB + base_z0[:, None, None, None]
        z0_mask = z0 < min(Z0BLOCK + z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = y1_offset + loop_y1 * Y1BLOCK_SUB + base_y1[None, :, None, None]
            y1_mask = y1 < min(Y1BLOCK + y1_offset, y1_numel)
            for loop_x2 in range(loops_x2):
                x2 = loop_x2 * X2BLOCK_SUB + base_x2[None, None, :, None]
                x2_mask = x2 < x2_numel
                r3 = base_r3[None, None, None, :]
                r3_mask = r3 < r3_numel
                tmp0 = tl.load(in_out_ptr0 + (r3 + ks0 * x2 + y1 * ks0 * ks0 + 64 * z0 * ks0 * ks0), r3_mask & x2_mask & y1_mask & z0_mask, other=0.0)
                tmp2 = tl.where(tl.broadcast_to(z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, Y1BLOCK_SUB, 1, 1]) & x2_mask.reshape([1, 1, X2BLOCK_SUB, 1]) & r3_mask.reshape([1, 1, 1, R3BLOCK_SUB]), [Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, R3BLOCK_SUB]), tmp0, float('-inf'))
                tmp3 = tl.max(tmp2, 3, propagate_nan=True).reshape(Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, 1)
                tmp4 = tl.full([1, 1, 1, 1], 0.125, tl.float32)
                tmp5 = tmp0 * tmp4
                tmp7 = tl.where(tl.broadcast_to(z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, Y1BLOCK_SUB, 1, 1]) & x2_mask.reshape([1, 1, X2BLOCK_SUB, 1]) & r3_mask.reshape([1, 1, 1, R3BLOCK_SUB]), [Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, R3BLOCK_SUB]), tmp5, float('-inf'))
                tmp8 = tl.max(tmp7, 3, propagate_nan=True).reshape(Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, 1)
                tmp9 = tl.load(in_ptr0 + (x2 + ks0 * y1 + 64 * ks0 * z0), x2_mask & y1_mask & z0_mask)
                tmp10 = tmp9 == 0
                tmp11 = tmp0 - tmp3
                tmp12 = tmp11 * tmp4
                tmp13 = tmp5 - tmp8
                tmp14 = tl.where(tmp10, tmp12, tmp13)
                tmp15 = libdevice.exp(tmp14)
                tmp17 = tl.where(tl.broadcast_to(z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, Y1BLOCK_SUB, 1, 1]) & x2_mask.reshape([1, 1, X2BLOCK_SUB, 1]) & r3_mask.reshape([1, 1, 1, R3BLOCK_SUB]), [Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, R3BLOCK_SUB]), tmp15, 0)
                tmp18 = tl.sum(tmp17, 3).reshape(Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, 1)
                tmp19 = tmp15 / tmp18
                tl.store(in_out_ptr0 + (r3 + ks0 * x2 + y1 * ks0 * ks0 + 64 * z0 * ks0 * ks0), tmp19, r3_mask & x2_mask & y1_mask & z0_mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xblock', type=int, default=512)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    torch.npu.set_device(0)
    signature = {
        'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'ks0': 'i64',
        'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'r3_numel': 'i32',
        'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32',
    }
    constants = dict(Z0BLOCK_SUB=1, Y1BLOCK_SUB=1, X2BLOCK_SUB=args.xblock, R3BLOCK_SUB=512)
    binary = triton.compile(
        ASTSource(triton_per_fused__softmax_all_amax_div_matmul__5, signature, constants),
        target=GPUTarget('npu', 'Ascend950PR_9579', 32),
        options=dict(num_warps=1, num_stages=2, debug=False,
                     compile_mode='simd_simt_template', multibuffer=True))
    print(f'COMPILE_PASS xblock={args.xblock}', flush=True)
    if args.check:
        torch.manual_seed(0)
        value = torch.randn((1, 64, 512, 512), device='npu')
        selector = torch.randint(0, 2, (1, 64, 512), device='npu', dtype=torch.int32).bool()
        expected = torch.softmax(value * 0.125, dim=-1)
        binary[(1, 64, 1)](value, selector, 512, 1, 64, 512, 512, 1, 1,
                           stream=torch.npu.current_stream().npu_stream)
        torch.npu.synchronize()
        torch.testing.assert_close(value, expected, atol=2e-6, rtol=2e-4)
        print(f'CHECK_PASS max_abs={(value - expected).abs().max().item()}', flush=True)
