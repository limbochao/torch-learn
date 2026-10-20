import torch, torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
@triton.jit
def triton_per_fused__softmax_add_clone_index_matm_13(in_out_ptr0, in_ptr0, in_ptr1, z0_numel, y1_numel, x2_numel, r3_numel, Z0BLOCK, Z0BLOCK_SUB: tl.constexpr, X2BLOCK_SUB: tl.constexpr):
    R3BLOCK_SUB: tl.constexpr = 64
    y1_numel = 4
    Y1BLOCK_SUB: tl.constexpr = 4
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0 = tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    base_y1 = tl.arange(0, Y1BLOCK_SUB)
    base_x2 = tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (x2_numel + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    base_r3 = tl.arange(0, R3BLOCK_SUB)
    loops_r3 = (r3_numel + R3BLOCK_SUB - 1) // R3BLOCK_SUB
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + loop_z0 * Z0BLOCK_SUB + base_z0[:, None, None, None]
        z0_mask = z0 < min(Z0BLOCK + z0_offset, z0_numel)
        y1 = base_y1[None, :, None, None]
        y1_mask = y1 < y1_numel
        for loop_x2 in range(loops_x2):
            x2 = loop_x2 * X2BLOCK_SUB + base_x2[None, None, :, None]
            x2_mask = x2 < x2_numel
            r3 = base_r3[None, None, None, :]
            r3_mask = r3 < r3_numel
            tmp0 = tl.load(in_out_ptr0 + (r3 + 49 * x2 + 2401 * y1 + 9604 * z0), r3_mask & x2_mask & y1_mask & z0_mask, other=0.0)
            tmp1 = tl.load(in_ptr0 + (r3 + 49 * x2), r3_mask & x2_mask, other=0.0)
            tmp2 = tl.full([Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, R3BLOCK_SUB], 169, tl.int32)
            tmp3 = tmp1 + tmp2
            tmp4 = tmp1 < 0
            tmp5 = tl.where(tmp4, tmp3, tmp1)
            tl.device_assert((0 <= tmp5) & (tmp5 < 169) | ~(r3_mask & x2_mask), 'index out of bounds: 0 <= tmp5 < 169')
            tmp7 = tl.load(in_ptr1 + (y1 + 4 * tmp5), r3_mask & x2_mask & y1_mask)
            tmp8 = tmp0 + tmp7
            tmp10 = tl.max(tmp8, 3, propagate_nan=True).reshape(Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, 1)
            tmp11 = tmp8 - tmp10
            tmp12 = libdevice.exp(tmp11)
            tmp14 = tl.sum(tmp12, 3).reshape(Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, 1)
            tmp15 = tmp12 / tmp14
            tl.store(in_out_ptr0 + (r3 + 49 * x2 + 2401 * y1 + 9604 * z0), tmp15, r3_mask & x2_mask & y1_mask & z0_mask)
