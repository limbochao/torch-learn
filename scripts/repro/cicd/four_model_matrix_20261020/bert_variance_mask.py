import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice

@triton.jit
def triton_per_fused_add_div_embedding_mean_mul_sl_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr2, y0_numel, x1_numel, r2_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    R2BLOCK_SUB: tl.constexpr = 1024
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    x1_offset = tl.program_id(1) * X1BLOCK
    base_x1= tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (X1BLOCK + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    base_r2= tl.arange(0, R2BLOCK_SUB)
    loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        for loop_x1 in range(loops_x1):
            x1 = x1_offset + (loop_x1 * X1BLOCK_SUB) + base_x1[None,:,None]
            x1_mask = x1 < min(X1BLOCK+x1_offset, x1_numel)
            r2 = base_r2[None,None,:]
            r2_mask = r2 < r2_numel
            tmp0 = tl.load(in_ptr0 + (x1 + 128*y0), x1_mask & y0_mask)
            tmp3 = tl.load(in_ptr2 + (r2 + 768*x1), r2_mask & x1_mask, other=0.0)
            tmp5 = tl.load(in_ptr3 + (x1 + 128*y0), x1_mask & y0_mask)
            tmp17 = tl.load(in_ptr5 + (r2), r2_mask, other=0.0)
            tmp25 = tl.load(in_ptr6 + (r2), r2_mask, other=0.0)
            tl.device_assert(((0 <= tmp0) & (tmp0 < 20005)) | ~(x1_mask & y0_mask), "index out of bounds: 0 <= tmp0 < 20005")
            tmp2 = tl.load(in_ptr1 + (r2 + 768*tmp0), r2_mask & x1_mask & y0_mask, other=0.0)
            tmp4 = tmp2 + tmp3
            tl.device_assert(((0 <= tmp5) & (tmp5 < 3)) | ~(x1_mask & y0_mask), "index out of bounds: 0 <= tmp5 < 3")
            tmp7 = tl.load(in_ptr4 + (r2 + 768*tmp5), r2_mask & x1_mask & y0_mask, other=0.0)
            tmp8 = tmp4 + tmp7
            tmp10 = tl.sum(tmp8, 2).reshape(Y0BLOCK_SUB, X1BLOCK_SUB, 1)
            tmp11 = tl.full([1, 1, 1], 768.0, tl.float32)
            tmp12 = (tmp10 / tmp11)
            tmp13 = tmp8 - tmp12
            tmp14 = tmp13 * tmp13
            tmp16 = tl.sum(tl.where(r2_mask, tmp14, 0.0), 2).reshape(Y0BLOCK_SUB, X1BLOCK_SUB, 1)
            tmp18 = tmp17 * tmp13
            tmp19 = tl.full([1, 1, 1], 767.0, tl.float32)
            tmp20 = (tmp16 / tmp19)
            tmp21 = tl.sqrt_rn(tmp20)
            tmp22 = tl.full([1, 1, 1], 1e-06, tl.float32)
            tmp23 = tmp21 + tmp22
            tmp24 = (tmp18 / tmp23)
            tmp26 = tmp24 + tmp25
            tl.store(out_ptr0 + (r2 + 768*x1 + 98304*y0), tmp8, r2_mask & x1_mask & y0_mask)
            tl.store(out_ptr2 + (r2 + 768*x1 + 98304*y0), tmp26, r2_mask & x1_mask & y0_mask)

if __name__ == '__main__':
    import argparse
    import json
    import torch
    import torch_npu
    parser = argparse.ArgumentParser()
    parser.add_argument('--compile-mode', default='simd_simt_template', choices=['simd_simt_template', 'simt_only'])
    args = parser.parse_args()
    torch.manual_seed(0)
    ids = torch.zeros((4, 128), device='npu', dtype=torch.int64)
    segments = torch.zeros_like(ids)
    words = torch.randn((4, 768), device='npu') * .1 + 1
    positions = torch.randn((1, 512, 768), device='npu') * .1 + 2
    segment_table = torch.randn((3, 768), device='npu') * .1 + 3
    weight = torch.ones(768, device='npu')
    bias = torch.zeros_like(weight)
    summed = torch.empty((4, 128, 768), device='npu')
    normalized = torch.empty_like(summed)
    value = words[ids] + positions[:, :128] + segment_table[segments]
    centered = value - value.mean(-1, keepdim=True)
    expected = weight * centered / (value.std(-1, keepdim=True, correction=1) + 1e-6) + bias
    triton_per_fused_add_div_embedding_mean_mul_sl_1[(4, 16)](
        ids, words, positions, segments, segment_table, weight, bias, summed, normalized,
        4, 128, 768, 1, 8, Y0BLOCK_SUB=1, X1BLOCK_SUB=1,
        num_warps=1, num_stages=2, compile_mode=args.compile_mode, multibuffer=False)
    torch.npu.synchronize()
    print(json.dumps({'compile_mode': args.compile_mode,
                      'sum_allclose': torch.allclose(summed, value, atol=2e-5, rtol=2e-5),
                      'allclose': torch.allclose(normalized, expected, atol=2e-5, rtol=2e-5),
                      'max_abs': (normalized - expected).abs().max().item()}))
