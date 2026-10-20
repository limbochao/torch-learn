def triton_per_fused_add_addmm_arange_embedding_na_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 1024
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[:,None]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        r1 = base_r1[None,:]
        r1_mask = r1 < r1_numel
        tmp0 = tl.load(in_ptr0 + (r1), r1_mask, other=0.0)
        tmp1 = tl.load(in_out_ptr0 + (r1 + 768*x0), r1_mask & x0_mask, other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp6 = tl.load(in_ptr3 + (r1 + 768*x0), r1_mask & x0_mask, other=0.0)
        tmp22 = tl.load(in_ptr4 + (r1), r1_mask, other=0.0)
        tmp24 = tl.load(in_ptr5 + (r1), r1_mask, other=0.0)
        tmp2 = tmp0 + tmp1
        tl.device_assert(((0 <= tmp3) & (tmp3 < 50257)) | ~(x0_mask), "index out of bounds: 0 <= tmp3 < 50257")
        tmp5 = tl.load(in_ptr2 + (r1 + 768*tmp3), r1_mask & x0_mask, other=0.0)
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 + tmp7
        tmp10 = tl.sum(tmp8, 1).reshape(X0BLOCK_SUB, 1)
        tmp11 = tl.full([1, 1], 768.0, tl.float32)
        tmp12 = (tmp10 / tmp11)
        tmp13 = tmp8 - tmp12
        tmp14 = tmp13 * tmp13
        tmp16 = tl.sum(tmp14, 1).reshape(X0BLOCK_SUB, 1)
        tmp17 = (tmp16 / tmp11)
        tmp18 = tl.full([1, 1], 1e-05, tl.float32)
        tmp19 = tmp17 + tmp18
        tmp20 = libdevice.rsqrt(tmp19)
        tmp21 = tmp13 * tmp20
        tmp23 = tmp21 * tmp22
        tmp25 = tmp23 + tmp24
        tl.store(in_out_ptr0 + (r1 + 768*x0), tmp8, r1_mask & x0_mask)
        tl.store(out_ptr1 + (r1 + 768*x0), tmp25, r1_mask & x0_mask)
