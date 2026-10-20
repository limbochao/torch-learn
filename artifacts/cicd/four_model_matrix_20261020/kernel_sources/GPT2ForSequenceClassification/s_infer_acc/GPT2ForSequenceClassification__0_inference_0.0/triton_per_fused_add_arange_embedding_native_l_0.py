def triton_per_fused_add_arange_embedding_native_l_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x0), x0_mask)
        tmp3 = tl.load(in_ptr2 + (r1 + 768*x0), r1_mask & x0_mask, other=0.0)
        tmp18 = tl.load(in_ptr3 + (r1), r1_mask, other=0.0)
        tmp20 = tl.load(in_ptr4 + (r1), r1_mask, other=0.0)
        tl.device_assert(((0 <= tmp0) & (tmp0 < 50257)) | ~(x0_mask), "index out of bounds: 0 <= tmp0 < 50257")
        tmp2 = tl.load(in_ptr1 + (r1 + 768*tmp0), r1_mask & x0_mask, other=0.0)
        tmp4 = tmp2 + tmp3
        tmp6 = tl.sum(tmp4, 1).reshape(X0BLOCK_SUB, 1)
        tmp7 = tl.full([1, 1], 768.0, tl.float32)
        tmp8 = (tmp6 / tmp7)
        tmp9 = tmp4 - tmp8
        tmp10 = tmp9 * tmp9
        tmp12 = tl.sum(tmp10, 1).reshape(X0BLOCK_SUB, 1)
        tmp13 = (tmp12 / tmp7)
        tmp14 = tl.full([1, 1], 1e-05, tl.float32)
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp9 * tmp16
        tmp19 = tmp17 * tmp18
        tmp21 = tmp19 + tmp20
        tl.store(out_ptr1 + (r1 + 768*x0), tmp21, r1_mask & x0_mask)
