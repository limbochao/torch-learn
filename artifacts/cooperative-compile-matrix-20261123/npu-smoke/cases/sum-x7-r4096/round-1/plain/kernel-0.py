def triton_per_fused_add_sum_0(in_ptr0, in_ptr1, out_ptr0, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 4096
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[:,None]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        r1 = base_r1[None,:]
        r1_mask = True
        tmp0 = tl.load(in_ptr0 + (r1 + 4096*x0), x0_mask, other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + 4096*x0), x0_mask, other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp2, 0)
        tmp5 = tl.sum(tmp4, 1).reshape(X0BLOCK_SUB, 1)
        tl.store(out_ptr0 + (x0), tmp5, x0_mask)
