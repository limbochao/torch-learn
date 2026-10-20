def triton_per_fused_clone_matmul_backward_sum_tra_11(in_ptr0, out_ptr0, out_ptr1, x4_numel, x5_numel, r2_numel, r3_numel, X4BLOCK, X5BLOCK, X5BLOCK_SUB : tl.constexpr):
    R2BLOCK_SUB: tl.constexpr = 4
    R3BLOCK_SUB: tl.constexpr = 128
    x4_offset = tl.program_id(0) * X4BLOCK
    x5_offset = tl.program_id(1) * X5BLOCK
    base_x5= tl.arange(0, X5BLOCK_SUB)
    loops_x5 = (X5BLOCK + X5BLOCK_SUB - 1) // X5BLOCK_SUB
    base_r2= tl.arange(0, R2BLOCK_SUB)
    loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
    base_r3= tl.arange(0, R3BLOCK_SUB)
    loops_r3 = (r3_numel + R3BLOCK_SUB - 1) // R3BLOCK_SUB
    for x4 in range(x4_offset, min(x4_offset + X4BLOCK, x4_numel)):
        for loop_x5 in range(loops_x5):
            x5 = x5_offset + (loop_x5 * X5BLOCK_SUB) + base_x5[None,None,:]
            x5_mask = x5 < min(X5BLOCK+x5_offset, x5_numel)
            for loop_r2 in range(loops_r2):
                r2 = base_r2[:,None,None]
                r2_mask = r2 < r2_numel
                r3 = base_r3[None,:,None]
                r3_mask = r3 < r3_numel
                tmp0 = tl.load(in_ptr0 + (x5 + 64*r3 + 8192*x4 + 98304*r2), x5_mask, other=0.0)
                tmp1 = tl.reshape(tmp0, [R2BLOCK_SUB* R3BLOCK_SUB* X5BLOCK_SUB])
                tmp3 = triton_helpers.promote_to_tensor(tl.sum(tmp1, 0))
                tl.store(out_ptr1 + (x5 + 64*x4 ), tmp3, x5_mask)
                tl.store(out_ptr0 + (x5 + 64*x4 + 768*r3 + 98304*r2), tmp0, x5_mask)
