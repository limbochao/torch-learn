import triton
import triton.language as tl
from torch_npu._inductor.runtime import triton_helpers
@triton.jit
def combo_store(in_ptr0, in_ptr1, out_ptr0, out_ptr1, X0BLOCK):
    pid = tl.program_id(0)
    num_blocks_0 = tl.cdiv(64, X0BLOCK)
    num_blocks_1 = num_blocks_0 + tl.cdiv(64, X0BLOCK)
    if pid < num_blocks_0:
        pid_offset = pid
        x0_numel = 64
        r2_numel = 32
        R2BLOCK_SUB: tl.constexpr = 32
        r1_numel = 128
        R1BLOCK_SUB: tl.constexpr = 128
        x0_offset = pid_offset * X0BLOCK
        base_r2= tl.arange(0, R2BLOCK_SUB)
        loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
        base_r1= tl.arange(0, R1BLOCK_SUB)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for x0 in range(x0_offset, min(x0_offset + X0BLOCK, x0_numel)):
            for loop_r2 in range(loops_r2):
                r2 = base_r2[:,None]
                r2_mask = r2 < r2_numel
                r1 = base_r1[None,:]
                r1_mask = r1 < r1_numel
                tmp0 = tl.load(in_ptr0 + (r1 + 128*x0 + 8192*r2), r1_mask & r2_mask, other=0.0)
                tmp1 = tl.reshape(tmp0, [R2BLOCK_SUB* R1BLOCK_SUB])
                tmp3 = tl.where(tl.reshape(tl.broadcast_to((r2_mask.reshape([R2BLOCK_SUB, 1]) & r1_mask.reshape([1, R1BLOCK_SUB])), [R2BLOCK_SUB, R1BLOCK_SUB]), [R2BLOCK_SUB* R1BLOCK_SUB]), tmp1, 0)
                tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
                tl.store(out_ptr0 + (x0 ), tmp4, None)
    elif pid < num_blocks_1:
        pid_offset = pid - num_blocks_0
        x0_numel = 64
        r2_numel = 32
        R2BLOCK_SUB: tl.constexpr = 32
        r1_numel = 128
        R1BLOCK_SUB: tl.constexpr = 128
        x0_offset = pid_offset * X0BLOCK
        base_r2= tl.arange(0, R2BLOCK_SUB)
        loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
        base_r1= tl.arange(0, R1BLOCK_SUB)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for x0 in range(x0_offset, min(x0_offset + X0BLOCK, x0_numel)):
            for loop_r2 in range(loops_r2):
                r2 = base_r2[:,None]
                r2_mask = r2 < r2_numel
                r1 = base_r1[None,:]
                r1_mask = r1 < r1_numel
                tmp5 = tl.load(in_ptr1 + (r1 + 128*x0 + 8192*r2), r1_mask & r2_mask, other=0.0)
                tmp6 = tl.reshape(tmp5, [R2BLOCK_SUB* R1BLOCK_SUB])
                tmp8 = tl.where(tl.reshape(tl.broadcast_to((r2_mask.reshape([R2BLOCK_SUB, 1]) & r1_mask.reshape([1, R1BLOCK_SUB])), [R2BLOCK_SUB, R1BLOCK_SUB]), [R2BLOCK_SUB* R1BLOCK_SUB]), tmp6, 0)
                tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
                tl.store(out_ptr1 + (x0 ), tmp9, None)
    else:
        pass
