import triton
import triton.language as tl
from torch_npu._inductor.runtime import triton_helpers
from torch_npu._inductor.runtime.triton_helpers import extension

@triton.jit
def triton_tem_fused__to_copy_bmm_2(arg_A, arg_B, out_ptr0, out_ptr1):
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 256
    BLOCK_K : tl.constexpr = 256
    GROUP_M : tl.constexpr = 1
    ALLOW_HF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    EVEN_K : tl.constexpr = True
    L1_COPIES : tl.constexpr = 2
    SHIFT_M : tl.constexpr = False
    SHIFT_N : tl.constexpr = True
    MASK_M : tl.constexpr = False
    MASK_N : tl.constexpr = False
    K_MAIN : tl.constexpr = 512
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 512
    N = 1490
    K = 512

    stride_aq = 262144
    stride_am = 512
    stride_ak = 1

    stride_bq = 762880
    stride_bk = 1490
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # A ragged M or N tail cannot be left to the store mask alone: on the last
    # batch the overhanging rows and columns run off the end of the tensor and
    # the MTE reports an out-of-range DDR address (aicore error 95).  Where the
    # extent covers at least one whole tile the last tile is pulled back to end
    # on it, as the persistent template does, and then no load and no store is
    # masked at all.  The pulled-back tile recomputes the rows or columns it now
    # shares with its predecessor and stores the same values into them, which an
    # elementwise epilogue is indifferent to.
    #
    # The clamp is on the scalar base and not on rm: `tl.minimum` on the vector
    # makes the backend materialize the whole 2-D index tensor in UB and every
    # wide tile then fails to fit.  An extent shorter than its own tile has
    # nothing to pull back to and keeps the mask.

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)


    rn = tl.minimum(pid_n * BLOCK_N, N - BLOCK_N) + tl.arange(0, BLOCK_N)





    # batch dimension index — precompute batch offsets before K-loop
    # to avoid redundant multiply inside the hot loop
    idx_q = tl.program_id(1).to(INDEX_DTYPE)
    a_batch_off = idx_q * stride_aq
    b_batch_off = idx_q * stride_bq


    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    # Whole K tiles only.  A ragged K overhangs on its last tile and on no
    # other, so that tile is peeled out below instead of every iteration
    # carrying k_mask: the mask is a compare, a 2-D `and` against m_mask/n_mask
    # and a fill on both operands, and at K=1023 with BLOCK_K=128 seven of the
    # eight iterations would pay all of it to mask nothing.
    for k_start in range(0, K_MAIN, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a_ptrs = A + (rm[:, None] * stride_am + offs_k[None, :] * stride_ak
                      + a_batch_off)
        b_ptrs = B + (offs_k[:, None] * stride_bk + rn[None, :] * stride_bn
                      + b_batch_off)
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        # A second L1 slot per operand, so iteration k+1's MTE2 runs under
        # iteration k's MAC instead of waiting on the same buffer.  The
        # persistent and batched templates both stage their operands this way;
        # this loop was the one leaving the two pipes serialized.
        extension.multibuffer(a, L1_COPIES)
        extension.multibuffer(b, L1_COPIES)

        acc = tl.dot(a, b, acc=acc, out_dtype=ACC_TYPE)



    # rematerialize rm, rn and idx_q to save registers, on the same bases the
    # loads used so a pulled-back tile stores where it read

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)


    rn = tl.minimum(pid_n * BLOCK_N, N - BLOCK_N) + tl.arange(0, BLOCK_N)

    idx_q = tl.program_id(1).to(INDEX_DTYPE)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    # Only an axis that is neither whole nor pulled back still overhangs.  A
    # tile that overhangs nowhere wants no mask rather than a true one: the
    # comparison and its 2-D `and` are per-tile work, and they were being paid
    # even on shapes every axis of which divides.

    mask = None


    # inductor generates a suffix
    xindex = idx_n + 1490*idx_m + 762880*idx_q
    tmp0 = acc.to(tl.bfloat16)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
    tmp2 = tmp1.to(tl.float32)
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp2, mask)
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp2, mask)
