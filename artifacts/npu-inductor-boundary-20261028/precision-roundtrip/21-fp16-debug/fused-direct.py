import triton
import triton.language as tl
from torch_npu._inductor.runtime import triton_helpers
from torch_npu._inductor.runtime.triton_helpers import extension

@triton.jit
def triton_tem_fused_addmm_relu_sub_0(in_ptr0, arg_B, out_ptr1):
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 1
    ALLOW_HF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    MASKED_M : tl.constexpr = False
    MASKED_N : tl.constexpr = False
    MASKED_K : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_B
    B = arg_B

    M = 128
    N = 128
    K = 128
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 128
    stride_ak = 1
    stride_bk = 128
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    # Use super-grouping (GROUP_M) for improved L2 cache hit rate
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)




    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)


        a = tl.load(A + (rm[:, None] * stride_am + offs_k[None, :] * stride_ak))


        b = tl.load(B + (offs_k[:, None] * stride_bk + rn[None, :] * stride_bn))

        acc += tl.dot(a, b, out_dtype=ACC_TYPE)

    # rm, rn, and m_mask are reused from above (no rematerialization).
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 128*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tmp2 = tmp1.to(tl.float16)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = triton_helpers.maximum(tmp4, tmp3)
    tmp6 = tl.full([1], 1.0, tl.float32)
    tmp7 = tmp5 - tmp6
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp7, mask)
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp7, mask)
