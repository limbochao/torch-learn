
from torch._dynamo.testing import rand_strided
from torch_npu._C import _npu_getCurrentRawStream as get_raw_stream
import torch


import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

import torch
import torch_npu
if not torch_npu.npu.is_initialized() and torch_npu.npu._is_in_bad_fork():
    torch_npu.npu._initialized = True
from torch_npu._inductor.runtime import triton_heuristics as triton_heuristics
from torch_npu._inductor.runtime import triton_helpers
from torch_npu._inductor.runtime.triton_helpers import libdevice, extension, math as tl_math

@triton_heuristics.template(
    num_stages=2,
    num_warps=4,
    triton_meta={'signature': {'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*fp32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'npu_compile_options': {'enable_hivm_batch_matmul': True, 'unit_flag': False}},
    inductor_meta={'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'backend_hash': '<redacted>', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_num_gb': 0.114425856, 'BLOCK_M': 128, 'BLOCK_N': 256},
)
@triton.jit
def triton_(arg_A, arg_B, out_ptr0, out_ptr1):
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 256
    BLOCK_K : tl.constexpr = 256
    GROUP_M : tl.constexpr = 1
    ALLOW_HF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    EVEN_K : tl.constexpr = True
    L1_COPIES : tl.constexpr = 0
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
    tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 1490*idx_m + 762880*idx_q, [BLOCK_M, BLOCK_N])), tmp2, mask)
    tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 1490*idx_m + 762880*idx_q, [BLOCK_M, BLOCK_N])), tmp2, mask)


def get_args():
    arg_0 = rand_strided((32, 512, 512), (262144, 512, 1), device='npu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((32, 512, 1490), (762880, 1490, 1), device='npu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((32, 512, 1490), (762880, 1490, 1), device='npu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((32, 512, 1490), (762880, 1490, 1), device='npu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 24, 32, 1,


def call(args):
    with torch.npu.utils.device(0):
        torch.npu.set_device(0)
        raw_stream0 = get_raw_stream(0)
        triton_.run(*args, stream=raw_stream0)


def benchmark_all_configs(args):
    with torch.npu.utils.device(0):
        torch.npu.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='npu', rep=40)
    num_gb = 0.114425856
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
