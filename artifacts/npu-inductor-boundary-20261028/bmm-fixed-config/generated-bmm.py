
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
    num_stages=4,
    num_warps=4,
    triton_meta={'signature': {'arg_A': '*fp16', 'arg_B': '*fp16', 'out_ptr0': '*fp16'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'npu_compile_options': {'unit_flag': False, 'enable_hivm_batch_matmul': True}},
    inductor_meta={'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'backend_hash': '<redacted>', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'BLOCK_M': 128, 'BLOCK_N': 256},
)
@triton.jit
def triton_npu_triton_bmm_batched(arg_A, arg_B, out_ptr0):
    BLOCK_Q : tl.constexpr = 2
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 256
    BLOCK_K : tl.constexpr = 144
    GROUPS_PER_PROGRAM : tl.constexpr = 55
    RAGGED_GROUPS : tl.constexpr = True
    N_GROUP : tl.constexpr = 28
    L1_COPIES : tl.constexpr = 2
    MASKED_M : tl.constexpr = False
    MASKED_N : tl.constexpr = True
    MASKED_K : tl.constexpr = False
    ALLOW_HF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 128
    N = 160
    K = 144
    BATCH = 3072

    stride_aq = 18432
    stride_am = 144
    stride_ak = 1

    stride_bq = 23040
    stride_bk = 160
    stride_bn = 1

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_q = tl.arange(0, BLOCK_Q)

    # Contiguous rather than strided by the program count, so one program walks
    # one slab of A and B instead of revisiting the whole tensor
    # GROUPS_PER_PROGRAM times.
    first_q = tl.program_id(0) * (GROUPS_PER_PROGRAM * BLOCK_Q)

    # Every group is one nd2nz -> nd2nz -> mmad -> fixpipe chain with no slack
    # in it; only a neighbouring group can fill the idle pipes, and only if the
    # compiler can see it, which means it has to be a loop body.

    # The group count has no divisor near the core count, which a power-of-two
    # batch never has, so every core gets a run and the last ones come up short.
    # A group is whole, so this is a trip count and nothing else, with no element
    # mask anywhere in the body.  Clamping the overrun onto the last group to
    # keep the count static measured the same and writes the same addresses from
    # two cores at once, so the shorter run is preferred.
    n_groups = tl.maximum(0, tl.minimum(GROUPS_PER_PROGRAM, (BATCH - first_q) // BLOCK_Q))

    for g in range(n_groups):
        q = first_q + g * BLOCK_Q + offs_q
        # See the persistent template: the batch-scaled offsets need the index
        # dtype, not q's.
        idx_q = q[:, None, None].to(INDEX_DTYPE)

        a_ptrs = A + idx_q * stride_aq + offs_m[None, :, None] * stride_am + offs_k[None, None, :] * stride_ak

        a = tl.load(a_ptrs)


        extension.multibuffer(a, L1_COPIES)


        b_ptrs = B + idx_q * stride_bq + offs_k[None, :, None] * stride_bk + offs_n[None, None, :] * stride_bn

        b = tl.load(b_ptrs, mask=(offs_k[None, :, None] < K) & (offs_n[None, None, :] < N), other=0.0)


        extension.multibuffer(b, L1_COPIES)


        acc = tl.zeros((BLOCK_Q, BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
        acc = tl.dot(a, b, acc=acc, out_dtype=ACC_TYPE)

        idx_m = offs_m[None, :, None]
        idx_n = offs_n[None, None, :]

        mask = idx_n < N

        xindex = idx_n + 160*idx_m + 20480*idx_q
        tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_Q, BLOCK_M, BLOCK_N])), acc, mask)