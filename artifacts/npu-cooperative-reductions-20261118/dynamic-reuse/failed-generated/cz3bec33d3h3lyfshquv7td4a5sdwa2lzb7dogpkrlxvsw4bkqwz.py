
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
triton_helpers.set_driver_to_gpu()

@triton_heuristics.cooperative_reduction(
    size_hints={'x': 1, 'r0_': 1048576},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'sem_ptr': '*u32', 'ws_ptr': '*u8', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr', 'RSPLIT': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'launch_cooperative_grid': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'CooperativeReductionGrid', 'kernel_name': 'triton_unk_fused_add_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'persistent_reduction': False, 'backend_hash': '139C9AD8FB0E3F910BF6CA8D4166949EB020778AFDF1D1A0ABA3AF3BC313E710', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'requires_no_linear_block_remap': True, 'npu_kernel_type': 'simd_simt_mix', 'split_axis': [0], 'tiling_axis': [0, 1], 'low_dims': [1], 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'axis_names': ['x', 'r0_']},
)
@triton.jit
def triton_unk_fused_add_sum_0(in_ptr0, in_ptr1, out_ptr0, sem_ptr, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr, RSPLIT : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    RSPLIT_NEXT_POWER_OF_2: tl.constexpr = triton_helpers.constexpr_next_power_of_2(RSPLIT)
    RSPLIT_IS_POWER_OF_2: tl.constexpr = RSPLIT == RSPLIT_NEXT_POWER_OF_2
    HAS_RSPLIT: tl.constexpr = RSPLIT > 1
    rsplit_id = tl.program_id(0)
    num_rblocks = (rnumel + RBLOCK - 1) // RBLOCK
    rsplit_chunk = (num_rblocks + RSPLIT - 1) // RSPLIT * RBLOCK
    rsplit_start = rsplit_chunk * rsplit_id
    rsplit_end = rsplit_chunk * (rsplit_id + 1)
    rsplit_end = tl.where(rsplit_end < rnumel, rsplit_end, rnumel)
    xoffset = tl.program_id(1) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    rsplit_arange = tl.arange(0, RSPLIT_NEXT_POWER_OF_2)[None, :]
    rsplit_mask = xmask if RSPLIT_IS_POWER_OF_2 else ((rsplit_arange < RSPLIT) & xmask)
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(rsplit_start, rsplit_end, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    if HAS_RSPLIT:
        tmp4_ws = (ws_ptr + 0).to(tl.pointer_type(tl.float32))
        tl.store(tmp4_ws + (xindex * RSPLIT + rsplit_id), tmp4, xindex < xnumel)
    if HAS_RSPLIT:
        triton_helpers.x_grid_barrier(sem_ptr + tl.program_id(1))
    if HAS_RSPLIT:
        tmp4_peers = tl.load(tmp4_ws + (xindex * RSPLIT + rsplit_arange), rsplit_mask, eviction_policy='evict_first', other=triton_helpers.if_mask(rsplit_mask, 0))
        tmp4 = tl.sum(tmp4_peers, 1)[:, None]
    if rsplit_id == (0 % RSPLIT):
        tl.store(out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp4, None)
