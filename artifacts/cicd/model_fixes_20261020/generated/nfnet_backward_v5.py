
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

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': 'fp32', 'in_ptr3': 'fp32', 'in_ptr4': '*fp32', 'in_ptr5': 'fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_add_mul_sigmoid_sigmoid_backward_silu_backward_sum_22', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'backend_hash': '<redacted>', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'requires_no_linear_block_remap': True, 'npu_kernel_type': 'simd_simt_mix', 'split_axis': [0], 'tiling_axis': [0, 1], 'low_dims': [1], 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'axis_names': ['x', 'r0_']},
)
@triton.jit
def triton_red_fused_add_mul_sigmoid_sigmoid_backward_silu_backward_sum_22(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = in_ptr2
    tmp6 = in_ptr3
    tmp13 = in_ptr5
    _tmp26 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + x0 + x0*(triton_helpers.div_floor_integer((-1) + ks0,  4))*(triton_helpers.div_floor_integer((-1) + ks0,  4)) + 2*x0*(triton_helpers.div_floor_integer((-1) + ks0,  4))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_out_ptr0 + (r0_1 + x0 + x0*(triton_helpers.div_floor_integer((-1) + ks0,  4))*(triton_helpers.div_floor_integer((-1) + ks0,  4)) + 2*x0*(triton_helpers.div_floor_integer((-1) + ks0,  4))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_out_ptr1 + (r0_1 + x0 + x0*(triton_helpers.div_floor_integer((-1) + ks0,  4))*(triton_helpers.div_floor_integer((-1) + ks0,  4)) + 2*x0*(triton_helpers.div_floor_integer((-1) + ks0,  4))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr4 + (r0_1 + x0 + x0*(triton_helpers.div_floor_integer((-1) + ks0,  4))*(triton_helpers.div_floor_integer((-1) + ks0,  4)) + 2*x0*(triton_helpers.div_floor_integer((-1) + ks0,  4))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = tmp3 * tmp4
        tmp7 = tmp5 * tmp6
        tmp9 = tmp7 + tmp8
        tmp12 = tmp10 + tmp11
        tmp14 = tmp12 * tmp13
        tmp15 = tl.sigmoid(tmp9)
        tmp16 = tmp14 * tmp15
        tmp17 = tl.full([1, 1], 1.0, tl.float32)
        tmp18 = tmp17 - tmp15
        tmp19 = tmp9 * tmp18
        tmp20 = tmp19 + tmp17
        tmp21 = tmp16 * tmp20
        tmp22 = tmp21 * tmp6
        tmp23 = tmp22 * tmp4
        tmp24 = tmp23 * tmp0
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(r0_mask & xmask, tmp27, _tmp26)
        tl.store(in_out_ptr1 + (r0_1 + x0 + x0*(triton_helpers.div_floor_integer((-1) + ks0,  4))*(triton_helpers.div_floor_integer((-1) + ks0,  4)) + 2*x0*(triton_helpers.div_floor_integer((-1) + ks0,  4))), tmp21, r0_mask & xmask)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tmp28 = tl.load(in_ptr1 + (x0), xmask)
    tmp29 = tl.sigmoid(tmp28)
    tmp30 = tl.full([1, 1], 1.0, tl.float32)
    tmp31 = tmp30 - tmp29
    tmp32 = tmp29 * tmp31
    tmp33 = tmp26 * tmp32
    tl.store(in_out_ptr2 + (x0), tmp33, xmask)
