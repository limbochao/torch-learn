
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
    size_hints={'x': 2048, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': 'fp64', 'in_ptr2': 'fp64', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_div_expand_hardtanh_backward_native_batch_norm_backward_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 3, 'num_reduction': 2, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'backend_hash': '<redacted>', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'requires_no_linear_block_remap': True, 'npu_kernel_type': 'simd_simt_mix', 'split_axis': [0], 'tiling_axis': [0, 1], 'low_dims': [1], 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'axis_names': ['x', 'r0_']},
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_div_expand_hardtanh_backward_native_batch_norm_backward_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp1 = in_ptr1
    tmp4 = in_ptr2
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    tmp18 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp22 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = (r0_index % ks0)
        r0_4 = r0_index // ks0
        tmp0 = tl.load(in_ptr0 + (r0_3 + x0 + 1280*r0_4 + x0*(triton_helpers.div_floor_integer((-1) + ks1,  32))*(triton_helpers.div_floor_integer((-1) + ks1,  32)) + 2*x0*(triton_helpers.div_floor_integer((-1) + ks1,  32)) + 1280*r0_4*(triton_helpers.div_floor_integer((-1) + ks1,  32))*(triton_helpers.div_floor_integer((-1) + ks1,  32)) + 2560*r0_4*(triton_helpers.div_floor_integer((-1) + ks1,  32))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + 1280*r0_4), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr4 + (r0_3 + x0 + 1280*r0_4 + x0*(triton_helpers.div_floor_integer((-1) + ks1,  32))*(triton_helpers.div_floor_integer((-1) + ks1,  32)) + 2*x0*(triton_helpers.div_floor_integer((-1) + ks1,  32)) + 1280*r0_4*(triton_helpers.div_floor_integer((-1) + ks1,  32))*(triton_helpers.div_floor_integer((-1) + ks1,  32)) + 2560*r0_4*(triton_helpers.div_floor_integer((-1) + ks1,  32))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 <= tmp2
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp0 >= tmp5
        tmp7 = tmp3 | tmp6
        tmp9 = (ks0).to(tl.float32)
        tmp10 = (tmp9).to(tl.float32)
        tmp11 = (tmp8 / tmp10)
        tmp12 = tl.full([1, 1], 0.0, tl.float32)
        tmp13 = tl.where(tmp7, tmp12, tmp11)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask & xmask, tmp16, _tmp15)
        tmp19 = tmp17 - tmp18
        tmp20 = tmp13 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(r0_mask & xmask, tmp23, _tmp22)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp15, xmask)
    tl.store(out_ptr1 + (x0), tmp22, xmask)
    tmp24 = tl.load(in_ptr6 + (x0), xmask)
    tmp25 = tmp22 * tmp24
    tl.store(out_ptr2 + (x0), tmp25, xmask)
