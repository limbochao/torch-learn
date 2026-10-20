
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
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__unsafe_view_add_addmm_clone_native_layer_norm_permute_roll_view_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 11, 'num_store': 1, 'num_reduction': 2, 'autotune_hints': set(), 'backend_hash': 'BACKEND_HASH_OMITTED', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'requires_no_linear_block_remap': True, 'npu_kernel_type': 'simd_simt_mix', 'split_axis': [0], 'tiling_axis': [0, 1], 'low_dims': [1], 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'axis_names': ['x', 'r0_']},
)
@triton.jit
def triton_red_fused__unsafe_view_add_addmm_clone_native_layer_norm_permute_roll_view_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 100352
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 28)
    x1 = xindex // 28
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (128*(r0_2 // 256) + 256*x0 + 7168*(((r0_2 // 128) % 2)) + 14336*x1 + ((r0_2 % 128))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((r0_2 % 128)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (128*(r0_2 // 256) + 256*x0 + 7168*(((r0_2 // 128) % 2)) + 14336*x1 + ((r0_2 % 128))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp8 = tl.full([1, 1], 512.0, tl.float32)
    tmp9 = (tmp6 / tmp8)
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp10 = tl.load(in_ptr0 + (128*(r0_2 // 256) + 256*x0 + 7168*(((r0_2 // 128) % 2)) + 14336*x1 + ((r0_2 % 128))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr1 + ((r0_2 % 128)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (128*(r0_2 // 256) + 256*x0 + 7168*(((r0_2 // 128) % 2)) + 14336*x1 + ((r0_2 % 128))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp10 + tmp13
        tmp15 = tmp14 - tmp9
        tmp16 = tmp15 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(r0_mask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp20 = tl.load(in_ptr0 + (128*(r0_2 // 256) + 256*x0 + 7168*(((r0_2 // 128) % 2)) + 14336*x1 + ((r0_2 % 128))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr1 + ((r0_2 % 128)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.load(in_ptr2 + (128*(r0_2 // 256) + 256*x0 + 7168*(((r0_2 // 128) % 2)) + 14336*x1 + ((r0_2 % 128))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp32 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp34 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp23 = tmp21 + tmp22
        tmp24 = tmp20 + tmp23
        tmp25 = tmp24 - tmp9
        tmp26 = tl.full([1, 1], 512.0, tl.float32)
        tmp27 = (tmp18 / tmp26)
        tmp28 = tl.full([1, 1], 1e-05, tl.float32)
        tmp29 = tmp27 + tmp28
        tmp30 = libdevice.rsqrt(tmp29)
        tmp31 = tmp25 * tmp30
        tmp33 = tmp31 * tmp32
        tmp35 = tmp33 + tmp34
        tl.store(out_ptr1 + (r0_2 + 512*x3), tmp35, r0_mask & xmask)
