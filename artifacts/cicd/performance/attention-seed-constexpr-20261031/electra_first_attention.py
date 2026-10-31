
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

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*i1', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*i1', 'out_ptr5': '*fp32', 'load_seed_offset': 'constexpr', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {'load_seed_offset': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__softmax_all_amax_div_isfinite_matmul_native_dropout_sub_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 6, 'num_reduction': 4, 'autotune_hints': set(), 'backend_hash': 'REDACTED', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'requires_no_linear_block_remap': True, 'npu_kernel_type': 'simd_simt_mix', 'split_axis': [0], 'tiling_axis': [0, 1], 'low_dims': [1], 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'axis_names': ['x', 'r0_']},
)
@triton.jit
def triton_per_fused__softmax_all_amax_div_isfinite_matmul_native_dropout_sub_4(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr5, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
    tmp1 = tl.full([1, 1], 0.125, tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2 == tmp2
    tmp4 = tl_math.abs(tmp2)
    tmp5 = tl.full([1, 1], float("inf"), tl.float32)
    tmp6 = tmp4 != tmp5
    tmp7 = tmp3 & tmp6
    tmp8 = tmp7 == 0
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, False)
    tmp12 = triton_helpers.any(tmp11, 1)[:, None].to(tl.int1)
    tmp13 = tmp12 == 0
    tmp14 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(r0_mask & xmask, tmp14, float("-inf"))
    tmp17 = triton_helpers.max2(tmp16, 1)[:, None].to(tl.float32)
    tmp18 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp20 = tl.where(r0_mask & xmask, tmp18, float("-inf"))
    tmp21 = triton_helpers.max2(tmp20, 1)[:, None].to(tl.float32)
    tmp22 = tmp0 - tmp17
    tmp23 = tmp22 * tmp1
    tmp24 = tmp2 - tmp21
    tmp25 = tl.where(tmp13, tmp23, tmp24)
    tmp26 = libdevice.exp(tmp25)
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
    tmp29 = tl.where(r0_mask & xmask, tmp27, 0)
    tmp30 = tl.sum(tmp29, 1)[:, None].to(tl.float32)
    tmp31 = tl.load(in_ptr1 + load_seed_offset)
    tmp32 = (r0_1 + 512*x0).to(tl.int32)
    tmp33 = tl.rand(tmp31, (tmp32).to(tl.uint32))
    tmp34 = tl.full([1, 1], 0.1, tl.float32)
    tmp35 = tmp33 > tmp34
    tmp36 = tmp35.to(tl.float32)
    tmp37 = (tmp26 / tmp30)
    tmp38 = tmp36 * tmp37
    tmp39 = tl.full([1, 1], 1.1111111111111112, tl.float32)
    tmp40 = tmp38 * tmp39
    tl.store(in_out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr4 + (r0_1 + 512*x0), tmp35, r0_mask & xmask)
    tl.store(out_ptr5 + (r0_1 + 512*x0), tmp40, r0_mask & xmask)
    tl.store(out_ptr0 + (x0), tmp17, xmask)
    tl.store(out_ptr1 + (x0), tmp21, xmask)
    tl.store(out_ptr2 + (x0), tmp30, xmask)
