
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

@triton_heuristics.reduction(
    size_hints={'r0': 473},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*i64', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'out_ptr0': '*i1', 'out_ptr1': '*i64', 'out_ptr2': '*i64', 'out_ptr3': '*i64', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_eq_index_masked_fill_max_ne_s_0', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['r0'], 'axis_static_values': (), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': None, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': (), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_red_fused_eq_index_masked_fill_max_ne_s_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, r0_numel, R0BLOCK_SUB : tl.constexpr):
    base_r0= tl.arange(0, R0BLOCK_SUB)
    loops_r0 = (r0_numel + R0BLOCK_SUB - 1) // R0BLOCK_SUB
    _tmp13 = tl.full([R0BLOCK_SUB], 0, tl.int64)
    _tmp17 = tl.full([R0BLOCK_SUB], 0, tl.int64)
    for loop_r0 in range(loops_r0):
        r0 = (loop_r0 * R0BLOCK_SUB) + base_r0
        r0_mask = r0 < r0_numel
        tmp0 = tl.load(in_ptr0 + (r0), r0_mask, other=0.0)
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tl.full([1], False, tl.int1)
        tmp4 = tmp2 == tmp3
        tmp5 = tl.full([R0BLOCK_SUB], 2869, tl.int32)
        tmp6 = tmp0 + tmp5
        tmp7 = tmp0 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp0)
        tl.device_assert(((0 <= tmp8) & (tmp8 < 2869)) | ~(r0_mask), "index out of bounds: 0 <= tmp8 < 2869")
        tmp10 = tl.load(in_ptr1 + (tmp8), r0_mask)
        tmp11 = tl.where(tmp4, tmp1, tmp10)
        tmp12 = tl.reshape(tmp11, [R0BLOCK_SUB])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(tl.reshape((r0_mask), [R0BLOCK_SUB]), tmp14, _tmp13)
        tmp15 = tmp2.to(tl.int64)
        tmp16 = tl.reshape(tmp15, [R0BLOCK_SUB])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(tl.reshape((r0_mask), [R0BLOCK_SUB]), tmp18, _tmp17)
        tl.store(out_ptr0 + (r0), tmp2, r0_mask)
        tl.store(out_ptr1 + (r0), tmp11, r0_mask)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(_tmp13, 0))
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(_tmp17, 0))
    tl.store(out_ptr2 + (tl.arange(0,1) ), tmp13, None)
    tl.store(out_ptr3 + (tl.arange(0,1) ), tmp17, None)
