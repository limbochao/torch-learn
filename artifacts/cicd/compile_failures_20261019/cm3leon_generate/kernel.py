
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

@triton_heuristics.persistent_reduction(
    size_hints={'r0': 1536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*i32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr4': '*fp32', 'r0_numel': 'i32', 'R0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_mul_na_1', 'mutated_arg_names': [], 'backend_hash': 'BACKEND_HASH_OMITTED', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['r0'], 'axis_static_values': (('r0', 1536),), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('R0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_mul_na_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr4, r0_numel, R0BLOCK):
    R0BLOCK_SUB: tl.constexpr = 2048
    r0_offset = tl.program_id(0) * R0BLOCK
    base_r0= tl.arange(0, R0BLOCK_SUB)
    loops_r0 = (R0BLOCK + R0BLOCK_SUB - 1) // R0BLOCK_SUB
    r0 = base_r0
    r0_mask = r0 < min(R0BLOCK+r0_offset, r0_numel)
    tmp0 = tl.load(in_ptr0 + tl.arange(0,1) +  (63))
    tmp1 = tl.reshape(tmp0, [1 ])
    tmp4 = tl.load(in_ptr2 + tl.arange(0,1) +  (63))
    tmp5 = tl.reshape(tmp4, [1 ])
    tmp7 = tl.load(in_ptr3 + tl.arange(0,1) +  (63))
    tmp8 = tl.reshape(tmp7, [1 ])
    tmp29 = tl.load(in_ptr5 + (r0), r0_mask, other=0.0)
    tmp31 = tl.load(in_ptr6 + (r0), r0_mask, other=0.0)
    tl.device_assert((0 <= tmp1) & (tmp1 < 2048), "index out of bounds: 0 <= tmp1 < 2048")
    tmp3 = tl.load(in_ptr1 + (r0 + 1536*tmp1), r0_mask, other=0.0)
    tmp6 = tmp5.to(tl.int32)
    tmp9 = tmp6 * tmp8
    tmp10 = tmp9.to(tl.int64)
    tmp11 = tl.full([1], 2047, tl.int64)
    tmp12 = tmp10 + tmp11
    tl.device_assert((0 <= tmp12) & (tmp12 < 4096), "index out of bounds: 0 <= tmp12 < 4096")
    tmp14 = tl.load(in_ptr4 + (r0 + 1536*tmp12), r0_mask, other=0.0)
    tmp15 = tmp3 + tmp14
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tl.full([1], 1536.0, tl.float32)
    tmp19 = (tmp17 / tmp18)
    tmp20 = tmp15 - tmp19
    tmp21 = tmp20 * tmp20
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp24 = (tmp23 / tmp18)
    tmp25 = tl.full([1], 1e-05, tl.float32)
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp20 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tl.store(out_ptr0 + (r0), tmp15, r0_mask)
    tl.store(out_ptr4 + (r0), tmp32, r0_mask)
