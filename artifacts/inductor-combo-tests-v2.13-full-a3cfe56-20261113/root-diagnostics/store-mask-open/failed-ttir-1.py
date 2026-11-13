
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
    size_hints={'r0': 2},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'r0_numel': 'i32', 'R0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '139C9AD8FB0E3F910BF6CA8D4166949EB020778AFDF1D1A0ABA3AF3BC313E710', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['r0'], 'axis_static_values': (('r0', 2),), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('R0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def Placeholder.KERNEL_NAME(in_out_ptr0, in_ptr0, ks0, ks1, ks2, r0_numel, R0BLOCK):
    R0BLOCK_SUB: tl.constexpr = 2
    r0_offset = tl.program_id(0) * R0BLOCK
    base_r0= tl.arange(0, R0BLOCK_SUB)
    loops_r0 = (R0BLOCK + R0BLOCK_SUB - 1) // R0BLOCK_SUB
    r0 = base_r0
    r0_mask = True
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tl.static_assert(tmp0.dtype == tl.float32)
    tmp2 = triton_helpers.promote_to_tensor(tl.sum(tmp0, 0))
    tl.static_assert(tmp2.dtype == tl.float32)
    tmp3 = ks0*ks1*ks2
    tl.static_assert(tmp3.dtype == tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tl.static_assert(tmp4.dtype == tl.float32)
    tmp5 = (tmp4).to(tl.float32)
    tl.static_assert(tmp5.dtype == tl.float32)
    tl.static_assert(tmp5.dtype == tl.float32)
    tmp6 = (tmp2 / tmp5)
    tl.static_assert(tmp6.dtype == tl.float32)
    tl.static_assert(tmp6.dtype == tl.float32)
    tl.store(in_out_ptr0 + (tl.arange(0,1)), tmp6, None)
