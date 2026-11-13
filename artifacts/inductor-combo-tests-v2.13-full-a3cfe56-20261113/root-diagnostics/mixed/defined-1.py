
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
    size_hints={'r0': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*fp32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': [], 'backend_hash': '139C9AD8FB0E3F910BF6CA8D4166949EB020778AFDF1D1A0ABA3AF3BC313E710', 'split_axis': [], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['r0'], 'axis_static_values': (('r0', 2048),), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': None, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': (), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': (), 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': 0, 'default_config': None, 'split_numels': [1, 1], 'no_split': [True, True], 'split_block': None, 'no_x_dim_0': True, 'xnumel_0': 1, 'no_x_dim_1': True, 'xnumel_1': 1}, 'combo_subkernel_meta': [{'axis_names': ['r0'], 'tiling_axis': [0], 'no_loop_axis': [], 'split_axis': [], 'low_dims': {0}, 'split_axis_dtype': None, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'size_hints': {'r0': 2048}, 'persistent_reduction': True}, {'axis_names': ['r0'], 'tiling_axis': [0], 'no_loop_axis': [], 'split_axis': [], 'low_dims': {0}, 'split_axis_dtype': None, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'size_hints': {'r0': 2048}, 'persistent_reduction': True}]}
)
@triton.jit
def Placeholder.KERNEL_NAME(in_ptr0, in_ptr1, out_ptr1, out_ptr3):
    pid = tl.program_id(0)
    num_blocks_0 = 1
    num_blocks_1 = num_blocks_0 + 1
    if pid < num_blocks_0:
        pid_offset = pid
        r0_numel = 2048
        R0BLOCK_SUB: tl.constexpr = 2048
        base_r0= tl.arange(0, R0BLOCK_SUB)
        loops_r0 = (r0_numel + R0BLOCK_SUB - 1) // R0BLOCK_SUB
        r0 = base_r0
        r0_mask = r0 < r0_numel
        tmp0 = tl.load(in_ptr0 + (r0), r0_mask, other=0.0)
        tmp2 = tl.where(tl.broadcast_to((r0_mask.reshape([R0BLOCK_SUB])), [R0BLOCK_SUB]), tmp0, 0)
        tmp3 = triton_helpers.promote_to_tensor(tl.sum(tmp2, 0))
        tl.store(out_ptr1 + (tl.arange(0,1) ), tmp3, None)
    elif pid < num_blocks_1:
        pid_offset = pid - num_blocks_0
        r0_numel = 2048
        R0BLOCK_SUB: tl.constexpr = 2048
        base_r0= tl.arange(0, R0BLOCK_SUB)
        loops_r0 = (r0_numel + R0BLOCK_SUB - 1) // R0BLOCK_SUB
        r0 = base_r0
        r0_mask = r0 < r0_numel
        tmp4 = tl.load(in_ptr1 + (r0), r0_mask, other=0.0)
        tmp6 = tl.where(tl.broadcast_to((r0_mask.reshape([R0BLOCK_SUB])), [R0BLOCK_SUB]), tmp4, 0)
        tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
        tl.store(out_ptr3 + (tl.arange(0,1) ), tmp7, None)
    else:
        pass
