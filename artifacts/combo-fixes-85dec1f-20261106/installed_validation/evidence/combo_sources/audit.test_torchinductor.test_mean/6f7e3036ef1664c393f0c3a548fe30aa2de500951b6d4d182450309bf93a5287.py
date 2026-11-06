
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

@triton_heuristics.pointwise(
    size_hints={'y0': 2, 'x1': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'Y0BLOCK': 'i32', 'Y0BLOCK_SUB': 'constexpr', 'X1BLOCK_SUB': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': [], 'backend_hash': '<removed>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 2), ('x1', 8)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': (), 'combo_grid_meta': {'num_kernels': 1, 'min_blocks': 0, 'default_config': {'XBLOCK': 'Y0BLOCK'}, 'split_numels': [2], 'no_split': [False], 'split_block': 'Y0BLOCK', 'no_x_dim_0': False, 'xnumel_0': 2}, 'combo_subkernel_meta': [{'axis_names': ['y0', 'x1'], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'split_axis': [0], 'low_dims': {1}, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'size_hints': {'y0': 2, 'x1': 8}, 'persistent_reduction': False}]}
)
@triton.jit
def Placeholder.KERNEL_NAME(in_ptr0, out_ptr0, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    pid = tl.program_id(0)
    num_blocks_0 = tl.cdiv(2, Y0BLOCK)
    if pid < num_blocks_0:
        pid_offset = pid
        y0_numel = 2
        x1_numel = 8
        y0_offset = pid_offset * Y0BLOCK
        base_y0= tl.arange(0, Y0BLOCK_SUB)
        loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
        base_x1= tl.arange(0, X1BLOCK_SUB)
        loops_x1 = (x1_numel + X1BLOCK_SUB - 1) // X1BLOCK_SUB
        for loop_y0 in range(loops_y0):
            y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
            y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
            for loop_x1 in range(loops_x1):
                x1 = (loop_x1 * X1BLOCK_SUB) + base_x1[None,:]
                x1_mask = x1 < x1_numel
                tmp0 = tl.load(in_ptr0 + (x1 + 32*y0), x1_mask & y0_mask)
                tl.static_assert(tmp0.dtype == tl.float32)
                tmp1 = tl.load(in_ptr0 + (8 + x1 + 32*y0), x1_mask & y0_mask)
                tl.static_assert(tmp1.dtype == tl.float32)
                tmp3 = tl.load(in_ptr0 + (16 + x1 + 32*y0), x1_mask & y0_mask)
                tl.static_assert(tmp3.dtype == tl.float32)
                tmp5 = tl.load(in_ptr0 + (24 + x1 + 32*y0), x1_mask & y0_mask)
                tl.static_assert(tmp5.dtype == tl.float32)
                tmp2 = tmp0 + tmp1
                tl.static_assert(tmp2.dtype == tl.float32)
                tl.static_assert(tmp2.dtype == tl.float32)
                tmp4 = tmp2 + tmp3
                tl.static_assert(tmp4.dtype == tl.float32)
                tl.static_assert(tmp4.dtype == tl.float32)
                tmp6 = tmp4 + tmp5
                tl.static_assert(tmp6.dtype == tl.float32)
                tl.static_assert(tmp6.dtype == tl.float32)
                tmp7 = 4.0
                tl.static_assert(tmp7.dtype == tl.float32)
                tl.static_assert(tmp7.dtype == tl.float32)
                tmp8 = (tmp6 / tmp7)
                tl.static_assert(tmp8.dtype == tl.float32)
                tl.static_assert(tmp8.dtype == tl.float32)
                tl.store(out_ptr0 + (x1 + 8*y0), tmp8, x1_mask & y0_mask)
    else:
        pass
