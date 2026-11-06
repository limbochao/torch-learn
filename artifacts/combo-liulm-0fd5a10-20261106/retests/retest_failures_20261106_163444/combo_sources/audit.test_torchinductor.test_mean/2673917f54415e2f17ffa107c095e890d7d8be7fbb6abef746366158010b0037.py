
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
    size_hints={'x0': 2, 'r1': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'X0BLOCK': 'i64', 'X0BLOCK_SUB': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<removed>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 2), ('r1', 32)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': (), 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': 0, 'default_config': {'XBLOCK': 'X0BLOCK'}, 'split_numels': [2, 8], 'no_split': [False, False], 'split_block': 'X0BLOCK', 'no_x_dim_0': False, 'xnumel_0': 2, 'no_x_dim_1': False, 'xnumel_1': 8}}
)
@triton.jit
def Placeholder.KERNEL_NAME(in_out_ptr0, in_ptr0, out_ptr0, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    pid = tl.program_id(0)
    num_blocks_0 = tl.cdiv(2, X0BLOCK)
    num_blocks_1 = num_blocks_0 + tl.cdiv(8, X0BLOCK)
    if pid < num_blocks_0:
        pid_offset = pid
        x0_numel = 2
        r1_numel = 32
        R1BLOCK_SUB: tl.constexpr = 32
        x0_offset = pid_offset.to(tl.int64) * X0BLOCK
        base_x0= tl.arange(0, X0BLOCK_SUB).to(tl.int64)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        base_r1= tl.arange(0, R1BLOCK_SUB).to(tl.int64)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[:,None]
            x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
            r1 = base_r1[None,:]
            r1_mask = r1 < r1_numel
            tmp0 = tl.load(in_ptr0 + (r1 + 32*x0), x0_mask, other=0.0)
            tl.static_assert(tmp0.dtype == tl.float32)
            tmp2 = tl.sum(tmp0, 1).reshape(X0BLOCK_SUB, 1)
            tl.static_assert(tmp2.dtype == tl.float32)
            tl.store(out_ptr0 + (x0 ), tmp2, x0_mask)
    elif pid < num_blocks_1:
        pid_offset = pid - num_blocks_0
        x0_numel = 8
        r1_numel = 8
        R1BLOCK_SUB: tl.constexpr = 8
        x0_offset = pid_offset.to(tl.int64) * X0BLOCK
        base_x0= tl.arange(0, X0BLOCK_SUB).to(tl.int64)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        base_r1= tl.arange(0, R1BLOCK_SUB).to(tl.int64)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[:,None]
            x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
            r1 = base_r1[None,:]
            r1_mask = r1 < r1_numel
            tmp3 = tl.load(in_ptr0 + (r1 + 8*x0), x0_mask, other=0.0)
            tl.static_assert(tmp3.dtype == tl.float32)
            tmp5 = tl.sum(tmp3, 1).reshape(X0BLOCK_SUB, 1)
            tl.static_assert(tmp5.dtype == tl.float32)
            tmp6 = 8.0
            tl.static_assert(tmp6.dtype == tl.float32)
            tl.static_assert(tmp6.dtype == tl.float32)
            tmp7 = (tmp5 / tmp6)
            tl.static_assert(tmp7.dtype == tl.float32)
            tl.static_assert(tmp7.dtype == tl.float32)
            tl.store(in_out_ptr0 + (x0), tmp7, x0_mask)
            tl.store(in_out_ptr0 + (x0 ), tmp5, x0_mask)
    else:
        pass
