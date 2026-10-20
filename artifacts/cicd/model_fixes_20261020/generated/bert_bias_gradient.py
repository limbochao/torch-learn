
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
    size_hints={'x4': 12, 'x5': 64, 'r2': 4, 'r3': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'x4_numel': 'i32', 'x5_numel': 'i32', 'r2_numel': 'i32', 'r3_numel': 'i32', 'X4BLOCK': 'i32', 'X5BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_matmul_backward_sum_tra_11', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [1, 2, 3], 'no_loop_axis': [], 'axis_names': ['x4', 'x5', 'r2', 'r3'], 'axis_static_values': (('x4', 12), ('x5', 64), ('r2', 4), ('r3', 128)), 'low_dims': {1}, 'numof_reduction_axis': 2, 'split_axis_dtype': torch.float32, 'dual_reduction': True, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('X4BLOCK', 'X5BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_clone_matmul_backward_sum_tra_11(in_ptr0, out_ptr0, out_ptr1, x4_numel, x5_numel, r2_numel, r3_numel, X4BLOCK, X5BLOCK, X5BLOCK_SUB : tl.constexpr):
    R2BLOCK_SUB: tl.constexpr = 4
    R3BLOCK_SUB: tl.constexpr = 128
    x4_offset = tl.program_id(0) * X4BLOCK
    x5_offset = tl.program_id(1) * X5BLOCK
    base_x5= tl.arange(0, X5BLOCK_SUB)
    loops_x5 = (X5BLOCK + X5BLOCK_SUB - 1) // X5BLOCK_SUB
    base_r2= tl.arange(0, R2BLOCK_SUB)
    loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
    base_r3= tl.arange(0, R3BLOCK_SUB)
    loops_r3 = (r3_numel + R3BLOCK_SUB - 1) // R3BLOCK_SUB
    for x4 in range(x4_offset, min(x4_offset + X4BLOCK, x4_numel)):
        for loop_x5 in range(loops_x5):
            x5 = x5_offset + (loop_x5 * X5BLOCK_SUB) + base_x5[None,None,:]
            x5_mask = x5 < min(X5BLOCK+x5_offset, x5_numel)
            for loop_r2 in range(loops_r2):
                r2 = base_r2[:,None,None]
                r2_mask = True
                r3 = base_r3[None,:,None]
                r3_mask = True
                tmp0 = tl.load(in_ptr0 + (x5 + 64*r3 + 8192*x4 + 98304*r2), x5_mask, other=0.0)
                tmp1 = tl.reshape(tmp0.permute([2, 0, 1]), [X5BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB])
                tmp3 = tl.where(tl.reshape((tl.broadcast_to((x5_mask.reshape([1, 1, X5BLOCK_SUB])), [R2BLOCK_SUB, R3BLOCK_SUB, X5BLOCK_SUB])).permute([2, 0, 1]), [X5BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB]), tmp1, 0)
                tmp4 = tl.sum(tmp3, 1).reshape(1, 1, X5BLOCK_SUB)
                tl.store(out_ptr1 + (x5 + 64*x4 ), tmp4, x5_mask)
                tl.store(out_ptr0 + (x5 + 64*x4 + 768*r3 + 98304*r2), tmp0, x5_mask)
