
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
    size_hints={'z0': 4, 'y1': 64, 'x2': 512, 'r3': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'ks0': 'i64', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'r3_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_all_amax_div_matmul__5', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [], 'axis_names': ['z0', 'y1', 'x2', 'r3'], 'axis_static_values': (('y1', 64),), 'low_dims': {2, 3}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__softmax_all_amax_div_matmul__5(in_out_ptr0, in_ptr0, ks0, z0_numel, y1_numel, x2_numel, r3_numel, Z0BLOCK, Y1BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr, R3BLOCK_SUB : tl.constexpr):
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(1) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (x2_numel + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    base_r3= tl.arange(0, R3BLOCK_SUB)
    loops_r3 = (r3_numel + R3BLOCK_SUB - 1) // R3BLOCK_SUB
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[None,:,None,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            for loop_x2 in range(loops_x2):
                x2 = (loop_x2 * X2BLOCK_SUB) + base_x2[None,None,:,None]
                x2_mask = x2 < x2_numel
                r3 = base_r3[None,None,None,:]
                r3_mask = r3 < r3_numel
                tmp0 = tl.load(in_out_ptr0 + (r3 + ks0*x2 + y1*ks0*ks0 + 64*z0*ks0*ks0), r3_mask & x2_mask & y1_mask & z0_mask, other=0.0)
                tmp2 = tl.where(tl.broadcast_to((z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, Y1BLOCK_SUB, 1, 1]) & x2_mask.reshape([1, 1, X2BLOCK_SUB, 1]) & r3_mask.reshape([1, 1, 1, R3BLOCK_SUB])), [Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, R3BLOCK_SUB]), tmp0, float("-inf"))
                tmp3 = tl.max(tmp2, 3, propagate_nan=True).reshape(Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, 1)
                tmp4 = tl.full([1, 1, 1, 1], 0.125, tl.float32)
                tmp5 = tmp0 * tmp4
                tmp7 = tl.where(tl.broadcast_to((z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, Y1BLOCK_SUB, 1, 1]) & x2_mask.reshape([1, 1, X2BLOCK_SUB, 1]) & r3_mask.reshape([1, 1, 1, R3BLOCK_SUB])), [Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, R3BLOCK_SUB]), tmp5, float("-inf"))
                tmp8 = tl.max(tmp7, 3, propagate_nan=True).reshape(Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, 1)
                tmp9 = tl.load(in_ptr0 + (x2 + ks0*y1 + 64*ks0*z0), x2_mask & y1_mask & z0_mask)
                tmp10 = tmp9 == 0
                tmp11 = tmp0 - tmp3
                tmp12 = tmp11 * tmp4
                tmp13 = tmp5 - tmp8
                tmp14 = tl.where(tmp10, tmp12, tmp13)
                tmp15 = libdevice.exp(tmp14)
                tmp17 = tl.where(tl.broadcast_to((z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, Y1BLOCK_SUB, 1, 1]) & x2_mask.reshape([1, 1, X2BLOCK_SUB, 1]) & r3_mask.reshape([1, 1, 1, R3BLOCK_SUB])), [Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, R3BLOCK_SUB]), tmp15, 0)
                tmp18 = tl.sum(tmp17, 3).reshape(Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, 1)
                tmp19 = (tmp15 / tmp18)
                tl.store(in_out_ptr0 + (r3 + ks0*x2 + y1*ks0*ks0 + 64*z0*ks0*ks0), tmp19, r3_mask & x2_mask & y1_mask & z0_mask)
