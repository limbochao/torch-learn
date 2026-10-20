
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
    size_hints={'z0': 512, 'y1': 4, 'x2': 49, 'r3': 49},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'r3_numel': 'i32', 'Z0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_clone_index_matm_13', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': 'BACKEND_HASH_OMITTED', 'split_axis': [0], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [1], 'axis_names': ['z0', 'y1', 'x2', 'r3'], 'axis_static_values': (('y1', 4), ('x2', 49), ('r3', 49)), 'low_dims': {1, 2, 3}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Z0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__softmax_add_clone_index_matm_13(in_out_ptr0, in_ptr0, in_ptr1, z0_numel, y1_numel, x2_numel, r3_numel, Z0BLOCK, Z0BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr):
    R3BLOCK_SUB: tl.constexpr = 64
    y1_numel = 4
    Y1BLOCK_SUB: tl.constexpr = 4
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    base_x2= tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (x2_numel + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    base_r3= tl.arange(0, R3BLOCK_SUB)
    loops_r3 = (r3_numel + R3BLOCK_SUB - 1) // R3BLOCK_SUB
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        y1 = base_y1[None,:,None,None]
        y1_mask = y1 < y1_numel
        for loop_x2 in range(loops_x2):
            x2 = (loop_x2 * X2BLOCK_SUB) + base_x2[None,None,:,None]
            x2_mask = x2 < x2_numel
            r3 = base_r3[None,None,None,:]
            r3_mask = r3 < r3_numel
            tmp0 = tl.load(in_out_ptr0 + (r3 + 49*x2 + 2401*y1 + 9604*z0), r3_mask & x2_mask & y1_mask & z0_mask, other=0.0)
            tmp1 = tl.load(in_ptr0 + (r3 + 49*x2), r3_mask & x2_mask, other=0.0)
            tmp2 = tl.full([Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, R3BLOCK_SUB], 169, tl.int32)
            tmp3 = tmp1 + tmp2
            tmp4 = tmp1 < 0
            tmp5 = tl.where(tmp4, tmp3, tmp1)
            tl.device_assert(((0 <= tmp5) & (tmp5 < 169)) | ~(r3_mask & x2_mask), "index out of bounds: 0 <= tmp5 < 169")
            tmp7 = tl.load(in_ptr1 + (y1 + 4*tmp5), r3_mask & x2_mask & y1_mask)
            tmp8 = tmp0 + tmp7
            tmp10 = tl.max(tmp8, 3, propagate_nan=True).reshape(Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, 1)
            tmp11 = tmp8 - tmp10
            tmp12 = libdevice.exp(tmp11)
            tmp14 = tl.sum(tmp12, 3).reshape(Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, 1)
            tmp15 = (tmp12 / tmp14)
            tl.store(in_out_ptr0 + (r3 + 49*x2 + 2401*y1 + 9604*z0), tmp15, r3_mask & x2_mask & y1_mask & z0_mask)
