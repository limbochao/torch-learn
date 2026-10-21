
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
    size_hints={'t0': 128, 'z1': 80, 'y2': 72, 'x3': 72},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 't0_numel': 'i32', 'z1_numel': 'i32', 'y2_numel': 'i32', 'x3_numel': 'i32', 'T0BLOCK': 'i32', 'Y2BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_permute_7', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 2], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [], 'axis_names': ['t0', 'z1', 'y2', 'x3'], 'axis_static_values': (('z1', 80),), 'low_dims': {1, 3}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('T0BLOCK', 'Y2BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_permute_7(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, t0_numel, z1_numel, y2_numel, x3_numel, T0BLOCK, Y2BLOCK, T0BLOCK_SUB : tl.constexpr, Z1BLOCK_SUB : tl.constexpr, Y2BLOCK_SUB : tl.constexpr, X3BLOCK_SUB : tl.constexpr):
    t0_offset = tl.program_id(0) * T0BLOCK
    base_t0= tl.arange(0, T0BLOCK_SUB)
    loops_t0 = (T0BLOCK + T0BLOCK_SUB - 1) // T0BLOCK_SUB
    base_z1= tl.arange(0, Z1BLOCK_SUB)
    loops_z1 = (z1_numel + Z1BLOCK_SUB - 1) // Z1BLOCK_SUB
    y2_offset = tl.program_id(1) * Y2BLOCK
    base_y2= tl.arange(0, Y2BLOCK_SUB)
    loops_y2 = (Y2BLOCK + Y2BLOCK_SUB - 1) // Y2BLOCK_SUB
    base_x3= tl.arange(0, X3BLOCK_SUB)
    loops_x3 = (x3_numel + X3BLOCK_SUB - 1) // X3BLOCK_SUB
    for loop_t0 in range(loops_t0):
        t0 = t0_offset + (loop_t0 * T0BLOCK_SUB) + base_t0[:,None,None,None]
        t0_mask = t0 < min(T0BLOCK+t0_offset, t0_numel)
        for loop_z1 in range(loops_z1):
            z1 = (loop_z1 * Z1BLOCK_SUB) + base_z1[None,:,None,None]
            z1_mask = z1 < z1_numel
            for loop_y2 in range(loops_y2):
                y2 = y2_offset + (loop_y2 * Y2BLOCK_SUB) + base_y2[None,None,:,None]
                y2_mask = y2 < min(Y2BLOCK+y2_offset, y2_numel)
                for loop_x3 in range(loops_x3):
                    x3 = (loop_x3 * X3BLOCK_SUB) + base_x3[None,None,None,:]
                    x3_mask = x3 < x3_numel
                    tmp0 = tl.load(in_ptr0 + (x3 + y2*((1) * ((1) >= (ks0)) + (ks0) * ((ks0) > (1))) + z1*((1) * ((1) >= (ks0)) + (ks0) * ((ks0) > (1)))*((1) * ((1) >= (ks0)) + (ks0) * ((ks0) > (1))) + 80*t0*((1) * ((1) >= (ks0)) + (ks0) * ((ks0) > (1)))*((1) * ((1) >= (ks0)) + (ks0) * ((ks0) > (1)))), t0_mask & x3_mask & y2_mask & z1_mask)
                    tmp1 = tl.load(in_ptr1 + (z1 + 80*x3 + 80*ks0*y2 + 80*ks1*t0), t0_mask & x3_mask & y2_mask & z1_mask)
                    tmp2 = tmp0 + tmp1
                    tl.store(out_ptr0 + (x3 + ks0*y2 + ks1*z1 + 80*ks1*t0), tmp2, t0_mask & x3_mask & y2_mask & z1_mask)
