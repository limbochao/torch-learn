
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
    size_hints={'y1': 8, 'y4': 7, 'y3': 448, 'x2': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'y1_numel': 'i32', 'y4_numel': 'i32', 'y3_numel': 'i32', 'x2_numel': 'i32', 'Y1BLOCK': 'i32', 'Y3BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_clone_native_layer__9', 'mutated_arg_names': [], 'backend_hash': 'BACKEND_HASH_OMITTED', 'split_axis': [0, 2], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [1, 3], 'axis_names': ['y1', 'y4', 'y3', 'x2'], 'axis_static_values': (('y1', 8), ('y4', 7), ('y3', 448), ('x2', 128)), 'low_dims': {1, 3}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Y1BLOCK', 'Y3BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_clone_native_layer__9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, y1_numel, y4_numel, y3_numel, x2_numel, Y1BLOCK, Y3BLOCK, Y1BLOCK_SUB : tl.constexpr, Y3BLOCK_SUB : tl.constexpr):
    y4_numel = 7
    Y4BLOCK_SUB: tl.constexpr = 7
    x2_numel = 128
    X2BLOCK_SUB: tl.constexpr = 128
    y1_offset = tl.program_id(0) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_y4= tl.arange(0, Y4BLOCK_SUB)
    y3_offset = tl.program_id(1) * Y3BLOCK
    base_y3= tl.arange(0, Y3BLOCK_SUB)
    loops_y3 = (Y3BLOCK + Y3BLOCK_SUB - 1) // Y3BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    for loop_y1 in range(loops_y1):
        y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[:,None,None,None]
        y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
        y4_2 = base_y4[None,None,:,None]
        y4 = base_y4[None,None,None,:]
        for loop_y3 in range(loops_y3):
            y3_1 = y3_offset + (loop_y3 * Y3BLOCK_SUB) + base_y3[None,:,None,None]
            y3 = y3_offset + (loop_y3 * Y3BLOCK_SUB) + base_y3[None,None,:,None]
            y3_mask = y3 < min(Y3BLOCK+y3_offset, y3_numel)
            y3_1_mask = y3_1 < min(Y3BLOCK+y3_offset, y3_numel)
            x2_3 = base_x2[None,None,None,:]
            x2 = base_x2[None,:,None,None]
            tmp0 = tl.load(in_ptr0 + (y4 + 7*y3 + 3136*x2 + 401408*y1), y1_mask & y3_mask)
            tmp1 = tl.load(in_ptr1 + (x2), None)
            tmp2 = tl.load(in_ptr2 + (x2 + 128*(((((y4 + 7*y3) % 56)) % 7)) + 896*(((y3 // 8) % 7)) + 6272*((y3 % 8)) + 50176*(y3 // 56) + 401408*y1), y1_mask & y3_mask)
            tmp5 = tl.load(in_ptr3 + (y4 + 7*y3 + 3136*y1), y1_mask & y3_mask)
            tmp7 = tl.load(in_ptr4 + (y4 + 7*y3 + 3136*y1), y1_mask & y3_mask)
            tmp14 = tl.load(in_ptr5 + (x2), None)
            tmp16 = tl.load(in_ptr6 + (x2), None)
            tmp3 = tmp1 + tmp2
            tmp4 = tmp0 + tmp3
            tmp6 = tmp4 - tmp5
            tmp8 = tl.full([1, 1, 1, 1], 128.0, tl.float32)
            tmp9 = (tmp7 / tmp8)
            tmp10 = tl.full([1, 1, 1, 1], 1e-05, tl.float32)
            tmp11 = tmp9 + tmp10
            tmp12 = libdevice.rsqrt(tmp11)
            tmp13 = tmp6 * tmp12
            tmp15 = tmp13 * tmp14
            tmp17 = tmp15 + tmp16
            tl.store(out_ptr0 + (x2_3 + 128*y4_2 + 896*y3_1 + 401408*y1), tmp17.permute([0, 2, 3, 1]), y1_mask & y3_1_mask)
