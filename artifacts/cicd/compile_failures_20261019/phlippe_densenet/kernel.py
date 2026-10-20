
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
    size_hints={'y0': 131072, 'x1': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y4BLOCK': 'i32', 'Y5BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_t_14', 'mutated_arg_names': [], 'backend_hash': 'BACKEND_HASH_OMITTED', 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 131072), ('x1', 16)), 'low_dims': set(), 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y4BLOCK', 'Y5BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_t_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, y0_numel, x1_numel, Y4BLOCK, Y5BLOCK, Y0BLOCK_SUB : tl.constexpr):
    x1_numel = 16
    X1BLOCK_SUB: tl.constexpr = 16
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (y0_numel + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0_0 = (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0 = (loop_y0 * Y0BLOCK_SUB) + base_y0[None,:]
        y0_mask = y0 < y0_numel
        y0_0_mask = y0_0 < y0_numel
        x1_1 = base_x1[None,:]
        x1 = base_x1[:,None]
        # Coordinate transformation for different expansion patterns
        # Transform for index_6
        y4 = y0 // 1024
        y3 = (y0 % 1024) // 16
        y2 = ((y0 % 1024) % 16)
        # Transform for index_7
        y4 = y0 // 1024
        y5 = (y0 % 1024)
        tmp0 = tl.load(in_ptr0 + (2*x1 + 64*y0), y0_mask)
        tmp1 = tl.load(in_ptr0 + (1 + 2*x1 + 64*y0), y0_mask)
        tmp3 = tl.load(in_ptr0 + (32 + 2*x1 + 64*y0), y0_mask)
        tmp5 = tl.load(in_ptr0 + (33 + 2*x1 + 64*y0), y0_mask)
        tmp9 = tl.load(in_ptr1 + (y3), None)
        tmp11 = tl.load(in_ptr2 + (y3), None)
        tmp18 = tl.load(in_ptr3 + (y3), None)
        tmp20 = tl.load(in_ptr4 + (y3), None)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tl.full([1, 1], 4.0, tl.float32)
        tmp8 = (tmp6 / tmp7)
        tmp10 = tmp8 - tmp9
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = tl.sqrt_rn(tmp13)
        tmp15 = tl.full([1, 1], 1.0, tl.float32)
        tmp16 = (tmp15 / tmp14)
        tmp17 = tmp10 * tmp16
        tmp19 = tmp17 * tmp18
        tmp21 = tmp19 + tmp20
        tmp22 = tl.full([1, 1], 0, tl.int32)
        tmp23 = triton_helpers.maximum(tmp22, tmp21)
        tl.store(out_ptr0 + (x1_1 + 16*y0_0), tmp8.permute([1, 0]), y0_0_mask)
        tl.store(out_ptr1 + (x1 + 16*y2 + 256*y3 + 16384*y4), tmp23, None)
        tl.store(out_ptr2 + (x1 + 16*y5 + 20480*y4), tmp8, None)
