
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
    size_hints={'x0': 512, 'r3': 8, 'r5': 7, 'r4': 2, 'r2': 14},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'x0_numel': 'i32', 'r3_numel': 'i32', 'r5_numel': 'i32', 'r4_numel': 'i32', 'r2_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_clone_native_lay_47', 'mutated_arg_names': [], 'backend_hash': 'BACKEND_HASH_OMITTED', 'split_axis': [0], 'tiling_axis': [0, 1, 2, 3, 4], 'no_loop_axis': [], 'axis_names': ['x0', 'r3', 'r5', 'r4', 'r2'], 'axis_static_values': (('x0', 512), ('r3', 8), ('r5', 7), ('r4', 2), ('r2', 14)), 'low_dims': {0}, 'numof_reduction_axis': 4, 'split_axis_dtype': torch.float32, 'dual_reduction': True, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__unsafe_view_clone_native_lay_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, x0_numel, r3_numel, r5_numel, r4_numel, r2_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R2BLOCK_SUB: tl.constexpr = 14
    R3BLOCK_SUB: tl.constexpr = 8
    R4BLOCK_SUB: tl.constexpr = 2
    R5BLOCK_SUB: tl.constexpr = 7
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r3= tl.arange(0, R3BLOCK_SUB)
    loops_r3 = (r3_numel + R3BLOCK_SUB - 1) // R3BLOCK_SUB
    base_r5= tl.arange(0, R5BLOCK_SUB)
    loops_r5 = (r5_numel + R5BLOCK_SUB - 1) // R5BLOCK_SUB
    base_r4= tl.arange(0, R4BLOCK_SUB)
    loops_r4 = (r4_numel + R4BLOCK_SUB - 1) // R4BLOCK_SUB
    base_r2= tl.arange(0, R2BLOCK_SUB)
    loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,None,None,None,:]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        for loop_r3 in range(loops_r3):
            r3 = base_r3[:,None,None,None,None]
            r3_mask = r3 < r3_numel
            for loop_r5 in range(loops_r5):
                r5 = base_r5[None,None,None,:,None]
                r5_mask = r5 < r5_numel
                for loop_r4 in range(loops_r4):
                    r4 = base_r4[None,None,:,None,None]
                    r4_mask = r4 < r4_numel
                    r2 = base_r2[None,:,None,None,None]
                    r2_mask = r2 < r2_numel
                    tmp0 = tl.load(in_ptr0 + (x0 + 512*r5 + 3584*r2 + 25088*r4 + 50176*(r2 // 7) + 100352*r3), x0_mask, other=0.0)
                    tmp1 = tl.load(in_ptr1 + (x0 + 512*r5 + 3584*r4 + 7168*r2 + 100352*r3), x0_mask, other=0.0)
                    tmp2 = tl.load(in_ptr2 + (r5 + 7*r4 + 14*r2 + 196*r3), None)
                    tmp4 = tl.load(in_ptr3 + (r5 + 7*r4 + 14*r2 + 196*r3), None)
                    tmp3 = tmp1 - tmp2
                    tmp5 = tmp3 * tmp4
                    tmp6 = tmp0 * tmp5
                    tmp7 = tl.reshape(tmp6.permute([4, 0, 1, 2, 3]), [X0BLOCK_SUB, R5BLOCK_SUB * R4BLOCK_SUB * R3BLOCK_SUB * R2BLOCK_SUB])
                    tmp9 = tl.sum(tmp7, 1).reshape(1, 1, 1, 1, X0BLOCK_SUB)
                    tmp10 = tl.reshape(tmp0.permute([4, 0, 1, 2, 3]), [X0BLOCK_SUB, R5BLOCK_SUB * R4BLOCK_SUB * R3BLOCK_SUB * R2BLOCK_SUB])
                    tmp12 = tl.sum(tmp10, 1).reshape(1, 1, 1, 1, X0BLOCK_SUB)
                    tl.store(out_ptr0 + (x0 ), tmp9, x0_mask)
                    tl.store(out_ptr1 + (x0 ), tmp12, x0_mask)
