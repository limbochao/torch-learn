
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
    size_hints={'z0': 16, 'y1': 12, 'x2': 512, 'r3': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*i1', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*i1', 'out_ptr5': '*fp32', 'load_seed_offset': 'i32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'r3_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_all_amax_div_isfinit_8', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': 'REDACTED', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [], 'axis_names': ['z0', 'y1', 'x2', 'r3'], 'axis_static_values': (('z0', 16), ('y1', 12), ('x2', 512), ('r3', 512)), 'low_dims': {3}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.bool, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__softmax_all_amax_div_isfinit_8(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr5, load_seed_offset, z0_numel, y1_numel, x2_numel, r3_numel, Z0BLOCK, Y1BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr):
    R3BLOCK_SUB: tl.constexpr = 512
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
                r3_mask = True
                tmp0 = tl.load(in_ptr0 + (r3 + 512*x2 + 262144*y1 + 3145728*z0), x2_mask & y1_mask & z0_mask, other=0.0)
                tmp1 = tl.full([1, 1, 1, 1], 0.125, tl.float32)
                tmp2 = tmp0 * tmp1
                tmp3 = tmp2 == tmp2
                tmp4 = tl_math.abs(tmp2)
                tmp5 = tl.full([1, 1, 1, 1], float("inf"), tl.float32)
                tmp6 = tmp4 != tmp5
                tmp7 = tmp3 & tmp6
                tmp8 = tmp7 == 0
                tmp10 = tl.where(tl.broadcast_to((z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, Y1BLOCK_SUB, 1, 1]) & x2_mask.reshape([1, 1, X2BLOCK_SUB, 1])), [Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, R3BLOCK_SUB]), tmp8, False)
                tmp11 = triton_helpers.any(tmp10, 3).reshape(Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, 1)
                tmp12 = tmp11 == 0
                tmp14 = tl.where(tl.broadcast_to((z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, Y1BLOCK_SUB, 1, 1]) & x2_mask.reshape([1, 1, X2BLOCK_SUB, 1])), [Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, R3BLOCK_SUB]), tmp0, float("-inf"))
                tmp15 = tl.max(tmp14, 3, propagate_nan=True).reshape(Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, 1)
                tmp17 = tl.where(tl.broadcast_to((z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, Y1BLOCK_SUB, 1, 1]) & x2_mask.reshape([1, 1, X2BLOCK_SUB, 1])), [Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, R3BLOCK_SUB]), tmp2, float("-inf"))
                tmp18 = tl.max(tmp17, 3, propagate_nan=True).reshape(Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, 1)
                tmp19 = tmp0 - tmp15
                tmp20 = tmp19 * tmp1
                tmp21 = tmp2 - tmp18
                tmp22 = tl.where(tmp12, tmp20, tmp21)
                tmp23 = libdevice.exp(tmp22)
                tmp25 = tl.where(tl.broadcast_to((z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, Y1BLOCK_SUB, 1, 1]) & x2_mask.reshape([1, 1, X2BLOCK_SUB, 1])), [Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, R3BLOCK_SUB]), tmp23, 0)
                tmp26 = tl.sum(tmp25, 3).reshape(Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB, 1)
                tmp27 = tl.load(in_ptr1 + load_seed_offset)
                tmp28 = r3 + 512*x2 + 262144*y1 + 3145728*z0
                tmp29 = tl.rand(tmp27, (tmp28).to(tl.uint32))
                tmp30 = tl.full([1, 1, 1, 1], 0.1, tl.float32)
                tmp31 = tmp29 > tmp30
                tmp32 = tmp31.to(tl.float32)
                tmp33 = (tmp23 / tmp26)
                tmp34 = tmp32 * tmp33
                tmp35 = tl.full([1, 1, 1, 1], 1.1111111111111112, tl.float32)
                tmp36 = tmp34 * tmp35
                tl.store(in_out_ptr0 + (x2 + 512*y1 + 6144*z0), tmp12, x2_mask & y1_mask & z0_mask)
                tl.store(out_ptr4 + (r3 + 512*x2 + 262144*y1 + 3145728*z0), tmp31, x2_mask & y1_mask & z0_mask)
                tl.store(out_ptr5 + (r3 + 512*x2 + 262144*y1 + 3145728*z0), tmp36, x2_mask & y1_mask & z0_mask)
                tl.store(out_ptr0 + (x2 + 512*y1 + 6144*z0 ), tmp15, x2_mask & y1_mask & z0_mask)
                tl.store(out_ptr1 + (x2 + 512*y1 + 6144*z0 ), tmp18, x2_mask & y1_mask & z0_mask)
                tl.store(out_ptr2 + (x2 + 512*y1 + 6144*z0 ), tmp26, x2_mask & y1_mask & z0_mask)
