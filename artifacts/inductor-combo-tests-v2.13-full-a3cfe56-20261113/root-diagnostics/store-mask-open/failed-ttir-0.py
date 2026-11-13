
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

@triton_heuristics.reduction(
    size_hints={'y0': 2, 'x1': 8, 'r2': 4},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'y0_numel': 'i32', 'x1_numel': 'i32', 'r2_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '139C9AD8FB0E3F910BF6CA8D4166949EB020778AFDF1D1A0ABA3AF3BC313E710', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['y0', 'x1', 'r2'], 'axis_static_values': (), 'low_dims': set(), 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def Placeholder.KERNEL_NAME(in_out_ptr0, in_ptr0, ks0, ks1, y0_numel, x1_numel, r2_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr, R2BLOCK_SUB : tl.constexpr):
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    x1_offset = tl.program_id(1) * X1BLOCK
    base_x1= tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (X1BLOCK + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    base_r2= tl.arange(0, R2BLOCK_SUB)
    loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
    for loop_y0 in range(loops_y0):
        for loop_x1 in range(loops_x1):
            x1 = x1_offset + (loop_x1 * X1BLOCK_SUB) + base_x1[None,:,None]
            x1_mask = x1 < min(X1BLOCK+x1_offset, x1_numel)
            _tmp3 = tl.full([Y0BLOCK_SUB, X1BLOCK_SUB, R2BLOCK_SUB], 0, tl.float32)
            for loop_r2 in range(loops_r2):
                r2 = (loop_r2 * R2BLOCK_SUB) + base_r2[None,None,:]
                r2_mask = r2 < r2_numel
                tmp0 = tl.load(in_ptr0 + (x0 + ks0*r2 + ks0*ks1*x1), r2_mask & x1_mask, other=0.0)
                tl.static_assert(tmp0.dtype == tl.float32)
                tmp1 = tl.broadcast_to(tmp0, [Y0BLOCK_SUB, X1BLOCK_SUB, R2BLOCK_SUB])
                tl.static_assert(tmp1.dtype == tl.float32)
                tmp2 = tl.reshape(tmp1, [Y0BLOCK_SUB, X1BLOCK_SUB, R2BLOCK_SUB])
                tl.static_assert(tmp2.dtype == tl.float32)
                tmp4 = _tmp3 + tmp2
                tl.static_assert(tmp4.dtype == tl.float32)
                tl.static_assert(tmp4.dtype == tl.float32)
                _tmp3 = tl.where(tl.reshape((r2_mask & x1_mask & y0_mask), [Y0BLOCK_SUB, X1BLOCK_SUB, R2BLOCK_SUB]), tmp4, _tmp3)
            tmp3 = tl.sum(_tmp3, 2).reshape(Y0BLOCK_SUB, X1BLOCK_SUB, 1)
            tmp5 = ks1
            tl.static_assert(tmp5.dtype == tl.float32)
            tmp6 = tmp5.to(tl.float32)
            tl.static_assert(tmp6.dtype == tl.float32)
            tmp7 = (tmp6).to(tl.float32)
            tl.static_assert(tmp7.dtype == tl.float32)
            tl.static_assert(tmp7.dtype == tl.float32)
            tmp8 = (tmp3 / tmp7)
            tl.static_assert(tmp8.dtype == tl.float32)
            tl.static_assert(tmp8.dtype == tl.float32)
            tl.store(in_out_ptr0 + (x0 + ks0*x1), tmp8, x1_mask)
