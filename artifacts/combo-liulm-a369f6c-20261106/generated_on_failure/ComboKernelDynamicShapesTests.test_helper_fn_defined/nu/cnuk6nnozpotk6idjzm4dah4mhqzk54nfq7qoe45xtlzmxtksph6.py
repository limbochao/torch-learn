
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
    size_hints={'x0': 64, 'r2': 32, 'r1': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'X0BLOCK': 'i64'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_0', 'mutated_arg_names': [], 'backend_hash': '<removed>', 'split_axis': [0], 'tiling_axis': [1, 2], 'no_loop_axis': [], 'axis_names': ['x0', 'r2', 'r1'], 'axis_static_values': (('x0', 64), ('r2', 32), ('r1', 128)), 'low_dims': {2}, 'numof_reduction_axis': 2, 'split_axis_dtype': torch.float32, 'dual_reduction': True, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': (), 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': 0, 'default_config': {'XBLOCK': 'X0BLOCK'}, 'split_numels': [64, 64], 'no_split': [False, False], 'split_block': 'X0BLOCK', 'no_x_dim_0': False, 'xnumel_0': 64, 'no_x_dim_1': False, 'xnumel_1': 64}}
)
@triton.jit
def triton_per_fused_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, X0BLOCK):
    pid = tl.program_id(0)
    num_blocks_0 = tl.cdiv(64, X0BLOCK)
    num_blocks_1 = num_blocks_0 + tl.cdiv(64, X0BLOCK)
    if pid < num_blocks_0:
        pid_offset = pid
        x0_numel = 64
        r2_numel = 32
        R2BLOCK_SUB: tl.constexpr = 32
        r1_numel = 128
        R1BLOCK_SUB: tl.constexpr = 128
        x0_offset = pid_offset.to(tl.int64) * X0BLOCK
        base_r2= tl.arange(0, R2BLOCK_SUB).to(tl.int64)
        loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
        base_r1= tl.arange(0, R1BLOCK_SUB).to(tl.int64)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for x0 in range(x0_offset, min(x0_offset + X0BLOCK, x0_numel)):
            for loop_r2 in range(loops_r2):
                r2 = base_r2[:,None]
                r2_mask = r2 < r2_numel
                r1 = base_r1[None,:]
                r1_mask = r1 < r1_numel
                tmp0 = tl.load(in_ptr0 + (r1 + 128*x0 + 8192*r2), None)
                tmp1 = tl.reshape(tmp0, [R2BLOCK_SUB* R1BLOCK_SUB])
                tmp3 = triton_helpers.promote_to_tensor(tl.sum(tmp1, 0))
                tl.store(out_ptr0 + (x0 ), tmp3, None)
    elif pid < num_blocks_1:
        pid_offset = pid - num_blocks_0
        x0_numel = 64
        r2_numel = 32
        R2BLOCK_SUB: tl.constexpr = 32
        r1_numel = 128
        R1BLOCK_SUB: tl.constexpr = 128
        x0_offset = pid_offset.to(tl.int64) * X0BLOCK
        base_r2= tl.arange(0, R2BLOCK_SUB).to(tl.int64)
        loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
        base_r1= tl.arange(0, R1BLOCK_SUB).to(tl.int64)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for x0 in range(x0_offset, min(x0_offset + X0BLOCK, x0_numel)):
            for loop_r2 in range(loops_r2):
                r2 = base_r2[:,None]
                r2_mask = r2 < r2_numel
                r1 = base_r1[None,:]
                r1_mask = r1 < r1_numel
                tmp4 = tl.load(in_ptr1 + (r1 + 128*x0 + 8192*r2), None)
                tmp5 = tl.reshape(tmp4, [R2BLOCK_SUB* R1BLOCK_SUB])
                tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
                tl.store(out_ptr1 + (x0 ), tmp7, None)
    else:
        pass
