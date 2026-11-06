
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

from torch._dynamo.testing import rand_strided
from torch_npu._C import _npu_getCurrentRawStreamNoWait as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'y0': 40, 'x1': 30}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'y0_numel_0': 'i64', 'x1_numel_0': 'i64', 'y0_numel_1': 'i64', 'x1_numel_1': 'i64', 'Y0BLOCK': 'i64', 'Y0BLOCK_SUB': 'constexpr', 'X1BLOCK_SUB': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'backend_hash': '<removed>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'axis_static_values': (), 'low_dims': {0, 1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': (), 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': {'XBLOCK': 'Y0BLOCK'}, 'split_numels': ['y0_numel_0', 'y0_numel_1'], 'no_split': [False, False], 'split_block': 'Y0BLOCK', 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}}
)
@triton.jit
def triton_poi_fused_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ks0, ks1, ks2, y0_numel_0, x1_numel_0, y0_numel_1, x1_numel_1, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    pid = tl.program_id(0)
    num_blocks_0 = tl.cdiv(y0_numel_0, Y0BLOCK)
    num_blocks_1 = num_blocks_0 + tl.cdiv(y0_numel_1, Y0BLOCK)
    if pid < num_blocks_0:
        pid_offset = pid
        y0_numel = y0_numel_0
        x1_numel = x1_numel_0
        y0_offset = pid_offset.to(tl.int64) * Y0BLOCK
        base_y0= tl.arange(0, Y0BLOCK_SUB).to(tl.int64)
        loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
        base_x1= tl.arange(0, X1BLOCK_SUB).to(tl.int64)
        loops_x1 = (x1_numel + X1BLOCK_SUB - 1) // X1BLOCK_SUB
        for loop_y0 in range(loops_y0):
            y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
            y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
            for loop_x1 in range(loops_x1):
                x1 = (loop_x1 * X1BLOCK_SUB) + base_x1[None,:]
                x1_mask = x1 < x1_numel
                tmp0 = tl.load(in_ptr0 + (x1 + ks0*y0), x1_mask & y0_mask)
                tmp1 = tl.load(in_ptr1 + (y0 + ks1*x1), x1_mask & y0_mask)
                tmp2 = tmp0 + tmp1
                tl.store(out_ptr0 + (x1 + ks0*y0), tmp2, x1_mask & y0_mask)
    elif pid < num_blocks_1:
        pid_offset = pid - num_blocks_0
        y0_numel = y0_numel_1
        x1_numel = x1_numel_1
        y0_offset = pid_offset.to(tl.int64) * Y0BLOCK
        base_y0= tl.arange(0, Y0BLOCK_SUB).to(tl.int64)
        loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
        base_x1= tl.arange(0, X1BLOCK_SUB).to(tl.int64)
        loops_x1 = (x1_numel + X1BLOCK_SUB - 1) // X1BLOCK_SUB
        for loop_y0 in range(loops_y0):
            y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
            y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
            for loop_x1 in range(loops_x1):
                x1 = (loop_x1 * X1BLOCK_SUB) + base_x1[None,:]
                x1_mask = x1 < x1_numel
                tmp3 = tl.load(in_ptr2 + (x1 + ks1*y0), x1_mask & y0_mask)
                tmp4 = tl.load(in_ptr3 + (y0 + ks2*x1), x1_mask & y0_mask)
                tmp5 = tmp3 + tmp4
                tl.store(out_ptr1 + (x1 + ks1*y0), tmp5, x1_mask & y0_mask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((40, 30), (30, 1), device='npu:0', dtype=torch.float32)
    arg_1 = rand_strided((40, 30), (1, 40), device='npu:0', dtype=torch.float32)
    arg_2 = rand_strided((36, 40), (40, 1), device='npu:0', dtype=torch.float32)
    arg_3 = rand_strided((36, 40), (1, 36), device='npu:0', dtype=torch.float32)
    arg_4 = rand_strided((40, 30), (30, 1), device='npu:0', dtype=torch.float32)
    arg_5 = rand_strided((36, 40), (40, 1), device='npu:0', dtype=torch.float32)
    arg_6 = 30
    arg_7 = 40
    arg_8 = 36
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, 40, 30, 36, 40,


def call(args):
    with torch.npu.utils.device(0):
        torch.npu.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.npu.utils.device(0):
        torch.npu.set_device(0)
        return triton_poi_fused_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=npu,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
