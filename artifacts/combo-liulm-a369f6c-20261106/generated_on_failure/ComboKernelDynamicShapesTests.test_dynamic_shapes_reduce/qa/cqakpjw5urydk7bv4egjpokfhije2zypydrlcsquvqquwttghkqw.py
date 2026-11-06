
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

@triton_heuristics.reduction(
    size_hints={'x0': 10, 'r1': 10},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i64', 'ks0': 'i64', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': [], 'backend_hash': '<removed>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': (), 'kernel_num_gb': 5.2e-07}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ks0, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr, R1BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        _tmp2 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], float("inf"), tl.float32)
        _tmp4 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], float("inf"), tl.float32)
        _tmp4_index = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 9223372036854775807, tl.int64)
        for loop_r1 in range(loops_r1):
            rindex = (loop_r1 * R1BLOCK_SUB) + base_r1
            r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[:,None]
            r1_mask = r1 < r1_numel
            tmp0 = tl.load(in_ptr0 + (x0 + ks0*r1), r1_mask & x0_mask, other=0.0)
            tmp1 = tl.reshape(tmp0, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp3 = triton_helpers.minimum(_tmp2, tmp1)
            _tmp2 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp3, _tmp2)
            _tmp4_next, _tmp4_index_next = triton_helpers.minimum_with_index(
                _tmp4, _tmp4_index, tmp1, r1
            )
            _tmp4 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), _tmp4_next, _tmp4)
            _tmp4_index = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), _tmp4_index_next, _tmp4_index)
        tmp2 = tl.min(_tmp2, 0).reshape(1, X0BLOCK_SUB)
        _, tmp4_tmp = triton_helpers.min_with_index(_tmp4, _tmp4_index, 0)
        tmp4 = tmp4_tmp.reshape(1, X0BLOCK_SUB)
        tl.store(out_ptr0 + (x0 ), tmp2, x0_mask)
        tl.store(out_ptr1 + (x0 ), tmp4, x0_mask)


def get_args():
    arg_0 = rand_strided((10, 10), (10, 1), device='npu:0', dtype=torch.float32)
    arg_1 = rand_strided((10,), (1,), device='npu:0', dtype=torch.float32)
    arg_2 = rand_strided((10,), (1,), device='npu:0', dtype=torch.int64)
    arg_3 = 10
    return arg_0, arg_1, arg_2, arg_3, 10, 10,


def call(args):
    with torch.npu.utils.device(0):
        torch.npu.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.npu.utils.device(0):
        torch.npu.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device=npu, rep=40)
    num_gb = 5.2e-07
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
