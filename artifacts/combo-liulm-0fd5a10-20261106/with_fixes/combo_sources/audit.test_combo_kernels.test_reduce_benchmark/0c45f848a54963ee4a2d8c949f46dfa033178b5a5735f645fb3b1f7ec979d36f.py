
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

@triton_heuristics.persistent_reduction(
    size_hints={'x0': 20, 'r1': 20},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i64', 'out_ptr2': '*fp32', 'out_ptr3': '*i64', 'out_ptr4': '*fp32', 'X0BLOCK': 'i64', 'X0BLOCK_SUB': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': [], 'backend_hash': '<removed>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 20), ('r1', 20)), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': (), 'combo_grid_meta': {'num_kernels': 3, 'min_blocks': 0, 'default_config': {'XBLOCK': 'X0BLOCK'}, 'split_numels': [20, 10, 10], 'no_split': [False, False, False], 'split_block': 'X0BLOCK', 'no_x_dim_0': False, 'xnumel_0': 20, 'no_x_dim_1': False, 'xnumel_1': 10, 'no_x_dim_2': False, 'xnumel_2': 10}}
)
@triton.jit
def Placeholder.KERNEL_NAME(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    pid = tl.program_id(0)
    num_blocks_0 = tl.cdiv(20, X0BLOCK)
    num_blocks_1 = num_blocks_0 + tl.cdiv(10, X0BLOCK)
    num_blocks_2 = num_blocks_1 + tl.cdiv(10, X0BLOCK)
    if pid < num_blocks_0:
        pid_offset = pid
        x0_numel = 20
        r1_numel = 20
        R1BLOCK_SUB: tl.constexpr = 20
        x0_offset = pid_offset.to(tl.int64) * X0BLOCK
        base_x0= tl.arange(0, X0BLOCK_SUB).to(tl.int64)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        base_r1= tl.arange(0, R1BLOCK_SUB).to(tl.int64)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
            x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
            rindex = base_r1
            r1 = base_r1[:,None]
            r1_mask = r1 < r1_numel
            tmp0 = tl.load(in_ptr0 + (x0 + 20*r1), x0_mask, other=0.0)
            tmp2 = tl.max(tmp0, 0, propagate_nan=True).reshape(1, X0BLOCK_SUB)
            tmp4 = tl.broadcast_to(r1.reshape([R1BLOCK_SUB,1]), tmp0.shape)
            _, tmp3_tmp = triton_helpers.max_with_index(tmp0, tmp4, 0)
            tmp3 = tmp3_tmp.reshape(1, X0BLOCK_SUB)
            tl.store(out_ptr0 + (x0 ), tmp2, x0_mask)
            tl.store(out_ptr1 + (x0 ), tmp3, x0_mask)
    elif pid < num_blocks_1:
        pid_offset = pid - num_blocks_0
        x0_numel = 10
        r1_numel = 10
        R1BLOCK_SUB: tl.constexpr = 10
        x0_offset = pid_offset.to(tl.int64) * X0BLOCK
        base_x0= tl.arange(0, X0BLOCK_SUB).to(tl.int64)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        base_r1= tl.arange(0, R1BLOCK_SUB).to(tl.int64)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
            x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
            rindex = base_r1
            r1 = base_r1[:,None]
            r1_mask = r1 < r1_numel
            tmp5 = tl.load(in_ptr1 + (x0 + 10*r1), x0_mask, other=0.0)
            tmp7 = tl.min(tmp5, 0).reshape(1, X0BLOCK_SUB)
            tmp9 = tl.broadcast_to(r1.reshape([R1BLOCK_SUB,1]), tmp5.shape)
            _, tmp8_tmp = triton_helpers.min_with_index(tmp5, tmp9, 0)
            tmp8 = tmp8_tmp.reshape(1, X0BLOCK_SUB)
            tl.store(out_ptr2 + (x0 ), tmp7, x0_mask)
            tl.store(out_ptr3 + (x0 ), tmp8, x0_mask)
    elif pid < num_blocks_2:
        pid_offset = pid - num_blocks_1
        x0_numel = 10
        r1_numel = 10
        R1BLOCK_SUB: tl.constexpr = 10
        x0_offset = pid_offset.to(tl.int64) * X0BLOCK
        base_x0= tl.arange(0, X0BLOCK_SUB).to(tl.int64)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        base_r1= tl.arange(0, R1BLOCK_SUB).to(tl.int64)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
            x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
            r1 = base_r1[:,None]
            r1_mask = r1 < r1_numel
            tmp10 = tl.load(in_ptr2 + (x0 + 10*r1), x0_mask, other=0.0)
            tmp12 = tl.sum(tmp10, 0).reshape(1, X0BLOCK_SUB)
            tl.store(out_ptr4 + (x0 ), tmp12, x0_mask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((20, 20), (20, 1), device='npu:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 10), (10, 1), device='npu:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 10), (10, 1), device='npu:0', dtype=torch.float32)
    arg_3 = rand_strided((20,), (1,), device='npu:0', dtype=torch.float32)
    arg_4 = rand_strided((20,), (1,), device='npu:0', dtype=torch.int64)
    arg_5 = rand_strided((10,), (1,), device='npu:0', dtype=torch.float32)
    arg_6 = rand_strided((10,), (1,), device='npu:0', dtype=torch.int64)
    arg_7 = rand_strided((10,), (1,), device='npu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7,


def call(args):
    with torch.npu.utils.device(0):
        torch.npu.set_device(0)
        stream0 = get_raw_stream(0)
        Placeholder.KERNEL_NAME.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.npu.utils.device(0):
        torch.npu.set_device(0)
        return Placeholder.KERNEL_NAME.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=npu,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
