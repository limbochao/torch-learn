
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
    size_hints={'x0': 32, 'r1': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i64', 'out_ptr2': '*fp32', 'x0_numel_0': 'i64', 'x0_numel_1': 'i64', 'x0_numel_2': 'i64', 'X0BLOCK': 'i64', 'X0BLOCK_SUB': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<removed>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('r1', 256),), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': (), 'combo_grid_meta': {'num_kernels': 3, 'min_blocks': None, 'default_config': {'XBLOCK': 'X0BLOCK'}, 'split_numels': ['x0_numel_0', 'x0_numel_1', 'x0_numel_2'], 'no_split': [False, False, False], 'split_block': 'X0BLOCK', 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None, 'no_x_dim_2': False, 'xnumel_2': None}}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, x0_numel_0, x0_numel_1, x0_numel_2, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    pid = tl.program_id(0)
    num_blocks_0 = tl.cdiv(x0_numel_0, X0BLOCK)
    num_blocks_1 = num_blocks_0 + tl.cdiv(x0_numel_1, X0BLOCK)
    num_blocks_2 = num_blocks_1 + tl.cdiv(x0_numel_2, X0BLOCK)
    if pid < num_blocks_0:
        pid_offset = pid
        x0_numel = x0_numel_0
        r1_numel = 256
        R1BLOCK_SUB: tl.constexpr = 256
        x0_offset = pid_offset.to(tl.int64) * X0BLOCK
        base_x0= tl.arange(0, X0BLOCK_SUB).to(tl.int64)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        base_r1= tl.arange(0, R1BLOCK_SUB).to(tl.int64)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[:,None]
            x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
            rindex = base_r1
            r1 = base_r1[None,:]
            r1_mask = r1 < r1_numel
            tmp0 = tl.load(in_ptr0 + (r1 + 256*x0), x0_mask, other=0.0)
            tmp2 = tl.max(tmp0, 1, propagate_nan=True).reshape(X0BLOCK_SUB, 1)
            tmp4 = tl.broadcast_to(r1.reshape([1,R1BLOCK_SUB]), tmp0.shape)
            _, tmp3_tmp = triton_helpers.max_with_index(tmp0, tmp4, 1)
            tmp3 = tmp3_tmp.reshape(X0BLOCK_SUB, 1)
            tl.store(out_ptr0 + (x0 ), tmp2, x0_mask)
            tl.store(out_ptr1 + (x0 ), tmp3, x0_mask)
    elif pid < num_blocks_1:
        pid_offset = pid - num_blocks_0
        x0_numel = x0_numel_1
        r1_numel = 128
        R1BLOCK_SUB: tl.constexpr = 128
        x0_offset = pid_offset.to(tl.int64) * X0BLOCK
        base_x0= tl.arange(0, X0BLOCK_SUB).to(tl.int64)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        base_r1= tl.arange(0, R1BLOCK_SUB).to(tl.int64)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[:,None]
            x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
            r1 = base_r1[None,:]
            r1_mask = r1 < r1_numel
            tmp5 = tl.load(in_ptr1 + (r1 + 128*x0), x0_mask, other=0.0)
            tmp7 = tl.sum(tmp5, 1).reshape(X0BLOCK_SUB, 1)
            tl.store(out_ptr2 + (x0 ), tmp7, x0_mask)
    elif pid < num_blocks_2:
        pid_offset = pid - num_blocks_1
        x0_numel = x0_numel_2
        r1_numel = 128
        R1BLOCK_SUB: tl.constexpr = 128
        x0_offset = pid_offset.to(tl.int64) * X0BLOCK
        base_x0= tl.arange(0, X0BLOCK_SUB).to(tl.int64)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        base_r1= tl.arange(0, R1BLOCK_SUB).to(tl.int64)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[:,None]
            x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
            r1 = base_r1[None,:]
            r1_mask = r1 < r1_numel
            tmp8 = tl.load(in_ptr2 + (r1 + 128*x0), x0_mask, other=0.0)
            tmp10 = tl.sum(tmp8, 1).reshape(X0BLOCK_SUB, 1)
            tmp11 = 128.0
            tmp12 = (tmp10 / tmp11)
            tl.store(in_out_ptr0 + (x0), tmp12, x0_mask)
            tl.store(in_out_ptr0 + (x0 ), tmp10, x0_mask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((32,), (1,), device='npu:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 256), (256, 1), device='npu:0', dtype=torch.float32)
    arg_2 = rand_strided((16, 128), (128, 1), device='npu:0', dtype=torch.float32)
    arg_3 = rand_strided((32, 128), (128, 1), device='npu:0', dtype=torch.float32)
    arg_4 = rand_strided((32,), (1,), device='npu:0', dtype=torch.float32)
    arg_5 = rand_strided((32,), (1,), device='npu:0', dtype=torch.int64)
    arg_6 = rand_strided((16,), (1,), device='npu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 32, 16, 32,


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
    ms = benchmarker.benchmark(call, fn_args=(args,), device=npu,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
