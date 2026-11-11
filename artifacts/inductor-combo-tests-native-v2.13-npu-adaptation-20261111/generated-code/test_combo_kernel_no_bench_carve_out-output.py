# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch_npu._C import _npu_getCurrentRawStreamNoWait as get_raw_stream
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch_npu._C import _npu_getCurrentRawStreamNoWait as get_raw_stream
import torch_npu
torch_npu.npu._initialized = torch_npu.npu.is_initialized()
has_initialized = False
import torch_npu._inductor.runtime.triton_heuristics as triton_heuristics

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
import torch_npu
empty_strided_npu = torch_npu._C._empty_strided_npu


# kernel path: /tmp/tmpn8lfcfer/su/csuhnyphhav6ow6ukaipyz47xq3z2blxophoybccsgnw7p63zdbf.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# SchedulerNodes: [ForeachKernelSchedulerNode(nodes=op2_op3)]

triton_for_fused_0 = async_compile.triton('triton_for_fused_0', '''
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

@triton_heuristics.foreach(
    size_hints={'x0': 2048},
    num_warps=8,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'X0BLOCK': 'i32', 'X0BLOCK_SUB': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'autotune_hints': set(), 'kernel_name': 'triton_for_fused_0', 'mutated_arg_names': [], 'backend_hash': '<removed>', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['x0'], 'axis_static_values': (('x0', 1024),), 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': (), 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': 0, 'default_config': {'XBLOCK': 'X0BLOCK'}, 'split_numels': [1024, 1024], 'no_split': [False, False], 'split_block': 'X0BLOCK', 'no_x_dim_0': False, 'xnumel_0': 1024, 'no_x_dim_1': False, 'xnumel_1': 1024}, 'combo_subkernel_meta': [{'axis_names': ['x0'], 'tiling_axis': [0], 'no_loop_axis': [], 'split_axis': [0], 'low_dims': {0}, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'size_hints': {'x0': 1024}, 'persistent_reduction': False}, {'axis_names': ['x0'], 'tiling_axis': [0], 'no_loop_axis': [], 'split_axis': [0], 'low_dims': {0}, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'size_hints': {'x0': 1024}, 'persistent_reduction': False}]},
)
@triton.jit
def triton_for_fused_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    pid = tl.program_id(0)
    num_blocks_0 = tl.cdiv(1024, X0BLOCK)
    num_blocks_1 = num_blocks_0 + tl.cdiv(1024, X0BLOCK)
    if pid < num_blocks_0:
        pid_offset = pid
        x0_numel = 1024
        x0_offset = pid_offset * X0BLOCK
        base_x0= tl.arange(0, X0BLOCK_SUB)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
            x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
            tmp0 = tl.load(in_ptr0 + (x0), x0_mask)
            tmp1 = tl.full([1], 2.0, tl.float32)
            tmp2 = tmp0 * tmp1
            tl.store(out_ptr0 + (x0), tmp2, x0_mask)
    elif pid < num_blocks_1:
        pid_offset = pid - num_blocks_0
        x0_numel = 1024
        x0_offset = pid_offset * X0BLOCK
        base_x0= tl.arange(0, X0BLOCK_SUB)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
            x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
            tmp3 = tl.load(in_ptr1 + (x0), x0_mask)
            tmp4 = tl.full([1], 1.0, tl.float32)
            tmp5 = tmp3 + tmp4
            tl.store(out_ptr1 + (x0), tmp5, x0_mask)
    else:
        pass
''', device_str='npu')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg1_1 = copy_if_misaligned(arg1_1)
            # Topologically Sorted Source Nodes: [bucketize], Original ATen: [aten.bucketize]
            buf0 = torch.ops.aten.bucketize.Tensor(arg0_1, arg1_1)
            del arg0_1
            del arg1_1
            buf1 = buf0
            assert_size_stride(buf1, (1024, ), (1, ), 'torch.ops.aten.bucketize.Tensor')
            assert_alignment(buf1, 16, 'torch.ops.aten.bucketize.Tensor')
            del buf0
            arg2_1 = copy_if_misaligned(arg2_1)
            arg3_1 = copy_if_misaligned(arg3_1)
            buf2 = empty_strided_npu((1024, ), (1, ), torch.float32)
            buf3 = empty_strided_npu((1024, ), (1, ), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            raw_stream0 = get_raw_stream(0)
            triton_for_fused_0.run(arg2_1, arg3_1, buf2, buf3, stream=raw_stream0)
            del arg2_1
            del arg3_1
        return (buf1, buf2, buf3, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((3, ), (1, ), device='npu:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg3_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))

