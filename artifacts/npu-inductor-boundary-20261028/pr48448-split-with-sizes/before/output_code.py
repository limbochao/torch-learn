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


# kernel path: /tmp/tmpvo6f5xii/jm/cjmzxkpyabs4lddccovnwednqhemjvh74cos5bjxnrne7gl7y4zo.py
# Topologically Sorted Source Nodes: [split, sum_1], Original ATen: [aten.split_with_sizes, aten.sum]
# Source node to ATen node mapping:
#   split => split_with_sizes
#   sum_1 => sum_1
# Graph fragment:
#   %arg1_1 : Tensor "f32[32][1]npu:0" = PlaceHolder[target=arg1_1]
#   %buf0 : Tensor  = PlaceHolder[target=buf0]
#   %split_with_sizes : [num_users=1] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%arg1_1, [%_local_scalar_dense, %_local_scalar_dense_1, %_local_scalar_dense_2]), kwargs = {})
#   %sum_1 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%getitem,), kwargs = {})
#   return %sum_1
# SchedulerNodes: [SchedulerNode(name='op13')]

triton_per_fused_split_with_sizes_sum_0 = async_compile.triton('triton_per_fused_split_with_sizes_sum_0', '''
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
    size_hints={'r0': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_split_with_sizes_sum_0', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['r0'], 'axis_static_values': (), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': None, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': (), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_split_with_sizes_sum_0(in_ptr0, out_ptr0, r0_numel, R0BLOCK_SUB : tl.constexpr):
    base_r0= tl.arange(0, R0BLOCK_SUB)
    loops_r0 = (r0_numel + R0BLOCK_SUB - 1) // R0BLOCK_SUB
    r0 = base_r0
    r0_mask = r0 < r0_numel
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp2 = triton_helpers.promote_to_tensor(tl.sum(tmp0, 0))
    tl.store(out_ptr0 + (tl.arange(0,1) ), tmp2, None)
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
        arg0_1, arg1_1 = args
        args.clear()
        u0 = reinterpret_tensor(arg0_1, (), (), 0).item()
        buf0 = None
        if not (u0 >= 0):
            raise RuntimeError('u0 >= 0')
        buf1 = None
        if not (u0 <= 32):
            raise RuntimeError('u0 <= 32')
        buf2 = None
        if not (0 <= u0):
            raise RuntimeError('0 <= u0')
        buf3 = None
        u1 = reinterpret_tensor(arg0_1, (), (), 1).item()
        buf4 = None
        if not (u1 >= 0):
            raise RuntimeError('u1 >= 0')
        buf5 = None
        if not (u1 <= 32):
            raise RuntimeError('u1 <= 32')
        buf6 = None
        if not (0 <= u1):
            raise RuntimeError('0 <= u1')
        buf7 = None
        u2 = reinterpret_tensor(arg0_1, (), (), 2).item()
        buf8 = None
        del arg0_1
        if not (u2 >= 0):
            raise RuntimeError('u2 >= 0')
        buf9 = None
        if not (u2 <= 32):
            raise RuntimeError('u2 <= 32')
        buf10 = None
        if not (0 <= u2):
            raise RuntimeError('0 <= u2')
        buf11 = None
        if not ((u0 + u1 + u2) == 32):
            raise RuntimeError('Eq(u0 + u1 + u2, 32)')
        buf12 = None
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg1_1 = copy_if_misaligned(arg1_1)
            buf13 = empty_strided_npu((), (), torch.float32)
            # Topologically Sorted Source Nodes: [split, sum_1], Original ATen: [aten.split_with_sizes, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_split_with_sizes_sum_0.run(arg1_1, buf13, u0, stream=raw_stream0)
            del arg1_1
        return (buf13, u0 + u1 + u2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((3, ), (1, ), device='cpu', dtype=torch.int64)
    arg1_1 = rand_strided((32, ), (1, ), device='npu:0', dtype=torch.float32)
    return [arg0_1, arg1_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
