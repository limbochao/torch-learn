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


# kernel path: /home/l30023782/PTA_213/combo-full-a3cfe56-20261113/cache-root-diagnostic/scalar-v2/gk/cgkkqdf4qvhcotqodfc665qgsuibwi5d6xxvlsj3kus2gr55bwsg.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# SchedulerNodes: [ForeachKernelSchedulerNode(nodes=op0_op1_op3_op4)]

triton_per_fused_0 = async_compile.triton('triton_per_fused_0', '''
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
    size_hints={'x0': 2048, 'r1': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_0', 'mutated_arg_names': [], 'backend_hash': '139C9AD8FB0E3F910BF6CA8D4166949EB020778AFDF1D1A0ABA3AF3BC313E710', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 2048), ('r1', 2048)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_0(in_ptr0, out_ptr0, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 2048
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[:,None]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        r1 = base_r1[None,:]
        r1_mask = True
        tmp0 = tl.load(in_ptr0 + (r1 + 2048*x0), x0_mask, other=0.0)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 1).reshape(X0BLOCK_SUB, 1)
        tl.store(out_ptr0 + (x0 ), tmp3, x0_mask)
''', device_str='npu')


# kernel path: /home/l30023782/PTA_213/combo-full-a3cfe56-20261113/cache-root-diagnostic/scalar-v2/zd/czdp5upillk645vv3cdh2vjgvzgfpkgqa7z34rjviqzehonruz3k.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# SchedulerNodes: [ForeachKernelSchedulerNode(nodes=op0_op1_op3_op4)]

triton_per_fused_1 = async_compile.triton('triton_per_fused_1', '''
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
    size_hints={'r0': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'r0_numel': 'i32', 'R0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_1', 'mutated_arg_names': [], 'backend_hash': '139C9AD8FB0E3F910BF6CA8D4166949EB020778AFDF1D1A0ABA3AF3BC313E710', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['r0'], 'axis_static_values': (('r0', 2048),), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('R0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, r0_numel, R0BLOCK):
    R0BLOCK_SUB: tl.constexpr = 2048
    r0_offset = tl.program_id(0) * R0BLOCK
    base_r0= tl.arange(0, R0BLOCK_SUB)
    loops_r0 = (R0BLOCK + R0BLOCK_SUB - 1) // R0BLOCK_SUB
    r0 = base_r0
    r0_mask = True
    tmp0 = tl.load(in_ptr0 + tl.arange(0,1) +  (0))
    tmp1 = tl.reshape(tmp0, [1 ])
    tmp2 = tl.load(in_ptr1 + tl.arange(0,1) +  (0))
    tmp3 = tl.reshape(tmp2, [1 ])
    tmp5 = tl.load(in_ptr2 + (r0), None)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5 * tmp4
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tl.store(out_ptr0 + (tl.arange(0,1)), tmp4, None)
    tl.store(out_ptr2 + (tl.arange(0,1) ), tmp8, None)
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
            arg2_1 = copy_if_misaligned(arg2_1)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg1_1 = copy_if_misaligned(arg1_1)
            arg3_1 = copy_if_misaligned(arg3_1)
            buf0 = empty_strided_npu((2048, ), (1, ), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_0.run(arg2_1, buf0, 2048, 2048, stream=raw_stream0)
            buf1 = empty_strided_npu((), (), torch.float32)
            buf4 = empty_strided_npu((), (), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_1.run(arg0_1, arg1_1, arg3_1, buf1, buf4, 2048, stream=raw_stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            del arg3_1
        return (buf1, buf0, buf4, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((), (), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((), (), device='npu:0', dtype=torch.float32)
    arg2_1 = rand_strided((2048, 2048), (2048, 1), device='npu:0', dtype=torch.float32)
    arg3_1 = rand_strided((2048, 1), (1, 1), device='npu:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
