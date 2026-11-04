# AOT ID: ['6_inference']
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


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add, aten.copy_]
# Source node to ATen node mapping:
#   add_ => add
# Graph fragment:
#   %copy_ : Tensor "f32[64][1]npu:0" = PlaceHolder[target=copy_]
#   %arg6_1 : Tensor "f32[64][1]npu:0" = PlaceHolder[target=arg6_1]
#   %add : Tensor "f32[64][1]npu:0" = PlaceHolder[target=add]
#   %add : Tensor "f32[64][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg6_1), kwargs = {alpha: -0.01})
#   %copy_ : Tensor "f32[64][1]npu:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg0_1, %add), kwargs = {})
#   return %add,%buf1
# SchedulerNodes: [SchedulerNode(name='op0'), SchedulerNode(name='op1')]

triton_poi_fused_add_copy__0 = async_compile.triton('triton_poi_fused_add_copy__0', '''
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

@triton_heuristics.pointwise(
    size_hints={'x0': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'x0_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy__0', 'mutated_arg_names': ['in_ptr0', 'out_ptr1'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['x0'], 'axis_static_values': (('x0', 64),), 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_copy__0(in_ptr0, in_ptr1, out_ptr1, x0_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        tmp0 = tl.load(in_ptr0 + (x0), x0_mask)
        tmp1 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp2 = tl.full([1], -0.01, tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 + tmp3
        tl.store(out_ptr1 + (x0), tmp4, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [add__1], Original ATen: [aten.add, aten.copy_]
# Source node to ATen node mapping:
#   add__1 => add_1
# Graph fragment:
#   %copy__1 : Tensor "f32[64, 64][64, 1]npu:0" = PlaceHolder[target=copy__1]
#   %arg7_1 : Tensor "f32[64, 64][64, 1]npu:0" = PlaceHolder[target=arg7_1]
#   %add_1 : Tensor "f32[64, 64][64, 1]npu:0" = PlaceHolder[target=add_1]
#   %add_1 : Tensor "f32[64, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %arg7_1), kwargs = {alpha: -0.01})
#   %copy__1 : Tensor "f32[64, 64][64, 1]npu:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg1_1, %add_1), kwargs = {})
#   return %add_1,%buf3
# SchedulerNodes: [SchedulerNode(name='op2'), SchedulerNode(name='op3')]

triton_poi_fused_add_copy__1 = async_compile.triton('triton_poi_fused_add_copy__1', '''
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

@triton_heuristics.pointwise(
    size_hints={'y0': 64, 'x1': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy__1', 'mutated_arg_names': ['in_ptr0', 'out_ptr1'], 'backend_hash': '<build-hash>', 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 64), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_copy__1(in_ptr0, in_ptr1, out_ptr1, y0_numel, x1_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    x1_offset = tl.program_id(1) * X1BLOCK
    base_x1= tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (X1BLOCK + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        for loop_x1 in range(loops_x1):
            x1 = x1_offset + (loop_x1 * X1BLOCK_SUB) + base_x1[None,:]
            x1_mask = x1 < min(X1BLOCK+x1_offset, x1_numel)
            tmp0 = tl.load(in_ptr0 + (x1 + 64*y0), x1_mask & y0_mask)
            tmp1 = tl.load(in_ptr1 + (x1 + 64*y0), x1_mask & y0_mask)
            tmp2 = tl.full([1, 1], -0.01, tl.float32)
            tmp3 = tmp1 * tmp2
            tmp4 = tmp0 + tmp3
            tl.store(out_ptr1 + (x1 + 64*y0), tmp4, x1_mask & y0_mask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg6_1 = copy_if_misaligned(arg6_1)
            # Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add, aten.copy_]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__0.run(arg0_1, arg6_1, arg0_1, 64, stream=raw_stream0)
            del arg0_1
            del arg6_1
            arg7_1 = copy_if_misaligned(arg7_1)
            # Topologically Sorted Source Nodes: [add__1], Original ATen: [aten.add, aten.copy_]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__1.run(arg1_1, arg7_1, arg1_1, 64, 64, stream=raw_stream0)
            del arg1_1
            del arg7_1
            arg8_1 = copy_if_misaligned(arg8_1)
            # Topologically Sorted Source Nodes: [add__2], Original ATen: [aten.add, aten.copy_]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__0.run(arg2_1, arg8_1, arg2_1, 64, stream=raw_stream0)
            del arg2_1
            del arg8_1
            arg9_1 = copy_if_misaligned(arg9_1)
            # Topologically Sorted Source Nodes: [add__3], Original ATen: [aten.add, aten.copy_]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__1.run(arg3_1, arg9_1, arg3_1, 64, 64, stream=raw_stream0)
            del arg3_1
            del arg9_1
            arg10_1 = copy_if_misaligned(arg10_1)
            # Topologically Sorted Source Nodes: [add__4], Original ATen: [aten.add, aten.copy_]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__0.run(arg4_1, arg10_1, arg4_1, 64, stream=raw_stream0)
            del arg10_1
            del arg4_1
            arg11_1 = copy_if_misaligned(arg11_1)
            # Topologically Sorted Source Nodes: [add__5], Original ATen: [aten.add, aten.copy_]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__1.run(arg5_1, arg11_1, arg5_1, 64, 64, stream=raw_stream0)
            del arg11_1
            del arg5_1
        return ()

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('basic_gnn_gcn', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
