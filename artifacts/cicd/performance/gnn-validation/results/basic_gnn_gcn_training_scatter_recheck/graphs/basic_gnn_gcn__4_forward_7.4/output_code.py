# AOT ID: ['4_forward']
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
# Topologically Sorted Source Nodes: [new_zeros], Original ATen: [aten.new_zeros]
# Source node to ATen node mapping:
#   new_zeros => full_default
# Graph fragment:
#   %full_default : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([10000, 64], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   return %full_default
# SchedulerNodes: [SchedulerNode(name='op1')]

triton_poi_fused_new_zeros_0 = async_compile.triton('triton_poi_fused_new_zeros_0', '''
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
    size_hints={'y0': 10000, 'x1': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_0', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 10000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_zeros_0(out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
    x1_numel = 64
    X1BLOCK_SUB: tl.constexpr = 64
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        x1 = base_x1[None,:]
        tmp0 = tl.full([1, 1], 0.0, tl.float32)
        tl.store(out_ptr0 + (x1 + 64*y0), tmp0, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [index, x_j, edge_index_i, view, out, view_1, index_1, out_1], Original ATen: [aten.select, aten.index_select, aten.view, aten.mul, aten.expand, aten.scatter_add]
# Source node to ATen node mapping:
#   edge_index_i => select_1
#   index => select
#   index_1 => expand
#   out => mul
#   out_1 => scatter_add
#   view => view
#   view_1 => view_1
#   x_j => index
# Graph fragment:
#   %primals_3 : Tensor "i64[2, 209978][209978, 1]npu:0" = PlaceHolder[target=primals_3]
#   %primals_4 : Tensor "f32[209978][1]npu:0" = PlaceHolder[target=primals_4]
#   %mm : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=mm]
#   %scatter_add : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=scatter_add]
#   %select : Tensor "i64[209978][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_3, 0, 0), kwargs = {})
#   %index : Tensor "f32[209978, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%mm, [%select]), kwargs = {})
#   %select_1 : Tensor "i64[209978][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_3, 0, 1), kwargs = {})
#   %view : Tensor "f32[209978, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_4, [-1, 1]), kwargs = {})
#   %mul : Tensor "f32[209978, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %index), kwargs = {})
#   %view_1 : Tensor "i64[209978, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [-1, 1]), kwargs = {})
#   %expand : Tensor "i64[209978, 64][1, 0]npu:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_1, [209978, 64]), kwargs = {})
#   %scatter_add : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default, 0, %expand, %mul), kwargs = {})
#   return %buf3
# SchedulerNodes: [SchedulerNode(name='op3')]

triton_poi_fused_expand_index_select_mul_scatt_1 = async_compile.triton('triton_poi_fused_expand_index_select_mul_scatt_1', '''
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
    size_hints={'y0': 209978, 'x1': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_expand_index_select_mul_scatt_1', 'mutated_arg_names': ['out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 209978), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_expand_index_select_mul_scatt_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
    x1_numel = 64
    X1BLOCK_SUB: tl.constexpr = 64
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        x1 = base_x1[None,:]
        x1_mask = x1 < x1_numel
        tmp0 = tl.load(in_ptr0 + (209978 + y0), y0_mask)
        tmp2 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp3 = tl.load(in_ptr0 + (y0), y0_mask)
        tl.device_assert(((0 <= tmp0) & (tmp0 < 10000)) | ~(y0_mask), "index out of bounds: 0 <= tmp0 < 10000")
        tmp4 = tl.full([Y0BLOCK_SUB, X1BLOCK_SUB], 10000, tl.int32)
        tmp5 = tmp3 + tmp4
        tmp6 = tmp3 < 0
        tmp7 = tl.where(tmp6, tmp5, tmp3)
        tl.device_assert(((0 <= tmp7) & (tmp7 < 10000)) | ~(y0_mask), "index out of bounds: 0 <= tmp7 < 10000")
        tmp9 = tl.load(in_ptr2 + (x1 + 64*tmp7), x1_mask & y0_mask)
        tmp10 = tmp2 * tmp9
        tl.atomic_add(out_ptr0 + (x1 + 64*tmp0), tmp10, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   out_2 => add
# Graph fragment:
#   %buf3 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=buf3]
#   %primals_5 : Tensor "f32[64][1]npu:0" = PlaceHolder[target=primals_5]
#   %add : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%scatter_add, %primals_5), kwargs = {})
#   return %add
# SchedulerNodes: [SchedulerNode(name='op4')]

triton_poi_fused_add_2 = async_compile.triton('triton_poi_fused_add_2', '''
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
    size_hints={'y0': 10000, 'x1': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_2', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 10000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_2(in_ptr0, in_ptr1, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
    x1_numel = 64
    X1BLOCK_SUB: tl.constexpr = 64
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        x1 = base_x1[None,:]
        tmp0 = tl.load(in_ptr0 + (x1 + 64*y0), y0_mask)
        tmp1 = tl.load(in_ptr1 + (x1), None)
        tmp2 = tmp0 + tmp1
        tl.store(out_ptr0 + (x1 + 64*y0), tmp2, y0_mask)
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
        primals_1, primals_2, primals_3, primals_4, primals_5 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            primals_2 = copy_if_misaligned(primals_2)
            buf0 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.t, aten.matmul]
            # [Provenance debug handles] extern_kernels.mm:1
            extern_kernels.mm(primals_2, reinterpret_tensor(primals_1, (64, 64), (1, 64), 0), out=buf0)
            buf1 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [new_zeros], Original ATen: [aten.new_zeros]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_new_zeros_0.run(buf1, 10000, 64, stream=raw_stream0)
            buf2 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [index, x_j, edge_index_i, view, out, view_1, index_1, out_1], Original ATen: [aten.select, aten.index_select, aten.view, aten.mul, aten.expand, aten.scatter_add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_new_zeros_0.run(buf2, 10000, 64, stream=raw_stream0)
            primals_3 = copy_if_misaligned(primals_3)
            primals_4 = copy_if_misaligned(primals_4)
            # Topologically Sorted Source Nodes: [index, x_j, edge_index_i, view, out, view_1, index_1, out_1], Original ATen: [aten.select, aten.index_select, aten.view, aten.mul, aten.expand, aten.scatter_add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_expand_index_select_mul_scatt_1.run(primals_3, primals_4, buf0, buf2, 209978, 64, stream=raw_stream0)
            buf4 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_2.run(buf2, primals_5, buf4, 10000, 64, stream=raw_stream0)
            del buf2
            del primals_5
        return (buf4, primals_1, primals_3, primals_4, buf1, reinterpret_tensor(primals_2, (64, 10000), (1, 64), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    primals_2 = rand_strided((10000, 64), (64, 1), device='npu:0', dtype=torch.float32)
    primals_3 = rand_strided((2, 209978), (209978, 1), device='npu:0', dtype=torch.int64)
    primals_4 = rand_strided((209978, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    return [primals_1, primals_2, primals_3, primals_4, primals_5]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('basic_gnn_gcn', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
