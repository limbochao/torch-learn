# AOT ID: ['2_inference']
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
# Topologically Sorted Source Nodes: [new_zeros, col, edge_weight, deg], Original ATen: [aten.new_zeros, aten.select, aten.ones, aten.scatter_add]
# Source node to ATen node mapping:
#   col => select_1
#   deg => scatter_add
#   edge_weight => full_default
#   new_zeros => full_default_1
# Graph fragment:
#   %full_default_1 : Tensor "f32[10000][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([10000], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %select_1 : Tensor "i64[209978][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 1), kwargs = {})
#   %full_default : Tensor "f32[209978][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([209978], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %scatter_add : Tensor "f32[10000][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_1, 0, %select_1, %full_default), kwargs = {})
#   return %scatter_add
# SchedulerNodes: [SchedulerNode(name='op0')]

triton_poi_fused_new_zeros_ones_scatter_add_se_0 = async_compile.triton('triton_poi_fused_new_zeros_ones_scatter_add_se_0', '''
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
    size_hints={'x0': 10000}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'x0_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_ones_scatter_add_se_0', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['x0'], 'axis_static_values': (('x0', 10000),), 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_zeros_ones_scatter_add_se_0(out_ptr0, x0_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        tmp0 = tl.full([1], 0.0, tl.float32)
        tl.store(out_ptr0 + (x0), tmp0, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [new_zeros, col, edge_weight, deg], Original ATen: [aten.new_zeros, aten.select, aten.ones, aten.scatter_add]
# Source node to ATen node mapping:
#   col => select_1
#   deg => scatter_add
#   edge_weight => full_default
#   new_zeros => full_default_1
# Graph fragment:
#   %arg0_1 : Tensor "i64[2, 209978][209978, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %scatter_add : Tensor "f32[10000][1]npu:0" = PlaceHolder[target=scatter_add]
#   %full_default_1 : Tensor "f32[10000][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([10000], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %select_1 : Tensor "i64[209978][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 1), kwargs = {})
#   %full_default : Tensor "f32[209978][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([209978], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %scatter_add : Tensor "f32[10000][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_1, 0, %select_1, %full_default), kwargs = {})
#   return %buf1
# SchedulerNodes: [SchedulerNode(name='op1')]

triton_poi_fused_new_zeros_ones_scatter_add_se_1 = async_compile.triton('triton_poi_fused_new_zeros_ones_scatter_add_se_1', '''
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
    size_hints={'x0': 209978}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'x0_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_ones_scatter_add_se_1', 'mutated_arg_names': ['out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['x0'], 'axis_static_values': (('x0', 209978),), 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_zeros_ones_scatter_add_se_1(in_ptr0, out_ptr0, x0_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        tmp0 = tl.load(in_ptr0 + (209978 + x0), x0_mask)
        tl.device_assert(((0 <= tmp0) & (tmp0 < 10000)) | ~(x0_mask), "index out of bounds: 0 <= tmp0 < 10000")
        tmp2 = tl.full([1], 1.0, tl.float32)
        tl.atomic_add(out_ptr0 + (tmp0), tmp2, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [col, deg_inv_sqrt, eq, masked_fill_, row, mul, getitem_3, edge_weight_1], Original ATen: [aten.select, aten.pow, aten.eq, aten.masked_fill, aten.mul, aten.index]
# Source node to ATen node mapping:
#   col => select_1
#   deg_inv_sqrt => pow_1
#   edge_weight_1 => mul_1
#   eq => eq
#   getitem_3 => index_1
#   masked_fill_ => full_default_2, where
#   mul => index
#   row => select
# Graph fragment:
#   %arg0_1 : Tensor "i64[2, 209978][209978, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %buf1 : Tensor "f32[10000][1]npu:0" = PlaceHolder[target=buf1]
#   %select_1 : Tensor "i64[209978][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 1), kwargs = {})
#   %pow_1 : Tensor "f32[10000][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%scatter_add, -0.5), kwargs = {})
#   %eq : Tensor "b8[10000][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%pow_1, inf), kwargs = {})
#   %full_default_2 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where : Tensor "f32[10000][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default_2, %pow_1), kwargs = {})
#   %select : Tensor "i64[209978][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 0, 0), kwargs = {})
#   %index : Tensor "f32[209978][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%where, [%select]), kwargs = {})
#   %index_1 : Tensor "f32[209978][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%where, [%select_1]), kwargs = {})
#   %mul_1 : Tensor "f32[209978][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%index, %index_1), kwargs = {})
#   return %mul_1
# SchedulerNodes: [SchedulerNode(name='op2')]

triton_poi_fused_eq_index_masked_fill_mul_pow__2 = async_compile.triton('triton_poi_fused_eq_index_masked_fill_mul_pow__2', '''
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
    size_hints={'x0': 209978}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'x0_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_eq_index_masked_fill_mul_pow__2', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['x0'], 'axis_static_values': (('x0', 209978),), 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_eq_index_masked_fill_mul_pow__2(in_ptr0, in_ptr1, out_ptr0, x0_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        tmp0 = tl.load(in_ptr0 + (x0), x0_mask)
        tmp13 = tl.load(in_ptr0 + (209978 + x0), x0_mask)
        tmp1 = tl.full([X0BLOCK_SUB], 10000, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 10000)) | ~(x0_mask), "index out of bounds: 0 <= tmp4 < 10000")
        tmp6 = tl.load(in_ptr1 + (tmp4), x0_mask)
        tmp7 = tl.full([1], -0.5, tl.float32)
        tmp8 = libdevice.pow(tmp6, tmp7)
        tmp9 = tl.full([1], float("inf"), tl.float32)
        tmp10 = tmp8 == tmp9
        tmp11 = tl.full([1], 0.0, tl.float32)
        tmp12 = tl.where(tmp10, tmp11, tmp8)
        tmp14 = tmp13 + tmp1
        tmp15 = tmp13 < 0
        tmp16 = tl.where(tmp15, tmp14, tmp13)
        tl.device_assert(((0 <= tmp16) & (tmp16 < 10000)) | ~(x0_mask), "index out of bounds: 0 <= tmp16 < 10000")
        tmp18 = tl.load(in_ptr1 + (tmp16), x0_mask)
        tmp19 = libdevice.pow(tmp18, tmp7)
        tmp20 = tmp19 == tmp9
        tmp21 = tl.where(tmp20, tmp11, tmp19)
        tmp22 = tmp12 * tmp21
        tl.store(out_ptr0 + (x0), tmp22, x0_mask)
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
        arg0_1, = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            buf0 = empty_strided_npu((10000, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [new_zeros, col, edge_weight, deg], Original ATen: [aten.new_zeros, aten.select, aten.ones, aten.scatter_add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_new_zeros_ones_scatter_add_se_0.run(buf0, 10000, stream=raw_stream0)
            arg0_1 = copy_if_misaligned(arg0_1)
            # Topologically Sorted Source Nodes: [new_zeros, col, edge_weight, deg], Original ATen: [aten.new_zeros, aten.select, aten.ones, aten.scatter_add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_new_zeros_ones_scatter_add_se_1.run(arg0_1, buf0, 209978, stream=raw_stream0)
            buf2 = empty_strided_npu((209978, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [col, deg_inv_sqrt, eq, masked_fill_, row, mul, getitem_3, edge_weight_1], Original ATen: [aten.select, aten.pow, aten.eq, aten.masked_fill, aten.mul, aten.index]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_eq_index_masked_fill_mul_pow__2.run(arg0_1, buf0, buf2, 209978, stream=raw_stream0)
            del arg0_1
            del buf0
        return (buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((2, 209978), (209978, 1), device='npu:0', dtype=torch.int64)
    return [arg0_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('basic_gnn_gcn', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
