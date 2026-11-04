# AOT ID: ['0_forward']
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
# Topologically Sorted Source Nodes: [edge_index_i, count, new_ones, scatter_add_], Original ATen: [aten.select, aten.new_zeros, aten.new_ones, aten.scatter_add]
# Source node to ATen node mapping:
#   count => full_default
#   edge_index_i => select_1
#   new_ones => full_default_1
#   scatter_add_ => scatter_add
# Graph fragment:
#   %select_1 : Tensor "i64[200000][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_2, 0, 1), kwargs = {})
#   %full_default : Tensor "f32[10000][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([10000], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %full_default_1 : Tensor "f32[200000][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([200000], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %scatter_add : Tensor "f32[10000][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default, 0, %select_1, %full_default_1), kwargs = {})
#   return %scatter_add
# SchedulerNodes: [SchedulerNode(name='op0')]

triton_poi_fused_new_ones_new_zeros_scatter_ad_0 = async_compile.triton('triton_poi_fused_new_ones_new_zeros_scatter_ad_0', '''
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
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_ones_new_zeros_scatter_ad_0', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['x0'], 'axis_static_values': (('x0', 10000),), 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_ones_new_zeros_scatter_ad_0(out_ptr0, x0_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
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
# Topologically Sorted Source Nodes: [edge_index_i, count, new_ones, scatter_add_], Original ATen: [aten.select, aten.new_zeros, aten.new_ones, aten.scatter_add]
# Source node to ATen node mapping:
#   count => full_default
#   edge_index_i => select_1
#   new_ones => full_default_1
#   scatter_add_ => scatter_add
# Graph fragment:
#   %primals_2 : Tensor "i64[2, 200000][200000, 1]npu:0" = PlaceHolder[target=primals_2]
#   %scatter_add : Tensor "f32[10000][1]npu:0" = PlaceHolder[target=scatter_add]
#   %select_1 : Tensor "i64[200000][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_2, 0, 1), kwargs = {})
#   %full_default : Tensor "f32[10000][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([10000], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %full_default_1 : Tensor "f32[200000][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([200000], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %scatter_add : Tensor "f32[10000][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default, 0, %select_1, %full_default_1), kwargs = {})
#   return %buf1
# SchedulerNodes: [SchedulerNode(name='op1')]

triton_poi_fused_new_ones_new_zeros_scatter_ad_1 = async_compile.triton('triton_poi_fused_new_ones_new_zeros_scatter_ad_1', '''
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
    size_hints={'x0': 200000}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'x0_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_ones_new_zeros_scatter_ad_1', 'mutated_arg_names': ['out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['x0'], 'axis_static_values': (('x0', 200000),), 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_ones_new_zeros_scatter_ad_1(in_ptr0, out_ptr0, x0_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        tmp0 = tl.load(in_ptr0 + (200000 + x0), x0_mask)
        tl.device_assert(((0 <= tmp0) & (tmp0 < 10000)) | ~(x0_mask), "index out of bounds: 0 <= tmp0 < 10000")
        tmp2 = tl.full([1], 1.0, tl.float32)
        tl.atomic_add(out_ptr0 + (tmp0), tmp2, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [index, edge_index_i, view, index_1, x_j_2, out_8], Original ATen: [aten.select, aten.view, aten.expand, aten.index_select, aten.scatter_add]
# Source node to ATen node mapping:
#   edge_index_i => select_1
#   index => select
#   index_1 => expand
#   out_8 => scatter_add_5
#   view => view
#   x_j_2 => index_2
# Graph fragment:
#   %select : Tensor "i64[200000][1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.select.int](args = (%primals_2, 0, 0), kwargs = {})
#   %select_1 : Tensor "i64[200000][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_2, 0, 1), kwargs = {})
#   %view : Tensor "i64[200000, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [-1, 1]), kwargs = {})
#   %expand : Tensor "i64[200000, 64][1, 0]npu:0"[num_users=3] = call_function[target=torch.ops.aten.expand.default](args = (%view, [200000, 64]), kwargs = {})
#   %index_2 : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%relu_1, [%select]), kwargs = {})
#   %scatter_add_5 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_2, 0, %expand, %index_2), kwargs = {})
#   return %scatter_add_5
# SchedulerNodes: [SchedulerNode(name='op15')]

triton_poi_fused_expand_index_select_scatter_a_2 = async_compile.triton('triton_poi_fused_expand_index_select_scatter_a_2', '''
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
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_expand_index_select_scatter_a_2', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 10000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_expand_index_select_scatter_a_2(out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
# Topologically Sorted Source Nodes: [index, x_j, edge_index_i, view, index_1, out], Original ATen: [aten.select, aten.index_select, aten.view, aten.expand, aten.scatter_add]
# Source node to ATen node mapping:
#   edge_index_i => select_1
#   index => select
#   index_1 => expand
#   out => scatter_add_1
#   view => view
#   x_j => index
# Graph fragment:
#   %primals_2 : Tensor "i64[2, 200000][200000, 1]npu:0" = PlaceHolder[target=primals_2]
#   %primals_1 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=primals_1]
#   %scatter_add_1 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=scatter_add_1]
#   %select : Tensor "i64[200000][1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.select.int](args = (%primals_2, 0, 0), kwargs = {})
#   %index : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%primals_1, [%select]), kwargs = {})
#   %select_1 : Tensor "i64[200000][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_2, 0, 1), kwargs = {})
#   %view : Tensor "i64[200000, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [-1, 1]), kwargs = {})
#   %expand : Tensor "i64[200000, 64][1, 0]npu:0"[num_users=3] = call_function[target=torch.ops.aten.expand.default](args = (%view, [200000, 64]), kwargs = {})
#   %scatter_add_1 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_2, 0, %expand, %index), kwargs = {})
#   return %buf4
# SchedulerNodes: [SchedulerNode(name='op4')]

triton_poi_fused_expand_index_select_scatter_a_3 = async_compile.triton('triton_poi_fused_expand_index_select_scatter_a_3', '''
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
    size_hints={'y0': 200000, 'x1': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_expand_index_select_scatter_a_3', 'mutated_arg_names': ['out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 200000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_expand_index_select_scatter_a_3(in_ptr0, in_ptr1, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (200000 + y0), y0_mask)
        tmp2 = tl.load(in_ptr0 + (y0), y0_mask)
        tl.device_assert(((0 <= tmp0) & (tmp0 < 10000)) | ~(y0_mask), "index out of bounds: 0 <= tmp0 < 10000")
        tmp3 = tl.full([Y0BLOCK_SUB, X1BLOCK_SUB], 10000, tl.int32)
        tmp4 = tmp2 + tmp3
        tmp5 = tmp2 < 0
        tmp6 = tl.where(tmp5, tmp4, tmp2)
        tl.device_assert(((0 <= tmp6) & (tmp6 < 10000)) | ~(y0_mask), "index out of bounds: 0 <= tmp6 < 10000")
        tmp8 = tl.load(in_ptr1 + (x1 + 64*tmp6), x1_mask & y0_mask)
        tl.atomic_add(out_ptr0 + (x1 + 64*tmp0), tmp8, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [count_1, view_1, expand_as_1, out_1], Original ATen: [aten.clamp, aten.view, aten.expand, aten.div]
# Source node to ATen node mapping:
#   count_1 => clamp_min
#   expand_as_1 => expand_1
#   out_1 => div
#   view_1 => view_1
# Graph fragment:
#   %buf4 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=buf4]
#   %buf1 : Tensor "f32[10000][1]npu:0" = PlaceHolder[target=buf1]
#   %clamp_min : Tensor "f32[10000][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%scatter_add, 1), kwargs = {})
#   %view_1 : Tensor "f32[10000, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clamp_min, [-1, 1]), kwargs = {})
#   %expand_1 : Tensor "f32[10000, 64][1, 0]npu:0"[num_users=3] = call_function[target=torch.ops.aten.expand.default](args = (%view_1, [10000, 64]), kwargs = {})
#   %div : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%scatter_add_1, %expand_1), kwargs = {})
#   return %div
# SchedulerNodes: [SchedulerNode(name='op5')]

triton_poi_fused_clamp_div_expand_view_4 = async_compile.triton('triton_poi_fused_clamp_div_expand_view_4', '''
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
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_div_expand_view_4', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 10000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_div_expand_view_4(in_ptr0, in_ptr1, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp1 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp2 = tl.full([1, 1], 1.0, tl.float32)
        tmp3 = triton_helpers.maximum(tmp1, tmp2)
        tmp4 = (tmp0 / tmp3)
        tl.store(out_ptr0 + (x1 + 64*y0), tmp4, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [, out_3, x], Original ATen: [aten.addmm, aten.add, aten.relu]
# Source node to ATen node mapping:
#    => add_tensor_1
#   out_3 => add
#   x => relu
# Graph fragment:
#   %primals_4 : Tensor "f32[64][1]npu:0" = PlaceHolder[target=primals_4]
#   %mm_default_1 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=mm_default_1]
#   %mm : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=mm]
#   %add_tensor_1 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_4, %mm_default_1), kwargs = {})
#   %add : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor_1, %mm), kwargs = {})
#   %relu : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add,), kwargs = {})
#   return %relu
# SchedulerNodes: [SchedulerNode(name='op8')]

triton_poi_fused_add_addmm_relu_5 = async_compile.triton('triton_poi_fused_add_addmm_relu_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 10000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_relu_5(in_out_ptr0, in_ptr0, in_ptr1, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x1), None)
        tmp1 = tl.load(in_out_ptr0 + (x1 + 64*y0), y0_mask)
        tmp3 = tl.load(in_ptr1 + (x1 + 64*y0), y0_mask)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.full([1, 1], 0, tl.int32)
        tmp6 = triton_helpers.maximum(tmp5, tmp4)
        tl.store(in_out_ptr0 + (x1 + 64*y0), tmp6, y0_mask)
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            buf0 = empty_strided_npu((10000, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [edge_index_i, count, new_ones, scatter_add_], Original ATen: [aten.select, aten.new_zeros, aten.new_ones, aten.scatter_add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_new_ones_new_zeros_scatter_ad_0.run(buf0, 10000, stream=raw_stream0)
            primals_2 = copy_if_misaligned(primals_2)
            # Topologically Sorted Source Nodes: [edge_index_i, count, new_ones, scatter_add_], Original ATen: [aten.select, aten.new_zeros, aten.new_ones, aten.scatter_add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_new_ones_new_zeros_scatter_ad_1.run(primals_2, buf0, 200000, stream=raw_stream0)
            buf15 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [index, edge_index_i, view, index_1, x_j_2, out_8], Original ATen: [aten.select, aten.view, aten.expand, aten.index_select, aten.scatter_add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_expand_index_select_scatter_a_2.run(buf15, 10000, 64, stream=raw_stream0)
            buf9 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [index, edge_index_i, view, index_1, x_j_1, out_4], Original ATen: [aten.select, aten.view, aten.expand, aten.index_select, aten.scatter_add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_expand_index_select_scatter_a_2.run(buf9, 10000, 64, stream=raw_stream0)
            primals_1 = copy_if_misaligned(primals_1)
            buf7 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.t, aten.matmul]
            # [Provenance debug handles] extern_kernels.mm:1
            extern_kernels.mm(primals_1, reinterpret_tensor(primals_5, (64, 64), (1, 64), 0), out=buf7)
            del primals_5
            buf3 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [index, x_j, edge_index_i, view, index_1, out], Original ATen: [aten.select, aten.index_select, aten.view, aten.expand, aten.scatter_add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_expand_index_select_scatter_a_2.run(buf3, 10000, 64, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [index, x_j, edge_index_i, view, index_1, out], Original ATen: [aten.select, aten.index_select, aten.view, aten.expand, aten.scatter_add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_expand_index_select_scatter_a_3.run(primals_2, primals_1, buf3, 200000, 64, stream=raw_stream0)
            buf5 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [count_1, view_1, expand_as_1, out_1], Original ATen: [aten.clamp, aten.view, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clamp_div_expand_view_4.run(buf3, buf0, buf5, 10000, 64, stream=raw_stream0)
            buf6 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [out_2, ], Original ATen: [aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:2
            extern_kernels.mm(buf5, reinterpret_tensor(primals_3, (64, 64), (1, 64), 0), out=buf6)
            del primals_3
            buf8 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [, out_3, x], Original ATen: [aten.addmm, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_relu_5.run(buf8, primals_4, buf7, 10000, 64, stream=raw_stream0)
            del primals_4
            # Topologically Sorted Source Nodes: [index, edge_index_i, view, index_1, x_j_1, out_4], Original ATen: [aten.select, aten.view, aten.expand, aten.index_select, aten.scatter_add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_expand_index_select_scatter_a_3.run(primals_2, buf8, buf9, 200000, 64, stream=raw_stream0)
            buf11 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [count_1, view_1, expand_as_1, out_5], Original ATen: [aten.clamp, aten.view, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clamp_div_expand_view_4.run(buf9, buf0, buf11, 10000, 64, stream=raw_stream0)
            buf12 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [out_6, ], Original ATen: [aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:3
            extern_kernels.mm(buf11, reinterpret_tensor(primals_6, (64, 64), (1, 64), 0), out=buf12)
            buf13 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.t, aten.matmul]
            # [Provenance debug handles] extern_kernels.mm:4
            extern_kernels.mm(buf8, reinterpret_tensor(primals_8, (64, 64), (1, 64), 0), out=buf13)
            buf14 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [, out_7, x_2], Original ATen: [aten.addmm, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_relu_5.run(buf14, primals_7, buf13, 10000, 64, stream=raw_stream0)
            del primals_7
            # Topologically Sorted Source Nodes: [index, edge_index_i, view, index_1, x_j_2, out_8], Original ATen: [aten.select, aten.view, aten.expand, aten.index_select, aten.scatter_add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_expand_index_select_scatter_a_3.run(primals_2, buf14, buf15, 200000, 64, stream=raw_stream0)
            buf17 = buf13; del buf13  # reuse
            # Topologically Sorted Source Nodes: [count_1, view_1, expand_as_1, out_9], Original ATen: [aten.clamp, aten.view, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clamp_div_expand_view_4.run(buf15, buf0, buf17, 10000, 64, stream=raw_stream0)
            del buf15
            # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten.t, aten.addmm]
            # [Provenance debug handles] torch.ops.aten.addmm.default:5
            buf18 = torch.ops.aten.addmm.default(primals_10, buf17, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), alpha=1, beta=1)
            del primals_10
            buf19 = buf18
            assert_size_stride(buf19, (10000, 64), (64, 1), 'torch.ops.aten.addmm.default')
            assert_alignment(buf19, 16, 'torch.ops.aten.addmm.default')
            del buf18
            # Topologically Sorted Source Nodes: [linear_5, ], Original ATen: [aten.t, aten.add]
            # [Provenance debug handles] torch.ops.aten.addmm.default:6
            buf20 = torch.ops.aten.addmm.default(buf19, buf14, reinterpret_tensor(primals_11, (64, 64), (1, 64), 0), alpha=1, beta=1)
            buf21 = buf20
            assert_size_stride(buf21, (10000, 64), (64, 1), 'torch.ops.aten.addmm.default')
            assert_alignment(buf21, 16, 'torch.ops.aten.addmm.default')
            del buf20
            buf2 = buf19; del buf19  # reuse
            # Topologically Sorted Source Nodes: [new_zeros_1], Original ATen: [aten.new_zeros]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_expand_index_select_scatter_a_2.run(buf2, 10000, 64, stream=raw_stream0)
        return (buf21, primals_2, primals_6, primals_8, primals_9, primals_11, buf0, buf2, buf5, buf8, buf11, buf14, buf17, reinterpret_tensor(primals_1, (64, 10000), (1, 64), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((10000, 64), (64, 1), device='npu:0', dtype=torch.float32)
    primals_2 = rand_strided((2, 200000), (200000, 1), device='npu:0', dtype=torch.int64)
    primals_3 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_5 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    primals_6 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    primals_9 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_11 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    return [primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('basic_gnn_sage', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
