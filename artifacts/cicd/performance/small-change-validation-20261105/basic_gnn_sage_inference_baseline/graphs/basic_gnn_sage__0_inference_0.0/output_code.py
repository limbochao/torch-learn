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


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [count], Original ATen: [aten.new_zeros]
# Source node to ATen node mapping:
#   count => full_default
# Graph fragment:
#   %full_default : Tensor "f32[10000][1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([10000], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   return %full_default
# SchedulerNodes: [SchedulerNode(name='op4')]

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
    size_hints={'x0': 10000}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'x0_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_0', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['x0'], 'axis_static_values': (('x0', 10000),), 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_zeros_0(out_ptr0, x0_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
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
# Topologically Sorted Source Nodes: [new_ones], Original ATen: [aten.new_ones]
# Source node to ATen node mapping:
#   new_ones => full_default_1
# Graph fragment:
#   %full_default_1 : Tensor "f32[200000][1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([200000], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   return %full_default_1
# SchedulerNodes: [SchedulerNode(name='op5')]

triton_poi_fused_new_ones_1 = async_compile.triton('triton_poi_fused_new_ones_1', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'x0_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_ones_1', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['x0'], 'axis_static_values': (('x0', 200000),), 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_ones_1(out_ptr0, x0_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        tmp0 = tl.full([1], 1.0, tl.float32)
        tl.store(out_ptr0 + (x0), tmp0, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [new_zeros_5], Original ATen: [aten.new_zeros]
# Source node to ATen node mapping:
#   new_zeros_5 => full_default_8
# Graph fragment:
#   %full_default_8 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([10000, 64], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   return %full_default_8
# SchedulerNodes: [SchedulerNode(name='op0')]

triton_poi_fused_new_zeros_2 = async_compile.triton('triton_poi_fused_new_zeros_2', '''
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
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_2', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 10000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_zeros_2(out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
# Topologically Sorted Source Nodes: [index, x_j], Original ATen: [aten.select, aten.index_select]
# Source node to ATen node mapping:
#   index => select
#   x_j => index
# Graph fragment:
#   %arg1_1 : Tensor "i64[2, 200000][200000, 1]npu:0" = PlaceHolder[target=arg1_1]
#   %arg0_1 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %select : Tensor "i64[200000][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg1_1, 0, 0), kwargs = {})
#   %index : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg0_1, [%select]), kwargs = {})
#   return %index
# SchedulerNodes: [SchedulerNode(name='op1')]

triton_poi_fused_index_select_select_3 = async_compile.triton('triton_poi_fused_index_select_select_3', '''
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
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_select_3', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 200000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_select_select_3(in_ptr0, in_ptr1, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (y0), y0_mask)
        tmp1 = tl.full([Y0BLOCK_SUB, X1BLOCK_SUB], 10000, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 10000)) | ~(y0_mask), "index out of bounds: 0 <= tmp4 < 10000")
        tmp6 = tl.load(in_ptr1 + (x1 + 64*tmp4), x1_mask & y0_mask)
        tl.store(out_ptr0 + (x1 + 64*y0), tmp6, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [count_1, view_1, expand_as_1, out_1], Original ATen: [aten.clamp, aten.view, aten.expand, aten.div]
# Source node to ATen node mapping:
#   count_1 => clamp_min
#   expand_as_1 => expand_1
#   out_1 => div
#   view_1 => view_1
# Graph fragment:
#   %scatter_add_1 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=scatter_add_1]
#   %scatter_add : Tensor "f32[10000][1]npu:0" = PlaceHolder[target=scatter_add]
#   %clamp_min : Tensor "f32[10000][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%scatter_add, 1), kwargs = {})
#   %view_1 : Tensor "f32[10000, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clamp_min, [-1, 1]), kwargs = {})
#   %expand_1 : Tensor "f32[10000, 64][1, 0]npu:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_1, [10000, 64]), kwargs = {})
#   %div : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%scatter_add_1, %expand_1), kwargs = {})
#   return %div
# SchedulerNodes: [SchedulerNode(name='op8')]

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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_div_expand_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 10000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_div_expand_view_4(in_out_ptr0, in_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp0 = tl.load(in_out_ptr0 + (x1 + 64*y0), y0_mask)
        tmp1 = tl.load(in_ptr0 + (y0), y0_mask)
        tmp2 = tl.full([1, 1], 1.0, tl.float32)
        tmp3 = triton_helpers.maximum(tmp1, tmp2)
        tmp4 = (tmp0 / tmp3)
        tl.store(in_out_ptr0 + (x1 + 64*y0), tmp4, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [, out_3, x], Original ATen: [aten.addmm, aten.add, aten.relu]
# Source node to ATen node mapping:
#    => add_tensor_1
#   out_3 => add
#   x => relu
# Graph fragment:
#   %arg3_1 : Tensor "f32[64][1]npu:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=mm_default_1]
#   %mm : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=mm]
#   %add_tensor_1 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %add : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor_1, %mm), kwargs = {})
#   %relu : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add,), kwargs = {})
#   return %relu
# SchedulerNodes: [SchedulerNode(name='op11')]

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


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [, out_7, x_2], Original ATen: [aten.addmm, aten.add, aten.relu]
# Source node to ATen node mapping:
#    => add_tensor
#   out_7 => add_1
#   x_2 => relu_1
# Graph fragment:
#   %arg6_1 : Tensor "f32[64][1]npu:0" = PlaceHolder[target=arg6_1]
#   %mm_default : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=mm_default]
#   %mm_1 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=mm_1]
#   %add_tensor : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg6_1, %mm_default), kwargs = {})
#   %add_1 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor, %mm_1), kwargs = {})
#   %relu_1 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
#   return %relu_1
# SchedulerNodes: [SchedulerNode(name='op28')]

triton_poi_fused_add_addmm_relu_6 = async_compile.triton('triton_poi_fused_add_addmm_relu_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_relu_6', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 10000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_relu_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp1 = tl.load(in_ptr1 + (x1 + 64*y0), y0_mask)
        tmp3 = tl.load(in_ptr2 + (x1 + 64*y0), y0_mask)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.full([1, 1], 0, tl.int32)
        tmp6 = triton_helpers.maximum(tmp5, tmp4)
        tl.store(out_ptr0 + (x1 + 64*y0), tmp6, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [, out_7, x_2, index_4, x_j_2], Original ATen: [aten.addmm, aten.add, aten.relu, aten.select, aten.index_select]
# Source node to ATen node mapping:
#    => add_tensor
#   index_4 => select_4
#   out_7 => add_1
#   x_2 => relu_1
#   x_j_2 => index_2
# Graph fragment:
#   %arg1_1 : Tensor "i64[2, 200000][200000, 1]npu:0" = PlaceHolder[target=arg1_1]
#   %arg6_1 : Tensor "f32[64][1]npu:0" = PlaceHolder[target=arg6_1]
#   %mm_default : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=mm_default]
#   %mm_1 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=mm_1]
#   %add_tensor : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg6_1, %mm_default), kwargs = {})
#   %add_1 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor, %mm_1), kwargs = {})
#   %relu_1 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
#   %select_4 : Tensor "i64[200000][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg1_1, 0, 0), kwargs = {})
#   %index_2 : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%relu_1, [%select_4]), kwargs = {})
#   return %index_2
# SchedulerNodes: [SchedulerNode(name='op20')]

triton_poi_fused_add_addmm_index_select_relu_s_7 = async_compile.triton('triton_poi_fused_add_addmm_index_select_relu_s_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_index_select_relu_s_7', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 200000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_index_select_relu_s_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (y0), y0_mask)
        tmp6 = tl.load(in_ptr1 + (x1), x1_mask)
        tmp1 = tl.full([Y0BLOCK_SUB, X1BLOCK_SUB], 10000, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 10000)) | ~(y0_mask), "index out of bounds: 0 <= tmp4 < 10000")
        tmp7 = tl.load(in_ptr2 + (x1 + 64*tmp4), x1_mask & y0_mask)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.load(in_ptr3 + (x1 + 64*tmp4), x1_mask & y0_mask)
        tmp10 = tmp8 + tmp9
        tmp11 = tl.full([1, 1], 0, tl.int32)
        tmp12 = triton_helpers.maximum(tmp11, tmp10)
        tl.store(out_ptr0 + (x1 + 64*y0), tmp12, x1_mask & y0_mask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            buf4 = empty_strided_npu((10000, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [count], Original ATen: [aten.new_zeros]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_new_zeros_0.run(buf4, 10000, stream=raw_stream0)
            buf5 = empty_strided_npu((200000, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [new_ones], Original ATen: [aten.new_ones]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_new_ones_1.run(buf5, 200000, stream=raw_stream0)
            arg1_1 = copy_if_misaligned(arg1_1)
            # Topologically Sorted Source Nodes: [edge_index_i, count, new_ones, scatter_add_], Original ATen: [aten.select, aten.new_zeros, aten.new_ones, aten.scatter_add]
            # [Provenance debug handles] torch.ops.aten.scatter_add.default:1
            buf6 = torch.ops.aten.scatter_add.default(buf4, 0, reinterpret_tensor(arg1_1, (200000, ), (1, ), 200000), buf5)
            buf7 = buf6
            assert_size_stride(buf7, (10000, ), (1, ), 'torch.ops.aten.scatter_add.default')
            assert_alignment(buf7, 16, 'torch.ops.aten.scatter_add.default')
            del buf6
            # Topologically Sorted Source Nodes: [edge_index_i_1, scatter_add__2], Original ATen: [aten.select, aten.scatter_add]
            # [Provenance debug handles] torch.ops.aten.scatter_add.default:2
            buf15 = torch.ops.aten.scatter_add.default(buf4, 0, reinterpret_tensor(arg1_1, (200000, ), (1, ), 200000), buf5)
            # Topologically Sorted Source Nodes: [edge_index_i_2, scatter_add__4], Original ATen: [aten.select, aten.scatter_add]
            # [Provenance debug handles] torch.ops.aten.scatter_add.default:3
            buf23 = torch.ops.aten.scatter_add.default(buf4, 0, reinterpret_tensor(arg1_1, (200000, ), (1, ), 200000), buf5)
            del buf4
            del buf5
            buf16 = buf15
            assert_size_stride(buf16, (10000, ), (1, ), 'torch.ops.aten.scatter_add.default')
            assert_alignment(buf16, 16, 'torch.ops.aten.scatter_add.default')
            del buf15
            buf24 = buf23
            assert_size_stride(buf24, (10000, ), (1, ), 'torch.ops.aten.scatter_add.default')
            assert_alignment(buf24, 16, 'torch.ops.aten.scatter_add.default')
            del buf23
            buf0 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [new_zeros_5], Original ATen: [aten.new_zeros]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_new_zeros_2.run(buf0, 10000, 64, stream=raw_stream0)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf10 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.t, aten.matmul]
            # [Provenance debug handles] extern_kernels.mm:4
            extern_kernels.mm(arg0_1, reinterpret_tensor(arg4_1, (64, 64), (1, 64), 0), out=buf10)
            del arg4_1
            buf1 = empty_strided_npu((200000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [index, x_j], Original ATen: [aten.select, aten.index_select]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_index_select_select_3.run(arg1_1, arg0_1, buf1, 200000, 64, stream=raw_stream0)
            del arg0_1
            # Topologically Sorted Source Nodes: [new_zeros_5, edge_index_i, view, index_1, index, x_j, out], Original ATen: [aten.new_zeros, aten.select, aten.view, aten.expand, aten.index_select, aten.scatter_add]
            # [Provenance debug handles] torch.ops.aten.scatter_add.default:5
            buf2 = torch.ops.aten.scatter_add.default(buf0, 0, reinterpret_tensor(arg1_1, (200000, 64), (1, 0), 200000), buf1)
            del buf1
            buf3 = buf2
            assert_size_stride(buf3, (10000, 64), (64, 1), 'torch.ops.aten.scatter_add.default')
            assert_alignment(buf3, 16, 'torch.ops.aten.scatter_add.default')
            del buf2
            buf8 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [count_1, view_1, expand_as_1, out_1], Original ATen: [aten.clamp, aten.view, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clamp_div_expand_view_4.run(buf8, buf7, 10000, 64, stream=raw_stream0)
            del buf7
            buf9 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [count_1, view_1, expand_as_1, out_1, out_2, ], Original ATen: [aten.clamp, aten.view, aten.expand, aten.div, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:6
            extern_kernels.mm(buf8, reinterpret_tensor(arg2_1, (64, 64), (1, 64), 0), out=buf9)
            del arg2_1
            del buf8
            buf11 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [, out_3, x], Original ATen: [aten.addmm, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_relu_5.run(buf11, arg3_1, buf10, 10000, 64, stream=raw_stream0)
            del arg3_1
            buf19 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.t, aten.matmul]
            # [Provenance debug handles] extern_kernels.mm:7
            extern_kernels.mm(buf11, reinterpret_tensor(arg7_1, (64, 64), (1, 64), 0), out=buf19)
            del arg7_1
            buf12 = empty_strided_npu((200000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [index_2, x_j_1], Original ATen: [aten.select, aten.index_select]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_index_select_select_3.run(arg1_1, buf11, buf12, 200000, 64, stream=raw_stream0)
            del buf11
            # Topologically Sorted Source Nodes: [edge_index_i_1, view_2, index_3, index_2, x_j_1, out_4], Original ATen: [aten.select, aten.view, aten.expand, aten.index_select, aten.scatter_add]
            # [Provenance debug handles] torch.ops.aten.scatter_add.default:8
            buf13 = torch.ops.aten.scatter_add.default(buf0, 0, reinterpret_tensor(arg1_1, (200000, 64), (1, 0), 200000), buf12)
            del buf12
            buf14 = buf13
            assert_size_stride(buf14, (10000, 64), (64, 1), 'torch.ops.aten.scatter_add.default')
            assert_alignment(buf14, 16, 'torch.ops.aten.scatter_add.default')
            del buf13
            buf17 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [count_3, view_3, expand_as_3, out_5], Original ATen: [aten.clamp, aten.view, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clamp_div_expand_view_4.run(buf17, buf16, 10000, 64, stream=raw_stream0)
            del buf16
            buf18 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [count_3, view_3, expand_as_3, out_5, out_6, ], Original ATen: [aten.clamp, aten.view, aten.expand, aten.div, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:9
            extern_kernels.mm(buf17, reinterpret_tensor(arg5_1, (64, 64), (1, 64), 0), out=buf18)
            del arg5_1
            buf28 = buf17; del buf17  # reuse
            # Topologically Sorted Source Nodes: [, out_7, x_2], Original ATen: [aten.addmm, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_relu_6.run(arg6_1, buf18, buf19, buf28, 10000, 64, stream=raw_stream0)
            buf20 = empty_strided_npu((200000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, out_7, x_2, index_4, x_j_2], Original ATen: [aten.addmm, aten.add, aten.relu, aten.select, aten.index_select]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_index_select_relu_s_7.run(arg1_1, arg6_1, buf18, buf19, buf20, 200000, 64, stream=raw_stream0)
            del arg6_1
            del buf18
            del buf19
            # Topologically Sorted Source Nodes: [edge_index_i_2, view_4, index_5, , out_7, x_2, index_4, x_j_2, out_8], Original ATen: [aten.select, aten.view, aten.expand, aten.addmm, aten.add, aten.relu, aten.index_select, aten.scatter_add]
            # [Provenance debug handles] torch.ops.aten.scatter_add.default:10
            buf21 = torch.ops.aten.scatter_add.default(buf0, 0, reinterpret_tensor(arg1_1, (200000, 64), (1, 0), 200000), buf20)
            del arg1_1
            del buf0
            del buf20
            buf22 = buf21
            assert_size_stride(buf22, (10000, 64), (64, 1), 'torch.ops.aten.scatter_add.default')
            assert_alignment(buf22, 16, 'torch.ops.aten.scatter_add.default')
            del buf21
            buf25 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [count_5, view_5, expand_as_5, out_9], Original ATen: [aten.clamp, aten.view, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clamp_div_expand_view_4.run(buf25, buf24, 10000, 64, stream=raw_stream0)
            del buf24
            # Topologically Sorted Source Nodes: [count_5, view_5, expand_as_5, out_9, out_10], Original ATen: [aten.clamp, aten.view, aten.expand, aten.div, aten.t, aten.addmm]
            # [Provenance debug handles] torch.ops.aten.addmm.default:11
            buf26 = torch.ops.aten.addmm.default(arg9_1, buf25, reinterpret_tensor(arg8_1, (64, 64), (1, 64), 0), alpha=1, beta=1)
            del arg8_1
            del arg9_1
            del buf25
            buf27 = buf26
            assert_size_stride(buf27, (10000, 64), (64, 1), 'torch.ops.aten.addmm.default')
            assert_alignment(buf27, 16, 'torch.ops.aten.addmm.default')
            del buf26
            # Topologically Sorted Source Nodes: [, out_7, x_2, linear_5], Original ATen: [aten.addmm, aten.add, aten.relu, aten.t]
            # [Provenance debug handles] torch.ops.aten.addmm.default:12
            buf29 = torch.ops.aten.addmm.default(buf27, buf28, reinterpret_tensor(arg10_1, (64, 64), (1, 64), 0), alpha=1, beta=1)
            del arg10_1
            del buf27
            del buf28
            buf30 = buf29
            assert_size_stride(buf30, (10000, 64), (64, 1), 'torch.ops.aten.addmm.default')
            assert_alignment(buf30, 16, 'torch.ops.aten.addmm.default')
            del buf29
        return (buf30, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((10000, 64), (64, 1), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((2, 200000), (200000, 1), device='npu:0', dtype=torch.int64)
    arg2_1 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('basic_gnn_sage', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
