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
# Topologically Sorted Source Nodes: [index, x_j, index_1, x_i, sub], Original ATen: [aten.select, aten.index_select, aten.sub]
# Source node to ATen node mapping:
#   index => select
#   index_1 => select_1
#   sub => sub
#   x_i => index_1
#   x_j => index
# Graph fragment:
#   %primals_2 : Tensor "i64[2, 200000][200000, 1]npu:0" = PlaceHolder[target=primals_2]
#   %primals_1 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=primals_1]
#   %select : Tensor "i64[200000][1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.select.int](args = (%primals_2, 0, 0), kwargs = {})
#   %index : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%primals_1, [%select]), kwargs = {})
#   %select_1 : Tensor "i64[200000][1]npu:0"[num_users=5] = call_function[target=torch.ops.aten.select.int](args = (%primals_2, 0, 1), kwargs = {})
#   %index_1 : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%primals_1, [%select_1]), kwargs = {})
#   %sub : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%index, %index_1), kwargs = {})
#   return %index_1,%sub
# SchedulerNodes: [SchedulerNode(name='op0'), SchedulerNode(name='op1')]

triton_poi_fused_index_select_select_sub_0 = async_compile.triton('triton_poi_fused_index_select_select_sub_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_select_sub_0', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 200000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_select_select_sub_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp7 = tl.load(in_ptr0 + (y0), y0_mask)
        tmp1 = tl.full([Y0BLOCK_SUB, X1BLOCK_SUB], 10000, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 10000)) | ~(y0_mask), "index out of bounds: 0 <= tmp4 < 10000")
        tmp6 = tl.load(in_ptr1 + (x1 + 64*tmp4), x1_mask & y0_mask)
        tmp8 = tmp7 + tmp1
        tmp9 = tmp7 < 0
        tmp10 = tl.where(tmp9, tmp8, tmp7)
        tl.device_assert(((0 <= tmp10) & (tmp10 < 10000)) | ~(y0_mask), "index out of bounds: 0 <= tmp10 < 10000")
        tmp12 = tl.load(in_ptr1 + (x1 + 64*tmp10), x1_mask & y0_mask)
        tmp13 = tmp12 - tmp6
        tl.store(out_ptr0 + (x1 + 128*y0), tmp6, x1_mask & y0_mask)
        tl.store(out_ptr1 + (x1 + 128*y0), tmp13, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [new_zeros], Original ATen: [aten.new_zeros]
# Source node to ATen node mapping:
#   new_zeros => full_default
# Graph fragment:
#   %full_default : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([10000, 64], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   return %full_default
# SchedulerNodes: [SchedulerNode(name='op7')]

triton_poi_fused_new_zeros_1 = async_compile.triton('triton_poi_fused_new_zeros_1', '''
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
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_1', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 10000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_zeros_1(out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
# Topologically Sorted Source Nodes: [, x_1], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#    => add_tensor_2
#   x_1 => relu
# Graph fragment:
#   %primals_4 : Tensor "f32[64][1]npu:0" = PlaceHolder[target=primals_4]
#   %mm_default_2 : Tensor "f32[200000, 64][64, 1]npu:0" = PlaceHolder[target=mm_default_2]
#   %add_tensor_2 : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_4, %mm_default_2), kwargs = {})
#   %relu : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_2,), kwargs = {})
#   return %relu
# SchedulerNodes: [SchedulerNode(name='op4')]

triton_poi_fused_addmm_relu_2 = async_compile.triton('triton_poi_fused_addmm_relu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 200000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_2(in_out_ptr0, in_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1, 1], 0, tl.int32)
        tmp4 = triton_helpers.maximum(tmp3, tmp2)
        tl.store(in_out_ptr0 + (x1 + 64*y0), tmp4, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [index, index_1, x_5, x_j_1, x_i_1, sub_1], Original ATen: [aten.select, aten.relu, aten.index_select, aten.sub]
# Source node to ATen node mapping:
#   index => select
#   index_1 => select_1
#   sub_1 => sub_1
#   x_5 => relu_1
#   x_i_1 => index_3
#   x_j_1 => index_2
# Graph fragment:
#   %primals_2 : Tensor "i64[2, 200000][200000, 1]npu:0" = PlaceHolder[target=primals_2]
#   %scatter_reduce : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=scatter_reduce]
#   %select : Tensor "i64[200000][1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.select.int](args = (%primals_2, 0, 0), kwargs = {})
#   %select_1 : Tensor "i64[200000][1]npu:0"[num_users=5] = call_function[target=torch.ops.aten.select.int](args = (%primals_2, 0, 1), kwargs = {})
#   %relu_1 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%scatter_reduce,), kwargs = {})
#   %index_2 : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%relu_1, [%select]), kwargs = {})
#   %index_3 : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%relu_1, [%select_1]), kwargs = {})
#   %sub_1 : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%index_2, %index_3), kwargs = {})
#   return %index_3,%sub_1
# SchedulerNodes: [SchedulerNode(name='op10'), SchedulerNode(name='op11')]

triton_poi_fused_index_select_relu_select_sub_3 = async_compile.triton('triton_poi_fused_index_select_relu_select_sub_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_relu_select_sub_3', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 200000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_select_relu_select_sub_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp9 = tl.load(in_ptr0 + (y0), y0_mask)
        tmp1 = tl.full([Y0BLOCK_SUB, X1BLOCK_SUB], 10000, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 10000)) | ~(y0_mask), "index out of bounds: 0 <= tmp4 < 10000")
        tmp6 = tl.load(in_ptr1 + (x1 + 64*tmp4), x1_mask & y0_mask)
        tmp7 = tl.full([1, 1], 0, tl.int32)
        tmp8 = triton_helpers.maximum(tmp7, tmp6)
        tmp10 = tmp9 + tmp1
        tmp11 = tmp9 < 0
        tmp12 = tl.where(tmp11, tmp10, tmp9)
        tl.device_assert(((0 <= tmp12) & (tmp12 < 10000)) | ~(y0_mask), "index out of bounds: 0 <= tmp12 < 10000")
        tmp14 = tl.load(in_ptr1 + (x1 + 64*tmp12), x1_mask & y0_mask)
        tmp15 = triton_helpers.maximum(tmp7, tmp14)
        tmp16 = tmp15 - tmp8
        tl.store(out_ptr0 + (x1 + 128*y0), tmp8, x1_mask & y0_mask)
        tl.store(out_ptr1 + (x1 + 128*y0), tmp16, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [x_5, le_3, eq_8], Original ATen: [aten.relu, aten.threshold_backward, aten.eq]
# Source node to ATen node mapping:
#   eq_8 => eq_8
#   le_3 => le_3
#   x_5 => relu_1
# Graph fragment:
#   %scatter_reduce : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=scatter_reduce]
#   %relu_1 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%scatter_reduce,), kwargs = {})
#   %le_3 : Tensor "b8[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_1, 0), kwargs = {})
#   %eq_8 : Tensor "b8[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%full_default, %scatter_reduce), kwargs = {})
#   return %le_3,%eq_8
# SchedulerNodes: [SchedulerNode(name='op33'), SchedulerNode(name='op34')]

triton_poi_fused_eq_relu_threshold_backward_4 = async_compile.triton('triton_poi_fused_eq_relu_threshold_backward_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_eq_relu_threshold_backward_4', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 10000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.bool, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_eq_relu_threshold_backward_4(in_ptr0, out_ptr0, out_ptr1, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = triton_helpers.maximum(tmp1, tmp0)
        tmp3 = tl.full([1, 1], 0.0, tl.float32)
        tmp4 = tmp2 <= tmp3
        tmp5 = tmp3 == tmp0
        tl.store(out_ptr0 + (x1 + 64*y0), tmp4, y0_mask)
        tl.store(out_ptr1 + (x1 + 64*y0), tmp5, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [index, index_1, view, index_2, x_12, x_j_2, x_i_2, sub_2, gather_2, eq_5, gather_4, eq_9], Original ATen: [aten.select, aten.view, aten.expand, aten.relu, aten.index_select, aten.sub, aten.gather, aten.eq]
# Source node to ATen node mapping:
#   eq_5 => eq_5
#   eq_9 => eq_9
#   gather_2 => gather_2
#   gather_4 => gather_4
#   index => select
#   index_1 => select_1
#   index_2 => expand
#   sub_2 => sub_2
#   view => view
#   x_12 => relu_3
#   x_i_2 => index_5
#   x_j_2 => index_4
# Graph fragment:
#   %primals_2 : Tensor "i64[2, 200000][200000, 1]npu:0" = PlaceHolder[target=primals_2]
#   %scatter_reduce_1 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=scatter_reduce_1]
#   %addmm_3 : Tensor "f32[200000, 64][64, 1]npu:0" = PlaceHolder[target=addmm_3]
#   %addmm_1 : Tensor "f32[200000, 64][64, 1]npu:0" = PlaceHolder[target=addmm_1]
#   %scatter_reduce : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=scatter_reduce]
#   %select : Tensor "i64[200000][1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.select.int](args = (%primals_2, 0, 0), kwargs = {})
#   %select_1 : Tensor "i64[200000][1]npu:0"[num_users=5] = call_function[target=torch.ops.aten.select.int](args = (%primals_2, 0, 1), kwargs = {})
#   %view : Tensor "i64[200000, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [-1, 1]), kwargs = {})
#   %expand : Tensor "i64[200000, 64][1, 0]npu:0"[num_users=6] = call_function[target=torch.ops.aten.expand.default](args = (%view, [200000, 64]), kwargs = {})
#   %relu_3 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%scatter_reduce_1,), kwargs = {})
#   %index_4 : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%relu_3, [%select]), kwargs = {})
#   %index_5 : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%relu_3, [%select_1]), kwargs = {})
#   %sub_2 : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%index_4, %index_5), kwargs = {})
#   %gather_2 : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%scatter_reduce_1, 0, %expand), kwargs = {})
#   %eq_5 : Tensor "b8[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%addmm_3, %gather_2), kwargs = {})
#   %gather_4 : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%scatter_reduce, 0, %expand), kwargs = {})
#   %eq_9 : Tensor "b8[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%addmm_1, %gather_4), kwargs = {})
#   return %index_5,%sub_2,%eq_5,%eq_9
# SchedulerNodes: [SchedulerNode(name='op19'), SchedulerNode(name='op20'), SchedulerNode(name='op32'), SchedulerNode(name='op35')]

triton_poi_fused_eq_expand_gather_index_select_5 = async_compile.triton('triton_poi_fused_eq_expand_gather_index_select_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*i1', 'out_ptr3': '*i1', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_eq_expand_gather_index_select_5', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 200000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_eq_expand_gather_index_select_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp9 = tl.load(in_ptr0 + (y0), y0_mask)
        tmp17 = tl.load(in_ptr2 + (x1 + 64*y0), x1_mask & y0_mask)
        tmp21 = tl.load(in_ptr3 + (x1 + 64*y0), x1_mask & y0_mask)
        tmp1 = tl.full([Y0BLOCK_SUB, X1BLOCK_SUB], 10000, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 10000)) | ~(y0_mask), "index out of bounds: 0 <= tmp4 < 10000")
        tmp6 = tl.load(in_ptr1 + (x1 + 64*tmp4), x1_mask & y0_mask)
        tmp7 = tl.full([1, 1], 0, tl.int32)
        tmp8 = triton_helpers.maximum(tmp7, tmp6)
        tmp10 = tmp9 + tmp1
        tmp11 = tmp9 < 0
        tmp12 = tl.where(tmp11, tmp10, tmp9)
        tl.device_assert(((0 <= tmp12) & (tmp12 < 10000)) | ~(y0_mask), "index out of bounds: 0 <= tmp12 < 10000")
        tmp14 = tl.load(in_ptr1 + (x1 + 64*tmp12), x1_mask & y0_mask)
        tmp15 = triton_helpers.maximum(tmp7, tmp14)
        tmp16 = tmp15 - tmp8
        tl.device_assert(((0 <= tmp0) & (tmp0 < 10000)) | ~(y0_mask), "index out of bounds: 0 <= tmp0 < 10000")
        tmp19 = tl.load(in_ptr1 + (x1 + 64*tmp0), x1_mask & y0_mask)
        tmp20 = tmp17 == tmp19
        tmp22 = tl.load(in_ptr4 + (x1 + 64*tmp0), x1_mask & y0_mask)
        tmp23 = tmp21 == tmp22
        tl.store(out_ptr0 + (x1 + 128*y0), tmp8, x1_mask & y0_mask)
        tl.store(out_ptr1 + (x1 + 128*y0), tmp16, x1_mask & y0_mask)
        tl.store(out_ptr2 + (x1 + 64*y0), tmp20, x1_mask & y0_mask)
        tl.store(out_ptr3 + (x1 + 64*y0), tmp23, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [index_1, view, index_2, gather, eq_1], Original ATen: [aten.select, aten.view, aten.expand, aten.gather, aten.eq]
# Source node to ATen node mapping:
#   eq_1 => eq_1
#   gather => gather
#   index_1 => select_1
#   index_2 => expand
#   view => view
# Graph fragment:
#   %addmm_5 : Tensor "f32[200000, 64][64, 1]npu:0" = PlaceHolder[target=addmm_5]
#   %primals_2 : Tensor "i64[2, 200000][200000, 1]npu:0" = PlaceHolder[target=primals_2]
#   %scatter_reduce_2 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=scatter_reduce_2]
#   %select_1 : Tensor "i64[200000][1]npu:0"[num_users=5] = call_function[target=torch.ops.aten.select.int](args = (%primals_2, 0, 1), kwargs = {})
#   %view : Tensor "i64[200000, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [-1, 1]), kwargs = {})
#   %expand : Tensor "i64[200000, 64][1, 0]npu:0"[num_users=6] = call_function[target=torch.ops.aten.expand.default](args = (%view, [200000, 64]), kwargs = {})
#   %gather : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%scatter_reduce_2, 0, %expand), kwargs = {})
#   %eq_1 : Tensor "b8[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%addmm_5, %gather), kwargs = {})
#   return %eq_1
# SchedulerNodes: [SchedulerNode(name='op29')]

triton_poi_fused_eq_expand_gather_select_view_6 = async_compile.triton('triton_poi_fused_eq_expand_gather_select_view_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*i1', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_eq_expand_gather_select_view_6', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 200000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.bool, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_eq_expand_gather_select_view_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x1 + 64*y0), x1_mask & y0_mask)
        tmp1 = tl.load(in_ptr1 + (200000 + y0), y0_mask)
        tl.device_assert(((0 <= tmp1) & (tmp1 < 10000)) | ~(y0_mask), "index out of bounds: 0 <= tmp1 < 10000")
        tmp3 = tl.load(in_ptr2 + (x1 + 64*tmp1), x1_mask & y0_mask)
        tmp4 = tmp0 == tmp3
        tl.store(out_ptr0 + (x1 + 64*y0), tmp4, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [eq], Original ATen: [aten.eq]
# Source node to ATen node mapping:
#   eq => eq
# Graph fragment:
#   %scatter_reduce_2 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=scatter_reduce_2]
#   %eq : Tensor "b8[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%full_default, %scatter_reduce_2), kwargs = {})
#   return %eq
# SchedulerNodes: [SchedulerNode(name='op28')]

triton_poi_fused_eq_7 = async_compile.triton('triton_poi_fused_eq_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_eq_7', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 10000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.bool, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_eq_7(in_ptr0, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp1 = tl.full([1, 1], 0.0, tl.float32)
        tmp2 = tmp1 == tmp0
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            primals_2 = copy_if_misaligned(primals_2)
            primals_1 = copy_if_misaligned(primals_1)
            buf2 = empty_strided_npu((200000, 128), (128, 1), torch.float32)
            buf0 = reinterpret_tensor(buf2, (200000, 64), (128, 1), 0)  # alias
            buf1 = reinterpret_tensor(buf2, (200000, 64), (128, 1), 64)  # alias
            # Topologically Sorted Source Nodes: [index, x_j, index_1, x_i, sub], Original ATen: [aten.select, aten.index_select, aten.sub]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_index_select_select_sub_0.run(primals_2, primals_1, buf0, buf1, 200000, 64, stream=raw_stream0)
            del primals_1
            buf7 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [new_zeros], Original ATen: [aten.new_zeros]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_new_zeros_1.run(buf7, 10000, 64, stream=raw_stream0)
            buf3 = empty_strided_npu((200000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, ], Original ATen: [aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:1
            extern_kernels.mm(buf2, reinterpret_tensor(primals_3, (128, 64), (1, 128), 0), out=buf3)
            del primals_3
            buf4 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [, x_1], Original ATen: [aten.addmm, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_2.run(buf4, primals_4, 200000, 64, stream=raw_stream0)
            del primals_4
            # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.t, aten.addmm]
            # [Provenance debug handles] torch.ops.aten.addmm.default:2
            buf5 = torch.ops.aten.addmm.default(primals_6, buf4, reinterpret_tensor(primals_5, (64, 64), (1, 64), 0), alpha=1, beta=1)
            del primals_6
            buf6 = buf5
            assert_size_stride(buf6, (200000, 64), (64, 1), 'torch.ops.aten.addmm.default')
            assert_alignment(buf6, 16, 'torch.ops.aten.addmm.default')
            del buf5
            # Topologically Sorted Source Nodes: [index_1, view, index_2, out], Original ATen: [aten.select, aten.view, aten.expand, aten.scatter_reduce]
            # [Provenance debug handles] torch.ops.aten.scatter_reduce.two:3
            buf8 = torch.ops.aten.scatter_reduce.two(buf7, 0, reinterpret_tensor(primals_2, (200000, 64), (1, 0), 200000), buf6, 'amax', include_self=False)
            buf9 = buf8
            assert_size_stride(buf9, (10000, 64), (64, 1), 'torch.ops.aten.scatter_reduce.two')
            assert_alignment(buf9, 16, 'torch.ops.aten.scatter_reduce.two')
            del buf8
            buf12 = empty_strided_npu((200000, 128), (128, 1), torch.float32)
            buf10 = reinterpret_tensor(buf12, (200000, 64), (128, 1), 0)  # alias
            buf11 = reinterpret_tensor(buf12, (200000, 64), (128, 1), 64)  # alias
            # Topologically Sorted Source Nodes: [index, index_1, x_5, x_j_1, x_i_1, sub_1], Original ATen: [aten.select, aten.relu, aten.index_select, aten.sub]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_index_select_relu_select_sub_3.run(primals_2, buf9, buf10, buf11, 200000, 64, stream=raw_stream0)
            buf33 = empty_strided_npu((10000, 64), (64, 1), torch.bool)
            buf34 = empty_strided_npu((10000, 64), (64, 1), torch.bool)
            # Topologically Sorted Source Nodes: [x_5, le_3, eq_8], Original ATen: [aten.relu, aten.threshold_backward, aten.eq]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_eq_relu_threshold_backward_4.run(buf9, buf33, buf34, 10000, 64, stream=raw_stream0)
            buf13 = empty_strided_npu((200000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_7, ], Original ATen: [aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:4
            extern_kernels.mm(buf12, reinterpret_tensor(primals_7, (128, 64), (1, 128), 0), out=buf13)
            buf14 = buf13; del buf13  # reuse
            # Topologically Sorted Source Nodes: [, x_8], Original ATen: [aten.addmm, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_2.run(buf14, primals_8, 200000, 64, stream=raw_stream0)
            del primals_8
            # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.t, aten.addmm]
            # [Provenance debug handles] torch.ops.aten.addmm.default:5
            buf15 = torch.ops.aten.addmm.default(primals_10, buf14, reinterpret_tensor(primals_9, (64, 64), (1, 64), 0), alpha=1, beta=1)
            del primals_10
            buf16 = buf15
            assert_size_stride(buf16, (200000, 64), (64, 1), 'torch.ops.aten.addmm.default')
            assert_alignment(buf16, 16, 'torch.ops.aten.addmm.default')
            del buf15
            # Topologically Sorted Source Nodes: [index_1, view, index_2, out_1], Original ATen: [aten.select, aten.view, aten.expand, aten.scatter_reduce]
            # [Provenance debug handles] torch.ops.aten.scatter_reduce.two:6
            buf17 = torch.ops.aten.scatter_reduce.two(buf7, 0, reinterpret_tensor(primals_2, (200000, 64), (1, 0), 200000), buf16, 'amax', include_self=False)
            buf18 = buf17
            assert_size_stride(buf18, (10000, 64), (64, 1), 'torch.ops.aten.scatter_reduce.two')
            assert_alignment(buf18, 16, 'torch.ops.aten.scatter_reduce.two')
            del buf17
            buf21 = empty_strided_npu((200000, 128), (128, 1), torch.float32)
            buf19 = reinterpret_tensor(buf21, (200000, 64), (128, 1), 0)  # alias
            buf20 = reinterpret_tensor(buf21, (200000, 64), (128, 1), 64)  # alias
            buf32 = empty_strided_npu((200000, 64), (64, 1), torch.bool)
            buf35 = empty_strided_npu((200000, 64), (64, 1), torch.bool)
            # Topologically Sorted Source Nodes: [index, index_1, view, index_2, x_12, x_j_2, x_i_2, sub_2, gather_2, eq_5, gather_4, eq_9], Original ATen: [aten.select, aten.view, aten.expand, aten.relu, aten.index_select, aten.sub, aten.gather, aten.eq]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_eq_expand_gather_index_select_5.run(primals_2, buf18, buf16, buf6, buf9, buf19, buf20, buf32, buf35, 200000, 64, stream=raw_stream0)
            del buf16
            del buf6
            del buf9
            buf30 = empty_strided_npu((10000, 64), (64, 1), torch.bool)
            buf31 = empty_strided_npu((10000, 64), (64, 1), torch.bool)
            # Topologically Sorted Source Nodes: [x_12, le_1, eq_4], Original ATen: [aten.relu, aten.threshold_backward, aten.eq]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_eq_relu_threshold_backward_4.run(buf18, buf30, buf31, 10000, 64, stream=raw_stream0)
            del buf18
            buf22 = empty_strided_npu((200000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_14, ], Original ATen: [aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:7
            extern_kernels.mm(buf21, reinterpret_tensor(primals_11, (128, 64), (1, 128), 0), out=buf22)
            buf23 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [, x_15], Original ATen: [aten.addmm, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_2.run(buf23, primals_12, 200000, 64, stream=raw_stream0)
            del primals_12
            # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten.t, aten.addmm]
            # [Provenance debug handles] torch.ops.aten.addmm.default:8
            buf24 = torch.ops.aten.addmm.default(primals_14, buf23, reinterpret_tensor(primals_13, (64, 64), (1, 64), 0), alpha=1, beta=1)
            del primals_14
            buf25 = buf24
            assert_size_stride(buf25, (200000, 64), (64, 1), 'torch.ops.aten.addmm.default')
            assert_alignment(buf25, 16, 'torch.ops.aten.addmm.default')
            del buf24
            # Topologically Sorted Source Nodes: [index_1, view, index_2, out_2], Original ATen: [aten.select, aten.view, aten.expand, aten.scatter_reduce]
            # [Provenance debug handles] torch.ops.aten.scatter_reduce.two:9
            buf26 = torch.ops.aten.scatter_reduce.two(buf7, 0, reinterpret_tensor(primals_2, (200000, 64), (1, 0), 200000), buf25, 'amax', include_self=False)
            buf27 = buf26
            assert_size_stride(buf27, (10000, 64), (64, 1), 'torch.ops.aten.scatter_reduce.two')
            assert_alignment(buf27, 16, 'torch.ops.aten.scatter_reduce.two')
            del buf26
            buf29 = empty_strided_npu((200000, 64), (64, 1), torch.bool)
            # Topologically Sorted Source Nodes: [index_1, view, index_2, gather, eq_1], Original ATen: [aten.select, aten.view, aten.expand, aten.gather, aten.eq]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_eq_expand_gather_select_view_6.run(buf25, primals_2, buf27, buf29, 200000, 64, stream=raw_stream0)
            del buf25
            buf28 = empty_strided_npu((10000, 64), (64, 1), torch.bool)
            # Topologically Sorted Source Nodes: [eq], Original ATen: [aten.eq]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_eq_7.run(buf27, buf28, 10000, 64, stream=raw_stream0)
        return (buf27, primals_5, primals_7, primals_9, primals_11, primals_13, reinterpret_tensor(primals_2, (200000, ), (1, ), 0), reinterpret_tensor(primals_2, (200000, ), (1, ), 200000), buf2, buf4, buf7, buf12, buf14, buf21, buf23, buf28, buf29, buf30, buf31, buf32, buf33, buf34, buf35, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((10000, 64), (64, 1), device='npu:0', dtype=torch.float32)
    primals_2 = rand_strided((2, 200000), (200000, 1), device='npu:0', dtype=torch.int64)
    primals_3 = rand_strided((64, 128), (128, 1), device='npu:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_5 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 128), (128, 1), device='npu:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_9 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_11 = rand_strided((64, 128), (128, 1), device='npu:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_13 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    return [primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('basic_gnn_edgecnn', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
