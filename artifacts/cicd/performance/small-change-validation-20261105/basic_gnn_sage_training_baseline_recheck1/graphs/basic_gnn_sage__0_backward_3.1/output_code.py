# AOT ID: ['0_backward']
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
# Topologically Sorted Source Nodes: [sum_1], Original ATen: [aten.sum]
# Source node to ATen node mapping:
#   sum_1 => sum_1
# Graph fragment:
#   %tangents_1 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=tangents_1]
#   %sum_1 : Tensor "f32[1, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%tangents_1, [0], True), kwargs = {})
#   return %sum_1
# SchedulerNodes: [SchedulerNode(name='op4')]

triton_red_fused_sum_0 = async_compile.triton('triton_red_fused_sum_0', '''
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

@triton_heuristics.reduction(
    size_hints={'x0': 64, 'r1': 10000},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_0', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 64), ('r1', 10000)), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_red_fused_sum_0(in_ptr0, out_ptr0, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr, R1BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        _tmp2 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        for loop_r1 in range(loops_r1):
            r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[:,None]
            r1_mask = r1 < r1_numel
            tmp0 = tl.load(in_ptr0 + (x0 + 64*r1), r1_mask & x0_mask, other=0.0)
            tmp1 = tl.reshape(tmp0, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp3 = _tmp2 + tmp1
            _tmp2 = tl.where(tl.reshape((r1_mask & x0_mask), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp3, _tmp2)
        tmp2 = tl.sum(_tmp2, 0).reshape(1, X0BLOCK_SUB)
        tl.store(out_ptr0 + (x0 ), tmp2, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [count_1, view_1, expand_as_1, div_3, edge_index_i, view, index_1, gather, index, index_put, div_4, gather_1, index_put_1], Original ATen: [aten.clamp, aten.view, aten.expand, aten.div, aten.select, aten.gather, aten.index_add]
# Source node to ATen node mapping:
#   count_1 => clamp_min
#   div_3 => div_3
#   div_4 => div_4
#   edge_index_i => select_1
#   expand_as_1 => expand_1
#   gather => gather
#   gather_1 => gather_1
#   index => select
#   index_1 => expand
#   index_put => index_put
#   index_put_1 => index_put_1
#   view => view
#   view_1 => view_1
# Graph fragment:
#   %full_default_2 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=full_default_2]
#   %clamp_min : Tensor "f32[10000][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%scatter_add, 1), kwargs = {})
#   %view_1 : Tensor "f32[10000, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clamp_min, [-1, 1]), kwargs = {})
#   %expand_1 : Tensor "f32[10000, 64][1, 0]npu:0"[num_users=2] = call_function[target=torch.ops.aten.expand.default](args = (%view_1, [10000, 64]), kwargs = {})
#   %div_3 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_5, %expand_1), kwargs = {})
#   %select_1 : Tensor "i64[200000][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_2, 0, 1), kwargs = {})
#   %view : Tensor "i64[200000, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [-1, 1]), kwargs = {})
#   %expand : Tensor "i64[200000, 64][1, 0]npu:0"[num_users=2] = call_function[target=torch.ops.aten.expand.default](args = (%view, [200000, 64]), kwargs = {})
#   %gather : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%div_3, 0, %expand), kwargs = {})
#   %select : Tensor "i64[200000][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_2, 0, 0), kwargs = {})
#   %index_put : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default_2, [%select], %gather, True), kwargs = {})
#   %div_4 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_9, %expand_1), kwargs = {})
#   %gather_1 : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%div_4, 0, %expand), kwargs = {})
#   %index_put_1 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default_2, [%select], %gather_1, True), kwargs = {})
#   return %index_put,%index_put_1
# SchedulerNodes: [SchedulerNode(name='op5'), SchedulerNode(name='op13')]

triton_poi_fused_clamp_div_expand_gather_index_1 = async_compile.triton('triton_poi_fused_clamp_div_expand_gather_index_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_div_expand_gather_index_1', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 10000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_div_expand_gather_index_1(in_ptr0, out_ptr0, out_ptr1, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tl.store(out_ptr0 + (x1 + 64*y0), tmp0, y0_mask)
        tl.store(out_ptr1 + (x1 + 64*y0), tmp0, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [count_1, view_1, expand_as_1, div_3, edge_index_i, view, index_1, gather, index, index_put], Original ATen: [aten.clamp, aten.view, aten.expand, aten.div, aten.select, aten.gather, aten.index_add]
# Source node to ATen node mapping:
#   count_1 => clamp_min
#   div_3 => div_3
#   edge_index_i => select_1
#   expand_as_1 => expand_1
#   gather => gather
#   index => select
#   index_1 => expand
#   index_put => index_put
#   view => view
#   view_1 => view_1
# Graph fragment:
#   %primals_2 : Tensor "i64[2, 200000][200000, 1]npu:0" = PlaceHolder[target=primals_2]
#   %mm_5 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=mm_5]
#   %scatter_add : Tensor "f32[10000][1]npu:0" = PlaceHolder[target=scatter_add]
#   %index_put : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=index_put]
#   %clamp_min : Tensor "f32[10000][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%scatter_add, 1), kwargs = {})
#   %view_1 : Tensor "f32[10000, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clamp_min, [-1, 1]), kwargs = {})
#   %expand_1 : Tensor "f32[10000, 64][1, 0]npu:0"[num_users=2] = call_function[target=torch.ops.aten.expand.default](args = (%view_1, [10000, 64]), kwargs = {})
#   %div_3 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_5, %expand_1), kwargs = {})
#   %select_1 : Tensor "i64[200000][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_2, 0, 1), kwargs = {})
#   %view : Tensor "i64[200000, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [-1, 1]), kwargs = {})
#   %expand : Tensor "i64[200000, 64][1, 0]npu:0"[num_users=2] = call_function[target=torch.ops.aten.expand.default](args = (%view, [200000, 64]), kwargs = {})
#   %gather : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%div_3, 0, %expand), kwargs = {})
#   %select : Tensor "i64[200000][1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_2, 0, 0), kwargs = {})
#   %index_put : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default_2, [%select], %gather, True), kwargs = {})
#   return %buf6
# SchedulerNodes: [SchedulerNode(name='op6')]

triton_poi_fused_clamp_div_expand_gather_index_2 = async_compile.triton('triton_poi_fused_clamp_div_expand_gather_index_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_div_expand_gather_index_2', 'mutated_arg_names': ['out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 200000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_div_expand_gather_index_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp6 = tl.load(in_ptr0 + (200000 + y0), y0_mask)
        tmp1 = tl.full([Y0BLOCK_SUB, X1BLOCK_SUB], 10000, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 10000)) | ~(y0_mask), "index out of bounds: 0 <= tmp4 < 10000")
        tl.device_assert(((0 <= tmp6) & (tmp6 < 10000)) | ~(y0_mask), "index out of bounds: 0 <= tmp6 < 10000")
        tmp8 = tl.load(in_ptr1 + (x1 + 64*tmp6), x1_mask & y0_mask)
        tmp9 = tl.load(in_ptr2 + (tmp6), y0_mask)
        tmp10 = tl.full([1, 1], 1.0, tl.float32)
        tmp11 = triton_helpers.maximum(tmp9, tmp10)
        tmp12 = (tmp8 / tmp11)
        tl.atomic_add(out_ptr0 + (x1 + 64*tmp4), tmp12, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [add_3, le, scalar_tensor, where, sum_2], Original ATen: [aten.add, aten.threshold_backward, aten.sum]
# Source node to ATen node mapping:
#   add_3 => add_3
#   le => le
#   scalar_tensor => full_default_10
#   sum_2 => sum_2
#   where => where
# Graph fragment:
#   %relu_1 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=relu_1]
#   %mm_3 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=mm_3]
#   %buf6 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=buf6]
#   %where : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=where]
#   %add_3 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_3, %index_put), kwargs = {})
#   %le : Tensor "b8[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_1, 0), kwargs = {})
#   %full_default_10 : Tensor "f32[][]npu:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=5] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default_10, %add_3), kwargs = {})
#   %sum_2 : Tensor "f32[1, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%where, [0], True), kwargs = {})
#   return %where,%sum_2
# SchedulerNodes: [SchedulerNode(name='op7'), SchedulerNode(name='op12')]

triton_red_fused_add_sum_threshold_backward_3 = async_compile.triton('triton_red_fused_add_sum_threshold_backward_3', '''
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

@triton_heuristics.reduction(
    size_hints={'x0': 64, 'r1': 10000},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_sum_threshold_backward_3', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 64), ('r1', 10000)), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_red_fused_add_sum_threshold_backward_3(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr, R1BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        _tmp8 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        for loop_r1 in range(loops_r1):
            r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[:,None]
            r1_mask = r1 < r1_numel
            tmp0 = tl.load(in_out_ptr0 + (x0 + 64*r1), r1_mask & x0_mask, other=0.0)
            tmp3 = tl.load(in_ptr0 + (x0 + 64*r1), r1_mask & x0_mask, other=0.0)
            tmp4 = tl.load(in_ptr1 + (x0 + 64*r1), r1_mask & x0_mask, other=0.0)
            tmp1 = tl.full([1, 1], 0.0, tl.float32)
            tmp2 = tmp0 <= tmp1
            tmp5 = tmp3 + tmp4
            tmp6 = tl.where(tmp2, tmp1, tmp5)
            tmp7 = tl.reshape(tmp6, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp9 = _tmp8 + tmp7
            _tmp8 = tl.where(tl.reshape((r1_mask & x0_mask), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp9, _tmp8)
            tl.store(in_out_ptr0 + (x0 + 64*r1), tmp6, r1_mask & x0_mask)
        tmp8 = tl.sum(_tmp8, 0).reshape(1, X0BLOCK_SUB)
        tl.store(out_ptr0 + (x0 ), tmp8, x0_mask)
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
        primals_2, primals_6, primals_8, primals_9, primals_11, scatter_add, full_default_2, div, relu, div_1, relu_1, div_2, permute_20, tangents_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            tangents_1 = copy_if_misaligned(tangents_1)
            buf0 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_5, permute_6, mm_3], Original ATen: [aten.t, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.mm:1
            extern_kernels.mm(tangents_1, primals_11, out=buf0)
            del primals_11
            buf1 = empty_strided_npu((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_7, mm_4], Original ATen: [aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.mm:2
            extern_kernels.mm(reinterpret_tensor(relu_1, (64, 10000), (1, 64), 0), tangents_1, out=buf1)
            buf2 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [out_10, permute_9, mm_5], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:3
            extern_kernels.mm(tangents_1, primals_9, out=buf2)
            del primals_9
            buf3 = empty_strided_npu((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_10, permute_12], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:4
            extern_kernels.mm(reinterpret_tensor(tangents_1, (64, 10000), (1, 64), 0), div_2, out=buf3)
            del div_2
            buf4 = empty_strided_npu((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sum_1], Original ATen: [aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_sum_0.run(tangents_1, buf4, 64, 10000, stream=raw_stream0)
            del tangents_1
            buf5 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            buf13 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [count_1, view_1, expand_as_1, div_3, edge_index_i, view, index_1, gather, index, index_put, div_4, gather_1, index_put_1], Original ATen: [aten.clamp, aten.view, aten.expand, aten.div, aten.select, aten.gather, aten.index_add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clamp_div_expand_gather_index_1.run(full_default_2, buf5, buf13, 10000, 64, stream=raw_stream0)
            del full_default_2
            # Topologically Sorted Source Nodes: [count_1, view_1, expand_as_1, div_3, edge_index_i, view, index_1, gather, index, index_put], Original ATen: [aten.clamp, aten.view, aten.expand, aten.div, aten.select, aten.gather, aten.index_add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clamp_div_expand_gather_index_2.run(primals_2, buf2, scatter_add, buf5, 200000, 64, stream=raw_stream0)
            del buf2
            buf7 = relu_1; del relu_1  # reuse
            buf12 = empty_strided_npu((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [add_3, le, scalar_tensor, where, sum_2], Original ATen: [aten.add, aten.threshold_backward, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_add_sum_threshold_backward_3.run(buf7, buf0, buf5, buf12, 64, 10000, stream=raw_stream0)
            buf8 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [linear_3, permute_13, mm_7], Original ATen: [aten.t, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.mm:5
            extern_kernels.mm(buf7, primals_8, out=buf8)
            del primals_8
            buf9 = empty_strided_npu((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_14, mm_8], Original ATen: [aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.mm:6
            extern_kernels.mm(reinterpret_tensor(relu, (64, 10000), (1, 64), 0), buf7, out=buf9)
            buf10 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [out_6, permute_16, mm_9], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:7
            extern_kernels.mm(buf7, primals_6, out=buf10)
            del primals_6
            buf11 = empty_strided_npu((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_17, permute_19], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:8
            extern_kernels.mm(reinterpret_tensor(buf7, (64, 10000), (1, 64), 0), div_1, out=buf11)
            del buf7
            del div_1
            # Topologically Sorted Source Nodes: [count_1, view_1, expand_as_1, edge_index_i, view, index_1, index, div_4, gather_1, index_put_1], Original ATen: [aten.clamp, aten.view, aten.expand, aten.select, aten.div, aten.gather, aten.index_add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clamp_div_expand_gather_index_2.run(primals_2, buf10, scatter_add, buf13, 200000, 64, stream=raw_stream0)
            del buf10
            del primals_2
            del scatter_add
            buf15 = relu; del relu  # reuse
            buf18 = empty_strided_npu((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [scalar_tensor, add_4, le_1, where_1, sum_3], Original ATen: [aten.threshold_backward, aten.add, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_add_sum_threshold_backward_3.run(buf15, buf8, buf13, buf18, 64, 10000, stream=raw_stream0)
            del buf13
            del buf8
            buf16 = empty_strided_npu((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mm_11], Original ATen: [aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.mm:9
            extern_kernels.mm(permute_20, buf15, out=buf16)
            del permute_20
            buf17 = empty_strided_npu((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_22, permute_24], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:10
            extern_kernels.mm(reinterpret_tensor(buf15, (64, 10000), (1, 64), 0), div, out=buf17)
            del buf15
            del div
        return (None, None, buf17, reinterpret_tensor(buf18, (64, ), (1, ), 0), reinterpret_tensor(buf16, (64, 64), (1, 64), 0), buf11, reinterpret_tensor(buf12, (64, ), (1, ), 0), reinterpret_tensor(buf9, (64, 64), (1, 64), 0), buf3, reinterpret_tensor(buf4, (64, ), (1, ), 0), reinterpret_tensor(buf1, (64, 64), (1, 64), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_2 = rand_strided((2, 200000), (200000, 1), device='npu:0', dtype=torch.int64)
    primals_6 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    primals_9 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    primals_11 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    scatter_add = rand_strided((10000, ), (1, ), device='npu:0', dtype=torch.float32)
    full_default_2 = rand_strided((10000, 64), (64, 1), device='npu:0', dtype=torch.float32)
    div = rand_strided((10000, 64), (64, 1), device='npu:0', dtype=torch.float32)
    relu = rand_strided((10000, 64), (64, 1), device='npu:0', dtype=torch.float32)
    div_1 = rand_strided((10000, 64), (64, 1), device='npu:0', dtype=torch.float32)
    relu_1 = rand_strided((10000, 64), (64, 1), device='npu:0', dtype=torch.float32)
    div_2 = rand_strided((10000, 64), (64, 1), device='npu:0', dtype=torch.float32)
    permute_20 = rand_strided((64, 10000), (1, 64), device='npu:0', dtype=torch.float32)
    tangents_1 = rand_strided((10000, 64), (64, 1), device='npu:0', dtype=torch.float32)
    return [primals_2, primals_6, primals_8, primals_9, primals_11, scatter_add, full_default_2, div, relu, div_1, relu_1, div_2, permute_20, tangents_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('basic_gnn_sage', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
