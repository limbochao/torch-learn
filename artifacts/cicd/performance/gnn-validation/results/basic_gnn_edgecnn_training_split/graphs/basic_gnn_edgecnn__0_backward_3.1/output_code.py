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
# Topologically Sorted Source Nodes: [convert_element_type], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   convert_element_type => convert_element_type
# Graph fragment:
#   %eq : Tensor "b8[10000, 64][64, 1]npu:0" = PlaceHolder[target=eq]
#   %convert_element_type : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%eq, torch.float32), kwargs = {})
#   return %convert_element_type
# SchedulerNodes: [SchedulerNode(name='op0')]

triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*i1', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 10000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp1 = tmp0.to(tl.float32)
        tl.store(out_ptr0 + (x1 + 64*y0), tmp1, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [convert_element_type_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   convert_element_type_1 => convert_element_type_1
# Graph fragment:
#   %eq_1 : Tensor "b8[200000, 64][64, 1]npu:0" = PlaceHolder[target=eq_1]
#   %convert_element_type_1 : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%eq_1, torch.float32), kwargs = {})
#   return %convert_element_type_1
# SchedulerNodes: [SchedulerNode(name='op1')]

triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*i1', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 200000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp1 = tmp0.to(tl.float32)
        tl.store(out_ptr0 + (x1 + 64*y0), tmp1, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [view, index_2, div, gather_1, mul_1, sum_1], Original ATen: [aten.view, aten.expand, aten.div, aten.gather, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   div => div
#   gather_1 => gather_1
#   index_2 => expand
#   mul_1 => mul_1
#   sum_1 => sum_1
#   view => view
# Graph fragment:
#   %eq_1 : Tensor "b8[200000, 64][64, 1]npu:0" = PlaceHolder[target=eq_1]
#   %select_1 : Tensor "i64[200000][1]npu:0" = PlaceHolder[target=select_1]
#   %tangents_1 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=tangents_1]
#   %scatter_add : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=scatter_add]
#   %mul_1 : Tensor "f32[200000, 64][64, 1]npu:0" = PlaceHolder[target=mul_1]
#   %view : Tensor "i64[200000, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [-1, 1]), kwargs = {})
#   %expand : Tensor "i64[200000, 64][1, 0]npu:0"[num_users=6] = call_function[target=torch.ops.aten.expand.default](args = (%view, [200000, 64]), kwargs = {})
#   %div : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %scatter_add), kwargs = {})
#   %gather_1 : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%div, 0, %expand), kwargs = {})
#   %mul_1 : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%eq_1, %gather_1), kwargs = {})
#   %sum_1 : Tensor "f32[1, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1, [0], True), kwargs = {})
#   return %mul_1,%sum_1
# SchedulerNodes: [SchedulerNode(name='op4'), SchedulerNode(name='op7')]

triton_red_fused_div_expand_gather_mul_sum_vie_2 = async_compile.triton('triton_red_fused_div_expand_gather_mul_sum_vie_2', '''
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
    size_hints={'x0': 64, 'r1': 200000},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_expand_gather_mul_sum_vie_2', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 64), ('r1', 200000)), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_red_fused_div_expand_gather_mul_sum_vie_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr, R1BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        _tmp9 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        for loop_r1 in range(loops_r1):
            r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[:,None]
            r1_mask = r1 < r1_numel
            tmp0 = tl.load(in_ptr0 + (x0 + 64*r1), r1_mask & x0_mask)
            tmp2 = tl.load(in_ptr1 + (r1), r1_mask, other=0.0)
            tmp1 = tmp0.to(tl.float32)
            tl.device_assert(((0 <= tmp2) & (tmp2 < 10000)) | ~(r1_mask), "index out of bounds: 0 <= tmp2 < 10000")
            tmp4 = tl.load(in_ptr2 + (x0 + 64*tmp2), r1_mask & x0_mask)
            tmp5 = tl.load(in_ptr3 + (x0 + 64*tmp2), r1_mask & x0_mask)
            tmp6 = (tmp4 / tmp5)
            tmp7 = tmp1 * tmp6
            tmp8 = tl.reshape(tmp7, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp10 = _tmp9 + tmp8
            _tmp9 = tl.where(tl.reshape((r1_mask & x0_mask), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp10, _tmp9)
            tl.store(out_ptr0 + (x0 + 64*r1), tmp7, r1_mask & x0_mask)
        tmp9 = tl.sum(_tmp9, 0).reshape(1, X0BLOCK_SUB)
        tl.store(out_ptr1 + (x0 ), tmp9, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [le, scalar_tensor, where], Original ATen: [aten.threshold_backward]
# Source node to ATen node mapping:
#   le => le
#   scalar_tensor => full_default_3
#   where => where
# Graph fragment:
#   %relu_4 : Tensor "f32[200000, 64][64, 1]npu:0" = PlaceHolder[target=relu_4]
#   %mm : Tensor "f32[200000, 64][64, 1]npu:0" = PlaceHolder[target=mm]
#   %le : Tensor "b8[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_4, 0), kwargs = {})
#   %full_default_3 : Tensor "f32[][]npu:0"[num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default_3, %mm), kwargs = {})
#   return %where
# SchedulerNodes: [SchedulerNode(name='op8')]

triton_poi_fused_threshold_backward_3 = async_compile.triton('triton_poi_fused_threshold_backward_3', '''
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
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_3', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 200000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_threshold_backward_3(in_out_ptr0, in_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp3 = tl.load(in_ptr0 + (x1 + 64*y0), y0_mask)
        tmp1 = tl.full([1, 1], 0.0, tl.float32)
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tl.store(in_out_ptr0 + (x1 + 64*y0), tmp4, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [sum_2], Original ATen: [aten.sum]
# Source node to ATen node mapping:
#   sum_2 => sum_2
# Graph fragment:
#   %where : Tensor "f32[200000, 64][64, 1]npu:0" = PlaceHolder[target=where]
#   %sum_2 : Tensor "f32[1, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%where, [0], True), kwargs = {})
#   return %buf11
# SchedulerNodes: [SchedulerNode(name='op11')]

triton_red_fused_sum_4 = async_compile.triton('triton_red_fused_sum_4', '''
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
    size_hints={'y0': 256, 'x1': 64, 'r2': 782},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'r2_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_4', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1', 'r2'], 'axis_static_values': (('y0', 256), ('x1', 64), ('r2', 782)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_red_fused_sum_4(in_ptr0, out_ptr0, y0_numel, x1_numel, r2_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr, R2BLOCK_SUB : tl.constexpr):
    x1_numel = 64
    X1BLOCK_SUB: tl.constexpr = 64
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    base_r2= tl.arange(0, R2BLOCK_SUB)
    loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        x1 = base_x1[None,None,:]
        x1_mask = x1 < x1_numel
        _tmp7 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
        for loop_r2 in range(loops_r2):
            r2 = (loop_r2 * R2BLOCK_SUB) + base_r2[None,:,None]
            r2_mask = r2 < r2_numel
            tmp0 = r2 + 782*y0
            tmp1 = tl.full([1, 1, 1], 200000, tl.int32)
            tmp2 = tmp0 < tmp1
            tmp3 = tmp2 & y0_mask
            tmp4 = tmp3 & r2_mask
            tmp5 = tl.load(in_ptr0 + (x1 + 64*r2 + 50048*y0), tmp4 & x1_mask, other=0.0)
            tmp6 = tl.reshape(tmp5, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
            tmp8 = _tmp7 + tmp6
            _tmp7 = tl.where(tl.reshape((r2_mask & x1_mask & y0_mask), [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp8, _tmp7)
        tmp7 = tl.sum(_tmp7, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
        tl.store(out_ptr0 + (x1 + 64*y0 ), tmp7, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [sum_2], Original ATen: [aten.sum]
# Source node to ATen node mapping:
#   sum_2 => sum_2
# Graph fragment:
#   %buf11 : Tensor "f32[1, 64, 256][16384, 1, 64]npu:0" = PlaceHolder[target=buf11]
#   %sum_2 : Tensor "f32[1, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%where, [0], True), kwargs = {})
#   return %sum_2
# SchedulerNodes: [SchedulerNode(name='op12')]

triton_per_fused_sum_5 = async_compile.triton('triton_per_fused_sum_5', '''
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
    size_hints={'x0': 64, 'r1': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_5', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 64), ('r1', 256)), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_sum_5(in_ptr0, out_ptr0, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 256
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        r1 = base_r1[:,None]
        r1_mask = True
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r1), x0_mask, other=0.0)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 0).reshape(1, X0BLOCK_SUB)
        tl.store(out_ptr0 + (x0 ), tmp3, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [slice_1, slice_2, neg, add, index_put, index_put_1, slice_3, slice_4, neg_1, add_2, index_put_2, index_put_3], Original ATen: [aten.slice, aten.neg, aten.add, aten.index_add]
# Source node to ATen node mapping:
#   add => add
#   add_2 => add_2
#   index_put => index_put
#   index_put_1 => index_put_1
#   index_put_2 => index_put_2
#   index_put_3 => index_put_3
#   neg => neg
#   neg_1 => neg_1
#   slice_1 => slice_1
#   slice_2 => slice_2
#   slice_3 => slice_3
#   slice_4 => slice_4
# Graph fragment:
#   %full_default : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=full_default]
#   %slice_1 : Tensor "f32[200000, 64][128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_2, 1, 0, 64), kwargs = {})
#   %slice_2 : Tensor "f32[200000, 64][128, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_2, 1, 64, 128), kwargs = {})
#   %neg : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_2,), kwargs = {})
#   %add : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_1, %neg), kwargs = {})
#   %index_put : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default, [%select_1], %add, True), kwargs = {})
#   %index_put_1 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default, [%select], %slice_2, True), kwargs = {})
#   %slice_3 : Tensor "f32[200000, 64][128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_6, 1, 0, 64), kwargs = {})
#   %slice_4 : Tensor "f32[200000, 64][128, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_6, 1, 64, 128), kwargs = {})
#   %neg_1 : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_4,), kwargs = {})
#   %add_2 : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_3, %neg_1), kwargs = {})
#   %index_put_2 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default, [%select_1], %add_2, True), kwargs = {})
#   %index_put_3 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default, [%select], %slice_4, True), kwargs = {})
#   return %index_put,%index_put_1,%index_put_2,%index_put_3
# SchedulerNodes: [SchedulerNode(name='op13'), SchedulerNode(name='op15'), SchedulerNode(name='op30'), SchedulerNode(name='op32')]

triton_poi_fused_add_index_add_neg_slice_6 = async_compile.triton('triton_poi_fused_add_index_add_neg_slice_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_index_add_neg_slice_6', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 10000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_index_add_neg_slice_6(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tl.store(out_ptr2 + (x1 + 64*y0), tmp0, y0_mask)
        tl.store(out_ptr3 + (x1 + 64*y0), tmp0, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [slice_1, slice_2, neg, add, index_put, index_put_1], Original ATen: [aten.slice, aten.neg, aten.add, aten.index_add]
# Source node to ATen node mapping:
#   add => add
#   index_put => index_put
#   index_put_1 => index_put_1
#   neg => neg
#   slice_1 => slice_1
#   slice_2 => slice_2
# Graph fragment:
#   %select_1 : Tensor "i64[200000][1]npu:0" = PlaceHolder[target=select_1]
#   %mm_2 : Tensor "f32[200000, 128][128, 1]npu:0" = PlaceHolder[target=mm_2]
#   %index_put : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=index_put]
#   %select : Tensor "i64[200000][1]npu:0" = PlaceHolder[target=select]
#   %index_put_1 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=index_put_1]
#   %slice_1 : Tensor "f32[200000, 64][128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_2, 1, 0, 64), kwargs = {})
#   %slice_2 : Tensor "f32[200000, 64][128, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_2, 1, 64, 128), kwargs = {})
#   %neg : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_2,), kwargs = {})
#   %add : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_1, %neg), kwargs = {})
#   %index_put : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default, [%select_1], %add, True), kwargs = {})
#   %index_put_1 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default, [%select], %slice_2, True), kwargs = {})
#   return %buf14,%buf16
# SchedulerNodes: [SchedulerNode(name='op14'), SchedulerNode(name='op16')]

triton_poi_fused_add_index_add_neg_slice_7 = async_compile.triton('triton_poi_fused_add_index_add_neg_slice_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_index_add_neg_slice_7', 'mutated_arg_names': ['out_ptr0', 'out_ptr1'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 200000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_index_add_neg_slice_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp6 = tl.load(in_ptr1 + (x1 + 128*y0), x1_mask & y0_mask)
        tmp7 = tl.load(in_ptr1 + (64 + x1 + 128*y0), x1_mask & y0_mask)
        tmp10 = tl.load(in_ptr2 + (y0), y0_mask)
        tmp1 = tl.full([Y0BLOCK_SUB, X1BLOCK_SUB], 10000, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 10000)) | ~(y0_mask), "index out of bounds: 0 <= tmp4 < 10000")
        tmp8 = -tmp7
        tmp9 = tmp6 + tmp8
        tmp11 = tmp10 + tmp1
        tmp12 = tmp10 < 0
        tmp13 = tl.where(tmp12, tmp11, tmp10)
        tl.device_assert(((0 <= tmp13) & (tmp13 < 10000)) | ~(y0_mask), "index out of bounds: 0 <= tmp13 < 10000")
        tl.atomic_add(out_ptr0 + (x1 + 64*tmp4), tmp9, x1_mask & y0_mask)
        tl.atomic_add(out_ptr1 + (x1 + 64*tmp13), tmp7, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [view, index_2, scalar_tensor, add_1, where_1, div_1, gather_3, mul_3, sum_3], Original ATen: [aten.view, aten.expand, aten.threshold_backward, aten.add, aten.div, aten.gather, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   add_1 => add_1
#   div_1 => div_1
#   gather_3 => gather_3
#   index_2 => expand
#   mul_3 => mul_3
#   scalar_tensor => full_default_3
#   sum_3 => sum_3
#   view => view
#   where_1 => where_1
# Graph fragment:
#   %eq_5 : Tensor "b8[200000, 64][64, 1]npu:0" = PlaceHolder[target=eq_5]
#   %select_1 : Tensor "i64[200000][1]npu:0" = PlaceHolder[target=select_1]
#   %le_1 : Tensor "b8[10000, 64][64, 1]npu:0" = PlaceHolder[target=le_1]
#   %buf14 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=buf14]
#   %buf16 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=buf16]
#   %scatter_add_1 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=scatter_add_1]
#   %mul_3 : Tensor "f32[200000, 64][64, 1]npu:0" = PlaceHolder[target=mul_3]
#   %view : Tensor "i64[200000, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [-1, 1]), kwargs = {})
#   %expand : Tensor "i64[200000, 64][1, 0]npu:0"[num_users=6] = call_function[target=torch.ops.aten.expand.default](args = (%view, [200000, 64]), kwargs = {})
#   %full_default_3 : Tensor "f32[][]npu:0"[num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %add_1 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%index_put, %index_put_1), kwargs = {})
#   %where_1 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_1, %full_default_3, %add_1), kwargs = {})
#   %div_1 : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%where_1, %scatter_add_1), kwargs = {})
#   %gather_3 : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%div_1, 0, %expand), kwargs = {})
#   %mul_3 : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%eq_5, %gather_3), kwargs = {})
#   %sum_3 : Tensor "f32[1, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_3, [0], True), kwargs = {})
#   return %mul_3,%sum_3
# SchedulerNodes: [SchedulerNode(name='op21'), SchedulerNode(name='op24')]

triton_red_fused_add_div_expand_gather_mul_sum_8 = async_compile.triton('triton_red_fused_add_div_expand_gather_mul_sum_8', '''
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
    size_hints={'x0': 64, 'r1': 200000},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i64', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_expand_gather_mul_sum_8', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 64), ('r1', 200000)), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_red_fused_add_div_expand_gather_mul_sum_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr, R1BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        _tmp14 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        for loop_r1 in range(loops_r1):
            r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[:,None]
            r1_mask = r1 < r1_numel
            tmp0 = tl.load(in_ptr0 + (x0 + 64*r1), r1_mask & x0_mask)
            tmp2 = tl.load(in_ptr1 + (r1), r1_mask, other=0.0)
            tmp1 = tmp0.to(tl.float32)
            tl.device_assert(((0 <= tmp2) & (tmp2 < 10000)) | ~(r1_mask), "index out of bounds: 0 <= tmp2 < 10000")
            tmp4 = tl.load(in_ptr2 + (x0 + 64*tmp2), r1_mask & x0_mask)
            tmp5 = tl.load(in_ptr3 + (x0 + 64*tmp2), r1_mask & x0_mask)
            tmp6 = tl.load(in_ptr4 + (x0 + 64*tmp2), r1_mask & x0_mask)
            tmp7 = tmp5 + tmp6
            tmp8 = tl.full([1, 1], 0.0, tl.float32)
            tmp9 = tl.where(tmp4, tmp8, tmp7)
            tmp10 = tl.load(in_ptr5 + (x0 + 64*tmp2), r1_mask & x0_mask)
            tmp11 = (tmp9 / tmp10)
            tmp12 = tmp1 * tmp11
            tmp13 = tl.reshape(tmp12, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp15 = _tmp14 + tmp13
            _tmp14 = tl.where(tl.reshape((r1_mask & x0_mask), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp15, _tmp14)
            tl.store(out_ptr0 + (x0 + 64*r1), tmp12, r1_mask & x0_mask)
        tmp14 = tl.sum(_tmp14, 0).reshape(1, X0BLOCK_SUB)
        tl.store(out_ptr1 + (x0 ), tmp14, x0_mask)
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
        primals_5, primals_7, primals_9, primals_11, primals_13, select, select_1, cat, relu, full_default, cat_1, relu_2, cat_2, relu_4, eq, eq_1, le_1, eq_4, eq_5, le_3, eq_8, eq_9, tangents_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            buf0 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type], Original ATen: [aten._to_copy]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_0.run(eq, buf0, 10000, 64, stream=raw_stream0)
            del eq
            buf1 = empty_strided_npu((200000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_1], Original ATen: [aten._to_copy]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(eq_1, buf1, 200000, 64, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [view, index_2, convert_element_type, convert_element_type_1, scatter_add], Original ATen: [aten.view, aten.expand, aten._to_copy, aten.scatter_add]
            # [Provenance debug handles] torch.ops.aten.scatter_add.default:1
            buf2 = torch.ops.aten.scatter_add.default(buf0, 0, reinterpret_tensor(select_1, (200000, 64), (1, 0), 0), buf1)
            del buf0
            buf3 = buf2
            assert_size_stride(buf3, (10000, 64), (64, 1), 'torch.ops.aten.scatter_add.default')
            assert_alignment(buf3, 16, 'torch.ops.aten.scatter_add.default')
            del buf2
            tangents_1 = copy_if_misaligned(tangents_1)
            buf4 = buf1; del buf1  # reuse
            buf7 = empty_strided_npu((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view, index_2, div, gather_1, mul_1, sum_1], Original ATen: [aten.view, aten.expand, aten.div, aten.gather, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_div_expand_gather_mul_sum_vie_2.run(eq_1, select_1, tangents_1, buf3, buf4, buf7, 64, 200000, stream=raw_stream0)
            del buf3
            del eq_1
            del tangents_1
            buf5 = empty_strided_npu((200000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_17, permute_6, mm], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:2
            extern_kernels.mm(buf4, primals_13, out=buf5)
            del primals_13
            buf6 = empty_strided_npu((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_7, permute_9], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:3
            extern_kernels.mm(reinterpret_tensor(buf4, (64, 200000), (1, 64), 0), relu_4, out=buf6)
            del buf4
            buf8 = relu_4; del relu_4  # reuse
            # Topologically Sorted Source Nodes: [le, scalar_tensor, where], Original ATen: [aten.threshold_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_threshold_backward_3.run(buf8, buf5, 200000, 64, stream=raw_stream0)
            del buf5
            buf9 = empty_strided_npu((200000, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_14, permute_10, mm_2], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:4
            extern_kernels.mm(buf8, primals_11, out=buf9)
            del primals_11
            buf10 = empty_strided_npu((64, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_11, permute_13], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:5
            extern_kernels.mm(reinterpret_tensor(buf8, (64, 200000), (1, 64), 0), cat_2, out=buf10)
            del cat_2
            buf11 = empty_strided_npu((1, 64, 256), (16384, 1, 64), torch.float32)
            # Topologically Sorted Source Nodes: [sum_2], Original ATen: [aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_sum_4.run(buf8, buf11, 256, 64, 782, stream=raw_stream0)
            buf12 = empty_strided_npu((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sum_2], Original ATen: [aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_sum_5.run(buf11, buf12, 64, 256, stream=raw_stream0)
            buf13 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            buf15 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            buf30 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            buf32 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [slice_1, slice_2, neg, add, index_put, index_put_1, slice_3, slice_4, neg_1, add_2, index_put_2, index_put_3], Original ATen: [aten.slice, aten.neg, aten.add, aten.index_add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_index_add_neg_slice_6.run(full_default, buf13, buf15, buf30, buf32, 10000, 64, stream=raw_stream0)
            del full_default
            # Topologically Sorted Source Nodes: [slice_1, slice_2, neg, add, index_put, index_put_1], Original ATen: [aten.slice, aten.neg, aten.add, aten.index_add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_index_add_neg_slice_7.run(select_1, buf9, select, buf13, buf15, 200000, 64, stream=raw_stream0)
            buf17 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2], Original ATen: [aten._to_copy]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_0.run(eq_4, buf17, 10000, 64, stream=raw_stream0)
            del eq_4
            buf18 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_3], Original ATen: [aten._to_copy]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(eq_5, buf18, 200000, 64, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [view, index_2, convert_element_type_2, convert_element_type_3, scatter_add_1], Original ATen: [aten.view, aten.expand, aten._to_copy, aten.scatter_add]
            # [Provenance debug handles] torch.ops.aten.scatter_add.default:6
            buf19 = torch.ops.aten.scatter_add.default(buf17, 0, reinterpret_tensor(select_1, (200000, 64), (1, 0), 0), buf18)
            del buf17
            buf20 = buf19
            assert_size_stride(buf20, (10000, 64), (64, 1), 'torch.ops.aten.scatter_add.default')
            assert_alignment(buf20, 16, 'torch.ops.aten.scatter_add.default')
            del buf19
            buf21 = buf18; del buf18  # reuse
            buf24 = empty_strided_npu((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view, index_2, scalar_tensor, add_1, where_1, div_1, gather_3, mul_3, sum_3], Original ATen: [aten.view, aten.expand, aten.threshold_backward, aten.add, aten.div, aten.gather, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_add_div_expand_gather_mul_sum_8.run(eq_5, select_1, le_1, buf13, buf15, buf20, buf21, buf24, 64, 200000, stream=raw_stream0)
            del buf13
            del buf15
            del eq_5
            del le_1
            buf22 = empty_strided_npu((200000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_10, permute_14, mm_4], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:7
            extern_kernels.mm(buf21, primals_9, out=buf22)
            del primals_9
            buf23 = empty_strided_npu((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_15, permute_17], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:8
            extern_kernels.mm(reinterpret_tensor(buf21, (64, 200000), (1, 64), 0), relu_2, out=buf23)
            del buf21
            buf25 = relu_2; del relu_2  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor, le_2, where_2], Original ATen: [aten.threshold_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_threshold_backward_3.run(buf25, buf22, 200000, 64, stream=raw_stream0)
            buf26 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [x_7, permute_18, mm_6], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:9
            extern_kernels.mm(buf25, primals_7, out=buf26)
            del primals_7
            buf27 = empty_strided_npu((64, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_19, permute_21], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:10
            extern_kernels.mm(reinterpret_tensor(buf25, (64, 200000), (1, 64), 0), cat_1, out=buf27)
            del cat_1
            buf28 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [sum_4], Original ATen: [aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_sum_4.run(buf25, buf28, 256, 64, 782, stream=raw_stream0)
            buf29 = empty_strided_npu((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sum_4], Original ATen: [aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_sum_5.run(buf28, buf29, 64, 256, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [slice_3, slice_4, neg_1, add_2, index_put_2, index_put_3], Original ATen: [aten.slice, aten.neg, aten.add, aten.index_add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_index_add_neg_slice_7.run(select_1, buf26, select, buf30, buf32, 200000, 64, stream=raw_stream0)
            del buf26
            del select
            buf34 = buf20; del buf20  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_4], Original ATen: [aten._to_copy]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_0.run(eq_8, buf34, 10000, 64, stream=raw_stream0)
            del eq_8
            buf35 = buf25; del buf25  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_5], Original ATen: [aten._to_copy]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(eq_9, buf35, 200000, 64, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [view, index_2, convert_element_type_4, convert_element_type_5, scatter_add_2], Original ATen: [aten.view, aten.expand, aten._to_copy, aten.scatter_add]
            # [Provenance debug handles] torch.ops.aten.scatter_add.default:11
            buf36 = torch.ops.aten.scatter_add.default(buf34, 0, reinterpret_tensor(select_1, (200000, 64), (1, 0), 0), buf35)
            del buf34
            buf37 = buf36
            assert_size_stride(buf37, (10000, 64), (64, 1), 'torch.ops.aten.scatter_add.default')
            assert_alignment(buf37, 16, 'torch.ops.aten.scatter_add.default')
            del buf36
            buf38 = buf35; del buf35  # reuse
            buf41 = empty_strided_npu((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view, index_2, scalar_tensor, add_3, where_3, div_2, gather_5, mul_5, sum_5], Original ATen: [aten.view, aten.expand, aten.threshold_backward, aten.add, aten.div, aten.gather, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_add_div_expand_gather_mul_sum_8.run(eq_9, select_1, le_3, buf30, buf32, buf37, buf38, buf41, 64, 200000, stream=raw_stream0)
            del buf30
            del buf32
            del buf37
            del eq_9
            del le_3
            del select_1
            buf39 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [x_3, permute_22, mm_8], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:12
            extern_kernels.mm(buf38, primals_5, out=buf39)
            del primals_5
            buf40 = empty_strided_npu((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_23, permute_25], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:13
            extern_kernels.mm(reinterpret_tensor(buf38, (64, 200000), (1, 64), 0), relu, out=buf40)
            del buf38
            buf42 = relu; del relu  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor, le_4, where_4], Original ATen: [aten.threshold_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_threshold_backward_3.run(buf42, buf39, 200000, 64, stream=raw_stream0)
            del buf39
            buf43 = empty_strided_npu((64, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [scalar_tensor, le_4, where_4, permute_26, permute_28], Original ATen: [aten.threshold_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:14
            extern_kernels.mm(reinterpret_tensor(buf42, (64, 200000), (1, 64), 0), cat, out=buf43)
            del cat
            buf44 = buf28; del buf28  # reuse
            # Topologically Sorted Source Nodes: [sum_6], Original ATen: [aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_sum_4.run(buf42, buf44, 256, 64, 782, stream=raw_stream0)
            del buf42
            buf45 = empty_strided_npu((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sum_6], Original ATen: [aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_sum_5.run(buf44, buf45, 64, 256, stream=raw_stream0)
            del buf44
        return (None, None, buf43, reinterpret_tensor(buf45, (64, ), (1, ), 0), buf40, reinterpret_tensor(buf41, (64, ), (1, ), 0), buf27, reinterpret_tensor(buf29, (64, ), (1, ), 0), buf23, reinterpret_tensor(buf24, (64, ), (1, ), 0), buf10, reinterpret_tensor(buf12, (64, ), (1, ), 0), buf6, reinterpret_tensor(buf7, (64, ), (1, ), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    _shared_storage_5 = rand_strided((400000,), (1,), device='npu:0', dtype=torch.int64)
    primals_5 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 128), (128, 1), device='npu:0', dtype=torch.float32)
    primals_9 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    primals_11 = rand_strided((64, 128), (128, 1), device='npu:0', dtype=torch.float32)
    primals_13 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    select = torch.as_strided(_shared_storage_5, (200000, ), (1, ))
    select_1 = torch.as_strided(_shared_storage_5, (200000, ), (1, ))
    cat = rand_strided((200000, 128), (128, 1), device='npu:0', dtype=torch.float32)
    relu = rand_strided((200000, 64), (64, 1), device='npu:0', dtype=torch.float32)
    full_default = rand_strided((10000, 64), (64, 1), device='npu:0', dtype=torch.float32)
    cat_1 = rand_strided((200000, 128), (128, 1), device='npu:0', dtype=torch.float32)
    relu_2 = rand_strided((200000, 64), (64, 1), device='npu:0', dtype=torch.float32)
    cat_2 = rand_strided((200000, 128), (128, 1), device='npu:0', dtype=torch.float32)
    relu_4 = rand_strided((200000, 64), (64, 1), device='npu:0', dtype=torch.float32)
    eq = rand_strided((10000, 64), (64, 1), device='npu:0', dtype=torch.bool)
    eq_1 = rand_strided((200000, 64), (64, 1), device='npu:0', dtype=torch.bool)
    le_1 = rand_strided((10000, 64), (64, 1), device='npu:0', dtype=torch.bool)
    eq_4 = rand_strided((10000, 64), (64, 1), device='npu:0', dtype=torch.bool)
    eq_5 = rand_strided((200000, 64), (64, 1), device='npu:0', dtype=torch.bool)
    le_3 = rand_strided((10000, 64), (64, 1), device='npu:0', dtype=torch.bool)
    eq_8 = rand_strided((10000, 64), (64, 1), device='npu:0', dtype=torch.bool)
    eq_9 = rand_strided((200000, 64), (64, 1), device='npu:0', dtype=torch.bool)
    tangents_1 = rand_strided((10000, 64), (64, 1), device='npu:0', dtype=torch.float32)
    return [primals_5, primals_7, primals_9, primals_11, primals_13, select, select_1, cat, relu, full_default, cat_1, relu_2, cat_2, relu_4, eq, eq_1, le_1, eq_4, eq_5, le_3, eq_8, eq_9, tangents_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('basic_gnn_edgecnn', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
