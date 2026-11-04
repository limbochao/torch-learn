# AOT ID: ['5_backward']
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
# SchedulerNodes: [SchedulerNode(name='op0')]

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
# Topologically Sorted Source Nodes: [edge_index_i, view_1, index_1, gather, view, mul_1, index, index_put], Original ATen: [aten.select, aten.view, aten.expand, aten.gather, aten.mul, aten.index_add]
# Source node to ATen node mapping:
#   edge_index_i => select_1
#   gather => gather
#   index => select
#   index_1 => expand
#   index_put => index_put
#   mul_1 => mul_1
#   view => view
#   view_1 => view_1
# Graph fragment:
#   %full_default : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=full_default]
#   %select_1 : Tensor "i64[209978][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_3, 0, 1), kwargs = {})
#   %view_1 : Tensor "i64[209978, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [-1, 1]), kwargs = {})
#   %expand : Tensor "i64[209978, 64][1, 0]npu:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_1, [209978, 64]), kwargs = {})
#   %gather : Tensor "f32[209978, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%tangents_1, 0, %expand), kwargs = {})
#   %view : Tensor "f32[209978, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_4, [-1, 1]), kwargs = {})
#   %mul_1 : Tensor "f32[209978, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gather, %view), kwargs = {})
#   %select : Tensor "i64[209978][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_3, 0, 0), kwargs = {})
#   %index_put : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default, [%select], %mul_1, True), kwargs = {})
#   return %index_put
# SchedulerNodes: [SchedulerNode(name='op1')]

triton_poi_fused_expand_gather_index_add_mul_s_1 = async_compile.triton('triton_poi_fused_expand_gather_index_add_mul_s_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_expand_gather_index_add_mul_s_1', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 10000), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_expand_gather_index_add_mul_s_1(in_out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tl.store(in_out_ptr0 + (x1 + 64*y0), tmp0, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [edge_index_i, view_1, index_1, gather, view, mul_1, index, index_put], Original ATen: [aten.select, aten.view, aten.expand, aten.gather, aten.mul, aten.index_add]
# Source node to ATen node mapping:
#   edge_index_i => select_1
#   gather => gather
#   index => select
#   index_1 => expand
#   index_put => index_put
#   mul_1 => mul_1
#   view => view
#   view_1 => view_1
# Graph fragment:
#   %primals_3 : Tensor "i64[2, 209978][209978, 1]npu:0" = PlaceHolder[target=primals_3]
#   %tangents_1 : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=tangents_1]
#   %primals_4 : Tensor "f32[209978][1]npu:0" = PlaceHolder[target=primals_4]
#   %index_put : Tensor "f32[10000, 64][64, 1]npu:0" = PlaceHolder[target=index_put]
#   %select_1 : Tensor "i64[209978][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_3, 0, 1), kwargs = {})
#   %view_1 : Tensor "i64[209978, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [-1, 1]), kwargs = {})
#   %expand : Tensor "i64[209978, 64][1, 0]npu:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_1, [209978, 64]), kwargs = {})
#   %gather : Tensor "f32[209978, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%tangents_1, 0, %expand), kwargs = {})
#   %view : Tensor "f32[209978, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_4, [-1, 1]), kwargs = {})
#   %mul_1 : Tensor "f32[209978, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gather, %view), kwargs = {})
#   %select : Tensor "i64[209978][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_3, 0, 0), kwargs = {})
#   %index_put : Tensor "f32[10000, 64][64, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default, [%select], %mul_1, True), kwargs = {})
#   return %buf2
# SchedulerNodes: [SchedulerNode(name='op2')]

triton_poi_fused_expand_gather_index_add_mul_s_2 = async_compile.triton('triton_poi_fused_expand_gather_index_add_mul_s_2', '''
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
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_expand_gather_index_add_mul_s_2', 'mutated_arg_names': ['out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 209978), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_expand_gather_index_add_mul_s_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp6 = tl.load(in_ptr0 + (209978 + y0), y0_mask)
        tmp9 = tl.load(in_ptr2 + (y0), y0_mask)
        tmp1 = tl.full([Y0BLOCK_SUB, X1BLOCK_SUB], 10000, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 10000)) | ~(y0_mask), "index out of bounds: 0 <= tmp4 < 10000")
        tl.device_assert(((0 <= tmp6) & (tmp6 < 10000)) | ~(y0_mask), "index out of bounds: 0 <= tmp6 < 10000")
        tmp8 = tl.load(in_ptr1 + (x1 + 64*tmp6), x1_mask & y0_mask)
        tmp10 = tmp8 * tmp9
        tl.atomic_add(out_ptr0 + (x1 + 64*tmp4), tmp10, x1_mask & y0_mask)
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
        primals_1, primals_3, primals_4, full_default, permute_2, tangents_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            tangents_1 = copy_if_misaligned(tangents_1)
            buf0 = empty_strided_npu((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sum_1], Original ATen: [aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_sum_0.run(tangents_1, buf0, 64, 10000, stream=raw_stream0)
            buf1 = full_default; del full_default  # reuse
            # Topologically Sorted Source Nodes: [edge_index_i, view_1, index_1, gather, view, mul_1, index, index_put], Original ATen: [aten.select, aten.view, aten.expand, aten.gather, aten.mul, aten.index_add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_expand_gather_index_add_mul_s_1.run(buf1, 10000, 64, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [edge_index_i, view_1, index_1, gather, view, mul_1, index, index_put], Original ATen: [aten.select, aten.view, aten.expand, aten.gather, aten.mul, aten.index_add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_expand_gather_index_add_mul_s_2.run(primals_3, tangents_1, primals_4, buf1, 209978, 64, stream=raw_stream0)
            del primals_3
            del primals_4
            del tangents_1
            buf4 = empty_strided_npu((64, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mm_2], Original ATen: [aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.mm:1
            extern_kernels.mm(permute_2, buf1, out=buf4)
            del permute_2
            buf3 = empty_strided_npu((10000, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, permute_1, mm_1], Original ATen: [aten.t, aten.matmul_backward]
            # [Provenance debug handles] extern_kernels.mm:2
            extern_kernels.mm(buf1, primals_1, out=buf3)
            del buf1
            del primals_1
        return (reinterpret_tensor(buf4, (64, 64), (1, 64), 0), buf3, None, None, reinterpret_tensor(buf0, (64, ), (1, ), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((64, 64), (64, 1), device='npu:0', dtype=torch.float32)
    primals_3 = rand_strided((2, 209978), (209978, 1), device='npu:0', dtype=torch.int64)
    primals_4 = rand_strided((209978, ), (1, ), device='npu:0', dtype=torch.float32)
    full_default = rand_strided((10000, 64), (64, 1), device='npu:0', dtype=torch.float32)
    permute_2 = rand_strided((64, 10000), (1, 64), device='npu:0', dtype=torch.float32)
    tangents_1 = rand_strided((10000, 64), (64, 1), device='npu:0', dtype=torch.float32)
    return [primals_1, primals_3, primals_4, full_default, permute_2, tangents_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('basic_gnn_gcn', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
