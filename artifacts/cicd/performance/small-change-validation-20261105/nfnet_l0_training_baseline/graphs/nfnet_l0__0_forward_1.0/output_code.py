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
# Topologically Sorted Source Nodes: [reshape, mul, view, batch_norm], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm => add, mul_1, mul_2, rsqrt, sub, unsqueeze, var_mean
#   mul => mul
#   reshape => view
#   view => view_1
# Graph fragment:
#   %primals_1 : Tensor "f32[16, 3, 3, 3][27, 9, 3, 1]npu:0" = PlaceHolder[target=primals_1]
#   %buf0 : Tensor "f32[1, 16, 1][16, 1, 16]npu:0" = PlaceHolder[target=buf0]
#   %getitem_1 : Tensor "f32[1, 16, 1][16, 1, 16]npu:0" = PlaceHolder[target=getitem_1]
#   %buf2 : Tensor "f32[1, 16, 1][16, 1, 16]npu:0" = PlaceHolder[target=buf2]
#   %primals_2 : Tensor "f32[16, 1, 1, 1][1, 1, 1, 1]npu:0" = PlaceHolder[target=primals_2]
#   %view : Tensor "f32[1, 16, 27][432, 27, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_1, [1, 16, -1]), kwargs = {})
#   %mul : Tensor "f32[16, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_2, 0.34412564994580647), kwargs = {})
#   %view_1 : Tensor "f32[16][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul, [-1]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add : Tensor "f32[1, 16, 1][16, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[1, 16, 1][16, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %sub : Tensor "f32[1, 16, 27][432, 27, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %mul_1 : Tensor "f32[1, 16, 27][432, 27, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %unsqueeze : Tensor "f32[16, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_1, -1), kwargs = {})
#   %mul_2 : Tensor "f32[1, 16, 27][432, 27, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze), kwargs = {})
#   return %buf0,%getitem_1,%buf2,%mul_2,%rsqrt
# SchedulerNodes: [SchedulerNode(name='op0'), SchedulerNode(name='op1'), SchedulerNode(name='op2'), SchedulerNode(name='op4'), SchedulerNode(name='op3')]

triton_per_fused__native_batch_norm_legit_mul__0 = async_compile.triton('triton_per_fused__native_batch_norm_legit_mul__0', '''
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
    size_hints={'x0': 16, 'r1': 27},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_mul__0', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 16), ('r1', 27)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_mul__0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 27
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
        tmp0 = tl.load(in_ptr0 + (r1 + 27*x0), x0_mask, other=0.0)
        tmp16 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 1).reshape(X0BLOCK_SUB, 1)
        tmp4 = tl.full([1, 1], 27.0, tl.float32)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tmp0 - tmp5
        tmp7 = tmp6 * tmp6
        tmp9 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp7, 0)
        tmp10 = tl.sum(tmp9, 1).reshape(X0BLOCK_SUB, 1)
        tmp11 = (tmp10 / tmp4)
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp6 * tmp14
        tmp17 = tl.full([1, 1], 0.34412564994580647, tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 * tmp18
        tl.store(in_out_ptr0 + (x0), tmp5, x0_mask)
        tl.store(out_ptr1 + (r1 + 27*x0), tmp19, x0_mask)
        tl.store(out_ptr2 + (x0), tmp14, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   input_2 => add_1, div, exp, neg
# Graph fragment:
#   %convolution : Tensor "f32[128, 16, 112, 112][200704, 12544, 112, 1]npu:0" = PlaceHolder[target=convolution]
#   %neg : Tensor "f32[128, 16, 112, 112][200704, 12544, 112, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convolution,), kwargs = {})
#   %exp : Tensor "f32[128, 16, 112, 112][200704, 12544, 112, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_1 : Tensor "f32[128, 16, 112, 112][200704, 12544, 112, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[128, 16, 112, 112][200704, 12544, 112, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution, %add_1), kwargs = {})
#   return %div
# SchedulerNodes: [SchedulerNode(name='op7')]

triton_poi_fused_silu_1 = async_compile.triton('triton_poi_fused_silu_1', '''
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
    size_hints={'t0': 128, 'z1': 16, 'y2': 112, 'x3': 112}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 't0_numel': 'i32', 'z1_numel': 'i32', 'y2_numel': 'i32', 'x3_numel': 'i32', 'T0BLOCK': 'i32', 'Z1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_1', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [3], 'axis_names': ['t0', 'z1', 'y2', 'x3'], 'axis_static_values': (('t0', 128), ('z1', 16), ('y2', 112), ('x3', 112)), 'low_dims': {3}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('T0BLOCK', 'Z1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_silu_1(in_ptr0, out_ptr0, t0_numel, z1_numel, y2_numel, x3_numel, T0BLOCK, Z1BLOCK, T0BLOCK_SUB : tl.constexpr, Z1BLOCK_SUB : tl.constexpr, Y2BLOCK_SUB : tl.constexpr):
    x3_numel = 112
    X3BLOCK_SUB: tl.constexpr = 112
    t0_offset = tl.program_id(0) * T0BLOCK
    base_t0= tl.arange(0, T0BLOCK_SUB)
    loops_t0 = (T0BLOCK + T0BLOCK_SUB - 1) // T0BLOCK_SUB
    z1_offset = tl.program_id(1) * Z1BLOCK
    base_z1= tl.arange(0, Z1BLOCK_SUB)
    loops_z1 = (Z1BLOCK + Z1BLOCK_SUB - 1) // Z1BLOCK_SUB
    base_y2= tl.arange(0, Y2BLOCK_SUB)
    loops_y2 = (y2_numel + Y2BLOCK_SUB - 1) // Y2BLOCK_SUB
    base_x3= tl.arange(0, X3BLOCK_SUB)
    for loop_t0 in range(loops_t0):
        t0 = t0_offset + (loop_t0 * T0BLOCK_SUB) + base_t0[:,None,None,None]
        t0_mask = t0 < min(T0BLOCK+t0_offset, t0_numel)
        for loop_z1 in range(loops_z1):
            z1 = z1_offset + (loop_z1 * Z1BLOCK_SUB) + base_z1[None,:,None,None]
            z1_mask = z1 < min(Z1BLOCK+z1_offset, z1_numel)
            for loop_y2 in range(loops_y2):
                y2 = (loop_y2 * Y2BLOCK_SUB) + base_y2[None,None,:,None]
                y2_mask = y2 < y2_numel
                x3 = base_x3[None,None,None,:]
                tmp0 = tl.load(in_ptr0 + (x3 + 112*y2 + 12544*z1 + 200704*t0), t0_mask & y2_mask & z1_mask)
                tmp1 = -tmp0
                tmp2 = libdevice.exp(tmp1)
                tmp3 = tl.full([1, 1, 1, 1], 1.0, tl.float32)
                tmp4 = tmp2 + tmp3
                tmp5 = (tmp0 / tmp4)
                tl.store(out_ptr0 + (x3 + 112*y2 + 12544*z1 + 200704*t0), tmp5, t0_mask & y2_mask & z1_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [reshape_1, mul_1, view_1, batch_norm_1], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_1 => add_2, mul_4, mul_5, rsqrt_1, sub_1, unsqueeze_1, var_mean_1
#   mul_1 => mul_3
#   reshape_1 => view_3
#   view_1 => view_4
# Graph fragment:
#   %primals_5 : Tensor "f32[32, 16, 3, 3][144, 9, 3, 1]npu:0" = PlaceHolder[target=primals_5]
#   %buf8 : Tensor "f32[1, 32, 1][32, 1, 32]npu:0" = PlaceHolder[target=buf8]
#   %getitem_3 : Tensor "f32[1, 32, 1][32, 1, 32]npu:0" = PlaceHolder[target=getitem_3]
#   %buf10 : Tensor "f32[1, 32, 1][32, 1, 32]npu:0" = PlaceHolder[target=buf10]
#   %primals_6 : Tensor "f32[32, 1, 1, 1][1, 1, 1, 1]npu:0" = PlaceHolder[target=primals_6]
#   %view_3 : Tensor "f32[1, 32, 144][4608, 144, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_5, [1, 32, -1]), kwargs = {})
#   %mul_3 : Tensor "f32[32, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_6, 0.1490107774734497), kwargs = {})
#   %view_4 : Tensor "f32[32][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_3, [-1]), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_3, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_2 : Tensor "f32[1, 32, 1][32, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 32, 1][32, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %sub_1 : Tensor "f32[1, 32, 144][4608, 144, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_3, %getitem_3), kwargs = {})
#   %mul_4 : Tensor "f32[1, 32, 144][4608, 144, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[32, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_4, -1), kwargs = {})
#   %mul_5 : Tensor "f32[1, 32, 144][4608, 144, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_1), kwargs = {})
#   return %buf8,%getitem_3,%buf10,%mul_5,%rsqrt_1
# SchedulerNodes: [SchedulerNode(name='op8'), SchedulerNode(name='op9'), SchedulerNode(name='op10'), SchedulerNode(name='op12'), SchedulerNode(name='op11')]

triton_per_fused__native_batch_norm_legit_mul__2 = async_compile.triton('triton_per_fused__native_batch_norm_legit_mul__2', '''
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
    size_hints={'x0': 32, 'r1': 144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_mul__2', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 32), ('r1', 144)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_mul__2(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 144
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
        tmp0 = tl.load(in_ptr0 + (r1 + 144*x0), x0_mask, other=0.0)
        tmp16 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 1).reshape(X0BLOCK_SUB, 1)
        tmp4 = tl.full([1, 1], 144.0, tl.float32)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tmp0 - tmp5
        tmp7 = tmp6 * tmp6
        tmp9 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp7, 0)
        tmp10 = tl.sum(tmp9, 1).reshape(X0BLOCK_SUB, 1)
        tmp11 = (tmp10 / tmp4)
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp6 * tmp14
        tmp17 = tl.full([1, 1], 0.1490107774734497, tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 * tmp18
        tl.store(in_out_ptr0 + (x0), tmp5, x0_mask)
        tl.store(out_ptr1 + (r1 + 144*x0), tmp19, x0_mask)
        tl.store(out_ptr2 + (x0), tmp14, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   input_4 => add_3, div_1, exp_1, neg_1
# Graph fragment:
#   %convolution_1 : Tensor "f32[128, 32, 112, 112][401408, 12544, 112, 1]npu:0" = PlaceHolder[target=convolution_1]
#   %neg_1 : Tensor "f32[128, 32, 112, 112][401408, 12544, 112, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convolution_1,), kwargs = {})
#   %exp_1 : Tensor "f32[128, 32, 112, 112][401408, 12544, 112, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_1,), kwargs = {})
#   %add_3 : Tensor "f32[128, 32, 112, 112][401408, 12544, 112, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_1, 1), kwargs = {})
#   %div_1 : Tensor "f32[128, 32, 112, 112][401408, 12544, 112, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_1, %add_3), kwargs = {})
#   return %div_1
# SchedulerNodes: [SchedulerNode(name='op15')]

triton_poi_fused_silu_3 = async_compile.triton('triton_poi_fused_silu_3', '''
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
    size_hints={'t0': 128, 'z1': 32, 'y2': 112, 'x3': 112}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 't0_numel': 'i32', 'z1_numel': 'i32', 'y2_numel': 'i32', 'x3_numel': 'i32', 'T0BLOCK': 'i32', 'Z1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_3', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [3], 'axis_names': ['t0', 'z1', 'y2', 'x3'], 'axis_static_values': (('t0', 128), ('z1', 32), ('y2', 112), ('x3', 112)), 'low_dims': {3}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('T0BLOCK', 'Z1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_silu_3(in_ptr0, out_ptr0, t0_numel, z1_numel, y2_numel, x3_numel, T0BLOCK, Z1BLOCK, T0BLOCK_SUB : tl.constexpr, Z1BLOCK_SUB : tl.constexpr, Y2BLOCK_SUB : tl.constexpr):
    x3_numel = 112
    X3BLOCK_SUB: tl.constexpr = 112
    t0_offset = tl.program_id(0) * T0BLOCK
    base_t0= tl.arange(0, T0BLOCK_SUB)
    loops_t0 = (T0BLOCK + T0BLOCK_SUB - 1) // T0BLOCK_SUB
    z1_offset = tl.program_id(1) * Z1BLOCK
    base_z1= tl.arange(0, Z1BLOCK_SUB)
    loops_z1 = (Z1BLOCK + Z1BLOCK_SUB - 1) // Z1BLOCK_SUB
    base_y2= tl.arange(0, Y2BLOCK_SUB)
    loops_y2 = (y2_numel + Y2BLOCK_SUB - 1) // Y2BLOCK_SUB
    base_x3= tl.arange(0, X3BLOCK_SUB)
    for loop_t0 in range(loops_t0):
        t0 = t0_offset + (loop_t0 * T0BLOCK_SUB) + base_t0[:,None,None,None]
        t0_mask = t0 < min(T0BLOCK+t0_offset, t0_numel)
        for loop_z1 in range(loops_z1):
            z1 = z1_offset + (loop_z1 * Z1BLOCK_SUB) + base_z1[None,:,None,None]
            z1_mask = z1 < min(Z1BLOCK+z1_offset, z1_numel)
            for loop_y2 in range(loops_y2):
                y2 = (loop_y2 * Y2BLOCK_SUB) + base_y2[None,None,:,None]
                y2_mask = y2 < y2_numel
                x3 = base_x3[None,None,None,:]
                tmp0 = tl.load(in_ptr0 + (x3 + 112*y2 + 12544*z1 + 401408*t0), t0_mask & y2_mask & z1_mask)
                tmp1 = -tmp0
                tmp2 = libdevice.exp(tmp1)
                tmp3 = tl.full([1, 1, 1, 1], 1.0, tl.float32)
                tmp4 = tmp2 + tmp3
                tmp5 = (tmp0 / tmp4)
                tl.store(out_ptr0 + (x3 + 112*y2 + 12544*z1 + 401408*t0), tmp5, t0_mask & y2_mask & z1_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [reshape_2, mul_2, view_2, batch_norm_2], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_2 => add_4, mul_7, mul_8, rsqrt_2, sub_2, unsqueeze_2, var_mean_2
#   mul_2 => mul_6
#   reshape_2 => view_6
#   view_2 => view_7
# Graph fragment:
#   %primals_8 : Tensor "f32[64, 32, 3, 3][288, 9, 3, 1]npu:0" = PlaceHolder[target=primals_8]
#   %buf16 : Tensor "f32[1, 64, 1][64, 1, 64]npu:0" = PlaceHolder[target=buf16]
#   %getitem_5 : Tensor "f32[1, 64, 1][64, 1, 64]npu:0" = PlaceHolder[target=getitem_5]
#   %buf18 : Tensor "f32[1, 64, 1][64, 1, 64]npu:0" = PlaceHolder[target=buf18]
#   %primals_9 : Tensor "f32[64, 1, 1, 1][1, 1, 1, 1]npu:0" = PlaceHolder[target=primals_9]
#   %view_6 : Tensor "f32[1, 64, 288][18432, 288, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_8, [1, 64, -1]), kwargs = {})
#   %mul_6 : Tensor "f32[64, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_9, 0.10536653122135592), kwargs = {})
#   %view_7 : Tensor "f32[64][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_6, [-1]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_6, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : Tensor "f32[1, 64, 1][64, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, 64, 1][64, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %sub_2 : Tensor "f32[1, 64, 288][18432, 288, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_6, %getitem_5), kwargs = {})
#   %mul_7 : Tensor "f32[1, 64, 288][18432, 288, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[64, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_7, -1), kwargs = {})
#   %mul_8 : Tensor "f32[1, 64, 288][18432, 288, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_2), kwargs = {})
#   return %buf16,%getitem_5,%buf18,%mul_8,%rsqrt_2
# SchedulerNodes: [SchedulerNode(name='op16'), SchedulerNode(name='op17'), SchedulerNode(name='op18'), SchedulerNode(name='op20'), SchedulerNode(name='op19')]

triton_per_fused__native_batch_norm_legit_mul__4 = async_compile.triton('triton_per_fused__native_batch_norm_legit_mul__4', '''
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
    size_hints={'x0': 64, 'r1': 288},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_mul__4', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 64), ('r1', 288)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_mul__4(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 288
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
        tmp0 = tl.load(in_ptr0 + (r1 + 288*x0), x0_mask, other=0.0)
        tmp16 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 1).reshape(X0BLOCK_SUB, 1)
        tmp4 = tl.full([1, 1], 288.0, tl.float32)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tmp0 - tmp5
        tmp7 = tmp6 * tmp6
        tmp9 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp7, 0)
        tmp10 = tl.sum(tmp9, 1).reshape(X0BLOCK_SUB, 1)
        tmp11 = (tmp10 / tmp4)
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp6 * tmp14
        tmp17 = tl.full([1, 1], 0.10536653122135592, tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 * tmp18
        tl.store(in_out_ptr0 + (x0), tmp5, x0_mask)
        tl.store(out_ptr1 + (r1 + 288*x0), tmp19, x0_mask)
        tl.store(out_ptr2 + (x0), tmp14, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   input_6 => add_5, div_2, exp_2, neg_2
# Graph fragment:
#   %convolution_2 : Tensor "f32[128, 64, 112, 112][802816, 12544, 112, 1]npu:0" = PlaceHolder[target=convolution_2]
#   %neg_2 : Tensor "f32[128, 64, 112, 112][802816, 12544, 112, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convolution_2,), kwargs = {})
#   %exp_2 : Tensor "f32[128, 64, 112, 112][802816, 12544, 112, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_2,), kwargs = {})
#   %add_5 : Tensor "f32[128, 64, 112, 112][802816, 12544, 112, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_2, 1), kwargs = {})
#   %div_2 : Tensor "f32[128, 64, 112, 112][802816, 12544, 112, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_2, %add_5), kwargs = {})
#   return %div_2
# SchedulerNodes: [SchedulerNode(name='op23')]

triton_poi_fused_silu_5 = async_compile.triton('triton_poi_fused_silu_5', '''
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
    size_hints={'t0': 128, 'z1': 64, 'y2': 112, 'x3': 112}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 't0_numel': 'i32', 'z1_numel': 'i32', 'y2_numel': 'i32', 'x3_numel': 'i32', 'T0BLOCK': 'i32', 'Z1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_5', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [3], 'axis_names': ['t0', 'z1', 'y2', 'x3'], 'axis_static_values': (('t0', 128), ('z1', 64), ('y2', 112), ('x3', 112)), 'low_dims': {3}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('T0BLOCK', 'Z1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_silu_5(in_ptr0, out_ptr0, t0_numel, z1_numel, y2_numel, x3_numel, T0BLOCK, Z1BLOCK, T0BLOCK_SUB : tl.constexpr, Z1BLOCK_SUB : tl.constexpr, Y2BLOCK_SUB : tl.constexpr):
    x3_numel = 112
    X3BLOCK_SUB: tl.constexpr = 112
    t0_offset = tl.program_id(0) * T0BLOCK
    base_t0= tl.arange(0, T0BLOCK_SUB)
    loops_t0 = (T0BLOCK + T0BLOCK_SUB - 1) // T0BLOCK_SUB
    z1_offset = tl.program_id(1) * Z1BLOCK
    base_z1= tl.arange(0, Z1BLOCK_SUB)
    loops_z1 = (Z1BLOCK + Z1BLOCK_SUB - 1) // Z1BLOCK_SUB
    base_y2= tl.arange(0, Y2BLOCK_SUB)
    loops_y2 = (y2_numel + Y2BLOCK_SUB - 1) // Y2BLOCK_SUB
    base_x3= tl.arange(0, X3BLOCK_SUB)
    for loop_t0 in range(loops_t0):
        t0 = t0_offset + (loop_t0 * T0BLOCK_SUB) + base_t0[:,None,None,None]
        t0_mask = t0 < min(T0BLOCK+t0_offset, t0_numel)
        for loop_z1 in range(loops_z1):
            z1 = z1_offset + (loop_z1 * Z1BLOCK_SUB) + base_z1[None,:,None,None]
            z1_mask = z1 < min(Z1BLOCK+z1_offset, z1_numel)
            for loop_y2 in range(loops_y2):
                y2 = (loop_y2 * Y2BLOCK_SUB) + base_y2[None,None,:,None]
                y2_mask = y2 < y2_numel
                x3 = base_x3[None,None,None,:]
                tmp0 = tl.load(in_ptr0 + (x3 + 112*y2 + 12544*z1 + 802816*t0), t0_mask & y2_mask & z1_mask)
                tmp1 = -tmp0
                tmp2 = libdevice.exp(tmp1)
                tmp3 = tl.full([1, 1, 1, 1], 1.0, tl.float32)
                tmp4 = tmp2 + tmp3
                tmp5 = (tmp0 / tmp4)
                tl.store(out_ptr0 + (x3 + 112*y2 + 12544*z1 + 802816*t0), tmp5, t0_mask & y2_mask & z1_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [reshape_3, mul_3, view_3, batch_norm_3], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_3 => add_6, mul_10, mul_11, rsqrt_3, sub_3, unsqueeze_3, var_mean_3
#   mul_3 => mul_9
#   reshape_3 => view_9
#   view_3 => view_10
# Graph fragment:
#   %primals_11 : Tensor "f32[128, 64, 3, 3][576, 9, 3, 1]npu:0" = PlaceHolder[target=primals_11]
#   %buf24 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=buf24]
#   %getitem_7 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=getitem_7]
#   %buf26 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=buf26]
#   %primals_12 : Tensor "f32[128, 1, 1, 1][1, 1, 1, 1]npu:0" = PlaceHolder[target=primals_12]
#   %view_9 : Tensor "f32[1, 128, 576][73728, 576, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_11, [1, 128, -1]), kwargs = {})
#   %mul_9 : Tensor "f32[128, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_12, 0.07450538873672485), kwargs = {})
#   %view_10 : Tensor "f32[128][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_9, [-1]), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_9, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_6 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %sub_3 : Tensor "f32[1, 128, 576][73728, 576, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_9, %getitem_7), kwargs = {})
#   %mul_10 : Tensor "f32[1, 128, 576][73728, 576, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[128, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_10, -1), kwargs = {})
#   %mul_11 : Tensor "f32[1, 128, 576][73728, 576, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_3), kwargs = {})
#   return %buf24,%getitem_7,%buf26,%mul_11,%rsqrt_3
# SchedulerNodes: [SchedulerNode(name='op24'), SchedulerNode(name='op25'), SchedulerNode(name='op26'), SchedulerNode(name='op28'), SchedulerNode(name='op27')]

triton_per_fused__native_batch_norm_legit_mul__6 = async_compile.triton('triton_per_fused__native_batch_norm_legit_mul__6', '''
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
    size_hints={'x0': 128, 'r1': 576},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_mul__6', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 128), ('r1', 576)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_mul__6(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 576
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
        tmp0 = tl.load(in_ptr0 + (r1 + 576*x0), x0_mask, other=0.0)
        tmp16 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 1).reshape(X0BLOCK_SUB, 1)
        tmp4 = tl.full([1, 1], 576.0, tl.float32)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tmp0 - tmp5
        tmp7 = tmp6 * tmp6
        tmp9 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp7, 0)
        tmp10 = tl.sum(tmp9, 1).reshape(X0BLOCK_SUB, 1)
        tmp11 = (tmp10 / tmp4)
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp6 * tmp14
        tmp17 = tl.full([1, 1], 0.07450538873672485, tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 * tmp18
        tl.store(in_out_ptr0 + (x0), tmp5, x0_mask)
        tl.store(out_ptr1 + (r1 + 576*x0), tmp19, x0_mask)
        tl.store(out_ptr2 + (x0), tmp14, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [silu_3, mul_12], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   mul_12 => clone_default
#   silu_3 => add_7, div_3, exp_3, neg_3
# Graph fragment:
#   %convolution_3 : Tensor "f32[128, 128, 56, 56][401408, 3136, 56, 1]npu:0" = PlaceHolder[target=convolution_3]
#   %neg_3 : Tensor "f32[128, 128, 56, 56][401408, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convolution_3,), kwargs = {})
#   %exp_3 : Tensor "f32[128, 128, 56, 56][401408, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_3,), kwargs = {})
#   %add_7 : Tensor "f32[128, 128, 56, 56][401408, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_3, 1), kwargs = {})
#   %div_3 : Tensor "f32[128, 128, 56, 56][401408, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_3, %add_7), kwargs = {})
#   %clone_default : Tensor "f32[128, 128, 56, 56][401408, 3136, 56, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%div_3,), kwargs = {})
#   return %clone_default
# SchedulerNodes: [SchedulerNode(name='op31')]

triton_poi_fused_silu_7 = async_compile.triton('triton_poi_fused_silu_7', '''
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
    size_hints={'t0': 128, 'z1': 128, 'y2': 56, 'x3': 56}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 't0_numel': 'i32', 'z1_numel': 'i32', 'y2_numel': 'i32', 'x3_numel': 'i32', 'T0BLOCK': 'i32', 'Z1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_7', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [3], 'axis_names': ['t0', 'z1', 'y2', 'x3'], 'axis_static_values': (('t0', 128), ('z1', 128), ('y2', 56), ('x3', 56)), 'low_dims': {3}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('T0BLOCK', 'Z1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_silu_7(in_ptr0, out_ptr0, t0_numel, z1_numel, y2_numel, x3_numel, T0BLOCK, Z1BLOCK, T0BLOCK_SUB : tl.constexpr, Z1BLOCK_SUB : tl.constexpr, Y2BLOCK_SUB : tl.constexpr):
    x3_numel = 56
    X3BLOCK_SUB: tl.constexpr = 56
    t0_offset = tl.program_id(0) * T0BLOCK
    base_t0= tl.arange(0, T0BLOCK_SUB)
    loops_t0 = (T0BLOCK + T0BLOCK_SUB - 1) // T0BLOCK_SUB
    z1_offset = tl.program_id(1) * Z1BLOCK
    base_z1= tl.arange(0, Z1BLOCK_SUB)
    loops_z1 = (Z1BLOCK + Z1BLOCK_SUB - 1) // Z1BLOCK_SUB
    base_y2= tl.arange(0, Y2BLOCK_SUB)
    loops_y2 = (y2_numel + Y2BLOCK_SUB - 1) // Y2BLOCK_SUB
    base_x3= tl.arange(0, X3BLOCK_SUB)
    for loop_t0 in range(loops_t0):
        t0 = t0_offset + (loop_t0 * T0BLOCK_SUB) + base_t0[:,None,None,None]
        t0_mask = t0 < min(T0BLOCK+t0_offset, t0_numel)
        for loop_z1 in range(loops_z1):
            z1 = z1_offset + (loop_z1 * Z1BLOCK_SUB) + base_z1[None,:,None,None]
            z1_mask = z1 < min(Z1BLOCK+z1_offset, z1_numel)
            for loop_y2 in range(loops_y2):
                y2 = (loop_y2 * Y2BLOCK_SUB) + base_y2[None,None,:,None]
                y2_mask = y2 < y2_numel
                x3 = base_x3[None,None,None,:]
                tmp0 = tl.load(in_ptr0 + (x3 + 56*y2 + 3136*z1 + 401408*t0), t0_mask & y2_mask & z1_mask)
                tmp1 = -tmp0
                tmp2 = libdevice.exp(tmp1)
                tmp3 = tl.full([1, 1, 1, 1], 1.0, tl.float32)
                tmp4 = tmp2 + tmp3
                tmp5 = (tmp0 / tmp4)
                tl.store(out_ptr0 + (x3 + 56*y2 + 3136*z1 + 401408*t0), tmp5, t0_mask & y2_mask & z1_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [reshape_4, mul_5, view_4, batch_norm_4], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_4 => add_8, mul_14, mul_15, rsqrt_4, sub_4, unsqueeze_4, var_mean_4
#   mul_5 => mul_13
#   reshape_4 => view_12
#   view_4 => view_13
# Graph fragment:
#   %primals_14 : Tensor "f32[256, 128, 1, 1][128, 1, 1, 1]npu:0" = PlaceHolder[target=primals_14]
#   %buf32 : Tensor "f32[1, 256, 1][256, 1, 256]npu:0" = PlaceHolder[target=buf32]
#   %getitem_9 : Tensor "f32[1, 256, 1][256, 1, 256]npu:0" = PlaceHolder[target=getitem_9]
#   %buf34 : Tensor "f32[1, 256, 1][256, 1, 256]npu:0" = PlaceHolder[target=buf34]
#   %primals_15 : Tensor "f32[256, 1, 1, 1][1, 1, 1, 1]npu:0" = PlaceHolder[target=primals_15]
#   %view_12 : Tensor "f32[1, 256, 128][32768, 128, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_14, [1, 256, -1]), kwargs = {})
#   %mul_13 : Tensor "f32[256, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_15, 0.1580497968320339), kwargs = {})
#   %view_13 : Tensor "f32[256][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_13, [-1]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_12, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_8 : Tensor "f32[1, 256, 1][256, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_4 : Tensor "f32[1, 256, 1][256, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %sub_4 : Tensor "f32[1, 256, 128][32768, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_12, %getitem_9), kwargs = {})
#   %mul_14 : Tensor "f32[1, 256, 128][32768, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[256, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_13, -1), kwargs = {})
#   %mul_15 : Tensor "f32[1, 256, 128][32768, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %unsqueeze_4), kwargs = {})
#   return %buf32,%getitem_9,%buf34,%mul_15,%rsqrt_4
# SchedulerNodes: [SchedulerNode(name='op32'), SchedulerNode(name='op33'), SchedulerNode(name='op34'), SchedulerNode(name='op36'), SchedulerNode(name='op35')]

triton_per_fused__native_batch_norm_legit_mul__8 = async_compile.triton('triton_per_fused__native_batch_norm_legit_mul__8', '''
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
    size_hints={'x0': 256, 'r1': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_mul__8', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 256), ('r1', 128)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_mul__8(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 128
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
        tmp0 = tl.load(in_ptr0 + (r1 + 128*x0), x0_mask, other=0.0)
        tmp16 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 1).reshape(X0BLOCK_SUB, 1)
        tmp4 = tl.full([1, 1], 128.0, tl.float32)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tmp0 - tmp5
        tmp7 = tmp6 * tmp6
        tmp9 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp7, 0)
        tmp10 = tl.sum(tmp9, 1).reshape(X0BLOCK_SUB, 1)
        tmp11 = (tmp10 / tmp4)
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp6 * tmp14
        tmp17 = tl.full([1, 1], 0.1580497968320339, tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 * tmp18
        tl.store(in_out_ptr0 + (x0), tmp5, x0_mask)
        tl.store(out_ptr1 + (r1 + 128*x0), tmp19, x0_mask)
        tl.store(out_ptr2 + (x0), tmp14, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [reshape_5, mul_6, view_5, batch_norm_5], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_5 => add_9, mul_17, mul_18, rsqrt_5, sub_5, unsqueeze_5, var_mean_5
#   mul_6 => mul_16
#   reshape_5 => view_15
#   view_5 => view_16
# Graph fragment:
#   %primals_17 : Tensor "f32[64, 128, 1, 1][128, 1, 1, 1]npu:0" = PlaceHolder[target=primals_17]
#   %buf39 : Tensor "f32[1, 64, 1][64, 1, 64]npu:0" = PlaceHolder[target=buf39]
#   %getitem_11 : Tensor "f32[1, 64, 1][64, 1, 64]npu:0" = PlaceHolder[target=getitem_11]
#   %buf41 : Tensor "f32[1, 64, 1][64, 1, 64]npu:0" = PlaceHolder[target=buf41]
#   %primals_18 : Tensor "f32[64, 1, 1, 1][1, 1, 1, 1]npu:0" = PlaceHolder[target=primals_18]
#   %view_15 : Tensor "f32[1, 64, 128][8192, 128, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_17, [1, 64, -1]), kwargs = {})
#   %mul_16 : Tensor "f32[64, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_18, 0.1580497968320339), kwargs = {})
#   %view_16 : Tensor "f32[64][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_16, [-1]), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_15, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_9 : Tensor "f32[1, 64, 1][64, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_5 : Tensor "f32[1, 64, 1][64, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %sub_5 : Tensor "f32[1, 64, 128][8192, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_15, %getitem_11), kwargs = {})
#   %mul_17 : Tensor "f32[1, 64, 128][8192, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_5), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[64, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_16, -1), kwargs = {})
#   %mul_18 : Tensor "f32[1, 64, 128][8192, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %unsqueeze_5), kwargs = {})
#   return %buf39,%getitem_11,%buf41,%mul_18,%rsqrt_5
# SchedulerNodes: [SchedulerNode(name='op39'), SchedulerNode(name='op40'), SchedulerNode(name='op41'), SchedulerNode(name='op43'), SchedulerNode(name='op42')]

triton_per_fused__native_batch_norm_legit_mul__9 = async_compile.triton('triton_per_fused__native_batch_norm_legit_mul__9', '''
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
    size_hints={'x0': 64, 'r1': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_mul__9', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 64), ('r1', 128)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_mul__9(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 128
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
        tmp0 = tl.load(in_ptr0 + (r1 + 128*x0), x0_mask, other=0.0)
        tmp16 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 1).reshape(X0BLOCK_SUB, 1)
        tmp4 = tl.full([1, 1], 128.0, tl.float32)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tmp0 - tmp5
        tmp7 = tmp6 * tmp6
        tmp9 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp7, 0)
        tmp10 = tl.sum(tmp9, 1).reshape(X0BLOCK_SUB, 1)
        tmp11 = (tmp10 / tmp4)
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp6 * tmp14
        tmp17 = tl.full([1, 1], 0.1580497968320339, tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 * tmp18
        tl.store(in_out_ptr0 + (x0), tmp5, x0_mask)
        tl.store(out_ptr1 + (r1 + 128*x0), tmp19, x0_mask)
        tl.store(out_ptr2 + (x0), tmp14, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [silu_4], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   silu_4 => add_10, div_4, exp_4, neg_4
# Graph fragment:
#   %convolution_5 : Tensor "f32[128, 64, 56, 56][200704, 3136, 56, 1]npu:0" = PlaceHolder[target=convolution_5]
#   %neg_4 : Tensor "f32[128, 64, 56, 56][200704, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convolution_5,), kwargs = {})
#   %exp_4 : Tensor "f32[128, 64, 56, 56][200704, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_4,), kwargs = {})
#   %add_10 : Tensor "f32[128, 64, 56, 56][200704, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_4, 1), kwargs = {})
#   %div_4 : Tensor "f32[128, 64, 56, 56][200704, 3136, 56, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_5, %add_10), kwargs = {})
#   return %div_4
# SchedulerNodes: [SchedulerNode(name='op46')]

triton_poi_fused_silu_10 = async_compile.triton('triton_poi_fused_silu_10', '''
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
    size_hints={'t0': 128, 'z1': 64, 'y2': 56, 'x3': 56}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 't0_numel': 'i32', 'z1_numel': 'i32', 'y2_numel': 'i32', 'x3_numel': 'i32', 'T0BLOCK': 'i32', 'Z1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_10', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [3], 'axis_names': ['t0', 'z1', 'y2', 'x3'], 'axis_static_values': (('t0', 128), ('z1', 64), ('y2', 56), ('x3', 56)), 'low_dims': {3}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('T0BLOCK', 'Z1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_silu_10(in_ptr0, out_ptr0, t0_numel, z1_numel, y2_numel, x3_numel, T0BLOCK, Z1BLOCK, T0BLOCK_SUB : tl.constexpr, Z1BLOCK_SUB : tl.constexpr, Y2BLOCK_SUB : tl.constexpr):
    x3_numel = 56
    X3BLOCK_SUB: tl.constexpr = 56
    t0_offset = tl.program_id(0) * T0BLOCK
    base_t0= tl.arange(0, T0BLOCK_SUB)
    loops_t0 = (T0BLOCK + T0BLOCK_SUB - 1) // T0BLOCK_SUB
    z1_offset = tl.program_id(1) * Z1BLOCK
    base_z1= tl.arange(0, Z1BLOCK_SUB)
    loops_z1 = (Z1BLOCK + Z1BLOCK_SUB - 1) // Z1BLOCK_SUB
    base_y2= tl.arange(0, Y2BLOCK_SUB)
    loops_y2 = (y2_numel + Y2BLOCK_SUB - 1) // Y2BLOCK_SUB
    base_x3= tl.arange(0, X3BLOCK_SUB)
    for loop_t0 in range(loops_t0):
        t0 = t0_offset + (loop_t0 * T0BLOCK_SUB) + base_t0[:,None,None,None]
        t0_mask = t0 < min(T0BLOCK+t0_offset, t0_numel)
        for loop_z1 in range(loops_z1):
            z1 = z1_offset + (loop_z1 * Z1BLOCK_SUB) + base_z1[None,:,None,None]
            z1_mask = z1 < min(Z1BLOCK+z1_offset, z1_numel)
            for loop_y2 in range(loops_y2):
                y2 = (loop_y2 * Y2BLOCK_SUB) + base_y2[None,None,:,None]
                y2_mask = y2 < y2_numel
                x3 = base_x3[None,None,None,:]
                tmp0 = tl.load(in_ptr0 + (x3 + 56*y2 + 3136*z1 + 200704*t0), t0_mask & y2_mask & z1_mask)
                tmp1 = -tmp0
                tmp2 = libdevice.exp(tmp1)
                tmp3 = tl.full([1, 1, 1, 1], 1.0, tl.float32)
                tmp4 = tmp2 + tmp3
                tmp5 = (tmp0 / tmp4)
                tl.store(out_ptr0 + (x3 + 56*y2 + 3136*z1 + 200704*t0), tmp5, t0_mask & y2_mask & z1_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [reshape_6, mul_7, view_6, batch_norm_6], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_6 => add_11, mul_20, mul_21, rsqrt_6, sub_6, unsqueeze_6, var_mean_6
#   mul_7 => mul_19
#   reshape_6 => view_18
#   view_6 => view_19
# Graph fragment:
#   %primals_20 : Tensor "f32[64, 64, 3, 3][576, 9, 3, 1]npu:0" = PlaceHolder[target=primals_20]
#   %buf47 : Tensor "f32[1, 64, 1][64, 1, 64]npu:0" = PlaceHolder[target=buf47]
#   %getitem_13 : Tensor "f32[1, 64, 1][64, 1, 64]npu:0" = PlaceHolder[target=getitem_13]
#   %buf49 : Tensor "f32[1, 64, 1][64, 1, 64]npu:0" = PlaceHolder[target=buf49]
#   %primals_21 : Tensor "f32[64, 1, 1, 1][1, 1, 1, 1]npu:0" = PlaceHolder[target=primals_21]
#   %view_18 : Tensor "f32[1, 64, 576][36864, 576, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_20, [1, 64, -1]), kwargs = {})
#   %mul_19 : Tensor "f32[64, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_21, 0.07450538873672485), kwargs = {})
#   %view_19 : Tensor "f32[64][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_19, [-1]), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_18, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_11 : Tensor "f32[1, 64, 1][64, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-05), kwargs = {})
#   %rsqrt_6 : Tensor "f32[1, 64, 1][64, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %sub_6 : Tensor "f32[1, 64, 576][36864, 576, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_18, %getitem_13), kwargs = {})
#   %mul_20 : Tensor "f32[1, 64, 576][36864, 576, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_6), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[64, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_19, -1), kwargs = {})
#   %mul_21 : Tensor "f32[1, 64, 576][36864, 576, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %unsqueeze_6), kwargs = {})
#   return %buf47,%getitem_13,%buf49,%mul_21,%rsqrt_6
# SchedulerNodes: [SchedulerNode(name='op47'), SchedulerNode(name='op48'), SchedulerNode(name='op49'), SchedulerNode(name='op51'), SchedulerNode(name='op50')]

triton_per_fused__native_batch_norm_legit_mul__11 = async_compile.triton('triton_per_fused__native_batch_norm_legit_mul__11', '''
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
    size_hints={'x0': 64, 'r1': 576},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_mul__11', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 64), ('r1', 576)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_mul__11(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 576
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
        tmp0 = tl.load(in_ptr0 + (r1 + 576*x0), x0_mask, other=0.0)
        tmp16 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 1).reshape(X0BLOCK_SUB, 1)
        tmp4 = tl.full([1, 1], 576.0, tl.float32)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tmp0 - tmp5
        tmp7 = tmp6 * tmp6
        tmp9 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp7, 0)
        tmp10 = tl.sum(tmp9, 1).reshape(X0BLOCK_SUB, 1)
        tmp11 = (tmp10 / tmp4)
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp6 * tmp14
        tmp17 = tl.full([1, 1], 0.07450538873672485, tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 * tmp18
        tl.store(in_out_ptr0 + (x0), tmp5, x0_mask)
        tl.store(out_ptr1 + (r1 + 576*x0), tmp19, x0_mask)
        tl.store(out_ptr2 + (x0), tmp14, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [reshape_8, mul_9, view_8, batch_norm_8], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_8 => add_15, mul_26, mul_27, rsqrt_8, sub_8, unsqueeze_8, var_mean_8
#   mul_9 => mul_25
#   reshape_8 => view_24
#   view_8 => view_25
# Graph fragment:
#   %primals_26 : Tensor "f32[256, 64, 1, 1][64, 1, 1, 1]npu:0" = PlaceHolder[target=primals_26]
#   %buf63 : Tensor "f32[1, 256, 1][256, 1, 256]npu:0" = PlaceHolder[target=buf63]
#   %getitem_17 : Tensor "f32[1, 256, 1][256, 1, 256]npu:0" = PlaceHolder[target=getitem_17]
#   %buf65 : Tensor "f32[1, 256, 1][256, 1, 256]npu:0" = PlaceHolder[target=buf65]
#   %primals_27 : Tensor "f32[256, 1, 1, 1][1, 1, 1, 1]npu:0" = PlaceHolder[target=primals_27]
#   %view_24 : Tensor "f32[1, 256, 64][16384, 64, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_26, [1, 256, -1]), kwargs = {})
#   %mul_25 : Tensor "f32[256, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_27, 0.22351616621017456), kwargs = {})
#   %view_25 : Tensor "f32[256][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_25, [-1]), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_24, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_15 : Tensor "f32[1, 256, 1][256, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_8 : Tensor "f32[1, 256, 1][256, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_15,), kwargs = {})
#   %sub_8 : Tensor "f32[1, 256, 64][16384, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_24, %getitem_17), kwargs = {})
#   %mul_26 : Tensor "f32[1, 256, 64][16384, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_8), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[256, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_25, -1), kwargs = {})
#   %mul_27 : Tensor "f32[1, 256, 64][16384, 64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, %unsqueeze_8), kwargs = {})
#   return %buf63,%getitem_17,%buf65,%mul_27,%rsqrt_8
# SchedulerNodes: [SchedulerNode(name='op63'), SchedulerNode(name='op64'), SchedulerNode(name='op65'), SchedulerNode(name='op67'), SchedulerNode(name='op66')]

triton_per_fused__native_batch_norm_legit_mul__12 = async_compile.triton('triton_per_fused__native_batch_norm_legit_mul__12', '''
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
    size_hints={'x0': 256, 'r1': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_mul__12', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 256), ('r1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_mul__12(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 64
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
        tmp0 = tl.load(in_ptr0 + (r1 + 64*x0), x0_mask, other=0.0)
        tmp16 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 1).reshape(X0BLOCK_SUB, 1)
        tmp4 = tl.full([1, 1], 64.0, tl.float32)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tmp0 - tmp5
        tmp7 = tmp6 * tmp6
        tmp9 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp7, 0)
        tmp10 = tl.sum(tmp9, 1).reshape(X0BLOCK_SUB, 1)
        tmp11 = (tmp10 / tmp4)
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp6 * tmp14
        tmp17 = tl.full([1, 1], 0.22351616621017456, tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 * tmp18
        tl.store(in_out_ptr0 + (x0), tmp5, x0_mask)
        tl.store(out_ptr1 + (r1 + 64*x0), tmp19, x0_mask)
        tl.store(out_ptr2 + (x0), tmp14, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [x_se], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se => mean
# Graph fragment:
#   %convolution_8 : Tensor "f32[128, 256, 56, 56][802816, 3136, 56, 1]npu:0" = PlaceHolder[target=convolution_8]
#   %buf70 : Tensor "f32[128, 256, 1, 1][256, 1, 32768, 32768]npu:0" = PlaceHolder[target=buf70]
#   %mean : Tensor "f32[128, 256, 1, 1][256, 1, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_8, [2, 3], True), kwargs = {})
#   return %buf70,%mean
# SchedulerNodes: [SchedulerNode(name='op70'), SchedulerNode(name='op71')]

triton_per_fused_mean_13 = async_compile.triton('triton_per_fused_mean_13', '''
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
    size_hints={'y0': 128, 'x1': 256, 'r2': 3136},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'r2_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_13', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['y0', 'x1', 'r2'], 'axis_static_values': (('y0', 128), ('x1', 256), ('r2', 3136)), 'low_dims': {2}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_mean_13(in_out_ptr0, in_ptr0, y0_numel, x1_numel, r2_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    R2BLOCK_SUB: tl.constexpr = 3136
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    x1_offset = tl.program_id(1) * X1BLOCK
    base_x1= tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (X1BLOCK + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    base_r2= tl.arange(0, R2BLOCK_SUB)
    loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        for loop_x1 in range(loops_x1):
            x1 = x1_offset + (loop_x1 * X1BLOCK_SUB) + base_x1[None,:,None]
            x1_mask = x1 < min(X1BLOCK+x1_offset, x1_numel)
            r2 = base_r2[None,None,:]
            r2_mask = True
            tmp0 = tl.load(in_ptr0 + (r2 + 3136*x1 + 802816*y0), x1_mask & y0_mask, other=0.0)
            tmp2 = tl.where(tl.broadcast_to((y0_mask.reshape([Y0BLOCK_SUB, 1, 1]) & x1_mask.reshape([1, X1BLOCK_SUB, 1])), [Y0BLOCK_SUB, X1BLOCK_SUB, R2BLOCK_SUB]), tmp0, 0)
            tmp3 = tl.sum(tmp2, 2).reshape(Y0BLOCK_SUB, X1BLOCK_SUB, 1)
            tmp4 = tl.full([1, 1, 1], 3136.0, tl.float32)
            tmp5 = (tmp3 / tmp4)
            tl.store(in_out_ptr0 + (x1 + 256*y0), tmp5, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [x_se_2], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   x_se_2 => relu
# Graph fragment:
#   %convolution_9 : Tensor "f32[128, 64, 1, 1][64, 1, 1, 1]npu:0" = PlaceHolder[target=convolution_9]
#   %relu : Tensor "f32[128, 64, 1, 1][64, 1, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_9,), kwargs = {})
#   return %relu
# SchedulerNodes: [SchedulerNode(name='op74')]

triton_poi_fused_relu_14 = async_compile.triton('triton_poi_fused_relu_14', '''
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
    size_hints={'y0': 128, 'x1': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 128), ('x1', 64)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_14(in_out_ptr0, y0_numel, x1_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
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
            tmp0 = tl.load(in_out_ptr0 + (x1 + 64*y0), x1_mask & y0_mask)
            tmp1 = tl.full([1, 1], 0, tl.int32)
            tmp2 = triton_helpers.maximum(tmp1, tmp0)
            tl.store(in_out_ptr0 + (x1 + 64*y0), tmp2, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [sigmoid, mul_10, out_5, mul_12, out_6, silu_7, out_7], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
# Source node to ATen node mapping:
#   mul_10 => mul_28
#   mul_12 => mul_30
#   out_5 => mul_29
#   out_6 => add_16
#   out_7 => mul_31
#   sigmoid => sigmoid
#   silu_7 => add_17, div_7, exp_7, neg_7
# Graph fragment:
#   %convolution_8 : Tensor "f32[128, 256, 56, 56][802816, 3136, 56, 1]npu:0" = PlaceHolder[target=convolution_8]
#   %convolution_10 : Tensor "f32[128, 256, 1, 1][256, 1, 1, 1]npu:0" = PlaceHolder[target=convolution_10]
#   %convolution_4 : Tensor "f32[128, 256, 56, 56][802816, 3136, 56, 1]npu:0" = PlaceHolder[target=convolution_4]
#   %sigmoid : Tensor "f32[128, 256, 1, 1][256, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_10,), kwargs = {})
#   %mul_28 : Tensor "f32[128, 256, 56, 56][802816, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_8, %sigmoid), kwargs = {})
#   %mul_29 : Tensor "f32[128, 256, 56, 56][802816, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, 2.0), kwargs = {})
#   %mul_30 : Tensor "f32[128, 256, 56, 56][802816, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, 0.2), kwargs = {})
#   %add_16 : Tensor "f32[128, 256, 56, 56][802816, 3136, 56, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %convolution_4), kwargs = {})
#   %neg_7 : Tensor "f32[128, 256, 56, 56][802816, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_16,), kwargs = {})
#   %exp_7 : Tensor "f32[128, 256, 56, 56][802816, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_7,), kwargs = {})
#   %add_17 : Tensor "f32[128, 256, 56, 56][802816, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_7, 1), kwargs = {})
#   %div_7 : Tensor "f32[128, 256, 56, 56][802816, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_16, %add_17), kwargs = {})
#   %mul_31 : Tensor "f32[128, 256, 56, 56][802816, 3136, 56, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_7, 0.9805806756909201), kwargs = {})
#   return %mul_31
# SchedulerNodes: [SchedulerNode(name='op77')]

triton_poi_fused_add_mul_sigmoid_silu_15 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_silu_15', '''
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
    size_hints={'y0': 32768, 'x1': 3136}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_silu_15', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 32768), ('x1', 3136)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_silu_15(in_ptr0, in_ptr1, in_ptr2, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (x1_numel + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        for loop_x1 in range(loops_x1):
            x1 = (loop_x1 * X1BLOCK_SUB) + base_x1[None,:]
            x1_mask = x1 < x1_numel
            tmp0 = tl.load(in_ptr0 + (x1 + 3136*y0), x1_mask & y0_mask)
            tmp1 = tl.load(in_ptr1 + (y0), y0_mask)
            tmp8 = tl.load(in_ptr2 + (x1 + 3136*y0), x1_mask & y0_mask)
            tmp2 = tl.sigmoid(tmp1)
            tmp3 = tmp0 * tmp2
            tmp4 = tl.full([1, 1], 2.0, tl.float32)
            tmp5 = tmp3 * tmp4
            tmp6 = tl.full([1, 1], 0.2, tl.float32)
            tmp7 = tmp5 * tmp6
            tmp9 = tmp7 + tmp8
            tmp10 = -tmp9
            tmp11 = libdevice.exp(tmp10)
            tmp12 = tl.full([1, 1], 1.0, tl.float32)
            tmp13 = tmp11 + tmp12
            tmp14 = (tmp9 / tmp13)
            tmp15 = tl.full([1, 1], 0.9805806756909201, tl.float32)
            tmp16 = tmp14 * tmp15
            tl.store(out_ptr0 + (x1 + 3136*y0), tmp16, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [avg_pool2d], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d => avg_pool2d
# Graph fragment:
#   %mul_31 : Tensor "f32[128, 256, 56, 56][802816, 3136, 56, 1]npu:0" = PlaceHolder[target=mul_31]
#   %avg_pool2d : Tensor "f32[128, 256, 28, 28][200704, 784, 28, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%mul_31, [2, 2], [2, 2], [0, 0], True, False), kwargs = {})
#   return %avg_pool2d
# SchedulerNodes: [SchedulerNode(name='op78')]

triton_poi_fused_avg_pool2d_16 = async_compile.triton('triton_poi_fused_avg_pool2d_16', '''
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
    size_hints={'y0': 917504, 'x1': 28}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_16', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 917504), ('x1', 28)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_16(in_ptr0, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
    x1_numel = 28
    X1BLOCK_SUB: tl.constexpr = 28
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        x1 = base_x1[None,:]
        tmp0 = tl.load(in_ptr0 + (2*x1 + 112*y0), y0_mask)
        tmp1 = tl.load(in_ptr0 + (1 + 2*x1 + 112*y0), y0_mask)
        tmp3 = tl.load(in_ptr0 + (56 + 2*x1 + 112*y0), y0_mask)
        tmp5 = tl.load(in_ptr0 + (57 + 2*x1 + 112*y0), y0_mask)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tl.full([1, 1], 4.0, tl.float32)
        tmp8 = (tmp6 / tmp7)
        tl.store(out_ptr0 + (x1 + 28*y0), tmp8, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [reshape_9, mul_14, view_9, batch_norm_9], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_9 => add_18, mul_33, mul_34, rsqrt_9, sub_9, unsqueeze_9, var_mean_9
#   mul_14 => mul_32
#   reshape_9 => view_27
#   view_9 => view_28
# Graph fragment:
#   %primals_33 : Tensor "f32[512, 256, 1, 1][256, 1, 1, 1]npu:0" = PlaceHolder[target=primals_33]
#   %buf79 : Tensor "f32[1, 512, 1][512, 1, 512]npu:0" = PlaceHolder[target=buf79]
#   %getitem_19 : Tensor "f32[1, 512, 1][512, 1, 512]npu:0" = PlaceHolder[target=getitem_19]
#   %buf81 : Tensor "f32[1, 512, 1][512, 1, 512]npu:0" = PlaceHolder[target=buf81]
#   %primals_34 : Tensor "f32[512, 1, 1, 1][1, 1, 1, 1]npu:0" = PlaceHolder[target=primals_34]
#   %view_27 : Tensor "f32[1, 512, 256][131072, 256, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_33, [1, 512, -1]), kwargs = {})
#   %mul_32 : Tensor "f32[512, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_34, 0.11175808310508728), kwargs = {})
#   %view_28 : Tensor "f32[512][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_32, [-1]), kwargs = {})
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_27, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_18 : Tensor "f32[1, 512, 1][512, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_18, 1e-05), kwargs = {})
#   %rsqrt_9 : Tensor "f32[1, 512, 1][512, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_18,), kwargs = {})
#   %sub_9 : Tensor "f32[1, 512, 256][131072, 256, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_27, %getitem_19), kwargs = {})
#   %mul_33 : Tensor "f32[1, 512, 256][131072, 256, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %rsqrt_9), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_28, -1), kwargs = {})
#   %mul_34 : Tensor "f32[1, 512, 256][131072, 256, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_33, %unsqueeze_9), kwargs = {})
#   return %buf79,%getitem_19,%buf81,%mul_34,%rsqrt_9
# SchedulerNodes: [SchedulerNode(name='op79'), SchedulerNode(name='op80'), SchedulerNode(name='op81'), SchedulerNode(name='op83'), SchedulerNode(name='op82')]

triton_per_fused__native_batch_norm_legit_mul__17 = async_compile.triton('triton_per_fused__native_batch_norm_legit_mul__17', '''
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
    size_hints={'x0': 512, 'r1': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_mul__17', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 512), ('r1', 256)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_mul__17(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 256
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
        tmp0 = tl.load(in_ptr0 + (r1 + 256*x0), x0_mask, other=0.0)
        tmp16 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 1).reshape(X0BLOCK_SUB, 1)
        tmp4 = tl.full([1, 1], 256.0, tl.float32)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tmp0 - tmp5
        tmp7 = tmp6 * tmp6
        tmp9 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp7, 0)
        tmp10 = tl.sum(tmp9, 1).reshape(X0BLOCK_SUB, 1)
        tmp11 = (tmp10 / tmp4)
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp6 * tmp14
        tmp17 = tl.full([1, 1], 0.11175808310508728, tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 * tmp18
        tl.store(in_out_ptr0 + (x0), tmp5, x0_mask)
        tl.store(out_ptr1 + (r1 + 256*x0), tmp19, x0_mask)
        tl.store(out_ptr2 + (x0), tmp14, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [reshape_10, mul_15, view_10, batch_norm_10], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_10 => add_19, mul_36, mul_37, rsqrt_10, sub_10, unsqueeze_10, var_mean_10
#   mul_15 => mul_35
#   reshape_10 => view_30
#   view_10 => view_31
# Graph fragment:
#   %primals_36 : Tensor "f32[128, 256, 1, 1][256, 1, 1, 1]npu:0" = PlaceHolder[target=primals_36]
#   %buf86 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=buf86]
#   %getitem_21 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=getitem_21]
#   %buf88 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=buf88]
#   %primals_37 : Tensor "f32[128, 1, 1, 1][1, 1, 1, 1]npu:0" = PlaceHolder[target=primals_37]
#   %view_30 : Tensor "f32[1, 128, 256][32768, 256, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_36, [1, 128, -1]), kwargs = {})
#   %mul_35 : Tensor "f32[128, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_37, 0.11175808310508728), kwargs = {})
#   %view_31 : Tensor "f32[128][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_35, [-1]), kwargs = {})
#   %var_mean_10 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_30, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_19 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_20, 1e-05), kwargs = {})
#   %rsqrt_10 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %sub_10 : Tensor "f32[1, 128, 256][32768, 256, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_30, %getitem_21), kwargs = {})
#   %mul_36 : Tensor "f32[1, 128, 256][32768, 256, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %rsqrt_10), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[128, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_31, -1), kwargs = {})
#   %mul_37 : Tensor "f32[1, 128, 256][32768, 256, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_36, %unsqueeze_10), kwargs = {})
#   return %buf86,%getitem_21,%buf88,%mul_37,%rsqrt_10
# SchedulerNodes: [SchedulerNode(name='op86'), SchedulerNode(name='op87'), SchedulerNode(name='op88'), SchedulerNode(name='op90'), SchedulerNode(name='op89')]

triton_per_fused__native_batch_norm_legit_mul__18 = async_compile.triton('triton_per_fused__native_batch_norm_legit_mul__18', '''
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
    size_hints={'x0': 128, 'r1': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_mul__18', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 128), ('r1', 256)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_mul__18(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 256
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
        tmp0 = tl.load(in_ptr0 + (r1 + 256*x0), x0_mask, other=0.0)
        tmp16 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 1).reshape(X0BLOCK_SUB, 1)
        tmp4 = tl.full([1, 1], 256.0, tl.float32)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tmp0 - tmp5
        tmp7 = tmp6 * tmp6
        tmp9 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp7, 0)
        tmp10 = tl.sum(tmp9, 1).reshape(X0BLOCK_SUB, 1)
        tmp11 = (tmp10 / tmp4)
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp6 * tmp14
        tmp17 = tl.full([1, 1], 0.11175808310508728, tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 * tmp18
        tl.store(in_out_ptr0 + (x0), tmp5, x0_mask)
        tl.store(out_ptr1 + (r1 + 256*x0), tmp19, x0_mask)
        tl.store(out_ptr2 + (x0), tmp14, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [silu_9], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   silu_9 => add_22, div_9, exp_9, neg_9
# Graph fragment:
#   %convolution_13 : Tensor "f32[128, 128, 28, 28][100352, 784, 28, 1]npu:0" = PlaceHolder[target=convolution_13]
#   %neg_9 : Tensor "f32[128, 128, 28, 28][100352, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convolution_13,), kwargs = {})
#   %exp_9 : Tensor "f32[128, 128, 28, 28][100352, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_9,), kwargs = {})
#   %add_22 : Tensor "f32[128, 128, 28, 28][100352, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_9, 1), kwargs = {})
#   %div_9 : Tensor "f32[128, 128, 28, 28][100352, 784, 28, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_13, %add_22), kwargs = {})
#   return %div_9
# SchedulerNodes: [SchedulerNode(name='op101')]

triton_poi_fused_silu_19 = async_compile.triton('triton_poi_fused_silu_19', '''
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
    size_hints={'t0': 128, 'z1': 128, 'y2': 28, 'x3': 28}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 't0_numel': 'i32', 'z1_numel': 'i32', 'y2_numel': 'i32', 'x3_numel': 'i32', 'T0BLOCK': 'i32', 'Z1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_19', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [2, 3], 'axis_names': ['t0', 'z1', 'y2', 'x3'], 'axis_static_values': (('t0', 128), ('z1', 128), ('y2', 28), ('x3', 28)), 'low_dims': {3}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('T0BLOCK', 'Z1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_silu_19(in_ptr0, out_ptr0, t0_numel, z1_numel, y2_numel, x3_numel, T0BLOCK, Z1BLOCK, T0BLOCK_SUB : tl.constexpr, Z1BLOCK_SUB : tl.constexpr):
    y2_numel = 28
    Y2BLOCK_SUB: tl.constexpr = 28
    x3_numel = 28
    X3BLOCK_SUB: tl.constexpr = 28
    t0_offset = tl.program_id(0) * T0BLOCK
    base_t0= tl.arange(0, T0BLOCK_SUB)
    loops_t0 = (T0BLOCK + T0BLOCK_SUB - 1) // T0BLOCK_SUB
    z1_offset = tl.program_id(1) * Z1BLOCK
    base_z1= tl.arange(0, Z1BLOCK_SUB)
    loops_z1 = (Z1BLOCK + Z1BLOCK_SUB - 1) // Z1BLOCK_SUB
    base_y2= tl.arange(0, Y2BLOCK_SUB)
    base_x3= tl.arange(0, X3BLOCK_SUB)
    for loop_t0 in range(loops_t0):
        t0 = t0_offset + (loop_t0 * T0BLOCK_SUB) + base_t0[:,None,None,None]
        t0_mask = t0 < min(T0BLOCK+t0_offset, t0_numel)
        for loop_z1 in range(loops_z1):
            z1 = z1_offset + (loop_z1 * Z1BLOCK_SUB) + base_z1[None,:,None,None]
            z1_mask = z1 < min(Z1BLOCK+z1_offset, z1_numel)
            y2 = base_y2[None,None,:,None]
            x3 = base_x3[None,None,None,:]
            tmp0 = tl.load(in_ptr0 + (x3 + 28*y2 + 784*z1 + 100352*t0), t0_mask & z1_mask)
            tmp1 = -tmp0
            tmp2 = libdevice.exp(tmp1)
            tmp3 = tl.full([1, 1, 1, 1], 1.0, tl.float32)
            tmp4 = tmp2 + tmp3
            tmp5 = (tmp0 / tmp4)
            tl.store(out_ptr0 + (x3 + 28*y2 + 784*z1 + 100352*t0), tmp5, t0_mask & z1_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [reshape_13, mul_18, view_13, batch_norm_13], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_13 => add_25, mul_45, mul_46, rsqrt_13, sub_13, unsqueeze_13, var_mean_13
#   mul_18 => mul_44
#   reshape_13 => view_39
#   view_13 => view_40
# Graph fragment:
#   %primals_45 : Tensor "f32[512, 128, 1, 1][128, 1, 1, 1]npu:0" = PlaceHolder[target=primals_45]
#   %buf110 : Tensor "f32[1, 512, 1][512, 1, 512]npu:0" = PlaceHolder[target=buf110]
#   %getitem_27 : Tensor "f32[1, 512, 1][512, 1, 512]npu:0" = PlaceHolder[target=getitem_27]
#   %buf112 : Tensor "f32[1, 512, 1][512, 1, 512]npu:0" = PlaceHolder[target=buf112]
#   %primals_46 : Tensor "f32[512, 1, 1, 1][1, 1, 1, 1]npu:0" = PlaceHolder[target=primals_46]
#   %view_39 : Tensor "f32[1, 512, 128][65536, 128, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_45, [1, 512, -1]), kwargs = {})
#   %mul_44 : Tensor "f32[512, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_46, 0.1580497968320339), kwargs = {})
#   %view_40 : Tensor "f32[512][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_44, [-1]), kwargs = {})
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_39, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_25 : Tensor "f32[1, 512, 1][512, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-05), kwargs = {})
#   %rsqrt_13 : Tensor "f32[1, 512, 1][512, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_25,), kwargs = {})
#   %sub_13 : Tensor "f32[1, 512, 128][65536, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_39, %getitem_27), kwargs = {})
#   %mul_45 : Tensor "f32[1, 512, 128][65536, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %rsqrt_13), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_40, -1), kwargs = {})
#   %mul_46 : Tensor "f32[1, 512, 128][65536, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_45, %unsqueeze_13), kwargs = {})
#   return %buf110,%getitem_27,%buf112,%mul_46,%rsqrt_13
# SchedulerNodes: [SchedulerNode(name='op110'), SchedulerNode(name='op111'), SchedulerNode(name='op112'), SchedulerNode(name='op114'), SchedulerNode(name='op113')]

triton_per_fused__native_batch_norm_legit_mul__20 = async_compile.triton('triton_per_fused__native_batch_norm_legit_mul__20', '''
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
    size_hints={'x0': 512, 'r1': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_mul__20', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 512), ('r1', 128)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_mul__20(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 128
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
        tmp0 = tl.load(in_ptr0 + (r1 + 128*x0), x0_mask, other=0.0)
        tmp16 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 1).reshape(X0BLOCK_SUB, 1)
        tmp4 = tl.full([1, 1], 128.0, tl.float32)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tmp0 - tmp5
        tmp7 = tmp6 * tmp6
        tmp9 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp7, 0)
        tmp10 = tl.sum(tmp9, 1).reshape(X0BLOCK_SUB, 1)
        tmp11 = (tmp10 / tmp4)
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp6 * tmp14
        tmp17 = tl.full([1, 1], 0.1580497968320339, tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 * tmp18
        tl.store(in_out_ptr0 + (x0), tmp5, x0_mask)
        tl.store(out_ptr1 + (r1 + 128*x0), tmp19, x0_mask)
        tl.store(out_ptr2 + (x0), tmp14, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [x_se_4], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_4 => mean_1
# Graph fragment:
#   %convolution_15 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0" = PlaceHolder[target=convolution_15]
#   %buf117 : Tensor "f32[128, 512, 1, 1][512, 1, 65536, 65536]npu:0" = PlaceHolder[target=buf117]
#   %mean_1 : Tensor "f32[128, 512, 1, 1][512, 1, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_15, [2, 3], True), kwargs = {})
#   return %buf117,%mean_1
# SchedulerNodes: [SchedulerNode(name='op117'), SchedulerNode(name='op118')]

triton_per_fused_mean_21 = async_compile.triton('triton_per_fused_mean_21', '''
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
    size_hints={'y0': 128, 'x1': 512, 'r2': 784},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'r2_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_21', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['y0', 'x1', 'r2'], 'axis_static_values': (('y0', 128), ('x1', 512), ('r2', 784)), 'low_dims': {2}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_mean_21(in_out_ptr0, in_ptr0, y0_numel, x1_numel, r2_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    R2BLOCK_SUB: tl.constexpr = 784
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    x1_offset = tl.program_id(1) * X1BLOCK
    base_x1= tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (X1BLOCK + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    base_r2= tl.arange(0, R2BLOCK_SUB)
    loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        for loop_x1 in range(loops_x1):
            x1 = x1_offset + (loop_x1 * X1BLOCK_SUB) + base_x1[None,:,None]
            x1_mask = x1 < min(X1BLOCK+x1_offset, x1_numel)
            r2 = base_r2[None,None,:]
            r2_mask = True
            tmp0 = tl.load(in_ptr0 + (r2 + 784*x1 + 401408*y0), x1_mask & y0_mask, other=0.0)
            tmp2 = tl.where(tl.broadcast_to((y0_mask.reshape([Y0BLOCK_SUB, 1, 1]) & x1_mask.reshape([1, X1BLOCK_SUB, 1])), [Y0BLOCK_SUB, X1BLOCK_SUB, R2BLOCK_SUB]), tmp0, 0)
            tmp3 = tl.sum(tmp2, 2).reshape(Y0BLOCK_SUB, X1BLOCK_SUB, 1)
            tmp4 = tl.full([1, 1, 1], 784.0, tl.float32)
            tmp5 = (tmp3 / tmp4)
            tl.store(in_out_ptr0 + (x1 + 512*y0), tmp5, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [x_se_6], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   x_se_6 => relu_1
# Graph fragment:
#   %convolution_16 : Tensor "f32[128, 128, 1, 1][128, 1, 1, 1]npu:0" = PlaceHolder[target=convolution_16]
#   %relu_1 : Tensor "f32[128, 128, 1, 1][128, 1, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   return %relu_1
# SchedulerNodes: [SchedulerNode(name='op121')]

triton_poi_fused_relu_22 = async_compile.triton('triton_poi_fused_relu_22', '''
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
    size_hints={'y0': 128, 'x1': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_22', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 128), ('x1', 128)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_22(in_out_ptr0, y0_numel, x1_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
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
            tmp0 = tl.load(in_out_ptr0 + (x1 + 128*y0), x1_mask & y0_mask)
            tmp1 = tl.full([1, 1], 0, tl.int32)
            tmp2 = triton_helpers.maximum(tmp1, tmp0)
            tl.store(in_out_ptr0 + (x1 + 128*y0), tmp2, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [sigmoid_1, mul_19, out_12, mul_21, out_13, silu_11, out_14], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
# Source node to ATen node mapping:
#   mul_19 => mul_47
#   mul_21 => mul_49
#   out_12 => mul_48
#   out_13 => add_26
#   out_14 => mul_50
#   sigmoid_1 => sigmoid_1
#   silu_11 => add_27, div_11, exp_11, neg_11
# Graph fragment:
#   %convolution_15 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0" = PlaceHolder[target=convolution_15]
#   %convolution_17 : Tensor "f32[128, 512, 1, 1][512, 1, 1, 1]npu:0" = PlaceHolder[target=convolution_17]
#   %convolution_11 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0" = PlaceHolder[target=convolution_11]
#   %sigmoid_1 : Tensor "f32[128, 512, 1, 1][512, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_17,), kwargs = {})
#   %mul_47 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_15, %sigmoid_1), kwargs = {})
#   %mul_48 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_47, 2.0), kwargs = {})
#   %mul_49 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_48, 0.2), kwargs = {})
#   %add_26 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_49, %convolution_11), kwargs = {})
#   %neg_11 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_26,), kwargs = {})
#   %exp_11 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_11,), kwargs = {})
#   %add_27 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_11, 1), kwargs = {})
#   %div_11 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_26, %add_27), kwargs = {})
#   %mul_50 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_11, 0.9805806756909201), kwargs = {})
#   return %mul_50
# SchedulerNodes: [SchedulerNode(name='op124')]

triton_poi_fused_add_mul_sigmoid_silu_23 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_silu_23', '''
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
    size_hints={'y0': 65536, 'x1': 784}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_silu_23', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 65536), ('x1', 784)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_silu_23(in_ptr0, in_ptr1, in_ptr2, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
    x1_numel = 784
    X1BLOCK_SUB: tl.constexpr = 784
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        x1 = base_x1[None,:]
        tmp0 = tl.load(in_ptr0 + (x1 + 784*y0), y0_mask)
        tmp1 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp8 = tl.load(in_ptr2 + (x1 + 784*y0), y0_mask)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = tl.full([1, 1], 2.0, tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full([1, 1], 0.2, tl.float32)
        tmp7 = tmp5 * tmp6
        tmp9 = tmp7 + tmp8
        tmp10 = -tmp9
        tmp11 = libdevice.exp(tmp10)
        tmp12 = tl.full([1, 1], 1.0, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = (tmp9 / tmp13)
        tmp15 = tl.full([1, 1], 0.9805806756909201, tl.float32)
        tmp16 = tmp14 * tmp15
        tl.store(out_ptr0 + (x1 + 784*y0), tmp16, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [reshape_14, mul_23, view_14, batch_norm_14], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_14 => add_28, mul_52, mul_53, rsqrt_14, sub_14, unsqueeze_14, var_mean_14
#   mul_23 => mul_51
#   reshape_14 => view_42
#   view_14 => view_43
# Graph fragment:
#   %primals_52 : Tensor "f32[128, 512, 1, 1][512, 1, 1, 1]npu:0" = PlaceHolder[target=primals_52]
#   %buf125 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=buf125]
#   %getitem_29 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=getitem_29]
#   %buf127 : Tensor "f32[1, 128, 1][128, 1, 128]npu:0" = PlaceHolder[target=buf127]
#   %primals_53 : Tensor "f32[128, 1, 1, 1][1, 1, 1, 1]npu:0" = PlaceHolder[target=primals_53]
#   %view_42 : Tensor "f32[1, 128, 512][65536, 512, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_52, [1, 128, -1]), kwargs = {})
#   %mul_51 : Tensor "f32[128, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_53, 0.07902489841601695), kwargs = {})
#   %view_43 : Tensor "f32[128][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_51, [-1]), kwargs = {})
#   %var_mean_14 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_42, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_28 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_28, 1e-05), kwargs = {})
#   %rsqrt_14 : Tensor "f32[1, 128, 1][128, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_28,), kwargs = {})
#   %sub_14 : Tensor "f32[1, 128, 512][65536, 512, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_42, %getitem_29), kwargs = {})
#   %mul_52 : Tensor "f32[1, 128, 512][65536, 512, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %rsqrt_14), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[128, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_43, -1), kwargs = {})
#   %mul_53 : Tensor "f32[1, 128, 512][65536, 512, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %unsqueeze_14), kwargs = {})
#   return %buf125,%getitem_29,%buf127,%mul_53,%rsqrt_14
# SchedulerNodes: [SchedulerNode(name='op125'), SchedulerNode(name='op126'), SchedulerNode(name='op127'), SchedulerNode(name='op129'), SchedulerNode(name='op128')]

triton_per_fused__native_batch_norm_legit_mul__24 = async_compile.triton('triton_per_fused__native_batch_norm_legit_mul__24', '''
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
    size_hints={'x0': 128, 'r1': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_mul__24', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 128), ('r1', 512)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_mul__24(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 512
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
        tmp0 = tl.load(in_ptr0 + (r1 + 512*x0), x0_mask, other=0.0)
        tmp16 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 1).reshape(X0BLOCK_SUB, 1)
        tmp4 = tl.full([1, 1], 512.0, tl.float32)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tmp0 - tmp5
        tmp7 = tmp6 * tmp6
        tmp9 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp7, 0)
        tmp10 = tl.sum(tmp9, 1).reshape(X0BLOCK_SUB, 1)
        tmp11 = (tmp10 / tmp4)
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp6 * tmp14
        tmp17 = tl.full([1, 1], 0.07902489841601695, tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 * tmp18
        tl.store(in_out_ptr0 + (x0), tmp5, x0_mask)
        tl.store(out_ptr1 + (r1 + 512*x0), tmp19, x0_mask)
        tl.store(out_ptr2 + (x0), tmp14, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [sigmoid_1, mul_19, out_12, mul_21, out_13, sigmoid_2, mul_27, out_19, mul_29, out_20, silu_15, out_21], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
# Source node to ATen node mapping:
#   mul_19 => mul_47
#   mul_21 => mul_49
#   mul_27 => mul_63
#   mul_29 => mul_65
#   out_12 => mul_48
#   out_13 => add_26
#   out_19 => mul_64
#   out_20 => add_35
#   out_21 => mul_66
#   sigmoid_1 => sigmoid_1
#   sigmoid_2 => sigmoid_2
#   silu_15 => add_36, div_15, exp_15, neg_15
# Graph fragment:
#   %convolution_21 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0" = PlaceHolder[target=convolution_21]
#   %convolution_23 : Tensor "f32[128, 512, 1, 1][512, 1, 1, 1]npu:0" = PlaceHolder[target=convolution_23]
#   %convolution_15 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0" = PlaceHolder[target=convolution_15]
#   %convolution_17 : Tensor "f32[128, 512, 1, 1][512, 1, 1, 1]npu:0" = PlaceHolder[target=convolution_17]
#   %convolution_11 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0" = PlaceHolder[target=convolution_11]
#   %add_35 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0" = PlaceHolder[target=add_35]
#   %sigmoid_1 : Tensor "f32[128, 512, 1, 1][512, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_17,), kwargs = {})
#   %mul_47 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_15, %sigmoid_1), kwargs = {})
#   %mul_48 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_47, 2.0), kwargs = {})
#   %mul_49 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_48, 0.2), kwargs = {})
#   %add_26 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_49, %convolution_11), kwargs = {})
#   %sigmoid_2 : Tensor "f32[128, 512, 1, 1][512, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_23,), kwargs = {})
#   %mul_63 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_21, %sigmoid_2), kwargs = {})
#   %mul_64 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_63, 2.0), kwargs = {})
#   %mul_65 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_64, 0.2), kwargs = {})
#   %add_35 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_65, %add_26), kwargs = {})
#   %neg_15 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_35,), kwargs = {})
#   %exp_15 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_15,), kwargs = {})
#   %add_36 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_15, 1), kwargs = {})
#   %div_15 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_35, %add_36), kwargs = {})
#   %mul_66 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_15, 0.9622504486493761), kwargs = {})
#   return %add_35,%mul_66
# SchedulerNodes: [SchedulerNode(name='op163'), SchedulerNode(name='op164')]

triton_poi_fused_add_mul_sigmoid_silu_25 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_silu_25', '''
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
    size_hints={'y0': 65536, 'x1': 784}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_silu_25', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 65536), ('x1', 784)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_silu_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
    x1_numel = 784
    X1BLOCK_SUB: tl.constexpr = 784
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        x1 = base_x1[None,:]
        tmp0 = tl.load(in_ptr0 + (x1 + 784*y0), y0_mask)
        tmp1 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp8 = tl.load(in_ptr2 + (x1 + 784*y0), y0_mask)
        tmp9 = tl.load(in_ptr3 + (y0), y0_mask)
        tmp14 = tl.load(in_ptr4 + (x1 + 784*y0), y0_mask)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = tl.full([1, 1], 2.0, tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full([1, 1], 0.2, tl.float32)
        tmp7 = tmp5 * tmp6
        tmp10 = tl.sigmoid(tmp9)
        tmp11 = tmp8 * tmp10
        tmp12 = tmp11 * tmp4
        tmp13 = tmp12 * tmp6
        tmp15 = tmp13 + tmp14
        tmp16 = tmp7 + tmp15
        tmp17 = -tmp16
        tmp18 = libdevice.exp(tmp17)
        tmp19 = tl.full([1, 1], 1.0, tl.float32)
        tmp20 = tmp18 + tmp19
        tmp21 = (tmp16 / tmp20)
        tmp22 = tl.full([1, 1], 0.9622504486493761, tl.float32)
        tmp23 = tmp21 * tmp22
        tl.store(out_ptr0 + (x1 + 784*y0), tmp16, y0_mask)
        tl.store(out_ptr1 + (x1 + 784*y0), tmp23, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [avg_pool2d_1], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d_1 => avg_pool2d_1
# Graph fragment:
#   %mul_66 : Tensor "f32[128, 512, 28, 28][401408, 784, 28, 1]npu:0" = PlaceHolder[target=mul_66]
#   %avg_pool2d_1 : Tensor "f32[128, 512, 14, 14][100352, 196, 14, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%mul_66, [2, 2], [2, 2], [0, 0], True, False), kwargs = {})
#   return %avg_pool2d_1
# SchedulerNodes: [SchedulerNode(name='op165')]

triton_poi_fused_avg_pool2d_26 = async_compile.triton('triton_poi_fused_avg_pool2d_26', '''
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
    size_hints={'y0': 917504, 'x1': 14}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_26', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 917504), ('x1', 14)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_26(in_ptr0, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
    x1_numel = 14
    X1BLOCK_SUB: tl.constexpr = 14
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        x1 = base_x1[None,:]
        tmp0 = tl.load(in_ptr0 + (2*x1 + 56*y0), y0_mask)
        tmp1 = tl.load(in_ptr0 + (1 + 2*x1 + 56*y0), y0_mask)
        tmp3 = tl.load(in_ptr0 + (28 + 2*x1 + 56*y0), y0_mask)
        tmp5 = tl.load(in_ptr0 + (29 + 2*x1 + 56*y0), y0_mask)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tl.full([1, 1], 4.0, tl.float32)
        tmp8 = (tmp6 / tmp7)
        tl.store(out_ptr0 + (x1 + 14*y0), tmp8, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [reshape_18, mul_31, view_18, batch_norm_18], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_18 => add_37, mul_68, mul_69, rsqrt_18, sub_18, unsqueeze_18, var_mean_18
#   mul_31 => mul_67
#   reshape_18 => view_54
#   view_18 => view_55
# Graph fragment:
#   %primals_68 : Tensor "f32[1536, 512, 1, 1][512, 1, 1, 1]npu:0" = PlaceHolder[target=primals_68]
#   %buf166 : Tensor "f32[1, 1536, 1][1536, 1, 1536]npu:0" = PlaceHolder[target=buf166]
#   %getitem_37 : Tensor "f32[1, 1536, 1][1536, 1, 1536]npu:0" = PlaceHolder[target=getitem_37]
#   %buf168 : Tensor "f32[1, 1536, 1][1536, 1, 1536]npu:0" = PlaceHolder[target=buf168]
#   %primals_69 : Tensor "f32[1536, 1, 1, 1][1, 1, 1, 1]npu:0" = PlaceHolder[target=primals_69]
#   %view_54 : Tensor "f32[1, 1536, 512][786432, 512, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_68, [1, 1536, -1]), kwargs = {})
#   %mul_67 : Tensor "f32[1536, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_69, 0.07902489841601695), kwargs = {})
#   %view_55 : Tensor "f32[1536][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_67, [-1]), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_54, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_37 : Tensor "f32[1, 1536, 1][1536, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_36, 1e-05), kwargs = {})
#   %rsqrt_18 : Tensor "f32[1, 1536, 1][1536, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_37,), kwargs = {})
#   %sub_18 : Tensor "f32[1, 1536, 512][786432, 512, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_54, %getitem_37), kwargs = {})
#   %mul_68 : Tensor "f32[1, 1536, 512][786432, 512, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %rsqrt_18), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[1536, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_55, -1), kwargs = {})
#   %mul_69 : Tensor "f32[1, 1536, 512][786432, 512, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_68, %unsqueeze_18), kwargs = {})
#   return %buf166,%getitem_37,%buf168,%mul_69,%rsqrt_18
# SchedulerNodes: [SchedulerNode(name='op166'), SchedulerNode(name='op167'), SchedulerNode(name='op168'), SchedulerNode(name='op170'), SchedulerNode(name='op169')]

triton_per_fused__native_batch_norm_legit_mul__27 = async_compile.triton('triton_per_fused__native_batch_norm_legit_mul__27', '''
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
    size_hints={'x0': 1536, 'r1': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_mul__27', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 1536), ('r1', 512)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_mul__27(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 512
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
        tmp0 = tl.load(in_ptr0 + (r1 + 512*x0), x0_mask, other=0.0)
        tmp16 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 1).reshape(X0BLOCK_SUB, 1)
        tmp4 = tl.full([1, 1], 512.0, tl.float32)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tmp0 - tmp5
        tmp7 = tmp6 * tmp6
        tmp9 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp7, 0)
        tmp10 = tl.sum(tmp9, 1).reshape(X0BLOCK_SUB, 1)
        tmp11 = (tmp10 / tmp4)
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp6 * tmp14
        tmp17 = tl.full([1, 1], 0.07902489841601695, tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 * tmp18
        tl.store(in_out_ptr0 + (x0), tmp5, x0_mask)
        tl.store(out_ptr1 + (r1 + 512*x0), tmp19, x0_mask)
        tl.store(out_ptr2 + (x0), tmp14, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [reshape_19, mul_32, view_19, batch_norm_19], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_19 => add_38, mul_71, mul_72, rsqrt_19, sub_19, unsqueeze_19, var_mean_19
#   mul_32 => mul_70
#   reshape_19 => view_57
#   view_19 => view_58
# Graph fragment:
#   %primals_71 : Tensor "f32[384, 512, 1, 1][512, 1, 1, 1]npu:0" = PlaceHolder[target=primals_71]
#   %buf173 : Tensor "f32[1, 384, 1][384, 1, 384]npu:0" = PlaceHolder[target=buf173]
#   %getitem_39 : Tensor "f32[1, 384, 1][384, 1, 384]npu:0" = PlaceHolder[target=getitem_39]
#   %buf175 : Tensor "f32[1, 384, 1][384, 1, 384]npu:0" = PlaceHolder[target=buf175]
#   %primals_72 : Tensor "f32[384, 1, 1, 1][1, 1, 1, 1]npu:0" = PlaceHolder[target=primals_72]
#   %view_57 : Tensor "f32[1, 384, 512][196608, 512, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_71, [1, 384, -1]), kwargs = {})
#   %mul_70 : Tensor "f32[384, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_72, 0.07902489841601695), kwargs = {})
#   %view_58 : Tensor "f32[384][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_70, [-1]), kwargs = {})
#   %var_mean_19 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_57, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_38 : Tensor "f32[1, 384, 1][384, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_38, 1e-05), kwargs = {})
#   %rsqrt_19 : Tensor "f32[1, 384, 1][384, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_38,), kwargs = {})
#   %sub_19 : Tensor "f32[1, 384, 512][196608, 512, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_57, %getitem_39), kwargs = {})
#   %mul_71 : Tensor "f32[1, 384, 512][196608, 512, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %rsqrt_19), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[384, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_58, -1), kwargs = {})
#   %mul_72 : Tensor "f32[1, 384, 512][196608, 512, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_71, %unsqueeze_19), kwargs = {})
#   return %buf173,%getitem_39,%buf175,%mul_72,%rsqrt_19
# SchedulerNodes: [SchedulerNode(name='op173'), SchedulerNode(name='op174'), SchedulerNode(name='op175'), SchedulerNode(name='op177'), SchedulerNode(name='op176')]

triton_per_fused__native_batch_norm_legit_mul__28 = async_compile.triton('triton_per_fused__native_batch_norm_legit_mul__28', '''
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
    size_hints={'x0': 384, 'r1': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_mul__28', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 384), ('r1', 512)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_mul__28(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 512
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
        tmp0 = tl.load(in_ptr0 + (r1 + 512*x0), x0_mask, other=0.0)
        tmp16 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 1).reshape(X0BLOCK_SUB, 1)
        tmp4 = tl.full([1, 1], 512.0, tl.float32)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tmp0 - tmp5
        tmp7 = tmp6 * tmp6
        tmp9 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp7, 0)
        tmp10 = tl.sum(tmp9, 1).reshape(X0BLOCK_SUB, 1)
        tmp11 = (tmp10 / tmp4)
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp6 * tmp14
        tmp17 = tl.full([1, 1], 0.07902489841601695, tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 * tmp18
        tl.store(in_out_ptr0 + (x0), tmp5, x0_mask)
        tl.store(out_ptr1 + (r1 + 512*x0), tmp19, x0_mask)
        tl.store(out_ptr2 + (x0), tmp14, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [silu_16], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   silu_16 => add_39, div_16, exp_16, neg_16
# Graph fragment:
#   %convolution_25 : Tensor "f32[128, 384, 28, 28][301056, 784, 28, 1]npu:0" = PlaceHolder[target=convolution_25]
#   %neg_16 : Tensor "f32[128, 384, 28, 28][301056, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convolution_25,), kwargs = {})
#   %exp_16 : Tensor "f32[128, 384, 28, 28][301056, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_16,), kwargs = {})
#   %add_39 : Tensor "f32[128, 384, 28, 28][301056, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_16, 1), kwargs = {})
#   %div_16 : Tensor "f32[128, 384, 28, 28][301056, 784, 28, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_25, %add_39), kwargs = {})
#   return %div_16
# SchedulerNodes: [SchedulerNode(name='op180')]

triton_poi_fused_silu_29 = async_compile.triton('triton_poi_fused_silu_29', '''
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
    size_hints={'t0': 128, 'z1': 384, 'y2': 28, 'x3': 28}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 't0_numel': 'i32', 'z1_numel': 'i32', 'y2_numel': 'i32', 'x3_numel': 'i32', 'T0BLOCK': 'i32', 'Z1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_29', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [2, 3], 'axis_names': ['t0', 'z1', 'y2', 'x3'], 'axis_static_values': (('t0', 128), ('z1', 384), ('y2', 28), ('x3', 28)), 'low_dims': {3}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('T0BLOCK', 'Z1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_silu_29(in_ptr0, out_ptr0, t0_numel, z1_numel, y2_numel, x3_numel, T0BLOCK, Z1BLOCK, T0BLOCK_SUB : tl.constexpr, Z1BLOCK_SUB : tl.constexpr):
    y2_numel = 28
    Y2BLOCK_SUB: tl.constexpr = 28
    x3_numel = 28
    X3BLOCK_SUB: tl.constexpr = 28
    t0_offset = tl.program_id(0) * T0BLOCK
    base_t0= tl.arange(0, T0BLOCK_SUB)
    loops_t0 = (T0BLOCK + T0BLOCK_SUB - 1) // T0BLOCK_SUB
    z1_offset = tl.program_id(1) * Z1BLOCK
    base_z1= tl.arange(0, Z1BLOCK_SUB)
    loops_z1 = (Z1BLOCK + Z1BLOCK_SUB - 1) // Z1BLOCK_SUB
    base_y2= tl.arange(0, Y2BLOCK_SUB)
    base_x3= tl.arange(0, X3BLOCK_SUB)
    for loop_t0 in range(loops_t0):
        t0 = t0_offset + (loop_t0 * T0BLOCK_SUB) + base_t0[:,None,None,None]
        t0_mask = t0 < min(T0BLOCK+t0_offset, t0_numel)
        for loop_z1 in range(loops_z1):
            z1 = z1_offset + (loop_z1 * Z1BLOCK_SUB) + base_z1[None,:,None,None]
            z1_mask = z1 < min(Z1BLOCK+z1_offset, z1_numel)
            y2 = base_y2[None,None,:,None]
            x3 = base_x3[None,None,None,:]
            tmp0 = tl.load(in_ptr0 + (x3 + 28*y2 + 784*z1 + 301056*t0), t0_mask & z1_mask)
            tmp1 = -tmp0
            tmp2 = libdevice.exp(tmp1)
            tmp3 = tl.full([1, 1, 1, 1], 1.0, tl.float32)
            tmp4 = tmp2 + tmp3
            tmp5 = (tmp0 / tmp4)
            tl.store(out_ptr0 + (x3 + 28*y2 + 784*z1 + 301056*t0), tmp5, t0_mask & z1_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [reshape_20, mul_33, view_20, batch_norm_20], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_20 => add_40, mul_74, mul_75, rsqrt_20, sub_20, unsqueeze_20, var_mean_20
#   mul_33 => mul_73
#   reshape_20 => view_60
#   view_20 => view_61
# Graph fragment:
#   %primals_74 : Tensor "f32[384, 64, 3, 3][576, 9, 3, 1]npu:0" = PlaceHolder[target=primals_74]
#   %buf181 : Tensor "f32[1, 384, 1][384, 1, 384]npu:0" = PlaceHolder[target=buf181]
#   %getitem_41 : Tensor "f32[1, 384, 1][384, 1, 384]npu:0" = PlaceHolder[target=getitem_41]
#   %buf183 : Tensor "f32[1, 384, 1][384, 1, 384]npu:0" = PlaceHolder[target=buf183]
#   %primals_75 : Tensor "f32[384, 1, 1, 1][1, 1, 1, 1]npu:0" = PlaceHolder[target=primals_75]
#   %view_60 : Tensor "f32[1, 384, 576][221184, 576, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_74, [1, 384, -1]), kwargs = {})
#   %mul_73 : Tensor "f32[384, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_75, 0.07450538873672485), kwargs = {})
#   %view_61 : Tensor "f32[384][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_73, [-1]), kwargs = {})
#   %var_mean_20 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_60, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_40 : Tensor "f32[1, 384, 1][384, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_40, 1e-05), kwargs = {})
#   %rsqrt_20 : Tensor "f32[1, 384, 1][384, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_40,), kwargs = {})
#   %sub_20 : Tensor "f32[1, 384, 576][221184, 576, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_60, %getitem_41), kwargs = {})
#   %mul_74 : Tensor "f32[1, 384, 576][221184, 576, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %rsqrt_20), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[384, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_61, -1), kwargs = {})
#   %mul_75 : Tensor "f32[1, 384, 576][221184, 576, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_74, %unsqueeze_20), kwargs = {})
#   return %buf181,%getitem_41,%buf183,%mul_75,%rsqrt_20
# SchedulerNodes: [SchedulerNode(name='op181'), SchedulerNode(name='op182'), SchedulerNode(name='op183'), SchedulerNode(name='op185'), SchedulerNode(name='op184')]

triton_per_fused__native_batch_norm_legit_mul__30 = async_compile.triton('triton_per_fused__native_batch_norm_legit_mul__30', '''
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
    size_hints={'x0': 384, 'r1': 576},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_mul__30', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 384), ('r1', 576)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_mul__30(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 576
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
        tmp0 = tl.load(in_ptr0 + (r1 + 576*x0), x0_mask, other=0.0)
        tmp16 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 1).reshape(X0BLOCK_SUB, 1)
        tmp4 = tl.full([1, 1], 576.0, tl.float32)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tmp0 - tmp5
        tmp7 = tmp6 * tmp6
        tmp9 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp7, 0)
        tmp10 = tl.sum(tmp9, 1).reshape(X0BLOCK_SUB, 1)
        tmp11 = (tmp10 / tmp4)
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp6 * tmp14
        tmp17 = tl.full([1, 1], 0.07450538873672485, tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 * tmp18
        tl.store(in_out_ptr0 + (x0), tmp5, x0_mask)
        tl.store(out_ptr1 + (r1 + 576*x0), tmp19, x0_mask)
        tl.store(out_ptr2 + (x0), tmp14, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [silu_17], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   silu_17 => add_41, div_17, exp_17, neg_17
# Graph fragment:
#   %convolution_26 : Tensor "f32[128, 384, 14, 14][75264, 196, 14, 1]npu:0" = PlaceHolder[target=convolution_26]
#   %neg_17 : Tensor "f32[128, 384, 14, 14][75264, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convolution_26,), kwargs = {})
#   %exp_17 : Tensor "f32[128, 384, 14, 14][75264, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_17,), kwargs = {})
#   %add_41 : Tensor "f32[128, 384, 14, 14][75264, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_17, 1), kwargs = {})
#   %div_17 : Tensor "f32[128, 384, 14, 14][75264, 196, 14, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_26, %add_41), kwargs = {})
#   return %div_17
# SchedulerNodes: [SchedulerNode(name='op188')]

triton_poi_fused_silu_31 = async_compile.triton('triton_poi_fused_silu_31', '''
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
    size_hints={'t0': 128, 'z1': 384, 'y2': 14, 'x3': 14}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 't0_numel': 'i32', 'z1_numel': 'i32', 'y2_numel': 'i32', 'x3_numel': 'i32', 'T0BLOCK': 'i32', 'Z1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_31', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [2, 3], 'axis_names': ['t0', 'z1', 'y2', 'x3'], 'axis_static_values': (('t0', 128), ('z1', 384), ('y2', 14), ('x3', 14)), 'low_dims': {3}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('T0BLOCK', 'Z1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_silu_31(in_ptr0, out_ptr0, t0_numel, z1_numel, y2_numel, x3_numel, T0BLOCK, Z1BLOCK, T0BLOCK_SUB : tl.constexpr, Z1BLOCK_SUB : tl.constexpr):
    y2_numel = 14
    Y2BLOCK_SUB: tl.constexpr = 14
    x3_numel = 14
    X3BLOCK_SUB: tl.constexpr = 14
    t0_offset = tl.program_id(0) * T0BLOCK
    base_t0= tl.arange(0, T0BLOCK_SUB)
    loops_t0 = (T0BLOCK + T0BLOCK_SUB - 1) // T0BLOCK_SUB
    z1_offset = tl.program_id(1) * Z1BLOCK
    base_z1= tl.arange(0, Z1BLOCK_SUB)
    loops_z1 = (Z1BLOCK + Z1BLOCK_SUB - 1) // Z1BLOCK_SUB
    base_y2= tl.arange(0, Y2BLOCK_SUB)
    base_x3= tl.arange(0, X3BLOCK_SUB)
    for loop_t0 in range(loops_t0):
        t0 = t0_offset + (loop_t0 * T0BLOCK_SUB) + base_t0[:,None,None,None]
        t0_mask = t0 < min(T0BLOCK+t0_offset, t0_numel)
        for loop_z1 in range(loops_z1):
            z1 = z1_offset + (loop_z1 * Z1BLOCK_SUB) + base_z1[None,:,None,None]
            z1_mask = z1 < min(Z1BLOCK+z1_offset, z1_numel)
            y2 = base_y2[None,None,:,None]
            x3 = base_x3[None,None,None,:]
            tmp0 = tl.load(in_ptr0 + (x3 + 14*y2 + 196*z1 + 75264*t0), t0_mask & z1_mask)
            tmp1 = -tmp0
            tmp2 = libdevice.exp(tmp1)
            tmp3 = tl.full([1, 1, 1, 1], 1.0, tl.float32)
            tmp4 = tmp2 + tmp3
            tmp5 = (tmp0 / tmp4)
            tl.store(out_ptr0 + (x3 + 14*y2 + 196*z1 + 75264*t0), tmp5, t0_mask & z1_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [reshape_22, mul_35, view_22, batch_norm_22], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_22 => add_44, mul_80, mul_81, rsqrt_22, sub_22, unsqueeze_22, var_mean_22
#   mul_35 => mul_79
#   reshape_22 => view_66
#   view_22 => view_67
# Graph fragment:
#   %primals_80 : Tensor "f32[1536, 384, 1, 1][384, 1, 1, 1]npu:0" = PlaceHolder[target=primals_80]
#   %buf197 : Tensor "f32[1, 1536, 1][1536, 1, 1536]npu:0" = PlaceHolder[target=buf197]
#   %getitem_45 : Tensor "f32[1, 1536, 1][1536, 1, 1536]npu:0" = PlaceHolder[target=getitem_45]
#   %buf199 : Tensor "f32[1, 1536, 1][1536, 1, 1536]npu:0" = PlaceHolder[target=buf199]
#   %primals_81 : Tensor "f32[1536, 1, 1, 1][1, 1, 1, 1]npu:0" = PlaceHolder[target=primals_81]
#   %view_66 : Tensor "f32[1, 1536, 384][589824, 384, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_80, [1, 1536, -1]), kwargs = {})
#   %mul_79 : Tensor "f32[1536, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_81, 0.09125009274634042), kwargs = {})
#   %view_67 : Tensor "f32[1536][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_79, [-1]), kwargs = {})
#   %var_mean_22 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_66, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_44 : Tensor "f32[1, 1536, 1][1536, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_44, 1e-05), kwargs = {})
#   %rsqrt_22 : Tensor "f32[1, 1536, 1][1536, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_44,), kwargs = {})
#   %sub_22 : Tensor "f32[1, 1536, 384][589824, 384, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_66, %getitem_45), kwargs = {})
#   %mul_80 : Tensor "f32[1, 1536, 384][589824, 384, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %rsqrt_22), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[1536, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_67, -1), kwargs = {})
#   %mul_81 : Tensor "f32[1, 1536, 384][589824, 384, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_80, %unsqueeze_22), kwargs = {})
#   return %buf197,%getitem_45,%buf199,%mul_81,%rsqrt_22
# SchedulerNodes: [SchedulerNode(name='op197'), SchedulerNode(name='op198'), SchedulerNode(name='op199'), SchedulerNode(name='op201'), SchedulerNode(name='op200')]

triton_per_fused__native_batch_norm_legit_mul__32 = async_compile.triton('triton_per_fused__native_batch_norm_legit_mul__32', '''
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
    size_hints={'x0': 1536, 'r1': 384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_mul__32', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 1536), ('r1', 384)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_mul__32(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 384
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
        tmp0 = tl.load(in_ptr0 + (r1 + 384*x0), x0_mask, other=0.0)
        tmp16 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 1).reshape(X0BLOCK_SUB, 1)
        tmp4 = tl.full([1, 1], 384.0, tl.float32)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tmp0 - tmp5
        tmp7 = tmp6 * tmp6
        tmp9 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp7, 0)
        tmp10 = tl.sum(tmp9, 1).reshape(X0BLOCK_SUB, 1)
        tmp11 = (tmp10 / tmp4)
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp6 * tmp14
        tmp17 = tl.full([1, 1], 0.09125009274634042, tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 * tmp18
        tl.store(in_out_ptr0 + (x0), tmp5, x0_mask)
        tl.store(out_ptr1 + (r1 + 384*x0), tmp19, x0_mask)
        tl.store(out_ptr2 + (x0), tmp14, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [x_se_12], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_12 => mean_3
# Graph fragment:
#   %convolution_28 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0" = PlaceHolder[target=convolution_28]
#   %buf204 : Tensor "f32[128, 1536, 1, 1][1536, 1, 196608, 196608]npu:0" = PlaceHolder[target=buf204]
#   %mean_3 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_28, [2, 3], True), kwargs = {})
#   return %buf204,%mean_3
# SchedulerNodes: [SchedulerNode(name='op204'), SchedulerNode(name='op205')]

triton_per_fused_mean_33 = async_compile.triton('triton_per_fused_mean_33', '''
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
    size_hints={'y0': 128, 'x1': 1536, 'r2': 196},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'r2_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_33', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['y0', 'x1', 'r2'], 'axis_static_values': (('y0', 128), ('x1', 1536), ('r2', 196)), 'low_dims': {2}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_mean_33(in_out_ptr0, in_ptr0, y0_numel, x1_numel, r2_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    R2BLOCK_SUB: tl.constexpr = 196
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    x1_offset = tl.program_id(1) * X1BLOCK
    base_x1= tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (X1BLOCK + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    base_r2= tl.arange(0, R2BLOCK_SUB)
    loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        for loop_x1 in range(loops_x1):
            x1 = x1_offset + (loop_x1 * X1BLOCK_SUB) + base_x1[None,:,None]
            x1_mask = x1 < min(X1BLOCK+x1_offset, x1_numel)
            r2 = base_r2[None,None,:]
            r2_mask = True
            tmp0 = tl.load(in_ptr0 + (r2 + 196*x1 + 301056*y0), x1_mask & y0_mask, other=0.0)
            tmp2 = tl.where(tl.broadcast_to((y0_mask.reshape([Y0BLOCK_SUB, 1, 1]) & x1_mask.reshape([1, X1BLOCK_SUB, 1])), [Y0BLOCK_SUB, X1BLOCK_SUB, R2BLOCK_SUB]), tmp0, 0)
            tmp3 = tl.sum(tmp2, 2).reshape(Y0BLOCK_SUB, X1BLOCK_SUB, 1)
            tmp4 = tl.full([1, 1, 1], 196.0, tl.float32)
            tmp5 = (tmp3 / tmp4)
            tl.store(in_out_ptr0 + (x1 + 1536*y0), tmp5, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [x_se_14], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   x_se_14 => relu_3
# Graph fragment:
#   %convolution_29 : Tensor "f32[128, 384, 1, 1][384, 1, 1, 1]npu:0" = PlaceHolder[target=convolution_29]
#   %relu_3 : Tensor "f32[128, 384, 1, 1][384, 1, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_29,), kwargs = {})
#   return %relu_3
# SchedulerNodes: [SchedulerNode(name='op208')]

triton_poi_fused_relu_34 = async_compile.triton('triton_poi_fused_relu_34', '''
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
    size_hints={'y0': 128, 'x1': 384}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_34', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 128), ('x1', 384)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_34(in_out_ptr0, y0_numel, x1_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
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
            tmp0 = tl.load(in_out_ptr0 + (x1 + 384*y0), x1_mask & y0_mask)
            tmp1 = tl.full([1, 1], 0, tl.int32)
            tmp2 = triton_helpers.maximum(tmp1, tmp0)
            tl.store(in_out_ptr0 + (x1 + 384*y0), tmp2, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [sigmoid_3, mul_36, out_26, mul_38, out_27, silu_19, out_28], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
# Source node to ATen node mapping:
#   mul_36 => mul_82
#   mul_38 => mul_84
#   out_26 => mul_83
#   out_27 => add_45
#   out_28 => mul_85
#   sigmoid_3 => sigmoid_3
#   silu_19 => add_46, div_19, exp_19, neg_19
# Graph fragment:
#   %convolution_28 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0" = PlaceHolder[target=convolution_28]
#   %convolution_30 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0" = PlaceHolder[target=convolution_30]
#   %convolution_24 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0" = PlaceHolder[target=convolution_24]
#   %sigmoid_3 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_30,), kwargs = {})
#   %mul_82 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_28, %sigmoid_3), kwargs = {})
#   %mul_83 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, 2.0), kwargs = {})
#   %mul_84 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_83, 0.2), kwargs = {})
#   %add_45 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_84, %convolution_24), kwargs = {})
#   %neg_19 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_45,), kwargs = {})
#   %exp_19 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_19,), kwargs = {})
#   %add_46 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_19, 1), kwargs = {})
#   %div_19 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_45, %add_46), kwargs = {})
#   %mul_85 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_19, 0.9805806756909201), kwargs = {})
#   return %mul_85
# SchedulerNodes: [SchedulerNode(name='op211')]

triton_poi_fused_add_mul_sigmoid_silu_35 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_silu_35', '''
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
    size_hints={'y0': 196608, 'x1': 196}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_silu_35', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 196608), ('x1', 196)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_silu_35(in_ptr0, in_ptr1, in_ptr2, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
    x1_numel = 196
    X1BLOCK_SUB: tl.constexpr = 196
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        x1 = base_x1[None,:]
        tmp0 = tl.load(in_ptr0 + (x1 + 196*y0), y0_mask)
        tmp1 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp8 = tl.load(in_ptr2 + (x1 + 196*y0), y0_mask)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = tl.full([1, 1], 2.0, tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full([1, 1], 0.2, tl.float32)
        tmp7 = tmp5 * tmp6
        tmp9 = tmp7 + tmp8
        tmp10 = -tmp9
        tmp11 = libdevice.exp(tmp10)
        tmp12 = tl.full([1, 1], 1.0, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = (tmp9 / tmp13)
        tmp15 = tl.full([1, 1], 0.9805806756909201, tl.float32)
        tmp16 = tmp14 * tmp15
        tl.store(out_ptr0 + (x1 + 196*y0), tmp16, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [reshape_23, mul_40, view_23, batch_norm_23], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_23 => add_47, mul_87, mul_88, rsqrt_23, sub_23, unsqueeze_23, var_mean_23
#   mul_40 => mul_86
#   reshape_23 => view_69
#   view_23 => view_70
# Graph fragment:
#   %primals_87 : Tensor "f32[384, 1536, 1, 1][1536, 1, 1, 1]npu:0" = PlaceHolder[target=primals_87]
#   %buf212 : Tensor "f32[1, 384, 1][384, 1, 384]npu:0" = PlaceHolder[target=buf212]
#   %getitem_47 : Tensor "f32[1, 384, 1][384, 1, 384]npu:0" = PlaceHolder[target=getitem_47]
#   %buf214 : Tensor "f32[1, 384, 1][384, 1, 384]npu:0" = PlaceHolder[target=buf214]
#   %primals_88 : Tensor "f32[384, 1, 1, 1][1, 1, 1, 1]npu:0" = PlaceHolder[target=primals_88]
#   %view_69 : Tensor "f32[1, 384, 1536][589824, 1536, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_87, [1, 384, -1]), kwargs = {})
#   %mul_86 : Tensor "f32[384, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_88, 0.04562504637317021), kwargs = {})
#   %view_70 : Tensor "f32[384][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_86, [-1]), kwargs = {})
#   %var_mean_23 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_69, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_47 : Tensor "f32[1, 384, 1][384, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_46, 1e-05), kwargs = {})
#   %rsqrt_23 : Tensor "f32[1, 384, 1][384, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_47,), kwargs = {})
#   %sub_23 : Tensor "f32[1, 384, 1536][589824, 1536, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_69, %getitem_47), kwargs = {})
#   %mul_87 : Tensor "f32[1, 384, 1536][589824, 1536, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %rsqrt_23), kwargs = {})
#   %unsqueeze_23 : Tensor "f32[384, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_70, -1), kwargs = {})
#   %mul_88 : Tensor "f32[1, 384, 1536][589824, 1536, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_87, %unsqueeze_23), kwargs = {})
#   return %buf212,%getitem_47,%buf214,%mul_88,%rsqrt_23
# SchedulerNodes: [SchedulerNode(name='op212'), SchedulerNode(name='op213'), SchedulerNode(name='op214'), SchedulerNode(name='op216'), SchedulerNode(name='op215')]

triton_per_fused__native_batch_norm_legit_mul__36 = async_compile.triton('triton_per_fused__native_batch_norm_legit_mul__36', '''
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
    size_hints={'x0': 384, 'r1': 1536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_mul__36', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 384), ('r1', 1536)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_mul__36(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 1536
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
        tmp0 = tl.load(in_ptr0 + (r1 + 1536*x0), x0_mask, other=0.0)
        tmp16 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 1).reshape(X0BLOCK_SUB, 1)
        tmp4 = tl.full([1, 1], 1536.0, tl.float32)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tmp0 - tmp5
        tmp7 = tmp6 * tmp6
        tmp9 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp7, 0)
        tmp10 = tl.sum(tmp9, 1).reshape(X0BLOCK_SUB, 1)
        tmp11 = (tmp10 / tmp4)
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp6 * tmp14
        tmp17 = tl.full([1, 1], 0.04562504637317021, tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 * tmp18
        tl.store(in_out_ptr0 + (x0), tmp5, x0_mask)
        tl.store(out_ptr1 + (r1 + 1536*x0), tmp19, x0_mask)
        tl.store(out_ptr2 + (x0), tmp14, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [sigmoid_3, mul_36, out_26, mul_38, out_27, sigmoid_4, mul_44, out_33, mul_46, out_34, silu_23, out_35], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
# Source node to ATen node mapping:
#   mul_36 => mul_82
#   mul_38 => mul_84
#   mul_44 => mul_98
#   mul_46 => mul_100
#   out_26 => mul_83
#   out_27 => add_45
#   out_33 => mul_99
#   out_34 => add_54
#   out_35 => mul_101
#   sigmoid_3 => sigmoid_3
#   sigmoid_4 => sigmoid_4
#   silu_23 => add_55, div_23, exp_23, neg_23
# Graph fragment:
#   %convolution_34 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0" = PlaceHolder[target=convolution_34]
#   %convolution_36 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0" = PlaceHolder[target=convolution_36]
#   %convolution_28 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0" = PlaceHolder[target=convolution_28]
#   %convolution_30 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0" = PlaceHolder[target=convolution_30]
#   %convolution_24 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0" = PlaceHolder[target=convolution_24]
#   %add_54 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0" = PlaceHolder[target=add_54]
#   %sigmoid_3 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_30,), kwargs = {})
#   %mul_82 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_28, %sigmoid_3), kwargs = {})
#   %mul_83 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, 2.0), kwargs = {})
#   %mul_84 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_83, 0.2), kwargs = {})
#   %add_45 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_84, %convolution_24), kwargs = {})
#   %sigmoid_4 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_36,), kwargs = {})
#   %mul_98 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_34, %sigmoid_4), kwargs = {})
#   %mul_99 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_98, 2.0), kwargs = {})
#   %mul_100 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_99, 0.2), kwargs = {})
#   %add_54 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_100, %add_45), kwargs = {})
#   %neg_23 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_54,), kwargs = {})
#   %exp_23 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_23,), kwargs = {})
#   %add_55 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_23, 1), kwargs = {})
#   %div_23 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_54, %add_55), kwargs = {})
#   %mul_101 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_23, 0.9622504486493761), kwargs = {})
#   return %add_54,%mul_101
# SchedulerNodes: [SchedulerNode(name='op250'), SchedulerNode(name='op251')]

triton_poi_fused_add_mul_sigmoid_silu_37 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_silu_37', '''
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
    size_hints={'y0': 196608, 'x1': 196}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_silu_37', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 196608), ('x1', 196)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_silu_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
    x1_numel = 196
    X1BLOCK_SUB: tl.constexpr = 196
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        x1 = base_x1[None,:]
        tmp0 = tl.load(in_ptr0 + (x1 + 196*y0), y0_mask)
        tmp1 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp8 = tl.load(in_ptr2 + (x1 + 196*y0), y0_mask)
        tmp9 = tl.load(in_ptr3 + (y0), y0_mask)
        tmp14 = tl.load(in_ptr4 + (x1 + 196*y0), y0_mask)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = tl.full([1, 1], 2.0, tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full([1, 1], 0.2, tl.float32)
        tmp7 = tmp5 * tmp6
        tmp10 = tl.sigmoid(tmp9)
        tmp11 = tmp8 * tmp10
        tmp12 = tmp11 * tmp4
        tmp13 = tmp12 * tmp6
        tmp15 = tmp13 + tmp14
        tmp16 = tmp7 + tmp15
        tmp17 = -tmp16
        tmp18 = libdevice.exp(tmp17)
        tmp19 = tl.full([1, 1], 1.0, tl.float32)
        tmp20 = tmp18 + tmp19
        tmp21 = (tmp16 / tmp20)
        tmp22 = tl.full([1, 1], 0.9622504486493761, tl.float32)
        tmp23 = tmp21 * tmp22
        tl.store(out_ptr0 + (x1 + 196*y0), tmp16, y0_mask)
        tl.store(out_ptr1 + (x1 + 196*y0), tmp23, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [sigmoid_5, mul_52, out_40, mul_54, out_41, silu_27, out_42], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
# Source node to ATen node mapping:
#   mul_52 => mul_114
#   mul_54 => mul_116
#   out_40 => mul_115
#   out_41 => add_63
#   out_42 => mul_117
#   sigmoid_5 => sigmoid_5
#   silu_27 => add_64, div_27, exp_27, neg_27
# Graph fragment:
#   %convolution_40 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0" = PlaceHolder[target=convolution_40]
#   %convolution_42 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0" = PlaceHolder[target=convolution_42]
#   %add_54 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0" = PlaceHolder[target=add_54]
#   %add_63 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0" = PlaceHolder[target=add_63]
#   %sigmoid_5 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_42,), kwargs = {})
#   %mul_114 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_40, %sigmoid_5), kwargs = {})
#   %mul_115 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_114, 2.0), kwargs = {})
#   %mul_116 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_115, 0.2), kwargs = {})
#   %add_63 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_116, %add_54), kwargs = {})
#   %neg_27 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_63,), kwargs = {})
#   %exp_27 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_27,), kwargs = {})
#   %add_64 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_27, 1), kwargs = {})
#   %div_27 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_63, %add_64), kwargs = {})
#   %mul_117 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_27, 0.9449111825230679), kwargs = {})
#   return %add_63,%mul_117
# SchedulerNodes: [SchedulerNode(name='op290'), SchedulerNode(name='op291')]

triton_poi_fused_add_mul_sigmoid_silu_38 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_silu_38', '''
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
    size_hints={'y0': 196608, 'x1': 196}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_silu_38', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 196608), ('x1', 196)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_silu_38(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
    x1_numel = 196
    X1BLOCK_SUB: tl.constexpr = 196
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        x1 = base_x1[None,:]
        tmp0 = tl.load(in_ptr0 + (x1 + 196*y0), y0_mask)
        tmp1 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp8 = tl.load(in_ptr2 + (x1 + 196*y0), y0_mask)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = tl.full([1, 1], 2.0, tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full([1, 1], 0.2, tl.float32)
        tmp7 = tmp5 * tmp6
        tmp9 = tmp7 + tmp8
        tmp10 = -tmp9
        tmp11 = libdevice.exp(tmp10)
        tmp12 = tl.full([1, 1], 1.0, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = (tmp9 / tmp13)
        tmp15 = tl.full([1, 1], 0.9449111825230679, tl.float32)
        tmp16 = tmp14 * tmp15
        tl.store(out_ptr0 + (x1 + 196*y0), tmp9, y0_mask)
        tl.store(out_ptr1 + (x1 + 196*y0), tmp16, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [sigmoid_6, mul_60, out_47, mul_62, out_48, silu_31, out_49], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
# Source node to ATen node mapping:
#   mul_60 => mul_130
#   mul_62 => mul_132
#   out_47 => mul_131
#   out_48 => add_72
#   out_49 => mul_133
#   sigmoid_6 => sigmoid_6
#   silu_31 => add_73, div_31, exp_31, neg_31
# Graph fragment:
#   %convolution_46 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0" = PlaceHolder[target=convolution_46]
#   %convolution_48 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0" = PlaceHolder[target=convolution_48]
#   %add_63 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0" = PlaceHolder[target=add_63]
#   %add_72 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0" = PlaceHolder[target=add_72]
#   %sigmoid_6 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_48,), kwargs = {})
#   %mul_130 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_46, %sigmoid_6), kwargs = {})
#   %mul_131 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_130, 2.0), kwargs = {})
#   %mul_132 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_131, 0.2), kwargs = {})
#   %add_72 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_132, %add_63), kwargs = {})
#   %neg_31 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_72,), kwargs = {})
#   %exp_31 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_31,), kwargs = {})
#   %add_73 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_31, 1), kwargs = {})
#   %div_31 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_72, %add_73), kwargs = {})
#   %mul_133 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_31, 0.9284766908852592), kwargs = {})
#   return %add_72,%mul_133
# SchedulerNodes: [SchedulerNode(name='op330'), SchedulerNode(name='op331')]

triton_poi_fused_add_mul_sigmoid_silu_39 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_silu_39', '''
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
    size_hints={'y0': 196608, 'x1': 196}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_silu_39', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 196608), ('x1', 196)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_silu_39(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
    x1_numel = 196
    X1BLOCK_SUB: tl.constexpr = 196
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        x1 = base_x1[None,:]
        tmp0 = tl.load(in_ptr0 + (x1 + 196*y0), y0_mask)
        tmp1 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp8 = tl.load(in_ptr2 + (x1 + 196*y0), y0_mask)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = tl.full([1, 1], 2.0, tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full([1, 1], 0.2, tl.float32)
        tmp7 = tmp5 * tmp6
        tmp9 = tmp7 + tmp8
        tmp10 = -tmp9
        tmp11 = libdevice.exp(tmp10)
        tmp12 = tl.full([1, 1], 1.0, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = (tmp9 / tmp13)
        tmp15 = tl.full([1, 1], 0.9284766908852592, tl.float32)
        tmp16 = tmp14 * tmp15
        tl.store(out_ptr0 + (x1 + 196*y0), tmp9, y0_mask)
        tl.store(out_ptr1 + (x1 + 196*y0), tmp16, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [sigmoid_7, mul_68, out_54, mul_70, out_55, silu_35, out_56], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
# Source node to ATen node mapping:
#   mul_68 => mul_146
#   mul_70 => mul_148
#   out_54 => mul_147
#   out_55 => add_81
#   out_56 => mul_149
#   sigmoid_7 => sigmoid_7
#   silu_35 => add_82, div_35, exp_35, neg_35
# Graph fragment:
#   %convolution_52 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0" = PlaceHolder[target=convolution_52]
#   %convolution_54 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0" = PlaceHolder[target=convolution_54]
#   %add_72 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0" = PlaceHolder[target=add_72]
#   %add_81 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0" = PlaceHolder[target=add_81]
#   %sigmoid_7 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_54,), kwargs = {})
#   %mul_146 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_52, %sigmoid_7), kwargs = {})
#   %mul_147 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_146, 2.0), kwargs = {})
#   %mul_148 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_147, 0.2), kwargs = {})
#   %add_81 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_148, %add_72), kwargs = {})
#   %neg_35 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_81,), kwargs = {})
#   %exp_35 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_35,), kwargs = {})
#   %add_82 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_35, 1), kwargs = {})
#   %div_35 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_81, %add_82), kwargs = {})
#   %mul_149 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_35, 0.9128709291752768), kwargs = {})
#   return %add_81,%mul_149
# SchedulerNodes: [SchedulerNode(name='op370'), SchedulerNode(name='op371')]

triton_poi_fused_add_mul_sigmoid_silu_40 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_silu_40', '''
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
    size_hints={'y0': 196608, 'x1': 196}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_silu_40', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 196608), ('x1', 196)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_silu_40(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
    x1_numel = 196
    X1BLOCK_SUB: tl.constexpr = 196
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        x1 = base_x1[None,:]
        tmp0 = tl.load(in_ptr0 + (x1 + 196*y0), y0_mask)
        tmp1 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp8 = tl.load(in_ptr2 + (x1 + 196*y0), y0_mask)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = tl.full([1, 1], 2.0, tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full([1, 1], 0.2, tl.float32)
        tmp7 = tmp5 * tmp6
        tmp9 = tmp7 + tmp8
        tmp10 = -tmp9
        tmp11 = libdevice.exp(tmp10)
        tmp12 = tl.full([1, 1], 1.0, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = (tmp9 / tmp13)
        tmp15 = tl.full([1, 1], 0.9128709291752768, tl.float32)
        tmp16 = tmp14 * tmp15
        tl.store(out_ptr0 + (x1 + 196*y0), tmp9, y0_mask)
        tl.store(out_ptr1 + (x1 + 196*y0), tmp16, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [sigmoid_8, mul_76, out_61, mul_78, out_62, silu_39, out_63], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
# Source node to ATen node mapping:
#   mul_76 => mul_162
#   mul_78 => mul_164
#   out_61 => mul_163
#   out_62 => add_90
#   out_63 => mul_165
#   sigmoid_8 => sigmoid_8
#   silu_39 => add_91, div_39, exp_39, neg_39
# Graph fragment:
#   %convolution_58 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0" = PlaceHolder[target=convolution_58]
#   %convolution_60 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0" = PlaceHolder[target=convolution_60]
#   %add_81 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0" = PlaceHolder[target=add_81]
#   %add_90 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0" = PlaceHolder[target=add_90]
#   %sigmoid_8 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_60,), kwargs = {})
#   %mul_162 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_58, %sigmoid_8), kwargs = {})
#   %mul_163 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_162, 2.0), kwargs = {})
#   %mul_164 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_163, 0.2), kwargs = {})
#   %add_90 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_164, %add_81), kwargs = {})
#   %neg_39 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_90,), kwargs = {})
#   %exp_39 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_39,), kwargs = {})
#   %add_91 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_39, 1), kwargs = {})
#   %div_39 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_90, %add_91), kwargs = {})
#   %mul_165 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_39, 0.8980265101338745), kwargs = {})
#   return %add_90,%mul_165
# SchedulerNodes: [SchedulerNode(name='op410'), SchedulerNode(name='op411')]

triton_poi_fused_add_mul_sigmoid_silu_41 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_silu_41', '''
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
    size_hints={'y0': 196608, 'x1': 196}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_silu_41', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 196608), ('x1', 196)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_silu_41(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
    x1_numel = 196
    X1BLOCK_SUB: tl.constexpr = 196
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        x1 = base_x1[None,:]
        tmp0 = tl.load(in_ptr0 + (x1 + 196*y0), y0_mask)
        tmp1 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp8 = tl.load(in_ptr2 + (x1 + 196*y0), y0_mask)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = tl.full([1, 1], 2.0, tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full([1, 1], 0.2, tl.float32)
        tmp7 = tmp5 * tmp6
        tmp9 = tmp7 + tmp8
        tmp10 = -tmp9
        tmp11 = libdevice.exp(tmp10)
        tmp12 = tl.full([1, 1], 1.0, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = (tmp9 / tmp13)
        tmp15 = tl.full([1, 1], 0.8980265101338745, tl.float32)
        tmp16 = tmp14 * tmp15
        tl.store(out_ptr0 + (x1 + 196*y0), tmp9, y0_mask)
        tl.store(out_ptr1 + (x1 + 196*y0), tmp16, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [avg_pool2d_2], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d_2 => avg_pool2d_2
# Graph fragment:
#   %mul_165 : Tensor "f32[128, 1536, 14, 14][301056, 196, 14, 1]npu:0" = PlaceHolder[target=mul_165]
#   %avg_pool2d_2 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%mul_165, [2, 2], [2, 2], [0, 0], True, False), kwargs = {})
#   return %avg_pool2d_2
# SchedulerNodes: [SchedulerNode(name='op412')]

triton_poi_fused_avg_pool2d_42 = async_compile.triton('triton_poi_fused_avg_pool2d_42', '''
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
    size_hints={'y0': 1376256, 'x1': 7}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_42', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 1376256), ('x1', 7)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_42(in_ptr0, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
    x1_numel = 7
    X1BLOCK_SUB: tl.constexpr = 7
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        x1 = base_x1[None,:]
        tmp0 = tl.load(in_ptr0 + (2*x1 + 28*y0), y0_mask)
        tmp1 = tl.load(in_ptr0 + (1 + 2*x1 + 28*y0), y0_mask)
        tmp3 = tl.load(in_ptr0 + (14 + 2*x1 + 28*y0), y0_mask)
        tmp5 = tl.load(in_ptr0 + (15 + 2*x1 + 28*y0), y0_mask)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tl.full([1, 1], 4.0, tl.float32)
        tmp8 = (tmp6 / tmp7)
        tl.store(out_ptr0 + (x1 + 7*y0), tmp8, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [reshape_43, mul_80, view_43, batch_norm_43], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_43 => add_92, mul_167, mul_168, rsqrt_43, sub_43, unsqueeze_43, var_mean_43
#   mul_80 => mul_166
#   reshape_43 => view_129
#   view_43 => view_130
# Graph fragment:
#   %primals_167 : Tensor "f32[1536, 1536, 1, 1][1536, 1, 1, 1]npu:0" = PlaceHolder[target=primals_167]
#   %buf413 : Tensor "f32[1, 1536, 1][1536, 1, 1536]npu:0" = PlaceHolder[target=buf413]
#   %getitem_87 : Tensor "f32[1, 1536, 1][1536, 1, 1536]npu:0" = PlaceHolder[target=getitem_87]
#   %buf415 : Tensor "f32[1, 1536, 1][1536, 1, 1536]npu:0" = PlaceHolder[target=buf415]
#   %primals_168 : Tensor "f32[1536, 1, 1, 1][1, 1, 1, 1]npu:0" = PlaceHolder[target=primals_168]
#   %view_129 : Tensor "f32[1, 1536, 1536][2359296, 1536, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_167, [1, 1536, -1]), kwargs = {})
#   %mul_166 : Tensor "f32[1536, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_168, 0.04562504637317021), kwargs = {})
#   %view_130 : Tensor "f32[1536][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_166, [-1]), kwargs = {})
#   %var_mean_43 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_129, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_92 : Tensor "f32[1, 1536, 1][1536, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_86, 1e-05), kwargs = {})
#   %rsqrt_43 : Tensor "f32[1, 1536, 1][1536, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_92,), kwargs = {})
#   %sub_43 : Tensor "f32[1, 1536, 1536][2359296, 1536, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_129, %getitem_87), kwargs = {})
#   %mul_167 : Tensor "f32[1, 1536, 1536][2359296, 1536, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %rsqrt_43), kwargs = {})
#   %unsqueeze_43 : Tensor "f32[1536, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_130, -1), kwargs = {})
#   %mul_168 : Tensor "f32[1, 1536, 1536][2359296, 1536, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_167, %unsqueeze_43), kwargs = {})
#   return %buf413,%getitem_87,%buf415,%mul_168,%rsqrt_43
# SchedulerNodes: [SchedulerNode(name='op413'), SchedulerNode(name='op414'), SchedulerNode(name='op415'), SchedulerNode(name='op417'), SchedulerNode(name='op416')]

triton_per_fused__native_batch_norm_legit_mul__43 = async_compile.triton('triton_per_fused__native_batch_norm_legit_mul__43', '''
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
    size_hints={'x0': 1536, 'r1': 1536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_mul__43', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 1536), ('r1', 1536)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_mul__43(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 1536
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
        tmp0 = tl.load(in_ptr0 + (r1 + 1536*x0), x0_mask, other=0.0)
        tmp16 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 1).reshape(X0BLOCK_SUB, 1)
        tmp4 = tl.full([1, 1], 1536.0, tl.float32)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tmp0 - tmp5
        tmp7 = tmp6 * tmp6
        tmp9 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp7, 0)
        tmp10 = tl.sum(tmp9, 1).reshape(X0BLOCK_SUB, 1)
        tmp11 = (tmp10 / tmp4)
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp6 * tmp14
        tmp17 = tl.full([1, 1], 0.04562504637317021, tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 * tmp18
        tl.store(in_out_ptr0 + (x0), tmp5, x0_mask)
        tl.store(out_ptr1 + (r1 + 1536*x0), tmp19, x0_mask)
        tl.store(out_ptr2 + (x0), tmp14, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [silu_41], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   silu_41 => add_96, div_41, exp_41, neg_41
# Graph fragment:
#   %convolution_63 : Tensor "f32[128, 384, 7, 7][18816, 49, 7, 1]npu:0" = PlaceHolder[target=convolution_63]
#   %neg_41 : Tensor "f32[128, 384, 7, 7][18816, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convolution_63,), kwargs = {})
#   %exp_41 : Tensor "f32[128, 384, 7, 7][18816, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_41,), kwargs = {})
#   %add_96 : Tensor "f32[128, 384, 7, 7][18816, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_41, 1), kwargs = {})
#   %div_41 : Tensor "f32[128, 384, 7, 7][18816, 49, 7, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_63, %add_96), kwargs = {})
#   return %div_41
# SchedulerNodes: [SchedulerNode(name='op435')]

triton_poi_fused_silu_44 = async_compile.triton('triton_poi_fused_silu_44', '''
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
    size_hints={'t0': 128, 'z1': 384, 'y2': 7, 'x3': 7}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 't0_numel': 'i32', 'z1_numel': 'i32', 'y2_numel': 'i32', 'x3_numel': 'i32', 'T0BLOCK': 'i32', 'Z1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_44', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [2, 3], 'axis_names': ['t0', 'z1', 'y2', 'x3'], 'axis_static_values': (('t0', 128), ('z1', 384), ('y2', 7), ('x3', 7)), 'low_dims': {3}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('T0BLOCK', 'Z1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_silu_44(in_ptr0, out_ptr0, t0_numel, z1_numel, y2_numel, x3_numel, T0BLOCK, Z1BLOCK, T0BLOCK_SUB : tl.constexpr, Z1BLOCK_SUB : tl.constexpr):
    y2_numel = 7
    Y2BLOCK_SUB: tl.constexpr = 7
    x3_numel = 7
    X3BLOCK_SUB: tl.constexpr = 7
    t0_offset = tl.program_id(0) * T0BLOCK
    base_t0= tl.arange(0, T0BLOCK_SUB)
    loops_t0 = (T0BLOCK + T0BLOCK_SUB - 1) // T0BLOCK_SUB
    z1_offset = tl.program_id(1) * Z1BLOCK
    base_z1= tl.arange(0, Z1BLOCK_SUB)
    loops_z1 = (Z1BLOCK + Z1BLOCK_SUB - 1) // Z1BLOCK_SUB
    base_y2= tl.arange(0, Y2BLOCK_SUB)
    base_x3= tl.arange(0, X3BLOCK_SUB)
    for loop_t0 in range(loops_t0):
        t0 = t0_offset + (loop_t0 * T0BLOCK_SUB) + base_t0[:,None,None,None]
        t0_mask = t0 < min(T0BLOCK+t0_offset, t0_numel)
        for loop_z1 in range(loops_z1):
            z1 = z1_offset + (loop_z1 * Z1BLOCK_SUB) + base_z1[None,:,None,None]
            z1_mask = z1 < min(Z1BLOCK+z1_offset, z1_numel)
            y2 = base_y2[None,None,:,None]
            x3 = base_x3[None,None,None,:]
            tmp0 = tl.load(in_ptr0 + (x3 + 7*y2 + 49*z1 + 18816*t0), t0_mask & z1_mask)
            tmp1 = -tmp0
            tmp2 = libdevice.exp(tmp1)
            tmp3 = tl.full([1, 1, 1, 1], 1.0, tl.float32)
            tmp4 = tmp2 + tmp3
            tmp5 = (tmp0 / tmp4)
            tl.store(out_ptr0 + (x3 + 7*y2 + 49*z1 + 18816*t0), tmp5, t0_mask & z1_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [x_se_36], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_36 => mean_9
# Graph fragment:
#   %convolution_65 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0" = PlaceHolder[target=convolution_65]
#   %buf451 : Tensor "f32[128, 1536, 1, 1][1536, 1, 196608, 196608]npu:0" = PlaceHolder[target=buf451]
#   %mean_9 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_65, [2, 3], True), kwargs = {})
#   return %buf451,%mean_9
# SchedulerNodes: [SchedulerNode(name='op451'), SchedulerNode(name='op452')]

triton_per_fused_mean_45 = async_compile.triton('triton_per_fused_mean_45', '''
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
    size_hints={'y0': 128, 'x1': 1536, 'r2': 49},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'r2_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_45', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['y0', 'x1', 'r2'], 'axis_static_values': (('y0', 128), ('x1', 1536), ('r2', 49)), 'low_dims': {2}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_mean_45(in_out_ptr0, in_ptr0, y0_numel, x1_numel, r2_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    R2BLOCK_SUB: tl.constexpr = 49
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    x1_offset = tl.program_id(1) * X1BLOCK
    base_x1= tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (X1BLOCK + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    base_r2= tl.arange(0, R2BLOCK_SUB)
    loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        for loop_x1 in range(loops_x1):
            x1 = x1_offset + (loop_x1 * X1BLOCK_SUB) + base_x1[None,:,None]
            x1_mask = x1 < min(X1BLOCK+x1_offset, x1_numel)
            r2 = base_r2[None,None,:]
            r2_mask = True
            tmp0 = tl.load(in_ptr0 + (r2 + 49*x1 + 75264*y0), x1_mask & y0_mask, other=0.0)
            tmp2 = tl.where(tl.broadcast_to((y0_mask.reshape([Y0BLOCK_SUB, 1, 1]) & x1_mask.reshape([1, X1BLOCK_SUB, 1])), [Y0BLOCK_SUB, X1BLOCK_SUB, R2BLOCK_SUB]), tmp0, 0)
            tmp3 = tl.sum(tmp2, 2).reshape(Y0BLOCK_SUB, X1BLOCK_SUB, 1)
            tmp4 = tl.full([1, 1, 1], 49.0, tl.float32)
            tmp5 = (tmp3 / tmp4)
            tl.store(in_out_ptr0 + (x1 + 1536*y0), tmp5, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [sigmoid_9, mul_85, out_68, mul_87, out_69, silu_43, out_70], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
# Source node to ATen node mapping:
#   mul_85 => mul_181
#   mul_87 => mul_183
#   out_68 => mul_182
#   out_69 => add_100
#   out_70 => mul_184
#   sigmoid_9 => sigmoid_9
#   silu_43 => add_101, div_43, exp_43, neg_43
# Graph fragment:
#   %convolution_65 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0" = PlaceHolder[target=convolution_65]
#   %convolution_67 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0" = PlaceHolder[target=convolution_67]
#   %convolution_61 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0" = PlaceHolder[target=convolution_61]
#   %sigmoid_9 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_67,), kwargs = {})
#   %mul_181 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_65, %sigmoid_9), kwargs = {})
#   %mul_182 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_181, 2.0), kwargs = {})
#   %mul_183 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_182, 0.2), kwargs = {})
#   %add_100 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_183, %convolution_61), kwargs = {})
#   %neg_43 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_100,), kwargs = {})
#   %exp_43 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_43,), kwargs = {})
#   %add_101 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_43, 1), kwargs = {})
#   %div_43 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_100, %add_101), kwargs = {})
#   %mul_184 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_43, 0.9805806756909201), kwargs = {})
#   return %mul_184
# SchedulerNodes: [SchedulerNode(name='op458')]

triton_poi_fused_add_mul_sigmoid_silu_46 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_silu_46', '''
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
    size_hints={'y0': 196608, 'x1': 49}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_silu_46', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 196608), ('x1', 49)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_silu_46(in_ptr0, in_ptr1, in_ptr2, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
    x1_numel = 49
    X1BLOCK_SUB: tl.constexpr = 49
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        x1 = base_x1[None,:]
        tmp0 = tl.load(in_ptr0 + (x1 + 49*y0), y0_mask)
        tmp1 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp8 = tl.load(in_ptr2 + (x1 + 49*y0), y0_mask)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = tl.full([1, 1], 2.0, tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full([1, 1], 0.2, tl.float32)
        tmp7 = tmp5 * tmp6
        tmp9 = tmp7 + tmp8
        tmp10 = -tmp9
        tmp11 = libdevice.exp(tmp10)
        tmp12 = tl.full([1, 1], 1.0, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = (tmp9 / tmp13)
        tmp15 = tl.full([1, 1], 0.9805806756909201, tl.float32)
        tmp16 = tmp14 * tmp15
        tl.store(out_ptr0 + (x1 + 49*y0), tmp16, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [sigmoid_9, mul_85, out_68, mul_87, out_69, sigmoid_10, mul_93, out_75, mul_95, out_76, silu_47, out_77], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
# Source node to ATen node mapping:
#   mul_85 => mul_181
#   mul_87 => mul_183
#   mul_93 => mul_197
#   mul_95 => mul_199
#   out_68 => mul_182
#   out_69 => add_100
#   out_75 => mul_198
#   out_76 => add_109
#   out_77 => mul_200
#   sigmoid_10 => sigmoid_10
#   sigmoid_9 => sigmoid_9
#   silu_47 => add_110, div_47, exp_47, neg_47
# Graph fragment:
#   %convolution_71 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0" = PlaceHolder[target=convolution_71]
#   %convolution_73 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0" = PlaceHolder[target=convolution_73]
#   %convolution_65 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0" = PlaceHolder[target=convolution_65]
#   %convolution_67 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0" = PlaceHolder[target=convolution_67]
#   %convolution_61 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0" = PlaceHolder[target=convolution_61]
#   %add_109 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0" = PlaceHolder[target=add_109]
#   %sigmoid_9 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_67,), kwargs = {})
#   %mul_181 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_65, %sigmoid_9), kwargs = {})
#   %mul_182 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_181, 2.0), kwargs = {})
#   %mul_183 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_182, 0.2), kwargs = {})
#   %add_100 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_183, %convolution_61), kwargs = {})
#   %sigmoid_10 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_73,), kwargs = {})
#   %mul_197 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_71, %sigmoid_10), kwargs = {})
#   %mul_198 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_197, 2.0), kwargs = {})
#   %mul_199 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_198, 0.2), kwargs = {})
#   %add_109 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_199, %add_100), kwargs = {})
#   %neg_47 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_109,), kwargs = {})
#   %exp_47 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_47,), kwargs = {})
#   %add_110 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_47, 1), kwargs = {})
#   %div_47 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_109, %add_110), kwargs = {})
#   %mul_200 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_47, 0.9622504486493761), kwargs = {})
#   return %add_109,%mul_200
# SchedulerNodes: [SchedulerNode(name='op497'), SchedulerNode(name='op498')]

triton_poi_fused_add_mul_sigmoid_silu_47 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_silu_47', '''
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
    size_hints={'y0': 196608, 'x1': 49}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_silu_47', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 196608), ('x1', 49)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_silu_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
    x1_numel = 49
    X1BLOCK_SUB: tl.constexpr = 49
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        x1 = base_x1[None,:]
        tmp0 = tl.load(in_ptr0 + (x1 + 49*y0), y0_mask)
        tmp1 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp8 = tl.load(in_ptr2 + (x1 + 49*y0), y0_mask)
        tmp9 = tl.load(in_ptr3 + (y0), y0_mask)
        tmp14 = tl.load(in_ptr4 + (x1 + 49*y0), y0_mask)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = tl.full([1, 1], 2.0, tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full([1, 1], 0.2, tl.float32)
        tmp7 = tmp5 * tmp6
        tmp10 = tl.sigmoid(tmp9)
        tmp11 = tmp8 * tmp10
        tmp12 = tmp11 * tmp4
        tmp13 = tmp12 * tmp6
        tmp15 = tmp13 + tmp14
        tmp16 = tmp7 + tmp15
        tmp17 = -tmp16
        tmp18 = libdevice.exp(tmp17)
        tmp19 = tl.full([1, 1], 1.0, tl.float32)
        tmp20 = tmp18 + tmp19
        tmp21 = (tmp16 / tmp20)
        tmp22 = tl.full([1, 1], 0.9622504486493761, tl.float32)
        tmp23 = tmp21 * tmp22
        tl.store(out_ptr0 + (x1 + 49*y0), tmp16, y0_mask)
        tl.store(out_ptr1 + (x1 + 49*y0), tmp23, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [sigmoid_11, mul_101, out_82, mul_103, out_83], Original ATen: [aten.sigmoid, aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_101 => mul_213
#   mul_103 => mul_215
#   out_82 => mul_214
#   out_83 => add_118
#   sigmoid_11 => sigmoid_11
# Graph fragment:
#   %convolution_77 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0" = PlaceHolder[target=convolution_77]
#   %convolution_79 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0" = PlaceHolder[target=convolution_79]
#   %add_109 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0" = PlaceHolder[target=add_109]
#   %sigmoid_11 : Tensor "f32[128, 1536, 1, 1][1536, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_79,), kwargs = {})
#   %mul_213 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_77, %sigmoid_11), kwargs = {})
#   %mul_214 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_213, 2.0), kwargs = {})
#   %mul_215 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_214, 0.2), kwargs = {})
#   %add_118 : Tensor "f32[128, 1536, 7, 7][75264, 49, 7, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_215, %add_109), kwargs = {})
#   return %add_118
# SchedulerNodes: [SchedulerNode(name='op537')]

triton_poi_fused_add_mul_sigmoid_48 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_48', '''
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
    size_hints={'y0': 196608, 'x1': 49}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_48', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 196608), ('x1', 49)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_48(in_ptr0, in_ptr1, in_ptr2, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
    x1_numel = 49
    X1BLOCK_SUB: tl.constexpr = 49
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        x1 = base_x1[None,:]
        tmp0 = tl.load(in_ptr0 + (x1 + 49*y0), y0_mask)
        tmp1 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp8 = tl.load(in_ptr2 + (x1 + 49*y0), y0_mask)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = tl.full([1, 1], 2.0, tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full([1, 1], 0.2, tl.float32)
        tmp7 = tmp5 * tmp6
        tmp9 = tmp7 + tmp8
        tl.store(out_ptr0 + (x1 + 49*y0), tmp9, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [reshape_56, mul_104, view_56, batch_norm_56], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   batch_norm_56 => add_119, mul_217, mul_218, rsqrt_56, sub_56, unsqueeze_56, var_mean_56
#   mul_104 => mul_216
#   reshape_56 => view_168
#   view_56 => view_169
# Graph fragment:
#   %primals_218 : Tensor "f32[2304, 1536, 1, 1][1536, 1, 1, 1]npu:0" = PlaceHolder[target=primals_218]
#   %buf538 : Tensor "f32[1, 2304, 1][2304, 1, 2304]npu:0" = PlaceHolder[target=buf538]
#   %getitem_113 : Tensor "f32[1, 2304, 1][2304, 1, 2304]npu:0" = PlaceHolder[target=getitem_113]
#   %buf540 : Tensor "f32[1, 2304, 1][2304, 1, 2304]npu:0" = PlaceHolder[target=buf540]
#   %primals_219 : Tensor "f32[2304, 1, 1, 1][1, 1, 1, 1]npu:0" = PlaceHolder[target=primals_219]
#   %view_168 : Tensor "f32[1, 2304, 1536][3538944, 1536, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_218, [1, 2304, -1]), kwargs = {})
#   %mul_216 : Tensor "f32[2304, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_219, 0.04562504637317021), kwargs = {})
#   %view_169 : Tensor "f32[2304][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_216, [-1]), kwargs = {})
#   %var_mean_56 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_168, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_119 : Tensor "f32[1, 2304, 1][2304, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_112, 1e-05), kwargs = {})
#   %rsqrt_56 : Tensor "f32[1, 2304, 1][2304, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_119,), kwargs = {})
#   %sub_56 : Tensor "f32[1, 2304, 1536][3538944, 1536, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_168, %getitem_113), kwargs = {})
#   %mul_217 : Tensor "f32[1, 2304, 1536][3538944, 1536, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_56, %rsqrt_56), kwargs = {})
#   %unsqueeze_56 : Tensor "f32[2304, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_169, -1), kwargs = {})
#   %mul_218 : Tensor "f32[1, 2304, 1536][3538944, 1536, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_217, %unsqueeze_56), kwargs = {})
#   return %buf538,%getitem_113,%buf540,%mul_218,%rsqrt_56
# SchedulerNodes: [SchedulerNode(name='op538'), SchedulerNode(name='op539'), SchedulerNode(name='op540'), SchedulerNode(name='op542'), SchedulerNode(name='op541')]

triton_per_fused__native_batch_norm_legit_mul__49 = async_compile.triton('triton_per_fused__native_batch_norm_legit_mul__49', '''
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
    size_hints={'x0': 2304, 'r1': 1536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_mul__49', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 2304), ('r1', 1536)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_mul__49(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 1536
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
        tmp0 = tl.load(in_ptr0 + (r1 + 1536*x0), x0_mask, other=0.0)
        tmp16 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 1).reshape(X0BLOCK_SUB, 1)
        tmp4 = tl.full([1, 1], 1536.0, tl.float32)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tmp0 - tmp5
        tmp7 = tmp6 * tmp6
        tmp9 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp7, 0)
        tmp10 = tl.sum(tmp9, 1).reshape(X0BLOCK_SUB, 1)
        tmp11 = (tmp10 / tmp4)
        tmp12 = tl.full([1, 1], 1e-05, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp6 * tmp14
        tmp17 = tl.full([1, 1], 0.04562504637317021, tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 * tmp18
        tl.store(in_out_ptr0 + (x0), tmp5, x0_mask)
        tl.store(out_ptr1 + (r1 + 1536*x0), tmp19, x0_mask)
        tl.store(out_ptr2 + (x0), tmp14, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_1 => add_120, div_51, exp_51, neg_51
#   x_2 => mean_12
# Graph fragment:
#   %convolution_80 : Tensor "f32[128, 2304, 7, 7][112896, 49, 7, 1]npu:0" = PlaceHolder[target=convolution_80]
#   %buf545 : Tensor "f32[128, 2304, 1, 1][2304, 1, 294912, 294912]npu:0" = PlaceHolder[target=buf545]
#   %neg_51 : Tensor "f32[128, 2304, 7, 7][112896, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convolution_80,), kwargs = {})
#   %exp_51 : Tensor "f32[128, 2304, 7, 7][112896, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_51,), kwargs = {})
#   %add_120 : Tensor "f32[128, 2304, 7, 7][112896, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_51, 1), kwargs = {})
#   %div_51 : Tensor "f32[128, 2304, 7, 7][112896, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_80, %add_120), kwargs = {})
#   %mean_12 : Tensor "f32[128, 2304, 1, 1][2304, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_51, [-1, -2], True), kwargs = {})
#   return %buf545,%mean_12
# SchedulerNodes: [SchedulerNode(name='op545'), SchedulerNode(name='op546')]

triton_per_fused_mean_silu_50 = async_compile.triton('triton_per_fused_mean_silu_50', '''
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
    size_hints={'y0': 128, 'x1': 2304, 'r2': 49},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'r2_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_50', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['y0', 'x1', 'r2'], 'axis_static_values': (('y0', 128), ('x1', 2304), ('r2', 49)), 'low_dims': {2}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_mean_silu_50(in_out_ptr0, in_ptr0, y0_numel, x1_numel, r2_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    R2BLOCK_SUB: tl.constexpr = 49
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    x1_offset = tl.program_id(1) * X1BLOCK
    base_x1= tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (X1BLOCK + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    base_r2= tl.arange(0, R2BLOCK_SUB)
    loops_r2 = (r2_numel + R2BLOCK_SUB - 1) // R2BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        for loop_x1 in range(loops_x1):
            x1 = x1_offset + (loop_x1 * X1BLOCK_SUB) + base_x1[None,:,None]
            x1_mask = x1 < min(X1BLOCK+x1_offset, x1_numel)
            r2 = base_r2[None,None,:]
            r2_mask = True
            tmp0 = tl.load(in_ptr0 + (r2 + 49*x1 + 112896*y0), x1_mask & y0_mask, other=0.0)
            tmp1 = -tmp0
            tmp2 = libdevice.exp(tmp1)
            tmp3 = tl.full([1, 1, 1], 1.0, tl.float32)
            tmp4 = tmp2 + tmp3
            tmp5 = (tmp0 / tmp4)
            tmp7 = tl.where(tl.broadcast_to((y0_mask.reshape([Y0BLOCK_SUB, 1, 1]) & x1_mask.reshape([1, X1BLOCK_SUB, 1])), [Y0BLOCK_SUB, X1BLOCK_SUB, R2BLOCK_SUB]), tmp5, 0)
            tmp8 = tl.sum(tmp7, 2).reshape(Y0BLOCK_SUB, X1BLOCK_SUB, 1)
            tmp9 = tl.full([1, 1, 1], 49.0, tl.float32)
            tmp10 = (tmp8 / tmp9)
            tl.store(in_out_ptr0 + (x1 + 2304*y0), tmp10, x1_mask & y0_mask)
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            buf0 = empty_strided_npu((1, 16, 1), (16, 1, 16), torch.float32)
            buf1 = buf0; del buf0  # reuse
            buf4 = empty_strided_npu((1, 16, 27), (432, 27, 1), torch.float32)
            buf3 = empty_strided_npu((1, 16, 1), (16, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape, mul, view, batch_norm], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__0.run(buf1, primals_1, primals_2, buf4, buf3, 16, 27, stream=raw_stream0)
            primals_4 = copy_if_misaligned(primals_4)
            # Topologically Sorted Source Nodes: [reshape, mul, view, batch_norm, weight, input_1], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:1
            buf5 = torch.ops.aten.convolution.default(primals_4, reinterpret_tensor(buf4, (16, 3, 3, 3), (27, 9, 3, 1), 0), primals_3, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
            del primals_3
            buf6 = buf5
            assert_size_stride(buf6, (128, 16, 112, 112), (200704, 12544, 112, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf6, 16, 'torch.ops.aten.convolution.default')
            del buf5
            buf7 = empty_strided_npu((128, 16, 112, 112), (200704, 12544, 112, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_1.run(buf6, buf7, 128, 16, 112, 112, stream=raw_stream0)
            buf8 = empty_strided_npu((1, 32, 1), (32, 1, 32), torch.float32)
            buf9 = buf8; del buf8  # reuse
            buf12 = empty_strided_npu((1, 32, 144), (4608, 144, 1), torch.float32)
            buf11 = empty_strided_npu((1, 32, 1), (32, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_1, mul_1, view_1, batch_norm_1], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__2.run(buf9, primals_5, primals_6, buf12, buf11, 32, 144, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_1, mul_1, view_1, batch_norm_1, weight_1, input_3], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:2
            buf13 = torch.ops.aten.convolution.default(buf7, reinterpret_tensor(buf12, (32, 16, 3, 3), (144, 9, 3, 1), 0), primals_7, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del primals_7
            buf14 = buf13
            assert_size_stride(buf14, (128, 32, 112, 112), (401408, 12544, 112, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf14, 16, 'torch.ops.aten.convolution.default')
            del buf13
            buf15 = empty_strided_npu((128, 32, 112, 112), (401408, 12544, 112, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_3.run(buf14, buf15, 128, 32, 112, 112, stream=raw_stream0)
            buf16 = empty_strided_npu((1, 64, 1), (64, 1, 64), torch.float32)
            buf17 = buf16; del buf16  # reuse
            buf20 = empty_strided_npu((1, 64, 288), (18432, 288, 1), torch.float32)
            buf19 = empty_strided_npu((1, 64, 1), (64, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_2, mul_2, view_2, batch_norm_2], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__4.run(buf17, primals_8, primals_9, buf20, buf19, 64, 288, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_2, mul_2, view_2, batch_norm_2, weight_2, input_5], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:3
            buf21 = torch.ops.aten.convolution.default(buf15, reinterpret_tensor(buf20, (64, 32, 3, 3), (288, 9, 3, 1), 0), primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del primals_10
            buf22 = buf21
            assert_size_stride(buf22, (128, 64, 112, 112), (802816, 12544, 112, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf22, 16, 'torch.ops.aten.convolution.default')
            del buf21
            buf23 = empty_strided_npu((128, 64, 112, 112), (802816, 12544, 112, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_5.run(buf22, buf23, 128, 64, 112, 112, stream=raw_stream0)
            buf24 = empty_strided_npu((1, 128, 1), (128, 1, 128), torch.float32)
            buf25 = buf24; del buf24  # reuse
            buf28 = empty_strided_npu((1, 128, 576), (73728, 576, 1), torch.float32)
            buf27 = empty_strided_npu((1, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_3, mul_3, view_3, batch_norm_3], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__6.run(buf25, primals_11, primals_12, buf28, buf27, 128, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_3, mul_3, view_3, batch_norm_3, weight_3, input_7], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:4
            buf29 = torch.ops.aten.convolution.default(buf23, reinterpret_tensor(buf28, (128, 64, 3, 3), (576, 9, 3, 1), 0), primals_13, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
            del primals_13
            buf30 = buf29
            assert_size_stride(buf30, (128, 128, 56, 56), (401408, 3136, 56, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf30, 16, 'torch.ops.aten.convolution.default')
            del buf29
            buf31 = empty_strided_npu((128, 128, 56, 56), (401408, 3136, 56, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_3, mul_12], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_7.run(buf30, buf31, 128, 128, 56, 56, stream=raw_stream0)
            buf32 = empty_strided_npu((1, 256, 1), (256, 1, 256), torch.float32)
            buf33 = buf32; del buf32  # reuse
            buf36 = empty_strided_npu((1, 256, 128), (32768, 128, 1), torch.float32)
            buf35 = empty_strided_npu((1, 256, 1), (256, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_4, mul_5, view_4, batch_norm_4], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__8.run(buf33, primals_14, primals_15, buf36, buf35, 256, 128, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_4, mul_5, view_4, batch_norm_4, weight_4, shortcut], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:5
            buf37 = torch.ops.aten.convolution.default(buf31, reinterpret_tensor(buf36, (256, 128, 1, 1), (128, 1, 0, 0), 0), primals_16, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_16
            buf38 = buf37
            assert_size_stride(buf38, (128, 256, 56, 56), (802816, 3136, 56, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf38, 16, 'torch.ops.aten.convolution.default')
            del buf37
            buf39 = empty_strided_npu((1, 64, 1), (64, 1, 64), torch.float32)
            buf40 = buf39; del buf39  # reuse
            buf43 = empty_strided_npu((1, 64, 128), (8192, 128, 1), torch.float32)
            buf42 = empty_strided_npu((1, 64, 1), (64, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_5, mul_6, view_5, batch_norm_5], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__9.run(buf40, primals_17, primals_18, buf43, buf42, 64, 128, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_5, mul_6, view_5, batch_norm_5, weight_5, out_1], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:6
            buf44 = torch.ops.aten.convolution.default(buf31, reinterpret_tensor(buf43, (64, 128, 1, 1), (128, 1, 0, 0), 0), primals_19, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_19
            buf45 = buf44
            assert_size_stride(buf45, (128, 64, 56, 56), (200704, 3136, 56, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf45, 16, 'torch.ops.aten.convolution.default')
            del buf44
            buf46 = empty_strided_npu((128, 64, 56, 56), (200704, 3136, 56, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_4], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_10.run(buf45, buf46, 128, 64, 56, 56, stream=raw_stream0)
            buf47 = empty_strided_npu((1, 64, 1), (64, 1, 64), torch.float32)
            buf48 = buf47; del buf47  # reuse
            buf51 = empty_strided_npu((1, 64, 576), (36864, 576, 1), torch.float32)
            buf50 = empty_strided_npu((1, 64, 1), (64, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_6, mul_7, view_6, batch_norm_6], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__11.run(buf48, primals_20, primals_21, buf51, buf50, 64, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_6, mul_7, view_6, batch_norm_6, weight_6, out_2], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:7
            buf52 = torch.ops.aten.convolution.default(buf46, reinterpret_tensor(buf51, (64, 64, 3, 3), (576, 9, 3, 1), 0), primals_22, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del primals_22
            buf53 = buf52
            assert_size_stride(buf53, (128, 64, 56, 56), (200704, 3136, 56, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf53, 16, 'torch.ops.aten.convolution.default')
            del buf52
            buf54 = empty_strided_npu((128, 64, 56, 56), (200704, 3136, 56, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_5], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_10.run(buf53, buf54, 128, 64, 56, 56, stream=raw_stream0)
            buf55 = empty_strided_npu((1, 64, 1), (64, 1, 64), torch.float32)
            buf56 = buf55; del buf55  # reuse
            buf59 = empty_strided_npu((1, 64, 576), (36864, 576, 1), torch.float32)
            buf58 = empty_strided_npu((1, 64, 1), (64, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_7, mul_8, view_7, batch_norm_7], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__11.run(buf56, primals_23, primals_24, buf59, buf58, 64, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_7, mul_8, view_7, batch_norm_7, weight_7, out_3], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:8
            buf60 = torch.ops.aten.convolution.default(buf54, reinterpret_tensor(buf59, (64, 64, 3, 3), (576, 9, 3, 1), 0), primals_25, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del primals_25
            buf61 = buf60
            assert_size_stride(buf61, (128, 64, 56, 56), (200704, 3136, 56, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf61, 16, 'torch.ops.aten.convolution.default')
            del buf60
            buf62 = empty_strided_npu((128, 64, 56, 56), (200704, 3136, 56, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_6], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_10.run(buf61, buf62, 128, 64, 56, 56, stream=raw_stream0)
            buf63 = empty_strided_npu((1, 256, 1), (256, 1, 256), torch.float32)
            buf64 = buf63; del buf63  # reuse
            buf67 = empty_strided_npu((1, 256, 64), (16384, 64, 1), torch.float32)
            buf66 = empty_strided_npu((1, 256, 1), (256, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_8, mul_9, view_8, batch_norm_8], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__12.run(buf64, primals_26, primals_27, buf67, buf66, 256, 64, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_8, mul_9, view_8, batch_norm_8, weight_8, out_4], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:9
            buf68 = torch.ops.aten.convolution.default(buf62, reinterpret_tensor(buf67, (256, 64, 1, 1), (64, 1, 0, 0), 0), primals_28, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_28
            buf69 = buf68
            assert_size_stride(buf69, (128, 256, 56, 56), (802816, 3136, 56, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf69, 16, 'torch.ops.aten.convolution.default')
            del buf68
            buf70 = empty_strided_npu((128, 256, 1, 1), (256, 1, 32768, 32768), torch.float32)
            buf71 = reinterpret_tensor(buf70, (128, 256, 1, 1), (256, 1, 1, 1), 0); del buf70  # reuse
            # Topologically Sorted Source Nodes: [x_se], Original ATen: [aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_mean_13.run(buf71, buf69, 128, 256, 3136, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_1], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:10
            buf72 = torch.ops.aten.convolution.default(buf71, primals_29, primals_30, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_30
            buf73 = buf72
            assert_size_stride(buf73, (128, 64, 1, 1), (64, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf73, 16, 'torch.ops.aten.convolution.default')
            del buf72
            buf74 = buf73; del buf73  # reuse
            # Topologically Sorted Source Nodes: [x_se_2], Original ATen: [aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_relu_14.run(buf74, 128, 64, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_3], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:11
            buf75 = torch.ops.aten.convolution.default(buf74, primals_31, primals_32, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_32
            buf76 = buf75
            assert_size_stride(buf76, (128, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf76, 16, 'torch.ops.aten.convolution.default')
            del buf75
            buf77 = empty_strided_npu((128, 256, 56, 56), (802816, 3136, 56, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sigmoid, mul_10, out_5, mul_12, out_6, silu_7, out_7], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_sigmoid_silu_15.run(buf69, buf76, buf38, buf77, 32768, 3136, stream=raw_stream0)
            buf78 = empty_strided_npu((128, 256, 28, 28), (200704, 784, 28, 1), torch.float32)
            # Topologically Sorted Source Nodes: [avg_pool2d], Original ATen: [aten.avg_pool2d]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_avg_pool2d_16.run(buf77, buf78, 917504, 28, stream=raw_stream0)
            buf79 = empty_strided_npu((1, 512, 1), (512, 1, 512), torch.float32)
            buf80 = buf79; del buf79  # reuse
            buf83 = empty_strided_npu((1, 512, 256), (131072, 256, 1), torch.float32)
            buf82 = empty_strided_npu((1, 512, 1), (512, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_9, mul_14, view_9, batch_norm_9], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__17.run(buf80, primals_33, primals_34, buf83, buf82, 512, 256, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_9, mul_14, view_9, batch_norm_9, weight_9, shortcut_1], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:12
            buf84 = torch.ops.aten.convolution.default(buf78, reinterpret_tensor(buf83, (512, 256, 1, 1), (256, 1, 0, 0), 0), primals_35, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_35
            buf85 = buf84
            assert_size_stride(buf85, (128, 512, 28, 28), (401408, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf85, 16, 'torch.ops.aten.convolution.default')
            del buf84
            buf86 = empty_strided_npu((1, 128, 1), (128, 1, 128), torch.float32)
            buf87 = buf86; del buf86  # reuse
            buf90 = empty_strided_npu((1, 128, 256), (32768, 256, 1), torch.float32)
            buf89 = empty_strided_npu((1, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_10, mul_15, view_10, batch_norm_10], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__18.run(buf87, primals_36, primals_37, buf90, buf89, 128, 256, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_10, mul_15, view_10, batch_norm_10, weight_10, out_8], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:13
            buf91 = torch.ops.aten.convolution.default(buf77, reinterpret_tensor(buf90, (128, 256, 1, 1), (256, 1, 0, 0), 0), primals_38, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_38
            buf92 = buf91
            assert_size_stride(buf92, (128, 128, 56, 56), (401408, 3136, 56, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf92, 16, 'torch.ops.aten.convolution.default')
            del buf91
            buf93 = empty_strided_npu((128, 128, 56, 56), (401408, 3136, 56, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_8], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_7.run(buf92, buf93, 128, 128, 56, 56, stream=raw_stream0)
            buf94 = empty_strided_npu((1, 128, 1), (128, 1, 128), torch.float32)
            buf95 = buf94; del buf94  # reuse
            buf98 = empty_strided_npu((1, 128, 576), (73728, 576, 1), torch.float32)
            buf97 = empty_strided_npu((1, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_11, mul_16, view_11, batch_norm_11], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__6.run(buf95, primals_39, primals_40, buf98, buf97, 128, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_11, mul_16, view_11, batch_norm_11, weight_11, out_9], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:14
            buf99 = torch.ops.aten.convolution.default(buf93, reinterpret_tensor(buf98, (128, 64, 3, 3), (576, 9, 3, 1), 0), primals_41, [2, 2], [1, 1], [1, 1], False, [0, 0], 2)
            del primals_41
            buf100 = buf99
            assert_size_stride(buf100, (128, 128, 28, 28), (100352, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf100, 16, 'torch.ops.aten.convolution.default')
            del buf99
            buf101 = empty_strided_npu((128, 128, 28, 28), (100352, 784, 28, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_9], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_19.run(buf100, buf101, 128, 128, 28, 28, stream=raw_stream0)
            buf102 = empty_strided_npu((1, 128, 1), (128, 1, 128), torch.float32)
            buf103 = buf102; del buf102  # reuse
            buf106 = empty_strided_npu((1, 128, 576), (73728, 576, 1), torch.float32)
            buf105 = empty_strided_npu((1, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_12, mul_17, view_12, batch_norm_12], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__6.run(buf103, primals_42, primals_43, buf106, buf105, 128, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_12, mul_17, view_12, batch_norm_12, weight_12, out_10], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:15
            buf107 = torch.ops.aten.convolution.default(buf101, reinterpret_tensor(buf106, (128, 64, 3, 3), (576, 9, 3, 1), 0), primals_44, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
            del primals_44
            buf108 = buf107
            assert_size_stride(buf108, (128, 128, 28, 28), (100352, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf108, 16, 'torch.ops.aten.convolution.default')
            del buf107
            buf109 = empty_strided_npu((128, 128, 28, 28), (100352, 784, 28, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_10], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_19.run(buf108, buf109, 128, 128, 28, 28, stream=raw_stream0)
            buf110 = empty_strided_npu((1, 512, 1), (512, 1, 512), torch.float32)
            buf111 = buf110; del buf110  # reuse
            buf114 = empty_strided_npu((1, 512, 128), (65536, 128, 1), torch.float32)
            buf113 = empty_strided_npu((1, 512, 1), (512, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_13, mul_18, view_13, batch_norm_13], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__20.run(buf111, primals_45, primals_46, buf114, buf113, 512, 128, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_13, mul_18, view_13, batch_norm_13, weight_13, out_11], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:16
            buf115 = torch.ops.aten.convolution.default(buf109, reinterpret_tensor(buf114, (512, 128, 1, 1), (128, 1, 0, 0), 0), primals_47, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_47
            buf116 = buf115
            assert_size_stride(buf116, (128, 512, 28, 28), (401408, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf116, 16, 'torch.ops.aten.convolution.default')
            del buf115
            buf117 = empty_strided_npu((128, 512, 1, 1), (512, 1, 65536, 65536), torch.float32)
            buf118 = reinterpret_tensor(buf117, (128, 512, 1, 1), (512, 1, 1, 1), 0); del buf117  # reuse
            # Topologically Sorted Source Nodes: [x_se_4], Original ATen: [aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_mean_21.run(buf118, buf116, 128, 512, 784, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_5], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:17
            buf119 = torch.ops.aten.convolution.default(buf118, primals_48, primals_49, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_49
            buf120 = buf119
            assert_size_stride(buf120, (128, 128, 1, 1), (128, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf120, 16, 'torch.ops.aten.convolution.default')
            del buf119
            buf121 = buf120; del buf120  # reuse
            # Topologically Sorted Source Nodes: [x_se_6], Original ATen: [aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_relu_22.run(buf121, 128, 128, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_7], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:18
            buf122 = torch.ops.aten.convolution.default(buf121, primals_50, primals_51, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_51
            buf123 = buf122
            assert_size_stride(buf123, (128, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf123, 16, 'torch.ops.aten.convolution.default')
            del buf122
            buf124 = empty_strided_npu((128, 512, 28, 28), (401408, 784, 28, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sigmoid_1, mul_19, out_12, mul_21, out_13, silu_11, out_14], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_sigmoid_silu_23.run(buf116, buf123, buf85, buf124, 65536, 784, stream=raw_stream0)
            buf125 = empty_strided_npu((1, 128, 1), (128, 1, 128), torch.float32)
            buf126 = buf125; del buf125  # reuse
            buf129 = empty_strided_npu((1, 128, 512), (65536, 512, 1), torch.float32)
            buf128 = empty_strided_npu((1, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_14, mul_23, view_14, batch_norm_14], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__24.run(buf126, primals_52, primals_53, buf129, buf128, 128, 512, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_14, mul_23, view_14, batch_norm_14, weight_14, out_15], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:19
            buf130 = torch.ops.aten.convolution.default(buf124, reinterpret_tensor(buf129, (128, 512, 1, 1), (512, 1, 0, 0), 0), primals_54, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_54
            buf131 = buf130
            assert_size_stride(buf131, (128, 128, 28, 28), (100352, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf131, 16, 'torch.ops.aten.convolution.default')
            del buf130
            buf132 = empty_strided_npu((128, 128, 28, 28), (100352, 784, 28, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_12], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_19.run(buf131, buf132, 128, 128, 28, 28, stream=raw_stream0)
            buf133 = empty_strided_npu((1, 128, 1), (128, 1, 128), torch.float32)
            buf134 = buf133; del buf133  # reuse
            buf137 = empty_strided_npu((1, 128, 576), (73728, 576, 1), torch.float32)
            buf136 = empty_strided_npu((1, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_15, mul_24, view_15, batch_norm_15], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__6.run(buf134, primals_55, primals_56, buf137, buf136, 128, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_15, mul_24, view_15, batch_norm_15, weight_15, out_16], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:20
            buf138 = torch.ops.aten.convolution.default(buf132, reinterpret_tensor(buf137, (128, 64, 3, 3), (576, 9, 3, 1), 0), primals_57, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
            del primals_57
            buf139 = buf138
            assert_size_stride(buf139, (128, 128, 28, 28), (100352, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf139, 16, 'torch.ops.aten.convolution.default')
            del buf138
            buf140 = empty_strided_npu((128, 128, 28, 28), (100352, 784, 28, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_13], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_19.run(buf139, buf140, 128, 128, 28, 28, stream=raw_stream0)
            buf141 = empty_strided_npu((1, 128, 1), (128, 1, 128), torch.float32)
            buf142 = buf141; del buf141  # reuse
            buf145 = empty_strided_npu((1, 128, 576), (73728, 576, 1), torch.float32)
            buf144 = empty_strided_npu((1, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_16, mul_25, view_16, batch_norm_16], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__6.run(buf142, primals_58, primals_59, buf145, buf144, 128, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_16, mul_25, view_16, batch_norm_16, weight_16, out_17], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:21
            buf146 = torch.ops.aten.convolution.default(buf140, reinterpret_tensor(buf145, (128, 64, 3, 3), (576, 9, 3, 1), 0), primals_60, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
            del primals_60
            buf147 = buf146
            assert_size_stride(buf147, (128, 128, 28, 28), (100352, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf147, 16, 'torch.ops.aten.convolution.default')
            del buf146
            buf148 = empty_strided_npu((128, 128, 28, 28), (100352, 784, 28, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_14], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_19.run(buf147, buf148, 128, 128, 28, 28, stream=raw_stream0)
            buf149 = empty_strided_npu((1, 512, 1), (512, 1, 512), torch.float32)
            buf150 = buf149; del buf149  # reuse
            buf153 = empty_strided_npu((1, 512, 128), (65536, 128, 1), torch.float32)
            buf152 = empty_strided_npu((1, 512, 1), (512, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_17, mul_26, view_17, batch_norm_17], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__20.run(buf150, primals_61, primals_62, buf153, buf152, 512, 128, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_17, mul_26, view_17, batch_norm_17, weight_17, out_18], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:22
            buf154 = torch.ops.aten.convolution.default(buf148, reinterpret_tensor(buf153, (512, 128, 1, 1), (128, 1, 0, 0), 0), primals_63, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_63
            buf155 = buf154
            assert_size_stride(buf155, (128, 512, 28, 28), (401408, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf155, 16, 'torch.ops.aten.convolution.default')
            del buf154
            buf156 = empty_strided_npu((128, 512, 1, 1), (512, 1, 65536, 65536), torch.float32)
            buf157 = reinterpret_tensor(buf156, (128, 512, 1, 1), (512, 1, 1, 1), 0); del buf156  # reuse
            # Topologically Sorted Source Nodes: [x_se_8], Original ATen: [aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_mean_21.run(buf157, buf155, 128, 512, 784, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_9], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:23
            buf158 = torch.ops.aten.convolution.default(buf157, primals_64, primals_65, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_65
            buf159 = buf158
            assert_size_stride(buf159, (128, 128, 1, 1), (128, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf159, 16, 'torch.ops.aten.convolution.default')
            del buf158
            buf160 = buf159; del buf159  # reuse
            # Topologically Sorted Source Nodes: [x_se_10], Original ATen: [aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_relu_22.run(buf160, 128, 128, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_11], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:24
            buf161 = torch.ops.aten.convolution.default(buf160, primals_66, primals_67, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_67
            buf162 = buf161
            assert_size_stride(buf162, (128, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf162, 16, 'torch.ops.aten.convolution.default')
            del buf161
            buf163 = empty_strided_npu((128, 512, 28, 28), (401408, 784, 28, 1), torch.float32)
            buf164 = empty_strided_npu((128, 512, 28, 28), (401408, 784, 28, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sigmoid_1, mul_19, out_12, mul_21, out_13, sigmoid_2, mul_27, out_19, mul_29, out_20, silu_15, out_21], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_sigmoid_silu_25.run(buf155, buf162, buf116, buf123, buf85, buf163, buf164, 65536, 784, stream=raw_stream0)
            buf165 = empty_strided_npu((128, 512, 14, 14), (100352, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [avg_pool2d_1], Original ATen: [aten.avg_pool2d]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_avg_pool2d_26.run(buf164, buf165, 917504, 14, stream=raw_stream0)
            buf166 = empty_strided_npu((1, 1536, 1), (1536, 1, 1536), torch.float32)
            buf167 = buf166; del buf166  # reuse
            buf170 = empty_strided_npu((1, 1536, 512), (786432, 512, 1), torch.float32)
            buf169 = empty_strided_npu((1, 1536, 1), (1536, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_18, mul_31, view_18, batch_norm_18], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__27.run(buf167, primals_68, primals_69, buf170, buf169, 1536, 512, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_18, mul_31, view_18, batch_norm_18, weight_18, shortcut_2], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:25
            buf171 = torch.ops.aten.convolution.default(buf165, reinterpret_tensor(buf170, (1536, 512, 1, 1), (512, 1, 0, 0), 0), primals_70, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_70
            buf172 = buf171
            assert_size_stride(buf172, (128, 1536, 14, 14), (301056, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf172, 16, 'torch.ops.aten.convolution.default')
            del buf171
            buf173 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf174 = buf173; del buf173  # reuse
            buf177 = empty_strided_npu((1, 384, 512), (196608, 512, 1), torch.float32)
            buf176 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_19, mul_32, view_19, batch_norm_19], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__28.run(buf174, primals_71, primals_72, buf177, buf176, 384, 512, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_19, mul_32, view_19, batch_norm_19, weight_19, out_22], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:26
            buf178 = torch.ops.aten.convolution.default(buf164, reinterpret_tensor(buf177, (384, 512, 1, 1), (512, 1, 0, 0), 0), primals_73, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_73
            buf179 = buf178
            assert_size_stride(buf179, (128, 384, 28, 28), (301056, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf179, 16, 'torch.ops.aten.convolution.default')
            del buf178
            buf180 = empty_strided_npu((128, 384, 28, 28), (301056, 784, 28, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_16], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_29.run(buf179, buf180, 128, 384, 28, 28, stream=raw_stream0)
            buf181 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf182 = buf181; del buf181  # reuse
            buf185 = empty_strided_npu((1, 384, 576), (221184, 576, 1), torch.float32)
            buf184 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_20, mul_33, view_20, batch_norm_20], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__30.run(buf182, primals_74, primals_75, buf185, buf184, 384, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_20, mul_33, view_20, batch_norm_20, weight_20, out_23], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:27
            buf186 = torch.ops.aten.convolution.default(buf180, reinterpret_tensor(buf185, (384, 64, 3, 3), (576, 9, 3, 1), 0), primals_76, [2, 2], [1, 1], [1, 1], False, [0, 0], 6)
            del primals_76
            buf187 = buf186
            assert_size_stride(buf187, (128, 384, 14, 14), (75264, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf187, 16, 'torch.ops.aten.convolution.default')
            del buf186
            buf188 = empty_strided_npu((128, 384, 14, 14), (75264, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_17], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_31.run(buf187, buf188, 128, 384, 14, 14, stream=raw_stream0)
            buf189 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf190 = buf189; del buf189  # reuse
            buf193 = empty_strided_npu((1, 384, 576), (221184, 576, 1), torch.float32)
            buf192 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_21, mul_34, view_21, batch_norm_21], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__30.run(buf190, primals_77, primals_78, buf193, buf192, 384, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_21, mul_34, view_21, batch_norm_21, weight_21, out_24], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:28
            buf194 = torch.ops.aten.convolution.default(buf188, reinterpret_tensor(buf193, (384, 64, 3, 3), (576, 9, 3, 1), 0), primals_79, [1, 1], [1, 1], [1, 1], False, [0, 0], 6)
            del primals_79
            buf195 = buf194
            assert_size_stride(buf195, (128, 384, 14, 14), (75264, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf195, 16, 'torch.ops.aten.convolution.default')
            del buf194
            buf196 = empty_strided_npu((128, 384, 14, 14), (75264, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_18], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_31.run(buf195, buf196, 128, 384, 14, 14, stream=raw_stream0)
            buf197 = empty_strided_npu((1, 1536, 1), (1536, 1, 1536), torch.float32)
            buf198 = buf197; del buf197  # reuse
            buf201 = empty_strided_npu((1, 1536, 384), (589824, 384, 1), torch.float32)
            buf200 = empty_strided_npu((1, 1536, 1), (1536, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_22, mul_35, view_22, batch_norm_22], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__32.run(buf198, primals_80, primals_81, buf201, buf200, 1536, 384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_22, mul_35, view_22, batch_norm_22, weight_22, out_25], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:29
            buf202 = torch.ops.aten.convolution.default(buf196, reinterpret_tensor(buf201, (1536, 384, 1, 1), (384, 1, 0, 0), 0), primals_82, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_82
            buf203 = buf202
            assert_size_stride(buf203, (128, 1536, 14, 14), (301056, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf203, 16, 'torch.ops.aten.convolution.default')
            del buf202
            buf204 = empty_strided_npu((128, 1536, 1, 1), (1536, 1, 196608, 196608), torch.float32)
            buf205 = reinterpret_tensor(buf204, (128, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf204  # reuse
            # Topologically Sorted Source Nodes: [x_se_12], Original ATen: [aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_mean_33.run(buf205, buf203, 128, 1536, 196, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_13], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:30
            buf206 = torch.ops.aten.convolution.default(buf205, primals_83, primals_84, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_84
            buf207 = buf206
            assert_size_stride(buf207, (128, 384, 1, 1), (384, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf207, 16, 'torch.ops.aten.convolution.default')
            del buf206
            buf208 = buf207; del buf207  # reuse
            # Topologically Sorted Source Nodes: [x_se_14], Original ATen: [aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_relu_34.run(buf208, 128, 384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_15], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:31
            buf209 = torch.ops.aten.convolution.default(buf208, primals_85, primals_86, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_86
            buf210 = buf209
            assert_size_stride(buf210, (128, 1536, 1, 1), (1536, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf210, 16, 'torch.ops.aten.convolution.default')
            del buf209
            buf211 = empty_strided_npu((128, 1536, 14, 14), (301056, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sigmoid_3, mul_36, out_26, mul_38, out_27, silu_19, out_28], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_sigmoid_silu_35.run(buf203, buf210, buf172, buf211, 196608, 196, stream=raw_stream0)
            buf212 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf213 = buf212; del buf212  # reuse
            buf216 = empty_strided_npu((1, 384, 1536), (589824, 1536, 1), torch.float32)
            buf215 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_23, mul_40, view_23, batch_norm_23], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__36.run(buf213, primals_87, primals_88, buf216, buf215, 384, 1536, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_23, mul_40, view_23, batch_norm_23, weight_23, out_29], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:32
            buf217 = torch.ops.aten.convolution.default(buf211, reinterpret_tensor(buf216, (384, 1536, 1, 1), (1536, 1, 0, 0), 0), primals_89, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_89
            buf218 = buf217
            assert_size_stride(buf218, (128, 384, 14, 14), (75264, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf218, 16, 'torch.ops.aten.convolution.default')
            del buf217
            buf219 = empty_strided_npu((128, 384, 14, 14), (75264, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_20], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_31.run(buf218, buf219, 128, 384, 14, 14, stream=raw_stream0)
            buf220 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf221 = buf220; del buf220  # reuse
            buf224 = empty_strided_npu((1, 384, 576), (221184, 576, 1), torch.float32)
            buf223 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_24, mul_41, view_24, batch_norm_24], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__30.run(buf221, primals_90, primals_91, buf224, buf223, 384, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_24, mul_41, view_24, batch_norm_24, weight_24, out_30], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:33
            buf225 = torch.ops.aten.convolution.default(buf219, reinterpret_tensor(buf224, (384, 64, 3, 3), (576, 9, 3, 1), 0), primals_92, [1, 1], [1, 1], [1, 1], False, [0, 0], 6)
            del primals_92
            buf226 = buf225
            assert_size_stride(buf226, (128, 384, 14, 14), (75264, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf226, 16, 'torch.ops.aten.convolution.default')
            del buf225
            buf227 = empty_strided_npu((128, 384, 14, 14), (75264, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_21], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_31.run(buf226, buf227, 128, 384, 14, 14, stream=raw_stream0)
            buf228 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf229 = buf228; del buf228  # reuse
            buf232 = empty_strided_npu((1, 384, 576), (221184, 576, 1), torch.float32)
            buf231 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_25, mul_42, view_25, batch_norm_25], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__30.run(buf229, primals_93, primals_94, buf232, buf231, 384, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_25, mul_42, view_25, batch_norm_25, weight_25, out_31], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:34
            buf233 = torch.ops.aten.convolution.default(buf227, reinterpret_tensor(buf232, (384, 64, 3, 3), (576, 9, 3, 1), 0), primals_95, [1, 1], [1, 1], [1, 1], False, [0, 0], 6)
            del primals_95
            buf234 = buf233
            assert_size_stride(buf234, (128, 384, 14, 14), (75264, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf234, 16, 'torch.ops.aten.convolution.default')
            del buf233
            buf235 = empty_strided_npu((128, 384, 14, 14), (75264, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_22], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_31.run(buf234, buf235, 128, 384, 14, 14, stream=raw_stream0)
            buf236 = empty_strided_npu((1, 1536, 1), (1536, 1, 1536), torch.float32)
            buf237 = buf236; del buf236  # reuse
            buf240 = empty_strided_npu((1, 1536, 384), (589824, 384, 1), torch.float32)
            buf239 = empty_strided_npu((1, 1536, 1), (1536, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_26, mul_43, view_26, batch_norm_26], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__32.run(buf237, primals_96, primals_97, buf240, buf239, 1536, 384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_26, mul_43, view_26, batch_norm_26, weight_26, out_32], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:35
            buf241 = torch.ops.aten.convolution.default(buf235, reinterpret_tensor(buf240, (1536, 384, 1, 1), (384, 1, 0, 0), 0), primals_98, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_98
            buf242 = buf241
            assert_size_stride(buf242, (128, 1536, 14, 14), (301056, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf242, 16, 'torch.ops.aten.convolution.default')
            del buf241
            buf243 = empty_strided_npu((128, 1536, 1, 1), (1536, 1, 196608, 196608), torch.float32)
            buf244 = reinterpret_tensor(buf243, (128, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf243  # reuse
            # Topologically Sorted Source Nodes: [x_se_16], Original ATen: [aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_mean_33.run(buf244, buf242, 128, 1536, 196, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_17], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:36
            buf245 = torch.ops.aten.convolution.default(buf244, primals_99, primals_100, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_100
            buf246 = buf245
            assert_size_stride(buf246, (128, 384, 1, 1), (384, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf246, 16, 'torch.ops.aten.convolution.default')
            del buf245
            buf247 = buf246; del buf246  # reuse
            # Topologically Sorted Source Nodes: [x_se_18], Original ATen: [aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_relu_34.run(buf247, 128, 384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_19], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:37
            buf248 = torch.ops.aten.convolution.default(buf247, primals_101, primals_102, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_102
            buf249 = buf248
            assert_size_stride(buf249, (128, 1536, 1, 1), (1536, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf249, 16, 'torch.ops.aten.convolution.default')
            del buf248
            buf250 = empty_strided_npu((128, 1536, 14, 14), (301056, 196, 14, 1), torch.float32)
            buf251 = empty_strided_npu((128, 1536, 14, 14), (301056, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sigmoid_3, mul_36, out_26, mul_38, out_27, sigmoid_4, mul_44, out_33, mul_46, out_34, silu_23, out_35], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_sigmoid_silu_37.run(buf242, buf249, buf203, buf210, buf172, buf250, buf251, 196608, 196, stream=raw_stream0)
            buf252 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf253 = buf252; del buf252  # reuse
            buf256 = empty_strided_npu((1, 384, 1536), (589824, 1536, 1), torch.float32)
            buf255 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_27, mul_48, view_27, batch_norm_27], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__36.run(buf253, primals_103, primals_104, buf256, buf255, 384, 1536, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_27, mul_48, view_27, batch_norm_27, weight_27, out_36], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:38
            buf257 = torch.ops.aten.convolution.default(buf251, reinterpret_tensor(buf256, (384, 1536, 1, 1), (1536, 1, 0, 0), 0), primals_105, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_105
            buf258 = buf257
            assert_size_stride(buf258, (128, 384, 14, 14), (75264, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf258, 16, 'torch.ops.aten.convolution.default')
            del buf257
            buf259 = empty_strided_npu((128, 384, 14, 14), (75264, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_24], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_31.run(buf258, buf259, 128, 384, 14, 14, stream=raw_stream0)
            buf260 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf261 = buf260; del buf260  # reuse
            buf264 = empty_strided_npu((1, 384, 576), (221184, 576, 1), torch.float32)
            buf263 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_28, mul_49, view_28, batch_norm_28], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__30.run(buf261, primals_106, primals_107, buf264, buf263, 384, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_28, mul_49, view_28, batch_norm_28, weight_28, out_37], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:39
            buf265 = torch.ops.aten.convolution.default(buf259, reinterpret_tensor(buf264, (384, 64, 3, 3), (576, 9, 3, 1), 0), primals_108, [1, 1], [1, 1], [1, 1], False, [0, 0], 6)
            del primals_108
            buf266 = buf265
            assert_size_stride(buf266, (128, 384, 14, 14), (75264, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf266, 16, 'torch.ops.aten.convolution.default')
            del buf265
            buf267 = empty_strided_npu((128, 384, 14, 14), (75264, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_25], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_31.run(buf266, buf267, 128, 384, 14, 14, stream=raw_stream0)
            buf268 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf269 = buf268; del buf268  # reuse
            buf272 = empty_strided_npu((1, 384, 576), (221184, 576, 1), torch.float32)
            buf271 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_29, mul_50, view_29, batch_norm_29], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__30.run(buf269, primals_109, primals_110, buf272, buf271, 384, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_29, mul_50, view_29, batch_norm_29, weight_29, out_38], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:40
            buf273 = torch.ops.aten.convolution.default(buf267, reinterpret_tensor(buf272, (384, 64, 3, 3), (576, 9, 3, 1), 0), primals_111, [1, 1], [1, 1], [1, 1], False, [0, 0], 6)
            del primals_111
            buf274 = buf273
            assert_size_stride(buf274, (128, 384, 14, 14), (75264, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf274, 16, 'torch.ops.aten.convolution.default')
            del buf273
            buf275 = empty_strided_npu((128, 384, 14, 14), (75264, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_26], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_31.run(buf274, buf275, 128, 384, 14, 14, stream=raw_stream0)
            buf276 = empty_strided_npu((1, 1536, 1), (1536, 1, 1536), torch.float32)
            buf277 = buf276; del buf276  # reuse
            buf280 = empty_strided_npu((1, 1536, 384), (589824, 384, 1), torch.float32)
            buf279 = empty_strided_npu((1, 1536, 1), (1536, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_30, mul_51, view_30, batch_norm_30], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__32.run(buf277, primals_112, primals_113, buf280, buf279, 1536, 384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_30, mul_51, view_30, batch_norm_30, weight_30, out_39], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:41
            buf281 = torch.ops.aten.convolution.default(buf275, reinterpret_tensor(buf280, (1536, 384, 1, 1), (384, 1, 0, 0), 0), primals_114, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_114
            buf282 = buf281
            assert_size_stride(buf282, (128, 1536, 14, 14), (301056, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf282, 16, 'torch.ops.aten.convolution.default')
            del buf281
            buf283 = empty_strided_npu((128, 1536, 1, 1), (1536, 1, 196608, 196608), torch.float32)
            buf284 = reinterpret_tensor(buf283, (128, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf283  # reuse
            # Topologically Sorted Source Nodes: [x_se_20], Original ATen: [aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_mean_33.run(buf284, buf282, 128, 1536, 196, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_21], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:42
            buf285 = torch.ops.aten.convolution.default(buf284, primals_115, primals_116, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_116
            buf286 = buf285
            assert_size_stride(buf286, (128, 384, 1, 1), (384, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf286, 16, 'torch.ops.aten.convolution.default')
            del buf285
            buf287 = buf286; del buf286  # reuse
            # Topologically Sorted Source Nodes: [x_se_22], Original ATen: [aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_relu_34.run(buf287, 128, 384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_23], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:43
            buf288 = torch.ops.aten.convolution.default(buf287, primals_117, primals_118, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_118
            buf289 = buf288
            assert_size_stride(buf289, (128, 1536, 1, 1), (1536, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf289, 16, 'torch.ops.aten.convolution.default')
            del buf288
            buf290 = empty_strided_npu((128, 1536, 14, 14), (301056, 196, 14, 1), torch.float32)
            buf291 = empty_strided_npu((128, 1536, 14, 14), (301056, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sigmoid_5, mul_52, out_40, mul_54, out_41, silu_27, out_42], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_sigmoid_silu_38.run(buf282, buf289, buf250, buf290, buf291, 196608, 196, stream=raw_stream0)
            buf292 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf293 = buf292; del buf292  # reuse
            buf296 = empty_strided_npu((1, 384, 1536), (589824, 1536, 1), torch.float32)
            buf295 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_31, mul_56, view_31, batch_norm_31], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__36.run(buf293, primals_119, primals_120, buf296, buf295, 384, 1536, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_31, mul_56, view_31, batch_norm_31, weight_31, out_43], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:44
            buf297 = torch.ops.aten.convolution.default(buf291, reinterpret_tensor(buf296, (384, 1536, 1, 1), (1536, 1, 0, 0), 0), primals_121, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_121
            buf298 = buf297
            assert_size_stride(buf298, (128, 384, 14, 14), (75264, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf298, 16, 'torch.ops.aten.convolution.default')
            del buf297
            buf299 = empty_strided_npu((128, 384, 14, 14), (75264, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_28], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_31.run(buf298, buf299, 128, 384, 14, 14, stream=raw_stream0)
            buf300 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf301 = buf300; del buf300  # reuse
            buf304 = empty_strided_npu((1, 384, 576), (221184, 576, 1), torch.float32)
            buf303 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_32, mul_57, view_32, batch_norm_32], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__30.run(buf301, primals_122, primals_123, buf304, buf303, 384, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_32, mul_57, view_32, batch_norm_32, weight_32, out_44], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:45
            buf305 = torch.ops.aten.convolution.default(buf299, reinterpret_tensor(buf304, (384, 64, 3, 3), (576, 9, 3, 1), 0), primals_124, [1, 1], [1, 1], [1, 1], False, [0, 0], 6)
            del primals_124
            buf306 = buf305
            assert_size_stride(buf306, (128, 384, 14, 14), (75264, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf306, 16, 'torch.ops.aten.convolution.default')
            del buf305
            buf307 = empty_strided_npu((128, 384, 14, 14), (75264, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_29], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_31.run(buf306, buf307, 128, 384, 14, 14, stream=raw_stream0)
            buf308 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf309 = buf308; del buf308  # reuse
            buf312 = empty_strided_npu((1, 384, 576), (221184, 576, 1), torch.float32)
            buf311 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_33, mul_58, view_33, batch_norm_33], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__30.run(buf309, primals_125, primals_126, buf312, buf311, 384, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_33, mul_58, view_33, batch_norm_33, weight_33, out_45], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:46
            buf313 = torch.ops.aten.convolution.default(buf307, reinterpret_tensor(buf312, (384, 64, 3, 3), (576, 9, 3, 1), 0), primals_127, [1, 1], [1, 1], [1, 1], False, [0, 0], 6)
            del primals_127
            buf314 = buf313
            assert_size_stride(buf314, (128, 384, 14, 14), (75264, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf314, 16, 'torch.ops.aten.convolution.default')
            del buf313
            buf315 = empty_strided_npu((128, 384, 14, 14), (75264, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_30], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_31.run(buf314, buf315, 128, 384, 14, 14, stream=raw_stream0)
            buf316 = empty_strided_npu((1, 1536, 1), (1536, 1, 1536), torch.float32)
            buf317 = buf316; del buf316  # reuse
            buf320 = empty_strided_npu((1, 1536, 384), (589824, 384, 1), torch.float32)
            buf319 = empty_strided_npu((1, 1536, 1), (1536, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_34, mul_59, view_34, batch_norm_34], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__32.run(buf317, primals_128, primals_129, buf320, buf319, 1536, 384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_34, mul_59, view_34, batch_norm_34, weight_34, out_46], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:47
            buf321 = torch.ops.aten.convolution.default(buf315, reinterpret_tensor(buf320, (1536, 384, 1, 1), (384, 1, 0, 0), 0), primals_130, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_130
            buf322 = buf321
            assert_size_stride(buf322, (128, 1536, 14, 14), (301056, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf322, 16, 'torch.ops.aten.convolution.default')
            del buf321
            buf323 = empty_strided_npu((128, 1536, 1, 1), (1536, 1, 196608, 196608), torch.float32)
            buf324 = reinterpret_tensor(buf323, (128, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf323  # reuse
            # Topologically Sorted Source Nodes: [x_se_24], Original ATen: [aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_mean_33.run(buf324, buf322, 128, 1536, 196, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_25], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:48
            buf325 = torch.ops.aten.convolution.default(buf324, primals_131, primals_132, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_132
            buf326 = buf325
            assert_size_stride(buf326, (128, 384, 1, 1), (384, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf326, 16, 'torch.ops.aten.convolution.default')
            del buf325
            buf327 = buf326; del buf326  # reuse
            # Topologically Sorted Source Nodes: [x_se_26], Original ATen: [aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_relu_34.run(buf327, 128, 384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_27], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:49
            buf328 = torch.ops.aten.convolution.default(buf327, primals_133, primals_134, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_134
            buf329 = buf328
            assert_size_stride(buf329, (128, 1536, 1, 1), (1536, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf329, 16, 'torch.ops.aten.convolution.default')
            del buf328
            buf330 = empty_strided_npu((128, 1536, 14, 14), (301056, 196, 14, 1), torch.float32)
            buf331 = empty_strided_npu((128, 1536, 14, 14), (301056, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sigmoid_6, mul_60, out_47, mul_62, out_48, silu_31, out_49], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_sigmoid_silu_39.run(buf322, buf329, buf290, buf330, buf331, 196608, 196, stream=raw_stream0)
            buf332 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf333 = buf332; del buf332  # reuse
            buf336 = empty_strided_npu((1, 384, 1536), (589824, 1536, 1), torch.float32)
            buf335 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_35, mul_64, view_35, batch_norm_35], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__36.run(buf333, primals_135, primals_136, buf336, buf335, 384, 1536, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_35, mul_64, view_35, batch_norm_35, weight_35, out_50], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:50
            buf337 = torch.ops.aten.convolution.default(buf331, reinterpret_tensor(buf336, (384, 1536, 1, 1), (1536, 1, 0, 0), 0), primals_137, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_137
            buf338 = buf337
            assert_size_stride(buf338, (128, 384, 14, 14), (75264, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf338, 16, 'torch.ops.aten.convolution.default')
            del buf337
            buf339 = empty_strided_npu((128, 384, 14, 14), (75264, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_32], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_31.run(buf338, buf339, 128, 384, 14, 14, stream=raw_stream0)
            buf340 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf341 = buf340; del buf340  # reuse
            buf344 = empty_strided_npu((1, 384, 576), (221184, 576, 1), torch.float32)
            buf343 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_36, mul_65, view_36, batch_norm_36], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__30.run(buf341, primals_138, primals_139, buf344, buf343, 384, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_36, mul_65, view_36, batch_norm_36, weight_36, out_51], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:51
            buf345 = torch.ops.aten.convolution.default(buf339, reinterpret_tensor(buf344, (384, 64, 3, 3), (576, 9, 3, 1), 0), primals_140, [1, 1], [1, 1], [1, 1], False, [0, 0], 6)
            del primals_140
            buf346 = buf345
            assert_size_stride(buf346, (128, 384, 14, 14), (75264, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf346, 16, 'torch.ops.aten.convolution.default')
            del buf345
            buf347 = empty_strided_npu((128, 384, 14, 14), (75264, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_33], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_31.run(buf346, buf347, 128, 384, 14, 14, stream=raw_stream0)
            buf348 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf349 = buf348; del buf348  # reuse
            buf352 = empty_strided_npu((1, 384, 576), (221184, 576, 1), torch.float32)
            buf351 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_37, mul_66, view_37, batch_norm_37], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__30.run(buf349, primals_141, primals_142, buf352, buf351, 384, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_37, mul_66, view_37, batch_norm_37, weight_37, out_52], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:52
            buf353 = torch.ops.aten.convolution.default(buf347, reinterpret_tensor(buf352, (384, 64, 3, 3), (576, 9, 3, 1), 0), primals_143, [1, 1], [1, 1], [1, 1], False, [0, 0], 6)
            del primals_143
            buf354 = buf353
            assert_size_stride(buf354, (128, 384, 14, 14), (75264, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf354, 16, 'torch.ops.aten.convolution.default')
            del buf353
            buf355 = empty_strided_npu((128, 384, 14, 14), (75264, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_34], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_31.run(buf354, buf355, 128, 384, 14, 14, stream=raw_stream0)
            buf356 = empty_strided_npu((1, 1536, 1), (1536, 1, 1536), torch.float32)
            buf357 = buf356; del buf356  # reuse
            buf360 = empty_strided_npu((1, 1536, 384), (589824, 384, 1), torch.float32)
            buf359 = empty_strided_npu((1, 1536, 1), (1536, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_38, mul_67, view_38, batch_norm_38], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__32.run(buf357, primals_144, primals_145, buf360, buf359, 1536, 384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_38, mul_67, view_38, batch_norm_38, weight_38, out_53], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:53
            buf361 = torch.ops.aten.convolution.default(buf355, reinterpret_tensor(buf360, (1536, 384, 1, 1), (384, 1, 0, 0), 0), primals_146, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_146
            buf362 = buf361
            assert_size_stride(buf362, (128, 1536, 14, 14), (301056, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf362, 16, 'torch.ops.aten.convolution.default')
            del buf361
            buf363 = empty_strided_npu((128, 1536, 1, 1), (1536, 1, 196608, 196608), torch.float32)
            buf364 = reinterpret_tensor(buf363, (128, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf363  # reuse
            # Topologically Sorted Source Nodes: [x_se_28], Original ATen: [aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_mean_33.run(buf364, buf362, 128, 1536, 196, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_29], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:54
            buf365 = torch.ops.aten.convolution.default(buf364, primals_147, primals_148, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_148
            buf366 = buf365
            assert_size_stride(buf366, (128, 384, 1, 1), (384, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf366, 16, 'torch.ops.aten.convolution.default')
            del buf365
            buf367 = buf366; del buf366  # reuse
            # Topologically Sorted Source Nodes: [x_se_30], Original ATen: [aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_relu_34.run(buf367, 128, 384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_31], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:55
            buf368 = torch.ops.aten.convolution.default(buf367, primals_149, primals_150, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_150
            buf369 = buf368
            assert_size_stride(buf369, (128, 1536, 1, 1), (1536, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf369, 16, 'torch.ops.aten.convolution.default')
            del buf368
            buf370 = empty_strided_npu((128, 1536, 14, 14), (301056, 196, 14, 1), torch.float32)
            buf371 = empty_strided_npu((128, 1536, 14, 14), (301056, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sigmoid_7, mul_68, out_54, mul_70, out_55, silu_35, out_56], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_sigmoid_silu_40.run(buf362, buf369, buf330, buf370, buf371, 196608, 196, stream=raw_stream0)
            buf372 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf373 = buf372; del buf372  # reuse
            buf376 = empty_strided_npu((1, 384, 1536), (589824, 1536, 1), torch.float32)
            buf375 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_39, mul_72, view_39, batch_norm_39], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__36.run(buf373, primals_151, primals_152, buf376, buf375, 384, 1536, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_39, mul_72, view_39, batch_norm_39, weight_39, out_57], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:56
            buf377 = torch.ops.aten.convolution.default(buf371, reinterpret_tensor(buf376, (384, 1536, 1, 1), (1536, 1, 0, 0), 0), primals_153, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_153
            buf378 = buf377
            assert_size_stride(buf378, (128, 384, 14, 14), (75264, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf378, 16, 'torch.ops.aten.convolution.default')
            del buf377
            buf379 = empty_strided_npu((128, 384, 14, 14), (75264, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_36], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_31.run(buf378, buf379, 128, 384, 14, 14, stream=raw_stream0)
            buf380 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf381 = buf380; del buf380  # reuse
            buf384 = empty_strided_npu((1, 384, 576), (221184, 576, 1), torch.float32)
            buf383 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_40, mul_73, view_40, batch_norm_40], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__30.run(buf381, primals_154, primals_155, buf384, buf383, 384, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_40, mul_73, view_40, batch_norm_40, weight_40, out_58], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:57
            buf385 = torch.ops.aten.convolution.default(buf379, reinterpret_tensor(buf384, (384, 64, 3, 3), (576, 9, 3, 1), 0), primals_156, [1, 1], [1, 1], [1, 1], False, [0, 0], 6)
            del primals_156
            buf386 = buf385
            assert_size_stride(buf386, (128, 384, 14, 14), (75264, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf386, 16, 'torch.ops.aten.convolution.default')
            del buf385
            buf387 = empty_strided_npu((128, 384, 14, 14), (75264, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_37], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_31.run(buf386, buf387, 128, 384, 14, 14, stream=raw_stream0)
            buf388 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf389 = buf388; del buf388  # reuse
            buf392 = empty_strided_npu((1, 384, 576), (221184, 576, 1), torch.float32)
            buf391 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_41, mul_74, view_41, batch_norm_41], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__30.run(buf389, primals_157, primals_158, buf392, buf391, 384, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_41, mul_74, view_41, batch_norm_41, weight_41, out_59], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:58
            buf393 = torch.ops.aten.convolution.default(buf387, reinterpret_tensor(buf392, (384, 64, 3, 3), (576, 9, 3, 1), 0), primals_159, [1, 1], [1, 1], [1, 1], False, [0, 0], 6)
            del primals_159
            buf394 = buf393
            assert_size_stride(buf394, (128, 384, 14, 14), (75264, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf394, 16, 'torch.ops.aten.convolution.default')
            del buf393
            buf395 = empty_strided_npu((128, 384, 14, 14), (75264, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_38], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_31.run(buf394, buf395, 128, 384, 14, 14, stream=raw_stream0)
            buf396 = empty_strided_npu((1, 1536, 1), (1536, 1, 1536), torch.float32)
            buf397 = buf396; del buf396  # reuse
            buf400 = empty_strided_npu((1, 1536, 384), (589824, 384, 1), torch.float32)
            buf399 = empty_strided_npu((1, 1536, 1), (1536, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_42, mul_75, view_42, batch_norm_42], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__32.run(buf397, primals_160, primals_161, buf400, buf399, 1536, 384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_42, mul_75, view_42, batch_norm_42, weight_42, out_60], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:59
            buf401 = torch.ops.aten.convolution.default(buf395, reinterpret_tensor(buf400, (1536, 384, 1, 1), (384, 1, 0, 0), 0), primals_162, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_162
            buf402 = buf401
            assert_size_stride(buf402, (128, 1536, 14, 14), (301056, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf402, 16, 'torch.ops.aten.convolution.default')
            del buf401
            buf403 = empty_strided_npu((128, 1536, 1, 1), (1536, 1, 196608, 196608), torch.float32)
            buf404 = reinterpret_tensor(buf403, (128, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf403  # reuse
            # Topologically Sorted Source Nodes: [x_se_32], Original ATen: [aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_mean_33.run(buf404, buf402, 128, 1536, 196, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_33], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:60
            buf405 = torch.ops.aten.convolution.default(buf404, primals_163, primals_164, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_164
            buf406 = buf405
            assert_size_stride(buf406, (128, 384, 1, 1), (384, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf406, 16, 'torch.ops.aten.convolution.default')
            del buf405
            buf407 = buf406; del buf406  # reuse
            # Topologically Sorted Source Nodes: [x_se_34], Original ATen: [aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_relu_34.run(buf407, 128, 384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_35], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:61
            buf408 = torch.ops.aten.convolution.default(buf407, primals_165, primals_166, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_166
            buf409 = buf408
            assert_size_stride(buf409, (128, 1536, 1, 1), (1536, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf409, 16, 'torch.ops.aten.convolution.default')
            del buf408
            buf410 = empty_strided_npu((128, 1536, 14, 14), (301056, 196, 14, 1), torch.float32)
            buf411 = empty_strided_npu((128, 1536, 14, 14), (301056, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sigmoid_8, mul_76, out_61, mul_78, out_62, silu_39, out_63], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_sigmoid_silu_41.run(buf402, buf409, buf370, buf410, buf411, 196608, 196, stream=raw_stream0)
            buf412 = empty_strided_npu((128, 1536, 7, 7), (75264, 49, 7, 1), torch.float32)
            # Topologically Sorted Source Nodes: [avg_pool2d_2], Original ATen: [aten.avg_pool2d]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_avg_pool2d_42.run(buf411, buf412, 1376256, 7, stream=raw_stream0)
            buf413 = empty_strided_npu((1, 1536, 1), (1536, 1, 1536), torch.float32)
            buf414 = buf413; del buf413  # reuse
            buf417 = empty_strided_npu((1, 1536, 1536), (2359296, 1536, 1), torch.float32)
            buf416 = empty_strided_npu((1, 1536, 1), (1536, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_43, mul_80, view_43, batch_norm_43], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__43.run(buf414, primals_167, primals_168, buf417, buf416, 1536, 1536, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_43, mul_80, view_43, batch_norm_43, weight_43, shortcut_3], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:62
            buf418 = torch.ops.aten.convolution.default(buf412, reinterpret_tensor(buf417, (1536, 1536, 1, 1), (1536, 1, 0, 0), 0), primals_169, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_169
            buf419 = buf418
            assert_size_stride(buf419, (128, 1536, 7, 7), (75264, 49, 7, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf419, 16, 'torch.ops.aten.convolution.default')
            del buf418
            buf420 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf421 = buf420; del buf420  # reuse
            buf424 = empty_strided_npu((1, 384, 1536), (589824, 1536, 1), torch.float32)
            buf423 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_44, mul_81, view_44, batch_norm_44], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__36.run(buf421, primals_170, primals_171, buf424, buf423, 384, 1536, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_44, mul_81, view_44, batch_norm_44, weight_44, out_64], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:63
            buf425 = torch.ops.aten.convolution.default(buf411, reinterpret_tensor(buf424, (384, 1536, 1, 1), (1536, 1, 0, 0), 0), primals_172, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_172
            buf426 = buf425
            assert_size_stride(buf426, (128, 384, 14, 14), (75264, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf426, 16, 'torch.ops.aten.convolution.default')
            del buf425
            buf427 = empty_strided_npu((128, 384, 14, 14), (75264, 196, 14, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_40], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_31.run(buf426, buf427, 128, 384, 14, 14, stream=raw_stream0)
            buf428 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf429 = buf428; del buf428  # reuse
            buf432 = empty_strided_npu((1, 384, 576), (221184, 576, 1), torch.float32)
            buf431 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_45, mul_82, view_45, batch_norm_45], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__30.run(buf429, primals_173, primals_174, buf432, buf431, 384, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_45, mul_82, view_45, batch_norm_45, weight_45, out_65], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:64
            buf433 = torch.ops.aten.convolution.default(buf427, reinterpret_tensor(buf432, (384, 64, 3, 3), (576, 9, 3, 1), 0), primals_175, [2, 2], [1, 1], [1, 1], False, [0, 0], 6)
            del primals_175
            buf434 = buf433
            assert_size_stride(buf434, (128, 384, 7, 7), (18816, 49, 7, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf434, 16, 'torch.ops.aten.convolution.default')
            del buf433
            buf435 = empty_strided_npu((128, 384, 7, 7), (18816, 49, 7, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_41], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_44.run(buf434, buf435, 128, 384, 7, 7, stream=raw_stream0)
            buf436 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf437 = buf436; del buf436  # reuse
            buf440 = empty_strided_npu((1, 384, 576), (221184, 576, 1), torch.float32)
            buf439 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_46, mul_83, view_46, batch_norm_46], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__30.run(buf437, primals_176, primals_177, buf440, buf439, 384, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_46, mul_83, view_46, batch_norm_46, weight_46, out_66], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:65
            buf441 = torch.ops.aten.convolution.default(buf435, reinterpret_tensor(buf440, (384, 64, 3, 3), (576, 9, 3, 1), 0), primals_178, [1, 1], [1, 1], [1, 1], False, [0, 0], 6)
            del primals_178
            buf442 = buf441
            assert_size_stride(buf442, (128, 384, 7, 7), (18816, 49, 7, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf442, 16, 'torch.ops.aten.convolution.default')
            del buf441
            buf443 = empty_strided_npu((128, 384, 7, 7), (18816, 49, 7, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_42], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_44.run(buf442, buf443, 128, 384, 7, 7, stream=raw_stream0)
            buf444 = empty_strided_npu((1, 1536, 1), (1536, 1, 1536), torch.float32)
            buf445 = buf444; del buf444  # reuse
            buf448 = empty_strided_npu((1, 1536, 384), (589824, 384, 1), torch.float32)
            buf447 = empty_strided_npu((1, 1536, 1), (1536, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_47, mul_84, view_47, batch_norm_47], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__32.run(buf445, primals_179, primals_180, buf448, buf447, 1536, 384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_47, mul_84, view_47, batch_norm_47, weight_47, out_67], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:66
            buf449 = torch.ops.aten.convolution.default(buf443, reinterpret_tensor(buf448, (1536, 384, 1, 1), (384, 1, 0, 0), 0), primals_181, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_181
            buf450 = buf449
            assert_size_stride(buf450, (128, 1536, 7, 7), (75264, 49, 7, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf450, 16, 'torch.ops.aten.convolution.default')
            del buf449
            buf451 = empty_strided_npu((128, 1536, 1, 1), (1536, 1, 196608, 196608), torch.float32)
            buf452 = reinterpret_tensor(buf451, (128, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf451  # reuse
            # Topologically Sorted Source Nodes: [x_se_36], Original ATen: [aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_mean_45.run(buf452, buf450, 128, 1536, 49, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_37], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:67
            buf453 = torch.ops.aten.convolution.default(buf452, primals_182, primals_183, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_183
            buf454 = buf453
            assert_size_stride(buf454, (128, 384, 1, 1), (384, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf454, 16, 'torch.ops.aten.convolution.default')
            del buf453
            buf455 = buf454; del buf454  # reuse
            # Topologically Sorted Source Nodes: [x_se_38], Original ATen: [aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_relu_34.run(buf455, 128, 384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_39], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:68
            buf456 = torch.ops.aten.convolution.default(buf455, primals_184, primals_185, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_185
            buf457 = buf456
            assert_size_stride(buf457, (128, 1536, 1, 1), (1536, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf457, 16, 'torch.ops.aten.convolution.default')
            del buf456
            buf458 = empty_strided_npu((128, 1536, 7, 7), (75264, 49, 7, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sigmoid_9, mul_85, out_68, mul_87, out_69, silu_43, out_70], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_sigmoid_silu_46.run(buf450, buf457, buf419, buf458, 196608, 49, stream=raw_stream0)
            buf459 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf460 = buf459; del buf459  # reuse
            buf463 = empty_strided_npu((1, 384, 1536), (589824, 1536, 1), torch.float32)
            buf462 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_48, mul_89, view_48, batch_norm_48], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__36.run(buf460, primals_186, primals_187, buf463, buf462, 384, 1536, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_48, mul_89, view_48, batch_norm_48, weight_48, out_71], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:69
            buf464 = torch.ops.aten.convolution.default(buf458, reinterpret_tensor(buf463, (384, 1536, 1, 1), (1536, 1, 0, 0), 0), primals_188, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_188
            buf465 = buf464
            assert_size_stride(buf465, (128, 384, 7, 7), (18816, 49, 7, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf465, 16, 'torch.ops.aten.convolution.default')
            del buf464
            buf466 = empty_strided_npu((128, 384, 7, 7), (18816, 49, 7, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_44], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_44.run(buf465, buf466, 128, 384, 7, 7, stream=raw_stream0)
            buf467 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf468 = buf467; del buf467  # reuse
            buf471 = empty_strided_npu((1, 384, 576), (221184, 576, 1), torch.float32)
            buf470 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_49, mul_90, view_49, batch_norm_49], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__30.run(buf468, primals_189, primals_190, buf471, buf470, 384, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_49, mul_90, view_49, batch_norm_49, weight_49, out_72], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:70
            buf472 = torch.ops.aten.convolution.default(buf466, reinterpret_tensor(buf471, (384, 64, 3, 3), (576, 9, 3, 1), 0), primals_191, [1, 1], [1, 1], [1, 1], False, [0, 0], 6)
            del primals_191
            buf473 = buf472
            assert_size_stride(buf473, (128, 384, 7, 7), (18816, 49, 7, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf473, 16, 'torch.ops.aten.convolution.default')
            del buf472
            buf474 = empty_strided_npu((128, 384, 7, 7), (18816, 49, 7, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_45], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_44.run(buf473, buf474, 128, 384, 7, 7, stream=raw_stream0)
            buf475 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf476 = buf475; del buf475  # reuse
            buf479 = empty_strided_npu((1, 384, 576), (221184, 576, 1), torch.float32)
            buf478 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_50, mul_91, view_50, batch_norm_50], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__30.run(buf476, primals_192, primals_193, buf479, buf478, 384, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_50, mul_91, view_50, batch_norm_50, weight_50, out_73], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:71
            buf480 = torch.ops.aten.convolution.default(buf474, reinterpret_tensor(buf479, (384, 64, 3, 3), (576, 9, 3, 1), 0), primals_194, [1, 1], [1, 1], [1, 1], False, [0, 0], 6)
            del primals_194
            buf481 = buf480
            assert_size_stride(buf481, (128, 384, 7, 7), (18816, 49, 7, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf481, 16, 'torch.ops.aten.convolution.default')
            del buf480
            buf482 = empty_strided_npu((128, 384, 7, 7), (18816, 49, 7, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_46], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_44.run(buf481, buf482, 128, 384, 7, 7, stream=raw_stream0)
            buf483 = empty_strided_npu((1, 1536, 1), (1536, 1, 1536), torch.float32)
            buf484 = buf483; del buf483  # reuse
            buf487 = empty_strided_npu((1, 1536, 384), (589824, 384, 1), torch.float32)
            buf486 = empty_strided_npu((1, 1536, 1), (1536, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_51, mul_92, view_51, batch_norm_51], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__32.run(buf484, primals_195, primals_196, buf487, buf486, 1536, 384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_51, mul_92, view_51, batch_norm_51, weight_51, out_74], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:72
            buf488 = torch.ops.aten.convolution.default(buf482, reinterpret_tensor(buf487, (1536, 384, 1, 1), (384, 1, 0, 0), 0), primals_197, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_197
            buf489 = buf488
            assert_size_stride(buf489, (128, 1536, 7, 7), (75264, 49, 7, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf489, 16, 'torch.ops.aten.convolution.default')
            del buf488
            buf490 = empty_strided_npu((128, 1536, 1, 1), (1536, 1, 196608, 196608), torch.float32)
            buf491 = reinterpret_tensor(buf490, (128, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf490  # reuse
            # Topologically Sorted Source Nodes: [x_se_40], Original ATen: [aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_mean_45.run(buf491, buf489, 128, 1536, 49, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_41], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:73
            buf492 = torch.ops.aten.convolution.default(buf491, primals_198, primals_199, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_199
            buf493 = buf492
            assert_size_stride(buf493, (128, 384, 1, 1), (384, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf493, 16, 'torch.ops.aten.convolution.default')
            del buf492
            buf494 = buf493; del buf493  # reuse
            # Topologically Sorted Source Nodes: [x_se_42], Original ATen: [aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_relu_34.run(buf494, 128, 384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_43], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:74
            buf495 = torch.ops.aten.convolution.default(buf494, primals_200, primals_201, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_201
            buf496 = buf495
            assert_size_stride(buf496, (128, 1536, 1, 1), (1536, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf496, 16, 'torch.ops.aten.convolution.default')
            del buf495
            buf497 = empty_strided_npu((128, 1536, 7, 7), (75264, 49, 7, 1), torch.float32)
            buf498 = empty_strided_npu((128, 1536, 7, 7), (75264, 49, 7, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sigmoid_9, mul_85, out_68, mul_87, out_69, sigmoid_10, mul_93, out_75, mul_95, out_76, silu_47, out_77], Original ATen: [aten.sigmoid, aten.mul, aten.add, aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_sigmoid_silu_47.run(buf489, buf496, buf450, buf457, buf419, buf497, buf498, 196608, 49, stream=raw_stream0)
            buf499 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf500 = buf499; del buf499  # reuse
            buf503 = empty_strided_npu((1, 384, 1536), (589824, 1536, 1), torch.float32)
            buf502 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_52, mul_97, view_52, batch_norm_52], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__36.run(buf500, primals_202, primals_203, buf503, buf502, 384, 1536, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_52, mul_97, view_52, batch_norm_52, weight_52, out_78], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:75
            buf504 = torch.ops.aten.convolution.default(buf498, reinterpret_tensor(buf503, (384, 1536, 1, 1), (1536, 1, 0, 0), 0), primals_204, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_204
            buf505 = buf504
            assert_size_stride(buf505, (128, 384, 7, 7), (18816, 49, 7, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf505, 16, 'torch.ops.aten.convolution.default')
            del buf504
            buf506 = empty_strided_npu((128, 384, 7, 7), (18816, 49, 7, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_48], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_44.run(buf505, buf506, 128, 384, 7, 7, stream=raw_stream0)
            buf507 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf508 = buf507; del buf507  # reuse
            buf511 = empty_strided_npu((1, 384, 576), (221184, 576, 1), torch.float32)
            buf510 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_53, mul_98, view_53, batch_norm_53], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__30.run(buf508, primals_205, primals_206, buf511, buf510, 384, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_53, mul_98, view_53, batch_norm_53, weight_53, out_79], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:76
            buf512 = torch.ops.aten.convolution.default(buf506, reinterpret_tensor(buf511, (384, 64, 3, 3), (576, 9, 3, 1), 0), primals_207, [1, 1], [1, 1], [1, 1], False, [0, 0], 6)
            del primals_207
            buf513 = buf512
            assert_size_stride(buf513, (128, 384, 7, 7), (18816, 49, 7, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf513, 16, 'torch.ops.aten.convolution.default')
            del buf512
            buf514 = empty_strided_npu((128, 384, 7, 7), (18816, 49, 7, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_49], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_44.run(buf513, buf514, 128, 384, 7, 7, stream=raw_stream0)
            buf515 = empty_strided_npu((1, 384, 1), (384, 1, 384), torch.float32)
            buf516 = buf515; del buf515  # reuse
            buf519 = empty_strided_npu((1, 384, 576), (221184, 576, 1), torch.float32)
            buf518 = empty_strided_npu((1, 384, 1), (384, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_54, mul_99, view_54, batch_norm_54], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__30.run(buf516, primals_208, primals_209, buf519, buf518, 384, 576, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_54, mul_99, view_54, batch_norm_54, weight_54, out_80], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:77
            buf520 = torch.ops.aten.convolution.default(buf514, reinterpret_tensor(buf519, (384, 64, 3, 3), (576, 9, 3, 1), 0), primals_210, [1, 1], [1, 1], [1, 1], False, [0, 0], 6)
            del primals_210
            buf521 = buf520
            assert_size_stride(buf521, (128, 384, 7, 7), (18816, 49, 7, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf521, 16, 'torch.ops.aten.convolution.default')
            del buf520
            buf522 = empty_strided_npu((128, 384, 7, 7), (18816, 49, 7, 1), torch.float32)
            # Topologically Sorted Source Nodes: [silu_50], Original ATen: [aten.silu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_44.run(buf521, buf522, 128, 384, 7, 7, stream=raw_stream0)
            buf523 = empty_strided_npu((1, 1536, 1), (1536, 1, 1536), torch.float32)
            buf524 = buf523; del buf523  # reuse
            buf527 = empty_strided_npu((1, 1536, 384), (589824, 384, 1), torch.float32)
            buf526 = empty_strided_npu((1, 1536, 1), (1536, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_55, mul_100, view_55, batch_norm_55], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__32.run(buf524, primals_211, primals_212, buf527, buf526, 1536, 384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_55, mul_100, view_55, batch_norm_55, weight_55, out_81], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:78
            buf528 = torch.ops.aten.convolution.default(buf522, reinterpret_tensor(buf527, (1536, 384, 1, 1), (384, 1, 0, 0), 0), primals_213, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_213
            buf529 = buf528
            assert_size_stride(buf529, (128, 1536, 7, 7), (75264, 49, 7, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf529, 16, 'torch.ops.aten.convolution.default')
            del buf528
            buf530 = empty_strided_npu((128, 1536, 1, 1), (1536, 1, 196608, 196608), torch.float32)
            buf531 = reinterpret_tensor(buf530, (128, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf530  # reuse
            # Topologically Sorted Source Nodes: [x_se_44], Original ATen: [aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_mean_45.run(buf531, buf529, 128, 1536, 49, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_45], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:79
            buf532 = torch.ops.aten.convolution.default(buf531, primals_214, primals_215, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_215
            buf533 = buf532
            assert_size_stride(buf533, (128, 384, 1, 1), (384, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf533, 16, 'torch.ops.aten.convolution.default')
            del buf532
            buf534 = buf533; del buf533  # reuse
            # Topologically Sorted Source Nodes: [x_se_46], Original ATen: [aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_relu_34.run(buf534, 128, 384, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_se_47], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:80
            buf535 = torch.ops.aten.convolution.default(buf534, primals_216, primals_217, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_217
            buf536 = buf535
            assert_size_stride(buf536, (128, 1536, 1, 1), (1536, 1, 1, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf536, 16, 'torch.ops.aten.convolution.default')
            del buf535
            buf537 = empty_strided_npu((128, 1536, 7, 7), (75264, 49, 7, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sigmoid_11, mul_101, out_82, mul_103, out_83], Original ATen: [aten.sigmoid, aten.mul, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_sigmoid_48.run(buf529, buf536, buf497, buf537, 196608, 49, stream=raw_stream0)
            buf538 = empty_strided_npu((1, 2304, 1), (2304, 1, 2304), torch.float32)
            buf539 = buf538; del buf538  # reuse
            buf542 = empty_strided_npu((1, 2304, 1536), (3538944, 1536, 1), torch.float32)
            buf541 = empty_strided_npu((1, 2304, 1), (2304, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [reshape_56, mul_104, view_56, batch_norm_56], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_mul__49.run(buf539, primals_218, primals_219, buf542, buf541, 2304, 1536, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [reshape_56, mul_104, view_56, batch_norm_56, weight_56, x], Original ATen: [aten.view, aten.mul, aten._native_batch_norm_legit, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:81
            buf543 = torch.ops.aten.convolution.default(buf537, reinterpret_tensor(buf542, (2304, 1536, 1, 1), (1536, 1, 0, 0), 0), primals_220, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del primals_220
            buf544 = buf543
            assert_size_stride(buf544, (128, 2304, 7, 7), (112896, 49, 7, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf544, 16, 'torch.ops.aten.convolution.default')
            del buf543
            buf545 = empty_strided_npu((128, 2304, 1, 1), (2304, 1, 294912, 294912), torch.float32)
            buf546 = buf545; del buf545  # reuse
            # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.silu, aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_mean_silu_50.run(buf546, buf544, 128, 2304, 49, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_1, x_2, x_3, x_5], Original ATen: [aten.silu, aten.mean, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] torch.ops.aten.addmm.default:82
            buf547 = torch.ops.aten.addmm.default(primals_222, reinterpret_tensor(buf546, (128, 2304), (2304, 1), 0), reinterpret_tensor(primals_221, (2304, 1000), (1, 2304), 0), alpha=1, beta=1)
            del primals_222
            buf548 = buf547
            assert_size_stride(buf548, (128, 1000), (1000, 1), 'torch.ops.aten.addmm.default')
            assert_alignment(buf548, 16, 'torch.ops.aten.addmm.default')
            del buf547
        return (buf548, primals_1, primals_2, primals_4, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_15, primals_17, primals_18, primals_20, primals_21, primals_23, primals_24, primals_26, primals_27, primals_29, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_66, primals_68, primals_69, primals_71, primals_72, primals_74, primals_75, primals_77, primals_78, primals_80, primals_81, primals_83, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_117, primals_119, primals_120, primals_122, primals_123, primals_125, primals_126, primals_128, primals_129, primals_131, primals_133, primals_135, primals_136, primals_138, primals_139, primals_141, primals_142, primals_144, primals_145, primals_147, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_165, primals_167, primals_168, primals_170, primals_171, primals_173, primals_174, primals_176, primals_177, primals_179, primals_180, primals_182, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_216, primals_218, primals_219, primals_221, reinterpret_tensor(buf3, (16, ), (1, ), 0), reinterpret_tensor(buf4, (16, 3, 3, 3), (27, 9, 3, 1), 0), buf6, buf7, reinterpret_tensor(buf11, (32, ), (1, ), 0), reinterpret_tensor(buf12, (32, 16, 3, 3), (144, 9, 3, 1), 0), buf14, buf15, reinterpret_tensor(buf19, (64, ), (1, ), 0), reinterpret_tensor(buf20, (64, 32, 3, 3), (288, 9, 3, 1), 0), buf22, buf23, reinterpret_tensor(buf27, (128, ), (1, ), 0), reinterpret_tensor(buf28, (128, 64, 3, 3), (576, 9, 3, 1), 0), buf30, buf31, reinterpret_tensor(buf35, (256, ), (1, ), 0), reinterpret_tensor(buf36, (256, 128, 1, 1), (128, 1, 1, 1), 0), buf38, reinterpret_tensor(buf42, (64, ), (1, ), 0), reinterpret_tensor(buf43, (64, 128, 1, 1), (128, 1, 1, 1), 0), buf45, buf46, reinterpret_tensor(buf50, (64, ), (1, ), 0), reinterpret_tensor(buf51, (64, 64, 3, 3), (576, 9, 3, 1), 0), buf53, buf54, reinterpret_tensor(buf58, (64, ), (1, ), 0), reinterpret_tensor(buf59, (64, 64, 3, 3), (576, 9, 3, 1), 0), buf61, buf62, reinterpret_tensor(buf66, (256, ), (1, ), 0), reinterpret_tensor(buf67, (256, 64, 1, 1), (64, 1, 1, 1), 0), buf69, buf71, buf74, buf76, buf77, buf78, reinterpret_tensor(buf82, (512, ), (1, ), 0), reinterpret_tensor(buf83, (512, 256, 1, 1), (256, 1, 1, 1), 0), buf85, reinterpret_tensor(buf89, (128, ), (1, ), 0), reinterpret_tensor(buf90, (128, 256, 1, 1), (256, 1, 1, 1), 0), buf92, buf93, reinterpret_tensor(buf97, (128, ), (1, ), 0), reinterpret_tensor(buf98, (128, 64, 3, 3), (576, 9, 3, 1), 0), buf100, buf101, reinterpret_tensor(buf105, (128, ), (1, ), 0), reinterpret_tensor(buf106, (128, 64, 3, 3), (576, 9, 3, 1), 0), buf108, buf109, reinterpret_tensor(buf113, (512, ), (1, ), 0), reinterpret_tensor(buf114, (512, 128, 1, 1), (128, 1, 1, 1), 0), buf116, buf118, buf121, buf123, buf124, reinterpret_tensor(buf128, (128, ), (1, ), 0), reinterpret_tensor(buf129, (128, 512, 1, 1), (512, 1, 1, 1), 0), buf131, buf132, reinterpret_tensor(buf136, (128, ), (1, ), 0), reinterpret_tensor(buf137, (128, 64, 3, 3), (576, 9, 3, 1), 0), buf139, buf140, reinterpret_tensor(buf144, (128, ), (1, ), 0), reinterpret_tensor(buf145, (128, 64, 3, 3), (576, 9, 3, 1), 0), buf147, buf148, reinterpret_tensor(buf152, (512, ), (1, ), 0), reinterpret_tensor(buf153, (512, 128, 1, 1), (128, 1, 1, 1), 0), buf155, buf157, buf160, buf162, buf163, buf164, buf165, reinterpret_tensor(buf169, (1536, ), (1, ), 0), reinterpret_tensor(buf170, (1536, 512, 1, 1), (512, 1, 1, 1), 0), buf172, reinterpret_tensor(buf176, (384, ), (1, ), 0), reinterpret_tensor(buf177, (384, 512, 1, 1), (512, 1, 1, 1), 0), buf179, buf180, reinterpret_tensor(buf184, (384, ), (1, ), 0), reinterpret_tensor(buf185, (384, 64, 3, 3), (576, 9, 3, 1), 0), buf187, buf188, reinterpret_tensor(buf192, (384, ), (1, ), 0), reinterpret_tensor(buf193, (384, 64, 3, 3), (576, 9, 3, 1), 0), buf195, buf196, reinterpret_tensor(buf200, (1536, ), (1, ), 0), reinterpret_tensor(buf201, (1536, 384, 1, 1), (384, 1, 1, 1), 0), buf203, buf205, buf208, buf210, buf211, reinterpret_tensor(buf215, (384, ), (1, ), 0), reinterpret_tensor(buf216, (384, 1536, 1, 1), (1536, 1, 1, 1), 0), buf218, buf219, reinterpret_tensor(buf223, (384, ), (1, ), 0), reinterpret_tensor(buf224, (384, 64, 3, 3), (576, 9, 3, 1), 0), buf226, buf227, reinterpret_tensor(buf231, (384, ), (1, ), 0), reinterpret_tensor(buf232, (384, 64, 3, 3), (576, 9, 3, 1), 0), buf234, buf235, reinterpret_tensor(buf239, (1536, ), (1, ), 0), reinterpret_tensor(buf240, (1536, 384, 1, 1), (384, 1, 1, 1), 0), buf242, buf244, buf247, buf249, buf250, buf251, reinterpret_tensor(buf255, (384, ), (1, ), 0), reinterpret_tensor(buf256, (384, 1536, 1, 1), (1536, 1, 1, 1), 0), buf258, buf259, reinterpret_tensor(buf263, (384, ), (1, ), 0), reinterpret_tensor(buf264, (384, 64, 3, 3), (576, 9, 3, 1), 0), buf266, buf267, reinterpret_tensor(buf271, (384, ), (1, ), 0), reinterpret_tensor(buf272, (384, 64, 3, 3), (576, 9, 3, 1), 0), buf274, buf275, reinterpret_tensor(buf279, (1536, ), (1, ), 0), reinterpret_tensor(buf280, (1536, 384, 1, 1), (384, 1, 1, 1), 0), buf282, buf284, buf287, buf289, buf290, buf291, reinterpret_tensor(buf295, (384, ), (1, ), 0), reinterpret_tensor(buf296, (384, 1536, 1, 1), (1536, 1, 1, 1), 0), buf298, buf299, reinterpret_tensor(buf303, (384, ), (1, ), 0), reinterpret_tensor(buf304, (384, 64, 3, 3), (576, 9, 3, 1), 0), buf306, buf307, reinterpret_tensor(buf311, (384, ), (1, ), 0), reinterpret_tensor(buf312, (384, 64, 3, 3), (576, 9, 3, 1), 0), buf314, buf315, reinterpret_tensor(buf319, (1536, ), (1, ), 0), reinterpret_tensor(buf320, (1536, 384, 1, 1), (384, 1, 1, 1), 0), buf322, buf324, buf327, buf329, buf330, buf331, reinterpret_tensor(buf335, (384, ), (1, ), 0), reinterpret_tensor(buf336, (384, 1536, 1, 1), (1536, 1, 1, 1), 0), buf338, buf339, reinterpret_tensor(buf343, (384, ), (1, ), 0), reinterpret_tensor(buf344, (384, 64, 3, 3), (576, 9, 3, 1), 0), buf346, buf347, reinterpret_tensor(buf351, (384, ), (1, ), 0), reinterpret_tensor(buf352, (384, 64, 3, 3), (576, 9, 3, 1), 0), buf354, buf355, reinterpret_tensor(buf359, (1536, ), (1, ), 0), reinterpret_tensor(buf360, (1536, 384, 1, 1), (384, 1, 1, 1), 0), buf362, buf364, buf367, buf369, buf370, buf371, reinterpret_tensor(buf375, (384, ), (1, ), 0), reinterpret_tensor(buf376, (384, 1536, 1, 1), (1536, 1, 1, 1), 0), buf378, buf379, reinterpret_tensor(buf383, (384, ), (1, ), 0), reinterpret_tensor(buf384, (384, 64, 3, 3), (576, 9, 3, 1), 0), buf386, buf387, reinterpret_tensor(buf391, (384, ), (1, ), 0), reinterpret_tensor(buf392, (384, 64, 3, 3), (576, 9, 3, 1), 0), buf394, buf395, reinterpret_tensor(buf399, (1536, ), (1, ), 0), reinterpret_tensor(buf400, (1536, 384, 1, 1), (384, 1, 1, 1), 0), buf402, buf404, buf407, buf409, buf410, buf411, buf412, reinterpret_tensor(buf416, (1536, ), (1, ), 0), reinterpret_tensor(buf417, (1536, 1536, 1, 1), (1536, 1, 1, 1), 0), buf419, reinterpret_tensor(buf423, (384, ), (1, ), 0), reinterpret_tensor(buf424, (384, 1536, 1, 1), (1536, 1, 1, 1), 0), buf426, buf427, reinterpret_tensor(buf431, (384, ), (1, ), 0), reinterpret_tensor(buf432, (384, 64, 3, 3), (576, 9, 3, 1), 0), buf434, buf435, reinterpret_tensor(buf439, (384, ), (1, ), 0), reinterpret_tensor(buf440, (384, 64, 3, 3), (576, 9, 3, 1), 0), buf442, buf443, reinterpret_tensor(buf447, (1536, ), (1, ), 0), reinterpret_tensor(buf448, (1536, 384, 1, 1), (384, 1, 1, 1), 0), buf450, buf452, buf455, buf457, buf458, reinterpret_tensor(buf462, (384, ), (1, ), 0), reinterpret_tensor(buf463, (384, 1536, 1, 1), (1536, 1, 1, 1), 0), buf465, buf466, reinterpret_tensor(buf470, (384, ), (1, ), 0), reinterpret_tensor(buf471, (384, 64, 3, 3), (576, 9, 3, 1), 0), buf473, buf474, reinterpret_tensor(buf478, (384, ), (1, ), 0), reinterpret_tensor(buf479, (384, 64, 3, 3), (576, 9, 3, 1), 0), buf481, buf482, reinterpret_tensor(buf486, (1536, ), (1, ), 0), reinterpret_tensor(buf487, (1536, 384, 1, 1), (384, 1, 1, 1), 0), buf489, buf491, buf494, buf496, buf497, buf498, reinterpret_tensor(buf502, (384, ), (1, ), 0), reinterpret_tensor(buf503, (384, 1536, 1, 1), (1536, 1, 1, 1), 0), buf505, buf506, reinterpret_tensor(buf510, (384, ), (1, ), 0), reinterpret_tensor(buf511, (384, 64, 3, 3), (576, 9, 3, 1), 0), buf513, buf514, reinterpret_tensor(buf518, (384, ), (1, ), 0), reinterpret_tensor(buf519, (384, 64, 3, 3), (576, 9, 3, 1), 0), buf521, buf522, reinterpret_tensor(buf526, (1536, ), (1, ), 0), reinterpret_tensor(buf527, (1536, 384, 1, 1), (384, 1, 1, 1), 0), buf529, buf531, buf534, buf536, buf537, reinterpret_tensor(buf541, (2304, ), (1, ), 0), reinterpret_tensor(buf542, (2304, 1536, 1, 1), (1536, 1, 1, 1), 0), buf544, reinterpret_tensor(buf546, (128, 2304), (2304, 1), 0), reinterpret_tensor(buf539, (1, 2304, 1), (2304, 1, 1), 0), reinterpret_tensor(buf524, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf516, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf508, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf500, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf484, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf476, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf468, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf460, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf445, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf437, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf429, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf421, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf414, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf397, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf389, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf381, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf373, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf357, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf349, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf341, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf333, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf317, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf309, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf301, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf293, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf277, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf269, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf261, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf253, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf237, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf229, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf221, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf213, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf198, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf190, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf182, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf174, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf167, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf150, (1, 512, 1), (512, 1, 1), 0), reinterpret_tensor(buf142, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf134, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf126, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf111, (1, 512, 1), (512, 1, 1), 0), reinterpret_tensor(buf103, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf95, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf87, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf80, (1, 512, 1), (512, 1, 1), 0), reinterpret_tensor(buf64, (1, 256, 1), (256, 1, 1), 0), reinterpret_tensor(buf56, (1, 64, 1), (64, 1, 1), 0), reinterpret_tensor(buf48, (1, 64, 1), (64, 1, 1), 0), reinterpret_tensor(buf40, (1, 64, 1), (64, 1, 1), 0), reinterpret_tensor(buf33, (1, 256, 1), (256, 1, 1), 0), reinterpret_tensor(buf25, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf17, (1, 64, 1), (64, 1, 1), 0), reinterpret_tensor(buf9, (1, 32, 1), (32, 1, 1), 0), reinterpret_tensor(buf1, (1, 16, 1), (16, 1, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_2 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_4 = rand_strided((128, 3, 224, 224), (150528, 50176, 224, 1), device='npu:0', dtype=torch.float32)
    primals_5 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_6 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_9 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_11 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_12 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_14 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_15 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_17 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_18 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_20 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_21 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_22 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_23 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_24 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_26 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_27 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_28 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_29 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_31 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_33 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_34 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_35 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_36 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_37 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_39 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_40 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_42 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_43 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_45 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_46 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_47 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_48 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_50 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_52 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_53 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_54 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_55 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_56 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_57 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_58 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_59 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_60 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_61 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_62 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_64 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_66 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_67 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_68 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_69 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_70 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_71 = rand_strided((384, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_72 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_73 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_74 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_75 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_76 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_77 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_78 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_79 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_80 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_81 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_82 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_83 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_84 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_85 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_86 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_87 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_88 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_89 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_90 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_91 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_92 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_93 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_94 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_95 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_96 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_97 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_98 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_99 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_100 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_101 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_102 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_103 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_104 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_105 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_106 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_107 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_108 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_109 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_110 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_111 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_112 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_113 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_114 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_115 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_116 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_117 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_118 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_119 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_120 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_121 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_122 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_123 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_124 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_125 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_126 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_127 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_128 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_129 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_130 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_131 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_132 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_133 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_134 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_135 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_136 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_137 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_138 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_139 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_140 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_141 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_142 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_143 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_144 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_145 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_146 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_147 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_148 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_149 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_150 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_151 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_152 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_153 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_154 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_155 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_156 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_157 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_158 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_159 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_160 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_161 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_162 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_163 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_164 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_165 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_166 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_167 = rand_strided((1536, 1536, 1, 1), (1536, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_168 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_169 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_170 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_171 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_172 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_173 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_174 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_175 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_176 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_177 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_178 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_179 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_180 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_181 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_182 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_183 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_184 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_185 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_186 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_187 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_188 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_189 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_190 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_191 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_192 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_193 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_194 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_195 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_196 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_197 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_198 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_199 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_200 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_201 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_202 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_203 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_204 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_205 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_206 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_207 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_208 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='npu:0', dtype=torch.float32)
    primals_209 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_210 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_211 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_212 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_213 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_214 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_215 = rand_strided((384, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_216 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_217 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_218 = rand_strided((2304, 1536, 1, 1), (1536, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_219 = rand_strided((2304, 1, 1, 1), (1, 1, 1, 1), device='npu:0', dtype=torch.float32)
    primals_220 = rand_strided((2304, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_221 = rand_strided((1000, 2304), (2304, 1), device='npu:0', dtype=torch.float32)
    primals_222 = rand_strided((1000, ), (1, ), device='npu:0', dtype=torch.float32)
    return [primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('nfnet_l0', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
