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


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/7g/c7ghxioarywrgjyejuqphx6sxth3xibi22sjs3ls33keqnoq5v5y.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.permute, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x => permute
#   x_1 => native_layer_norm
# Graph fragment:
#   %convolution : Tensor "f32[8, 80, 72, 72][414720, 5184, 72, 1]npu:0" = PlaceHolder[target=convolution]
#   %buf2 : Tensor "f32[8, 72, 72, 1][5184, 72, 1, 41472]npu:0" = PlaceHolder[target=buf2]
#   %buf3 : Tensor "f32[8, 72, 72, 1][5184, 72, 1, 41472]npu:0" = PlaceHolder[target=buf3]
#   %permute : Tensor "f32[8, 72, 72, 80][414720, 72, 1, 5184]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convolution, [0, 2, 3, 1]), kwargs = {})
#   %native_layer_norm : [num_users=1] = call_function[target=torch.ops.aten.native_layer_norm.default](args = (%permute, [80], %arg3_1, %arg4_1, 1e-06), kwargs = {})
#   return %buf2,%buf3,%buf4
# SchedulerNodes: [SchedulerNode(name='op2'), SchedulerNode(name='op3'), SchedulerNode(name='op4')]

triton_per_fused_native_layer_norm_permute_0 = async_compile.triton('triton_per_fused_native_layer_norm_permute_0', '''
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
    size_hints={'z0': 8, 'y1': 72, 'x2': 72, 'r3': 80},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'r3_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_permute_0', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [], 'axis_names': ['z0', 'y1', 'x2', 'r3'], 'axis_static_values': (('z0', 8), ('y1', 72), ('x2', 72), ('r3', 80)), 'low_dims': {2}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_native_layer_norm_permute_0(in_out_ptr0, in_ptr0, out_ptr0, z0_numel, y1_numel, x2_numel, r3_numel, Z0BLOCK, Y1BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr):
    R3BLOCK_SUB: tl.constexpr = 80
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(1) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (x2_numel + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    base_r3= tl.arange(0, R3BLOCK_SUB)
    loops_r3 = (r3_numel + R3BLOCK_SUB - 1) // R3BLOCK_SUB
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[None,None,:,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            for loop_x2 in range(loops_x2):
                x2 = (loop_x2 * X2BLOCK_SUB) + base_x2[None,None,None,:]
                x2_mask = x2 < x2_numel
                r3 = base_r3[None,:,None,None]
                r3_mask = True
                tmp0 = tl.load(in_ptr0 + (x2 + 72*y1 + 5184*r3 + 414720*z0), x2_mask & y1_mask & z0_mask, other=0.0)
                tmp2 = tl.where(tl.broadcast_to((z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, 1, Y1BLOCK_SUB, 1]) & x2_mask.reshape([1, 1, 1, X2BLOCK_SUB])), [Z0BLOCK_SUB, R3BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB]), tmp0, 0)
                tmp3 = tl.sum(tmp2, 1).reshape(Z0BLOCK_SUB, 1, Y1BLOCK_SUB, X2BLOCK_SUB)
                tmp4 = tl.full([1, 1, 1, 1], 80.0, tl.float32)
                tmp5 = (tmp3 / tmp4)
                tmp6 = tmp0 - tmp5
                tmp7 = tmp6 * tmp6
                tmp9 = tl.where(tl.broadcast_to((z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, 1, Y1BLOCK_SUB, 1]) & x2_mask.reshape([1, 1, 1, X2BLOCK_SUB])), [Z0BLOCK_SUB, R3BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB]), tmp7, 0)
                tmp10 = tl.sum(tmp9, 1).reshape(Z0BLOCK_SUB, 1, Y1BLOCK_SUB, X2BLOCK_SUB)
                tl.store(in_out_ptr0 + (x2 + 72*y1 + 5184*z0), tmp5, x2_mask & y1_mask & z0_mask)
                tl.store(out_ptr0 + (x2 + 72*y1 + 5184*z0 ), tmp10, x2_mask & y1_mask & z0_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/wa/cwaj6pbme4y2pxhoqqznpxuedg6jjo36yzrwvjuhzy56j3rszjfx.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.permute, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x => permute
#   x_1 => native_layer_norm
# Graph fragment:
#   %convolution : Tensor "f32[8, 80, 72, 72][414720, 5184, 72, 1]npu:0" = PlaceHolder[target=convolution]
#   %buf3 : Tensor "f32[8, 72, 72, 1][5184, 72, 1, 41472]npu:0" = PlaceHolder[target=buf3]
#   %buf4 : Tensor "f32[8, 72, 72, 1][5184, 72, 1, 41472]npu:0" = PlaceHolder[target=buf4]
#   %arg3_1 : Tensor "f32[80][1]npu:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[80][1]npu:0" = PlaceHolder[target=arg4_1]
#   %permute : Tensor "f32[8, 72, 72, 80][414720, 72, 1, 5184]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convolution, [0, 2, 3, 1]), kwargs = {})
#   %native_layer_norm : [num_users=1] = call_function[target=torch.ops.aten.native_layer_norm.default](args = (%permute, [80], %arg3_1, %arg4_1, 1e-06), kwargs = {})
#   return %getitem
# SchedulerNodes: [SchedulerNode(name='op5')]

triton_poi_fused_native_layer_norm_permute_1 = async_compile.triton('triton_poi_fused_native_layer_norm_permute_1', '''
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
    size_hints={'y1': 8, 'y0': 5184, 'x2': 80},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'y1_numel': 'i32', 'y0_numel': 'i32', 'x2_numel': 'i32', 'Y1BLOCK': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_permute_1', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [2], 'axis_names': ['y1', 'y0', 'x2'], 'axis_static_values': (('y1', 8), ('y0', 5184), ('x2', 80)), 'low_dims': {1, 2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Y1BLOCK', 'Y0BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_permute_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, y1_numel, y0_numel, x2_numel, Y1BLOCK, Y0BLOCK, Y1BLOCK_SUB : tl.constexpr, Y0BLOCK_SUB : tl.constexpr):
    x2_numel = 80
    X2BLOCK_SUB: tl.constexpr = 80
    y1_offset = tl.program_id(0) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    y0_offset = tl.program_id(1) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    for loop_y1 in range(loops_y1):
        y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[:,None,None]
        y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
        for loop_y0 in range(loops_y0):
            y0_1 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[None,:,None]
            y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[None,None,:]
            y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
            y0_1_mask = y0_1 < min(Y0BLOCK+y0_offset, y0_numel)
            x2_2 = base_x2[None,None,:]
            x2 = base_x2[None,:,None]
            tmp0 = tl.load(in_ptr0 + (y0 + 5184*x2 + 414720*y1), y0_mask & y1_mask)
            tmp1 = tl.load(in_ptr1 + (y0 + 5184*y1), y0_mask & y1_mask)
            tmp3 = tl.load(in_ptr2 + (y0 + 5184*y1), y0_mask & y1_mask)
            tmp10 = tl.load(in_ptr3 + (x2), None)
            tmp12 = tl.load(in_ptr4 + (x2), None)
            tmp2 = tmp0 - tmp1
            tmp4 = tl.full([1, 1, 1], 80.0, tl.float32)
            tmp5 = (tmp3 / tmp4)
            tmp6 = tl.full([1, 1, 1], 1e-06, tl.float32)
            tmp7 = tmp5 + tmp6
            tmp8 = libdevice.rsqrt(tmp7)
            tmp9 = tmp2 * tmp8
            tmp11 = tmp9 * tmp10
            tmp13 = tmp11 + tmp12
            tl.store(out_ptr0 + (x2_2 + 80*y0_1 + 414720*y1), tmp13.permute([0, 2, 1]), y0_1_mask & y1_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/sh/cshfi7akj7ekrhconebg3yzq2ofirmupqnossfr2j3slxknde2w5.py
# Topologically Sorted Source Nodes: [x_g], Original ATen: [aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   x_g => pow_1, sum_1
# Graph fragment:
#   %gelu : Tensor "f32[8, 320, 72, 72][1658880, 5184, 72, 1]npu:0" = PlaceHolder[target=gelu]
#   %pow_1 : Tensor "f32[8, 320, 72, 72][1658880, 5184, 72, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%gelu, 2), kwargs = {})
#   %sum_1 : Tensor "f32[8, 320, 1, 1][320, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [2, 3], True), kwargs = {})
#   return %sum_1
# SchedulerNodes: [SchedulerNode(name='op16')]

triton_red_fused_linalg_vector_norm_2 = async_compile.triton('triton_red_fused_linalg_vector_norm_2', '''
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
    size_hints={'y0': 8, 'x1': 320, 'r2': 5184},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'r2_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_linalg_vector_norm_2', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['y0', 'x1', 'r2'], 'axis_static_values': (('y0', 8), ('x1', 320), ('r2', 5184)), 'low_dims': {2}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_red_fused_linalg_vector_norm_2(in_ptr0, out_ptr0, y0_numel, x1_numel, r2_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr, R2BLOCK_SUB : tl.constexpr):
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
            _tmp3 = tl.full([Y0BLOCK_SUB, X1BLOCK_SUB, R2BLOCK_SUB], 0, tl.float32)
            for loop_r2 in range(loops_r2):
                r2 = (loop_r2 * R2BLOCK_SUB) + base_r2[None,None,:]
                r2_mask = r2 < r2_numel
                tmp0 = tl.load(in_ptr0 + (r2 + 5184*x1 + 1658880*y0), r2_mask & x1_mask & y0_mask, other=0.0)
                tmp1 = tmp0 * tmp0
                tmp2 = tl.reshape(tmp1, [Y0BLOCK_SUB, X1BLOCK_SUB, R2BLOCK_SUB])
                tmp4 = _tmp3 + tmp2
                _tmp3 = tl.where(tl.reshape((r2_mask & x1_mask & y0_mask), [Y0BLOCK_SUB, X1BLOCK_SUB, R2BLOCK_SUB]), tmp4, _tmp3)
            tmp3 = tl.sum(_tmp3, 2).reshape(Y0BLOCK_SUB, X1BLOCK_SUB, 1)
            tl.store(out_ptr0 + (x1 + 320*y0 ), tmp3, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/ta/ctadv2atcwebacxj2iswtjfzc74c46c2bfm73xgq4twejby2mbsd.py
# Topologically Sorted Source Nodes: [x_g, mean], Original ATen: [aten.linalg_vector_norm, aten.mean]
# Source node to ATen node mapping:
#   mean => mean
#   x_g => pow_2
# Graph fragment:
#   %sum_1 : Tensor "f32[8, 320, 1, 1][320, 1, 2560, 2560]npu:0" = PlaceHolder[target=sum_1]
#   %pow_2 : Tensor "f32[8, 320, 1, 1][320, 1, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %mean : Tensor "f32[8, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [1], True), kwargs = {})
#   return %buf17
# SchedulerNodes: [SchedulerNode(name='op17')]

triton_per_fused_linalg_vector_norm_mean_3 = async_compile.triton('triton_per_fused_linalg_vector_norm_mean_3', '''
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
    size_hints={'x0': 8, 'r1': 320},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_linalg_vector_norm_mean_3', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 8), ('r1', 320)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_linalg_vector_norm_mean_3(in_ptr0, out_ptr0, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 320
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
        tmp0 = tl.load(in_ptr0 + (r1 + 320*x0), x0_mask, other=0.0)
        tmp1 = tl.sqrt_rn(tmp0)
        tmp3 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp1, 0)
        tmp4 = tl.sum(tmp3, 1).reshape(X0BLOCK_SUB, 1)
        tl.store(out_ptr0 + (x0 ), tmp4, x0_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/px/cpxai37gssddpjyo3mdn6qpuvls2ztdujfgbsg4ezxlmyugegn4j.py
# Topologically Sorted Source Nodes: [view, view_1, x_g, mean, add, x_n, mul, addcmul, x_10], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul]
# Source node to ATen node mapping:
#   add => add
#   addcmul => addcmul
#   mean => mean
#   mul => mul
#   view => view
#   view_1 => view_1
#   x_10 => add_1
#   x_g => pow_2
#   x_n => div
# Graph fragment:
#   %gelu : Tensor "f32[8, 320, 72, 72][1658880, 5184, 72, 1]npu:0" = PlaceHolder[target=gelu]
#   %arg11_1 : Tensor "f32[320][1]npu:0" = PlaceHolder[target=arg11_1]
#   %arg12_1 : Tensor "f32[320][1]npu:0" = PlaceHolder[target=arg12_1]
#   %sum_1 : Tensor "f32[8, 320, 1, 1][320, 1, 2560, 2560]npu:0" = PlaceHolder[target=sum_1]
#   %buf17 : Tensor "f32[8, 1, 1, 1][1, 8, 8, 8]npu:0" = PlaceHolder[target=buf17]
#   %view : Tensor "f32[1, 320, 1, 1][320, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg11_1, [1, -1, 1, 1]), kwargs = {})
#   %view_1 : Tensor "f32[1, 320, 1, 1][320, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg12_1, [1, -1, 1, 1]), kwargs = {})
#   %pow_2 : Tensor "f32[8, 320, 1, 1][320, 1, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %mean : Tensor "f32[8, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [1], True), kwargs = {})
#   %add : Tensor "f32[8, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %div : Tensor "f32[8, 320, 1, 1][320, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%pow_2, %add), kwargs = {})
#   %mul : Tensor "f32[8, 320, 72, 72][1658880, 5184, 72, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gelu, %div), kwargs = {})
#   %addcmul : Tensor "f32[8, 320, 72, 72][1658880, 5184, 72, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.addcmul.default](args = (%view, %view_1, %mul), kwargs = {})
#   %add_1 : Tensor "f32[8, 320, 72, 72][1658880, 5184, 72, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%gelu, %addcmul), kwargs = {})
#   return %add_1
# SchedulerNodes: [SchedulerNode(name='op18')]

triton_poi_fused_add_addcmul_div_linalg_vector_4 = async_compile.triton('triton_poi_fused_add_addcmul_div_linalg_vector_4', '''
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
    size_hints={'z0': 8, 'y1': 320, 'x2': 5184},
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addcmul_div_linalg_vector_4', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['z0', 'y1', 'x2'], 'axis_static_values': (('z0', 8), ('y1', 320), ('x2', 5184)), 'low_dims': {2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addcmul_div_linalg_vector_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, z0_numel, y1_numel, x2_numel, Z0BLOCK, Y1BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr):
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(1) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (x2_numel + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[None,:,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            for loop_x2 in range(loops_x2):
                x2 = (loop_x2 * X2BLOCK_SUB) + base_x2[None,None,:]
                x2_mask = x2 < x2_numel
                tmp0 = tl.load(in_out_ptr0 + (x2 + 5184*y1 + 1658880*z0), x2_mask & y1_mask & z0_mask)
                tmp1 = tl.load(in_ptr0 + (y1), y1_mask)
                tmp2 = tl.load(in_ptr1 + (y1), y1_mask)
                tmp3 = tl.load(in_ptr2 + (y1 + 320*z0), y1_mask & z0_mask)
                tmp5 = tl.load(in_ptr3 + (z0), z0_mask)
                tmp4 = tl.sqrt_rn(tmp3)
                tmp6 = tl.full([1, 1, 1], 320.0, tl.float32)
                tmp7 = (tmp5 / tmp6)
                tmp8 = tl.full([1, 1, 1], 1e-06, tl.float32)
                tmp9 = tmp7 + tmp8
                tmp10 = (tmp4 / tmp9)
                tmp11 = tmp0 * tmp10
                tmp12 = tmp2 * tmp11
                tmp13 = tl.full([1, 1, 1], 1.0, tl.float32)
                tmp14 = tmp13 * tmp12
                tmp15 = tmp1 + tmp14
                tmp16 = tmp0 + tmp15
                tl.store(in_out_ptr0 + (x2 + 5184*y1 + 1658880*z0), tmp16, x2_mask & y1_mask & z0_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/kg/ckgm56td72be2blbgssivjebybg6iy3bexv2g4ggo4cdlaqznfxd.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, x_13], Original ATen: [aten.permute, aten.native_layer_norm, aten.add]
# Source node to ATen node mapping:
#   x => permute
#   x_1 => native_layer_norm
#   x_13 => add_2
#   x_2 => permute_1
# Graph fragment:
#   %convolution_3 : Tensor "f32[8, 80, 72, 72][414720, 5184, 72, 1]npu:0" = PlaceHolder[target=convolution_3]
#   %getitem : Tensor "f32[8, 72, 72, 80][414720, 5760, 80, 1]npu:0" = PlaceHolder[target=getitem]
#   %permute : Tensor "f32[8, 72, 72, 80][414720, 72, 1, 5184]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convolution, [0, 2, 3, 1]), kwargs = {})
#   %native_layer_norm : [num_users=1] = call_function[target=torch.ops.aten.native_layer_norm.default](args = (%permute, [80], %arg3_1, %arg4_1, 1e-06), kwargs = {})
#   %permute_1 : Tensor "f32[8, 80, 72, 72][414720, 1, 5760, 80]npu:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%getitem, [0, 3, 1, 2]), kwargs = {})
#   %add_2 : Tensor "f32[8, 80, 72, 72][414720, 5184, 72, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_3, %permute_1), kwargs = {})
#   return %add_2
# SchedulerNodes: [SchedulerNode(name='op21')]

triton_poi_fused_add_native_layer_norm_permute_5 = async_compile.triton('triton_poi_fused_add_native_layer_norm_permute_5', '''
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
    size_hints={'y1': 8, 'y0': 80, 'x2': 5184},
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'y1_numel': 'i32', 'y0_numel': 'i32', 'x2_numel': 'i32', 'Y1BLOCK': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_permute_5', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['y1', 'y0', 'x2'], 'axis_static_values': (('y1', 8), ('y0', 80), ('x2', 5184)), 'low_dims': {1, 2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Y1BLOCK', 'Y0BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_permute_5(in_out_ptr0, in_ptr0, y1_numel, y0_numel, x2_numel, Y1BLOCK, Y0BLOCK, Y1BLOCK_SUB : tl.constexpr, Y0BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr):
    y1_offset = tl.program_id(0) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    y0_offset = tl.program_id(1) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (x2_numel + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    for loop_y1 in range(loops_y1):
        y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[:,None,None]
        y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
        for loop_y0 in range(loops_y0):
            y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[None,:,None]
            y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
            for loop_x2 in range(loops_x2):
                x2 = (loop_x2 * X2BLOCK_SUB) + base_x2[None,None,:]
                x2_mask = x2 < x2_numel
                tmp0 = tl.load(in_out_ptr0 + (x2 + 5184*y0 + 414720*y1), x2_mask & y0_mask & y1_mask)
                tmp1 = tl.load(in_ptr0 + (y0 + 80*x2 + 414720*y1), x2_mask & y0_mask & y1_mask)
                tmp2 = tmp0 + tmp1
                tl.store(in_out_ptr0 + (x2 + 5184*y0 + 414720*y1), tmp2, x2_mask & y0_mask & y1_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/2v/c2vsqeayasf4ndbpcf32dbxzm2ni5cctytsuiuy55tem7fpl33nh.py
# Topologically Sorted Source Nodes: [x_24, x_25, x_26], Original ATen: [aten.add, aten.permute, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_24 => add_5
#   x_25 => permute_6
#   x_26 => native_layer_norm_3
# Graph fragment:
#   %convolution_6 : Tensor "f32[8, 80, 72, 72][414720, 5184, 72, 1]npu:0" = PlaceHolder[target=convolution_6]
#   %add_2 : Tensor "f32[8, 80, 72, 72][414720, 5184, 72, 1]npu:0" = PlaceHolder[target=add_2]
#   %buf37 : Tensor "f32[8, 72, 72, 1][5184, 72, 1, 41472]npu:0" = PlaceHolder[target=buf37]
#   %buf38 : Tensor "f32[8, 72, 72, 1][5184, 72, 1, 41472]npu:0" = PlaceHolder[target=buf38]
#   %add_5 : Tensor "f32[8, 80, 72, 72][414720, 5184, 72, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_6, %add_2), kwargs = {})
#   %permute_6 : Tensor "f32[8, 72, 72, 80][414720, 72, 1, 5184]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_5, [0, 2, 3, 1]), kwargs = {})
#   %native_layer_norm_3 : [num_users=1] = call_function[target=torch.ops.aten.native_layer_norm.default](args = (%permute_6, [80], %arg25_1, %arg26_1, 1e-06), kwargs = {})
#   return %buf37,%buf38,%buf39
# SchedulerNodes: [SchedulerNode(name='op37'), SchedulerNode(name='op38'), SchedulerNode(name='op39')]

triton_per_fused_add_native_layer_norm_permute_6 = async_compile.triton('triton_per_fused_add_native_layer_norm_permute_6', '''
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
    size_hints={'z0': 8, 'y1': 72, 'x2': 72, 'r3': 80},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'r3_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_permute_6', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [], 'axis_names': ['z0', 'y1', 'x2', 'r3'], 'axis_static_values': (('z0', 8), ('y1', 72), ('x2', 72), ('r3', 80)), 'low_dims': {2}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_permute_6(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, z0_numel, y1_numel, x2_numel, r3_numel, Z0BLOCK, Y1BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr):
    R3BLOCK_SUB: tl.constexpr = 80
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(1) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (x2_numel + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    base_r3= tl.arange(0, R3BLOCK_SUB)
    loops_r3 = (r3_numel + R3BLOCK_SUB - 1) // R3BLOCK_SUB
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[None,None,:,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            for loop_x2 in range(loops_x2):
                x2 = (loop_x2 * X2BLOCK_SUB) + base_x2[None,None,None,:]
                x2_mask = x2 < x2_numel
                r3 = base_r3[None,:,None,None]
                r3_mask = True
                tmp0 = tl.load(in_ptr0 + (x2 + 72*y1 + 5184*r3 + 414720*z0), x2_mask & y1_mask & z0_mask, other=0.0)
                tmp1 = tl.load(in_ptr1 + (x2 + 72*y1 + 5184*r3 + 414720*z0), x2_mask & y1_mask & z0_mask, other=0.0)
                tmp2 = tmp0 + tmp1
                tmp4 = tl.where(tl.broadcast_to((z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, 1, Y1BLOCK_SUB, 1]) & x2_mask.reshape([1, 1, 1, X2BLOCK_SUB])), [Z0BLOCK_SUB, R3BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB]), tmp2, 0)
                tmp5 = tl.sum(tmp4, 1).reshape(Z0BLOCK_SUB, 1, Y1BLOCK_SUB, X2BLOCK_SUB)
                tmp6 = tl.full([1, 1, 1, 1], 80.0, tl.float32)
                tmp7 = (tmp5 / tmp6)
                tmp8 = tmp2 - tmp7
                tmp9 = tmp8 * tmp8
                tmp11 = tl.where(tl.broadcast_to((z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, 1, Y1BLOCK_SUB, 1]) & x2_mask.reshape([1, 1, 1, X2BLOCK_SUB])), [Z0BLOCK_SUB, R3BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB]), tmp9, 0)
                tmp12 = tl.sum(tmp11, 1).reshape(Z0BLOCK_SUB, 1, Y1BLOCK_SUB, X2BLOCK_SUB)
                tl.store(in_out_ptr0 + (x2 + 72*y1 + 5184*z0), tmp7, x2_mask & y1_mask & z0_mask)
                tl.store(out_ptr0 + (x2 + 72*y1 + 5184*z0 ), tmp12, x2_mask & y1_mask & z0_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/hy/chysh6462ii224jnwgua67aacofdhz4mqkfntijggd6gqfxzxafa.py
# Topologically Sorted Source Nodes: [x_24, x_25, x_26], Original ATen: [aten.add, aten.permute, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_24 => add_5
#   x_25 => permute_6
#   x_26 => native_layer_norm_3
# Graph fragment:
#   %convolution_6 : Tensor "f32[8, 80, 72, 72][414720, 5184, 72, 1]npu:0" = PlaceHolder[target=convolution_6]
#   %add_2 : Tensor "f32[8, 80, 72, 72][414720, 5184, 72, 1]npu:0" = PlaceHolder[target=add_2]
#   %buf38 : Tensor "f32[8, 72, 72, 1][5184, 72, 1, 41472]npu:0" = PlaceHolder[target=buf38]
#   %buf39 : Tensor "f32[8, 72, 72, 1][5184, 72, 1, 41472]npu:0" = PlaceHolder[target=buf39]
#   %arg25_1 : Tensor "f32[80][1]npu:0" = PlaceHolder[target=arg25_1]
#   %arg26_1 : Tensor "f32[80][1]npu:0" = PlaceHolder[target=arg26_1]
#   %add_5 : Tensor "f32[8, 80, 72, 72][414720, 5184, 72, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_6, %add_2), kwargs = {})
#   %permute_6 : Tensor "f32[8, 72, 72, 80][414720, 72, 1, 5184]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_5, [0, 2, 3, 1]), kwargs = {})
#   %native_layer_norm_3 : [num_users=1] = call_function[target=torch.ops.aten.native_layer_norm.default](args = (%permute_6, [80], %arg25_1, %arg26_1, 1e-06), kwargs = {})
#   return %getitem_9
# SchedulerNodes: [SchedulerNode(name='op40')]

triton_poi_fused_add_native_layer_norm_permute_7 = async_compile.triton('triton_poi_fused_add_native_layer_norm_permute_7', '''
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
    size_hints={'y1': 8, 'y0': 80, 'x2': 5184},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'y1_numel': 'i32', 'y0_numel': 'i32', 'x2_numel': 'i32', 'Y1BLOCK': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_permute_7', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['y1', 'y0', 'x2'], 'axis_static_values': (('y1', 8), ('y0', 80), ('x2', 5184)), 'low_dims': {1, 2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Y1BLOCK', 'Y0BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_permute_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, y1_numel, y0_numel, x2_numel, Y1BLOCK, Y0BLOCK, Y1BLOCK_SUB : tl.constexpr, Y0BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr):
    y1_offset = tl.program_id(0) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    y0_offset = tl.program_id(1) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (x2_numel + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    for loop_y1 in range(loops_y1):
        y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[:,None,None]
        y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
        for loop_y0 in range(loops_y0):
            y0_2 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[None,None,:]
            y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[None,:,None]
            y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
            y0_2_mask = y0_2 < min(Y0BLOCK+y0_offset, y0_numel)
            for loop_x2 in range(loops_x2):
                x2_1 = (loop_x2 * X2BLOCK_SUB) + base_x2[None,:,None]
                x2 = (loop_x2 * X2BLOCK_SUB) + base_x2[None,None,:]
                x2_mask = x2 < x2_numel
                x2_1_mask = x2_1 < x2_numel
                tmp0 = tl.load(in_ptr0 + (x2 + 5184*y0 + 414720*y1), x2_mask & y0_mask & y1_mask)
                tmp1 = tl.load(in_ptr1 + (x2 + 5184*y0 + 414720*y1), x2_mask & y0_mask & y1_mask)
                tmp3 = tl.load(in_ptr2 + (x2 + 5184*y1), x2_mask & y1_mask)
                tmp5 = tl.load(in_ptr3 + (x2 + 5184*y1), x2_mask & y1_mask)
                tmp12 = tl.load(in_ptr4 + (y0), y0_mask)
                tmp14 = tl.load(in_ptr5 + (y0), y0_mask)
                tmp2 = tmp0 + tmp1
                tmp4 = tmp2 - tmp3
                tmp6 = tl.full([1, 1, 1], 80.0, tl.float32)
                tmp7 = (tmp5 / tmp6)
                tmp8 = tl.full([1, 1, 1], 1e-06, tl.float32)
                tmp9 = tmp7 + tmp8
                tmp10 = libdevice.rsqrt(tmp9)
                tmp11 = tmp4 * tmp10
                tmp13 = tmp11 * tmp12
                tmp15 = tmp13 + tmp14
                tl.store(out_ptr0 + (y0_2 + 80*x2_1 + 414720*y1), tmp15.permute([0, 2, 1]), x2_1_mask & y0_2_mask & y1_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/h7/ch7vgrsbd5lrbqbn7guqspx6x6eligqppg4idkz65m7jlvazrumv.py
# Topologically Sorted Source Nodes: [x_29, x_30], Original ATen: [aten.permute, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_29 => permute_8
#   x_30 => native_layer_norm_4
# Graph fragment:
#   %convolution_8 : Tensor "f32[8, 160, 36, 36][207360, 1296, 36, 1]npu:0" = PlaceHolder[target=convolution_8]
#   %buf45 : Tensor "f32[8, 36, 36, 1][1296, 36, 1, 10368]npu:0" = PlaceHolder[target=buf45]
#   %buf46 : Tensor "f32[8, 36, 36, 1][1296, 36, 1, 10368]npu:0" = PlaceHolder[target=buf46]
#   %permute_8 : Tensor "f32[8, 36, 36, 160][207360, 36, 1, 1296]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convolution_8, [0, 2, 3, 1]), kwargs = {})
#   %native_layer_norm_4 : [num_users=1] = call_function[target=torch.ops.aten.native_layer_norm.default](args = (%permute_8, [160], %arg31_1, %arg32_1, 1e-06), kwargs = {})
#   return %buf45,%buf46,%buf47
# SchedulerNodes: [SchedulerNode(name='op45'), SchedulerNode(name='op46'), SchedulerNode(name='op47')]

triton_per_fused_native_layer_norm_permute_8 = async_compile.triton('triton_per_fused_native_layer_norm_permute_8', '''
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
    size_hints={'z0': 8, 'y1': 36, 'x2': 36, 'r3': 160},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'r3_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_permute_8', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [], 'axis_names': ['z0', 'y1', 'x2', 'r3'], 'axis_static_values': (('z0', 8), ('y1', 36), ('x2', 36), ('r3', 160)), 'low_dims': {2}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_native_layer_norm_permute_8(in_out_ptr0, in_ptr0, out_ptr0, z0_numel, y1_numel, x2_numel, r3_numel, Z0BLOCK, Y1BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr):
    R3BLOCK_SUB: tl.constexpr = 160
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(1) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (x2_numel + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    base_r3= tl.arange(0, R3BLOCK_SUB)
    loops_r3 = (r3_numel + R3BLOCK_SUB - 1) // R3BLOCK_SUB
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[None,None,:,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            for loop_x2 in range(loops_x2):
                x2 = (loop_x2 * X2BLOCK_SUB) + base_x2[None,None,None,:]
                x2_mask = x2 < x2_numel
                r3 = base_r3[None,:,None,None]
                r3_mask = True
                tmp0 = tl.load(in_ptr0 + (x2 + 36*y1 + 1296*r3 + 207360*z0), x2_mask & y1_mask & z0_mask, other=0.0)
                tmp2 = tl.where(tl.broadcast_to((z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, 1, Y1BLOCK_SUB, 1]) & x2_mask.reshape([1, 1, 1, X2BLOCK_SUB])), [Z0BLOCK_SUB, R3BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB]), tmp0, 0)
                tmp3 = tl.sum(tmp2, 1).reshape(Z0BLOCK_SUB, 1, Y1BLOCK_SUB, X2BLOCK_SUB)
                tmp4 = tl.full([1, 1, 1, 1], 160.0, tl.float32)
                tmp5 = (tmp3 / tmp4)
                tmp6 = tmp0 - tmp5
                tmp7 = tmp6 * tmp6
                tmp9 = tl.where(tl.broadcast_to((z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, 1, Y1BLOCK_SUB, 1]) & x2_mask.reshape([1, 1, 1, X2BLOCK_SUB])), [Z0BLOCK_SUB, R3BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB]), tmp7, 0)
                tmp10 = tl.sum(tmp9, 1).reshape(Z0BLOCK_SUB, 1, Y1BLOCK_SUB, X2BLOCK_SUB)
                tl.store(in_out_ptr0 + (x2 + 36*y1 + 1296*z0), tmp5, x2_mask & y1_mask & z0_mask)
                tl.store(out_ptr0 + (x2 + 36*y1 + 1296*z0 ), tmp10, x2_mask & y1_mask & z0_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/qh/cqhy2iwpuk3hfykqj6hsthrbuhky7rywz7rwsa56ufq2m2qbmgkm.py
# Topologically Sorted Source Nodes: [x_29, x_30], Original ATen: [aten.permute, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_29 => permute_8
#   x_30 => native_layer_norm_4
# Graph fragment:
#   %convolution_8 : Tensor "f32[8, 160, 36, 36][207360, 1296, 36, 1]npu:0" = PlaceHolder[target=convolution_8]
#   %buf46 : Tensor "f32[8, 36, 36, 1][1296, 36, 1, 10368]npu:0" = PlaceHolder[target=buf46]
#   %buf47 : Tensor "f32[8, 36, 36, 1][1296, 36, 1, 10368]npu:0" = PlaceHolder[target=buf47]
#   %arg31_1 : Tensor "f32[160][1]npu:0" = PlaceHolder[target=arg31_1]
#   %arg32_1 : Tensor "f32[160][1]npu:0" = PlaceHolder[target=arg32_1]
#   %permute_8 : Tensor "f32[8, 36, 36, 160][207360, 36, 1, 1296]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convolution_8, [0, 2, 3, 1]), kwargs = {})
#   %native_layer_norm_4 : [num_users=1] = call_function[target=torch.ops.aten.native_layer_norm.default](args = (%permute_8, [160], %arg31_1, %arg32_1, 1e-06), kwargs = {})
#   return %getitem_12
# SchedulerNodes: [SchedulerNode(name='op48')]

triton_poi_fused_native_layer_norm_permute_9 = async_compile.triton('triton_poi_fused_native_layer_norm_permute_9', '''
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
    size_hints={'y1': 8, 'y0': 1296, 'x2': 160},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'y1_numel': 'i32', 'y0_numel': 'i32', 'x2_numel': 'i32', 'Y1BLOCK': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_permute_9', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [2], 'axis_names': ['y1', 'y0', 'x2'], 'axis_static_values': (('y1', 8), ('y0', 1296), ('x2', 160)), 'low_dims': {1, 2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Y1BLOCK', 'Y0BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_permute_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, y1_numel, y0_numel, x2_numel, Y1BLOCK, Y0BLOCK, Y1BLOCK_SUB : tl.constexpr, Y0BLOCK_SUB : tl.constexpr):
    x2_numel = 160
    X2BLOCK_SUB: tl.constexpr = 160
    y1_offset = tl.program_id(0) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    y0_offset = tl.program_id(1) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    for loop_y1 in range(loops_y1):
        y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[:,None,None]
        y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
        for loop_y0 in range(loops_y0):
            y0_1 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[None,:,None]
            y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[None,None,:]
            y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
            y0_1_mask = y0_1 < min(Y0BLOCK+y0_offset, y0_numel)
            x2_2 = base_x2[None,None,:]
            x2 = base_x2[None,:,None]
            tmp0 = tl.load(in_ptr0 + (y0 + 1296*x2 + 207360*y1), y0_mask & y1_mask)
            tmp1 = tl.load(in_ptr1 + (y0 + 1296*y1), y0_mask & y1_mask)
            tmp3 = tl.load(in_ptr2 + (y0 + 1296*y1), y0_mask & y1_mask)
            tmp10 = tl.load(in_ptr3 + (x2), None)
            tmp12 = tl.load(in_ptr4 + (x2), None)
            tmp2 = tmp0 - tmp1
            tmp4 = tl.full([1, 1, 1], 160.0, tl.float32)
            tmp5 = (tmp3 / tmp4)
            tmp6 = tl.full([1, 1, 1], 1e-06, tl.float32)
            tmp7 = tmp5 + tmp6
            tmp8 = libdevice.rsqrt(tmp7)
            tmp9 = tmp2 * tmp8
            tmp11 = tmp9 * tmp10
            tmp13 = tmp11 + tmp12
            tl.store(out_ptr0 + (x2_2 + 160*y0_1 + 207360*y1), tmp13.permute([0, 2, 1]), y0_1_mask & y1_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/ta/ctakoybmcivc3ap6en6oxnglwshoygdmmgedscmfspougbnivj44.py
# Topologically Sorted Source Nodes: [x_g_2], Original ATen: [aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   x_g_2 => pow_5, sum_3
# Graph fragment:
#   %gelu_2 : Tensor "f32[8, 640, 36, 36][829440, 1296, 36, 1]npu:0" = PlaceHolder[target=gelu_2]
#   %pow_5 : Tensor "f32[8, 640, 36, 36][829440, 1296, 36, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%gelu_2, 2), kwargs = {})
#   %sum_3 : Tensor "f32[8, 640, 1, 1][640, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_5, [2, 3], True), kwargs = {})
#   return %sum_3
# SchedulerNodes: [SchedulerNode(name='op53')]

triton_per_fused_linalg_vector_norm_10 = async_compile.triton('triton_per_fused_linalg_vector_norm_10', '''
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
    size_hints={'y0': 8, 'x1': 640, 'r2': 1296},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'r2_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_linalg_vector_norm_10', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['y0', 'x1', 'r2'], 'axis_static_values': (('y0', 8), ('x1', 640), ('r2', 1296)), 'low_dims': {2}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_linalg_vector_norm_10(in_ptr0, out_ptr0, y0_numel, x1_numel, r2_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    R2BLOCK_SUB: tl.constexpr = 1296
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
            tmp0 = tl.load(in_ptr0 + (r2 + 1296*x1 + 829440*y0), x1_mask & y0_mask, other=0.0)
            tmp1 = tmp0 * tmp0
            tmp3 = tl.where(tl.broadcast_to((y0_mask.reshape([Y0BLOCK_SUB, 1, 1]) & x1_mask.reshape([1, X1BLOCK_SUB, 1])), [Y0BLOCK_SUB, X1BLOCK_SUB, R2BLOCK_SUB]), tmp1, 0)
            tmp4 = tl.sum(tmp3, 2).reshape(Y0BLOCK_SUB, X1BLOCK_SUB, 1)
            tl.store(out_ptr0 + (x1 + 640*y0 ), tmp4, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/ic/cic2pgiq6cyxr5qdhfu57fly4kg5msnib55nzfiagi5wy6pw2ik3.py
# Topologically Sorted Source Nodes: [x_g_2, mean_2], Original ATen: [aten.linalg_vector_norm, aten.mean]
# Source node to ATen node mapping:
#   mean_2 => mean_2
#   x_g_2 => pow_6
# Graph fragment:
#   %sum_3 : Tensor "f32[8, 640, 1, 1][640, 1, 5120, 5120]npu:0" = PlaceHolder[target=sum_3]
#   %pow_6 : Tensor "f32[8, 640, 1, 1][640, 1, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %mean_2 : Tensor "f32[8, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_6, [1], True), kwargs = {})
#   return %buf54
# SchedulerNodes: [SchedulerNode(name='op54')]

triton_per_fused_linalg_vector_norm_mean_11 = async_compile.triton('triton_per_fused_linalg_vector_norm_mean_11', '''
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
    size_hints={'x0': 8, 'r1': 640},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_linalg_vector_norm_mean_11', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 8), ('r1', 640)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_linalg_vector_norm_mean_11(in_ptr0, out_ptr0, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 640
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
        tmp0 = tl.load(in_ptr0 + (r1 + 640*x0), x0_mask, other=0.0)
        tmp1 = tl.sqrt_rn(tmp0)
        tmp3 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp1, 0)
        tmp4 = tl.sum(tmp3, 1).reshape(X0BLOCK_SUB, 1)
        tl.store(out_ptr0 + (x0 ), tmp4, x0_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/7b/c7bn5pgjxkvgl7sw2ld7mc3wi7zx23lnvllm7kc6zogykrm6uuy7.py
# Topologically Sorted Source Nodes: [view_4, view_5, x_g_2, mean_2, add_6, x_n_2, mul_2, addcmul_2, x_35], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul]
# Source node to ATen node mapping:
#   add_6 => add_6
#   addcmul_2 => addcmul_2
#   mean_2 => mean_2
#   mul_2 => mul_2
#   view_4 => view_4
#   view_5 => view_5
#   x_35 => add_7
#   x_g_2 => pow_6
#   x_n_2 => div_2
# Graph fragment:
#   %gelu_2 : Tensor "f32[8, 640, 36, 36][829440, 1296, 36, 1]npu:0" = PlaceHolder[target=gelu_2]
#   %arg35_1 : Tensor "f32[640][1]npu:0" = PlaceHolder[target=arg35_1]
#   %arg36_1 : Tensor "f32[640][1]npu:0" = PlaceHolder[target=arg36_1]
#   %sum_3 : Tensor "f32[8, 640, 1, 1][640, 1, 5120, 5120]npu:0" = PlaceHolder[target=sum_3]
#   %buf54 : Tensor "f32[8, 1, 1, 1][1, 8, 8, 8]npu:0" = PlaceHolder[target=buf54]
#   %view_4 : Tensor "f32[1, 640, 1, 1][640, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg35_1, [1, -1, 1, 1]), kwargs = {})
#   %view_5 : Tensor "f32[1, 640, 1, 1][640, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg36_1, [1, -1, 1, 1]), kwargs = {})
#   %pow_6 : Tensor "f32[8, 640, 1, 1][640, 1, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %mean_2 : Tensor "f32[8, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_6, [1], True), kwargs = {})
#   %add_6 : Tensor "f32[8, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-06), kwargs = {})
#   %div_2 : Tensor "f32[8, 640, 1, 1][640, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%pow_6, %add_6), kwargs = {})
#   %mul_2 : Tensor "f32[8, 640, 36, 36][829440, 1296, 36, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gelu_2, %div_2), kwargs = {})
#   %addcmul_2 : Tensor "f32[8, 640, 36, 36][829440, 1296, 36, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.addcmul.default](args = (%view_4, %view_5, %mul_2), kwargs = {})
#   %add_7 : Tensor "f32[8, 640, 36, 36][829440, 1296, 36, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%gelu_2, %addcmul_2), kwargs = {})
#   return %add_7
# SchedulerNodes: [SchedulerNode(name='op55')]

triton_poi_fused_add_addcmul_div_linalg_vector_12 = async_compile.triton('triton_poi_fused_add_addcmul_div_linalg_vector_12', '''
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
    size_hints={'z0': 8, 'y1': 640, 'x2': 1296},
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addcmul_div_linalg_vector_12', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['z0', 'y1', 'x2'], 'axis_static_values': (('z0', 8), ('y1', 640), ('x2', 1296)), 'low_dims': {2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addcmul_div_linalg_vector_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, z0_numel, y1_numel, x2_numel, Z0BLOCK, Y1BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr):
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(1) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (x2_numel + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[None,:,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            for loop_x2 in range(loops_x2):
                x2 = (loop_x2 * X2BLOCK_SUB) + base_x2[None,None,:]
                x2_mask = x2 < x2_numel
                tmp0 = tl.load(in_out_ptr0 + (x2 + 1296*y1 + 829440*z0), x2_mask & y1_mask & z0_mask)
                tmp1 = tl.load(in_ptr0 + (y1), y1_mask)
                tmp2 = tl.load(in_ptr1 + (y1), y1_mask)
                tmp3 = tl.load(in_ptr2 + (y1 + 640*z0), y1_mask & z0_mask)
                tmp5 = tl.load(in_ptr3 + (z0), z0_mask)
                tmp4 = tl.sqrt_rn(tmp3)
                tmp6 = tl.full([1, 1, 1], 640.0, tl.float32)
                tmp7 = (tmp5 / tmp6)
                tmp8 = tl.full([1, 1, 1], 1e-06, tl.float32)
                tmp9 = tmp7 + tmp8
                tmp10 = (tmp4 / tmp9)
                tmp11 = tmp0 * tmp10
                tmp12 = tmp2 * tmp11
                tmp13 = tl.full([1, 1, 1], 1.0, tl.float32)
                tmp14 = tmp13 * tmp12
                tmp15 = tmp1 + tmp14
                tmp16 = tmp0 + tmp15
                tl.store(in_out_ptr0 + (x2 + 1296*y1 + 829440*z0), tmp16, x2_mask & y1_mask & z0_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/zh/czh5ygqozn4l2i5tkxarmu6ofq27ge7zblsayqrsf7wd2umgcfxw.py
# Topologically Sorted Source Nodes: [x_38], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_38 => add_8
# Graph fragment:
#   %convolution_10 : Tensor "f32[8, 160, 36, 36][207360, 1296, 36, 1]npu:0" = PlaceHolder[target=convolution_10]
#   %convolution_7 : Tensor "f32[8, 160, 36, 36][207360, 1296, 36, 1]npu:0" = PlaceHolder[target=convolution_7]
#   %add_8 : Tensor "f32[8, 160, 36, 36][207360, 1296, 36, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_10, %convolution_7), kwargs = {})
#   return %add_8
# SchedulerNodes: [SchedulerNode(name='op58')]

triton_poi_fused_add_13 = async_compile.triton('triton_poi_fused_add_13', '''
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
    size_hints={'t0': 8, 'z1': 160, 'y2': 36, 'x3': 36},
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 't0_numel': 'i32', 'z1_numel': 'i32', 'y2_numel': 'i32', 'x3_numel': 'i32', 'T0BLOCK': 'i32', 'Z1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_13', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [3], 'axis_names': ['t0', 'z1', 'y2', 'x3'], 'axis_static_values': (('t0', 8), ('z1', 160), ('y2', 36), ('x3', 36)), 'low_dims': {3}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('T0BLOCK', 'Z1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_13(in_out_ptr0, in_ptr0, t0_numel, z1_numel, y2_numel, x3_numel, T0BLOCK, Z1BLOCK, T0BLOCK_SUB : tl.constexpr, Z1BLOCK_SUB : tl.constexpr, Y2BLOCK_SUB : tl.constexpr):
    x3_numel = 36
    X3BLOCK_SUB: tl.constexpr = 36
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
                tmp0 = tl.load(in_out_ptr0 + (x3 + 36*y2 + 1296*z1 + 207360*t0), t0_mask & y2_mask & z1_mask)
                tmp1 = tl.load(in_ptr0 + (x3 + 36*y2 + 1296*z1 + 207360*t0), t0_mask & y2_mask & z1_mask)
                tmp2 = tmp0 + tmp1
                tl.store(in_out_ptr0 + (x3 + 36*y2 + 1296*z1 + 207360*t0), tmp2, t0_mask & y2_mask & z1_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/3s/c3sybg5h5umst2dhlqens7jfcscgyyqkv6wby7wdzn2htpx5dle2.py
# Topologically Sorted Source Nodes: [x_49, x_50, x_51], Original ATen: [aten.add, aten.permute, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_49 => add_11
#   x_50 => permute_12
#   x_51 => native_layer_norm_6
# Graph fragment:
#   %convolution_13 : Tensor "f32[8, 160, 36, 36][207360, 1296, 36, 1]npu:0" = PlaceHolder[target=convolution_13]
#   %add_8 : Tensor "f32[8, 160, 36, 36][207360, 1296, 36, 1]npu:0" = PlaceHolder[target=add_8]
#   %buf74 : Tensor "f32[8, 36, 36, 1][1296, 36, 1, 10368]npu:0" = PlaceHolder[target=buf74]
#   %buf75 : Tensor "f32[8, 36, 36, 1][1296, 36, 1, 10368]npu:0" = PlaceHolder[target=buf75]
#   %add_11 : Tensor "f32[8, 160, 36, 36][207360, 1296, 36, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_13, %add_8), kwargs = {})
#   %permute_12 : Tensor "f32[8, 36, 36, 160][207360, 36, 1, 1296]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_11, [0, 2, 3, 1]), kwargs = {})
#   %native_layer_norm_6 : [num_users=1] = call_function[target=torch.ops.aten.native_layer_norm.default](args = (%permute_12, [160], %arg49_1, %arg50_1, 1e-06), kwargs = {})
#   return %buf74,%buf75,%buf76
# SchedulerNodes: [SchedulerNode(name='op74'), SchedulerNode(name='op75'), SchedulerNode(name='op76')]

triton_per_fused_add_native_layer_norm_permute_14 = async_compile.triton('triton_per_fused_add_native_layer_norm_permute_14', '''
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
    size_hints={'z0': 8, 'y1': 36, 'x2': 36, 'r3': 160},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'r3_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_permute_14', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [], 'axis_names': ['z0', 'y1', 'x2', 'r3'], 'axis_static_values': (('z0', 8), ('y1', 36), ('x2', 36), ('r3', 160)), 'low_dims': {2}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_permute_14(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, z0_numel, y1_numel, x2_numel, r3_numel, Z0BLOCK, Y1BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr):
    R3BLOCK_SUB: tl.constexpr = 160
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(1) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (x2_numel + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    base_r3= tl.arange(0, R3BLOCK_SUB)
    loops_r3 = (r3_numel + R3BLOCK_SUB - 1) // R3BLOCK_SUB
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[None,None,:,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            for loop_x2 in range(loops_x2):
                x2 = (loop_x2 * X2BLOCK_SUB) + base_x2[None,None,None,:]
                x2_mask = x2 < x2_numel
                r3 = base_r3[None,:,None,None]
                r3_mask = True
                tmp0 = tl.load(in_ptr0 + (x2 + 36*y1 + 1296*r3 + 207360*z0), x2_mask & y1_mask & z0_mask, other=0.0)
                tmp1 = tl.load(in_ptr1 + (x2 + 36*y1 + 1296*r3 + 207360*z0), x2_mask & y1_mask & z0_mask, other=0.0)
                tmp2 = tmp0 + tmp1
                tmp4 = tl.where(tl.broadcast_to((z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, 1, Y1BLOCK_SUB, 1]) & x2_mask.reshape([1, 1, 1, X2BLOCK_SUB])), [Z0BLOCK_SUB, R3BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB]), tmp2, 0)
                tmp5 = tl.sum(tmp4, 1).reshape(Z0BLOCK_SUB, 1, Y1BLOCK_SUB, X2BLOCK_SUB)
                tmp6 = tl.full([1, 1, 1, 1], 160.0, tl.float32)
                tmp7 = (tmp5 / tmp6)
                tmp8 = tmp2 - tmp7
                tmp9 = tmp8 * tmp8
                tmp11 = tl.where(tl.broadcast_to((z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, 1, Y1BLOCK_SUB, 1]) & x2_mask.reshape([1, 1, 1, X2BLOCK_SUB])), [Z0BLOCK_SUB, R3BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB]), tmp9, 0)
                tmp12 = tl.sum(tmp11, 1).reshape(Z0BLOCK_SUB, 1, Y1BLOCK_SUB, X2BLOCK_SUB)
                tl.store(in_out_ptr0 + (x2 + 36*y1 + 1296*z0), tmp7, x2_mask & y1_mask & z0_mask)
                tl.store(out_ptr0 + (x2 + 36*y1 + 1296*z0 ), tmp12, x2_mask & y1_mask & z0_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/tw/ctwfspmmixyhdpue7dkuqlvc3mzitp2qktcipqigh7o3ruvkcu57.py
# Topologically Sorted Source Nodes: [x_49, x_50, x_51], Original ATen: [aten.add, aten.permute, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_49 => add_11
#   x_50 => permute_12
#   x_51 => native_layer_norm_6
# Graph fragment:
#   %convolution_13 : Tensor "f32[8, 160, 36, 36][207360, 1296, 36, 1]npu:0" = PlaceHolder[target=convolution_13]
#   %add_8 : Tensor "f32[8, 160, 36, 36][207360, 1296, 36, 1]npu:0" = PlaceHolder[target=add_8]
#   %buf75 : Tensor "f32[8, 36, 36, 1][1296, 36, 1, 10368]npu:0" = PlaceHolder[target=buf75]
#   %buf76 : Tensor "f32[8, 36, 36, 1][1296, 36, 1, 10368]npu:0" = PlaceHolder[target=buf76]
#   %arg49_1 : Tensor "f32[160][1]npu:0" = PlaceHolder[target=arg49_1]
#   %arg50_1 : Tensor "f32[160][1]npu:0" = PlaceHolder[target=arg50_1]
#   %add_11 : Tensor "f32[8, 160, 36, 36][207360, 1296, 36, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_13, %add_8), kwargs = {})
#   %permute_12 : Tensor "f32[8, 36, 36, 160][207360, 36, 1, 1296]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_11, [0, 2, 3, 1]), kwargs = {})
#   %native_layer_norm_6 : [num_users=1] = call_function[target=torch.ops.aten.native_layer_norm.default](args = (%permute_12, [160], %arg49_1, %arg50_1, 1e-06), kwargs = {})
#   return %getitem_18
# SchedulerNodes: [SchedulerNode(name='op77')]

triton_poi_fused_add_native_layer_norm_permute_15 = async_compile.triton('triton_poi_fused_add_native_layer_norm_permute_15', '''
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
    size_hints={'y1': 8, 'y0': 160, 'x2': 1296},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'y1_numel': 'i32', 'y0_numel': 'i32', 'x2_numel': 'i32', 'Y1BLOCK': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_permute_15', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['y1', 'y0', 'x2'], 'axis_static_values': (('y1', 8), ('y0', 160), ('x2', 1296)), 'low_dims': {1, 2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Y1BLOCK', 'Y0BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_permute_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, y1_numel, y0_numel, x2_numel, Y1BLOCK, Y0BLOCK, Y1BLOCK_SUB : tl.constexpr, Y0BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr):
    y1_offset = tl.program_id(0) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    y0_offset = tl.program_id(1) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (x2_numel + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    for loop_y1 in range(loops_y1):
        y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[:,None,None]
        y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
        for loop_y0 in range(loops_y0):
            y0_2 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[None,None,:]
            y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[None,:,None]
            y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
            y0_2_mask = y0_2 < min(Y0BLOCK+y0_offset, y0_numel)
            for loop_x2 in range(loops_x2):
                x2_1 = (loop_x2 * X2BLOCK_SUB) + base_x2[None,:,None]
                x2 = (loop_x2 * X2BLOCK_SUB) + base_x2[None,None,:]
                x2_mask = x2 < x2_numel
                x2_1_mask = x2_1 < x2_numel
                tmp0 = tl.load(in_ptr0 + (x2 + 1296*y0 + 207360*y1), x2_mask & y0_mask & y1_mask)
                tmp1 = tl.load(in_ptr1 + (x2 + 1296*y0 + 207360*y1), x2_mask & y0_mask & y1_mask)
                tmp3 = tl.load(in_ptr2 + (x2 + 1296*y1), x2_mask & y1_mask)
                tmp5 = tl.load(in_ptr3 + (x2 + 1296*y1), x2_mask & y1_mask)
                tmp12 = tl.load(in_ptr4 + (y0), y0_mask)
                tmp14 = tl.load(in_ptr5 + (y0), y0_mask)
                tmp2 = tmp0 + tmp1
                tmp4 = tmp2 - tmp3
                tmp6 = tl.full([1, 1, 1], 160.0, tl.float32)
                tmp7 = (tmp5 / tmp6)
                tmp8 = tl.full([1, 1, 1], 1e-06, tl.float32)
                tmp9 = tmp7 + tmp8
                tmp10 = libdevice.rsqrt(tmp9)
                tmp11 = tmp4 * tmp10
                tmp13 = tmp11 * tmp12
                tmp15 = tmp13 + tmp14
                tl.store(out_ptr0 + (y0_2 + 160*x2_1 + 207360*y1), tmp15.permute([0, 2, 1]), x2_1_mask & y0_2_mask & y1_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/xe/cxeeocesjokib7n5qazobksutcn64pesvvgh6tmjey6youlx7tmu.py
# Topologically Sorted Source Nodes: [x_54, x_55], Original ATen: [aten.permute, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_54 => permute_14
#   x_55 => native_layer_norm_7
# Graph fragment:
#   %convolution_15 : Tensor "f32[8, 320, 18, 18][103680, 324, 18, 1]npu:0" = PlaceHolder[target=convolution_15]
#   %buf82 : Tensor "f32[8, 18, 18, 1][324, 18, 1, 2592]npu:0" = PlaceHolder[target=buf82]
#   %buf83 : Tensor "f32[8, 18, 18, 1][324, 18, 1, 2592]npu:0" = PlaceHolder[target=buf83]
#   %permute_14 : Tensor "f32[8, 18, 18, 320][103680, 18, 1, 324]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convolution_15, [0, 2, 3, 1]), kwargs = {})
#   %native_layer_norm_7 : [num_users=1] = call_function[target=torch.ops.aten.native_layer_norm.default](args = (%permute_14, [320], %arg55_1, %arg56_1, 1e-06), kwargs = {})
#   return %buf82,%buf83,%buf84
# SchedulerNodes: [SchedulerNode(name='op82'), SchedulerNode(name='op83'), SchedulerNode(name='op84')]

triton_per_fused_native_layer_norm_permute_16 = async_compile.triton('triton_per_fused_native_layer_norm_permute_16', '''
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
    size_hints={'z0': 8, 'y1': 18, 'x2': 18, 'r3': 320},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'r3_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_permute_16', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [], 'axis_names': ['z0', 'y1', 'x2', 'r3'], 'axis_static_values': (('z0', 8), ('y1', 18), ('x2', 18), ('r3', 320)), 'low_dims': {2}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_native_layer_norm_permute_16(in_out_ptr0, in_ptr0, out_ptr0, z0_numel, y1_numel, x2_numel, r3_numel, Z0BLOCK, Y1BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr):
    R3BLOCK_SUB: tl.constexpr = 320
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(1) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (x2_numel + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    base_r3= tl.arange(0, R3BLOCK_SUB)
    loops_r3 = (r3_numel + R3BLOCK_SUB - 1) // R3BLOCK_SUB
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[None,None,:,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            for loop_x2 in range(loops_x2):
                x2 = (loop_x2 * X2BLOCK_SUB) + base_x2[None,None,None,:]
                x2_mask = x2 < x2_numel
                r3 = base_r3[None,:,None,None]
                r3_mask = True
                tmp0 = tl.load(in_ptr0 + (x2 + 18*y1 + 324*r3 + 103680*z0), x2_mask & y1_mask & z0_mask, other=0.0)
                tmp2 = tl.where(tl.broadcast_to((z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, 1, Y1BLOCK_SUB, 1]) & x2_mask.reshape([1, 1, 1, X2BLOCK_SUB])), [Z0BLOCK_SUB, R3BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB]), tmp0, 0)
                tmp3 = tl.sum(tmp2, 1).reshape(Z0BLOCK_SUB, 1, Y1BLOCK_SUB, X2BLOCK_SUB)
                tmp4 = tl.full([1, 1, 1, 1], 320.0, tl.float32)
                tmp5 = (tmp3 / tmp4)
                tmp6 = tmp0 - tmp5
                tmp7 = tmp6 * tmp6
                tmp9 = tl.where(tl.broadcast_to((z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, 1, Y1BLOCK_SUB, 1]) & x2_mask.reshape([1, 1, 1, X2BLOCK_SUB])), [Z0BLOCK_SUB, R3BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB]), tmp7, 0)
                tmp10 = tl.sum(tmp9, 1).reshape(Z0BLOCK_SUB, 1, Y1BLOCK_SUB, X2BLOCK_SUB)
                tl.store(in_out_ptr0 + (x2 + 18*y1 + 324*z0), tmp5, x2_mask & y1_mask & z0_mask)
                tl.store(out_ptr0 + (x2 + 18*y1 + 324*z0 ), tmp10, x2_mask & y1_mask & z0_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/hw/chwhz54cbkqadbh32dl7ma2jf2zdx2di5rvf2krs5ztqannotv22.py
# Topologically Sorted Source Nodes: [x_54, x_55], Original ATen: [aten.permute, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_54 => permute_14
#   x_55 => native_layer_norm_7
# Graph fragment:
#   %convolution_15 : Tensor "f32[8, 320, 18, 18][103680, 324, 18, 1]npu:0" = PlaceHolder[target=convolution_15]
#   %buf83 : Tensor "f32[8, 18, 18, 1][324, 18, 1, 2592]npu:0" = PlaceHolder[target=buf83]
#   %buf84 : Tensor "f32[8, 18, 18, 1][324, 18, 1, 2592]npu:0" = PlaceHolder[target=buf84]
#   %arg55_1 : Tensor "f32[320][1]npu:0" = PlaceHolder[target=arg55_1]
#   %arg56_1 : Tensor "f32[320][1]npu:0" = PlaceHolder[target=arg56_1]
#   %permute_14 : Tensor "f32[8, 18, 18, 320][103680, 18, 1, 324]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convolution_15, [0, 2, 3, 1]), kwargs = {})
#   %native_layer_norm_7 : [num_users=1] = call_function[target=torch.ops.aten.native_layer_norm.default](args = (%permute_14, [320], %arg55_1, %arg56_1, 1e-06), kwargs = {})
#   return %getitem_21
# SchedulerNodes: [SchedulerNode(name='op85')]

triton_poi_fused_native_layer_norm_permute_17 = async_compile.triton('triton_poi_fused_native_layer_norm_permute_17', '''
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
    size_hints={'y1': 8, 'y0': 324, 'x2': 320},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'y1_numel': 'i32', 'y0_numel': 'i32', 'x2_numel': 'i32', 'Y1BLOCK': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_permute_17', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [2], 'axis_names': ['y1', 'y0', 'x2'], 'axis_static_values': (('y1', 8), ('y0', 324), ('x2', 320)), 'low_dims': {1, 2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Y1BLOCK', 'Y0BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_permute_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, y1_numel, y0_numel, x2_numel, Y1BLOCK, Y0BLOCK, Y1BLOCK_SUB : tl.constexpr, Y0BLOCK_SUB : tl.constexpr):
    x2_numel = 320
    X2BLOCK_SUB: tl.constexpr = 320
    y1_offset = tl.program_id(0) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    y0_offset = tl.program_id(1) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    for loop_y1 in range(loops_y1):
        y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[:,None,None]
        y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
        for loop_y0 in range(loops_y0):
            y0_1 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[None,:,None]
            y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[None,None,:]
            y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
            y0_1_mask = y0_1 < min(Y0BLOCK+y0_offset, y0_numel)
            x2_2 = base_x2[None,None,:]
            x2 = base_x2[None,:,None]
            tmp0 = tl.load(in_ptr0 + (y0 + 324*x2 + 103680*y1), y0_mask & y1_mask)
            tmp1 = tl.load(in_ptr1 + (y0 + 324*y1), y0_mask & y1_mask)
            tmp3 = tl.load(in_ptr2 + (y0 + 324*y1), y0_mask & y1_mask)
            tmp10 = tl.load(in_ptr3 + (x2), None)
            tmp12 = tl.load(in_ptr4 + (x2), None)
            tmp2 = tmp0 - tmp1
            tmp4 = tl.full([1, 1, 1], 320.0, tl.float32)
            tmp5 = (tmp3 / tmp4)
            tmp6 = tl.full([1, 1, 1], 1e-06, tl.float32)
            tmp7 = tmp5 + tmp6
            tmp8 = libdevice.rsqrt(tmp7)
            tmp9 = tmp2 * tmp8
            tmp11 = tmp9 * tmp10
            tmp13 = tmp11 + tmp12
            tl.store(out_ptr0 + (x2_2 + 320*y0_1 + 103680*y1), tmp13.permute([0, 2, 1]), y0_1_mask & y1_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/jo/cjoylvjudn7rybx7dqyft63hultnyldhndvu6pid4gjqejd5c74i.py
# Topologically Sorted Source Nodes: [x_g_4], Original ATen: [aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   x_g_4 => pow_9, sum_5
# Graph fragment:
#   %gelu_4 : Tensor "f32[8, 1280, 18, 18][414720, 324, 18, 1]npu:0" = PlaceHolder[target=gelu_4]
#   %pow_9 : Tensor "f32[8, 1280, 18, 18][414720, 324, 18, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%gelu_4, 2), kwargs = {})
#   %sum_5 : Tensor "f32[8, 1280, 1, 1][1280, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_9, [2, 3], True), kwargs = {})
#   return %sum_5
# SchedulerNodes: [SchedulerNode(name='op90')]

triton_per_fused_linalg_vector_norm_18 = async_compile.triton('triton_per_fused_linalg_vector_norm_18', '''
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
    size_hints={'y0': 8, 'x1': 1280, 'r2': 324},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'r2_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_linalg_vector_norm_18', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['y0', 'x1', 'r2'], 'axis_static_values': (('y0', 8), ('x1', 1280), ('r2', 324)), 'low_dims': {2}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_linalg_vector_norm_18(in_ptr0, out_ptr0, y0_numel, x1_numel, r2_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    R2BLOCK_SUB: tl.constexpr = 324
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
            tmp0 = tl.load(in_ptr0 + (r2 + 324*x1 + 414720*y0), x1_mask & y0_mask, other=0.0)
            tmp1 = tmp0 * tmp0
            tmp3 = tl.where(tl.broadcast_to((y0_mask.reshape([Y0BLOCK_SUB, 1, 1]) & x1_mask.reshape([1, X1BLOCK_SUB, 1])), [Y0BLOCK_SUB, X1BLOCK_SUB, R2BLOCK_SUB]), tmp1, 0)
            tmp4 = tl.sum(tmp3, 2).reshape(Y0BLOCK_SUB, X1BLOCK_SUB, 1)
            tl.store(out_ptr0 + (x1 + 1280*y0 ), tmp4, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/yj/cyj2fj6eq3bw7pgrnksfr3ivkykfyojisstgefqum3qcwdzoiqwy.py
# Topologically Sorted Source Nodes: [x_g_4, mean_4], Original ATen: [aten.linalg_vector_norm, aten.mean]
# Source node to ATen node mapping:
#   mean_4 => mean_4
#   x_g_4 => pow_10
# Graph fragment:
#   %sum_5 : Tensor "f32[8, 1280, 1, 1][1280, 1, 10240, 10240]npu:0" = PlaceHolder[target=sum_5]
#   %pow_10 : Tensor "f32[8, 1280, 1, 1][1280, 1, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_5, 0.5), kwargs = {})
#   %mean_4 : Tensor "f32[8, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_10, [1], True), kwargs = {})
#   return %buf91
# SchedulerNodes: [SchedulerNode(name='op91')]

triton_per_fused_linalg_vector_norm_mean_19 = async_compile.triton('triton_per_fused_linalg_vector_norm_mean_19', '''
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
    size_hints={'x0': 8, 'r1': 1280},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_linalg_vector_norm_mean_19', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 8), ('r1', 1280)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_linalg_vector_norm_mean_19(in_ptr0, out_ptr0, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 1280
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
        tmp0 = tl.load(in_ptr0 + (r1 + 1280*x0), x0_mask, other=0.0)
        tmp1 = tl.sqrt_rn(tmp0)
        tmp3 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp1, 0)
        tmp4 = tl.sum(tmp3, 1).reshape(X0BLOCK_SUB, 1)
        tl.store(out_ptr0 + (x0 ), tmp4, x0_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/uc/cucr7oqjv6u5trdmf26suty5gekjnty2j5pbcwzly6npvdq5dhpf.py
# Topologically Sorted Source Nodes: [view_8, view_9, x_g_4, mean_4, add_12, x_n_4, mul_4, addcmul_4, x_60], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul]
# Source node to ATen node mapping:
#   add_12 => add_12
#   addcmul_4 => addcmul_4
#   mean_4 => mean_4
#   mul_4 => mul_4
#   view_8 => view_8
#   view_9 => view_9
#   x_60 => add_13
#   x_g_4 => pow_10
#   x_n_4 => div_4
# Graph fragment:
#   %gelu_4 : Tensor "f32[8, 1280, 18, 18][414720, 324, 18, 1]npu:0" = PlaceHolder[target=gelu_4]
#   %arg59_1 : Tensor "f32[1280][1]npu:0" = PlaceHolder[target=arg59_1]
#   %arg60_1 : Tensor "f32[1280][1]npu:0" = PlaceHolder[target=arg60_1]
#   %sum_5 : Tensor "f32[8, 1280, 1, 1][1280, 1, 10240, 10240]npu:0" = PlaceHolder[target=sum_5]
#   %buf91 : Tensor "f32[8, 1, 1, 1][1, 8, 8, 8]npu:0" = PlaceHolder[target=buf91]
#   %view_8 : Tensor "f32[1, 1280, 1, 1][1280, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg59_1, [1, -1, 1, 1]), kwargs = {})
#   %view_9 : Tensor "f32[1, 1280, 1, 1][1280, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg60_1, [1, -1, 1, 1]), kwargs = {})
#   %pow_10 : Tensor "f32[8, 1280, 1, 1][1280, 1, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_5, 0.5), kwargs = {})
#   %mean_4 : Tensor "f32[8, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_10, [1], True), kwargs = {})
#   %add_12 : Tensor "f32[8, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_4, 1e-06), kwargs = {})
#   %div_4 : Tensor "f32[8, 1280, 1, 1][1280, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%pow_10, %add_12), kwargs = {})
#   %mul_4 : Tensor "f32[8, 1280, 18, 18][414720, 324, 18, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gelu_4, %div_4), kwargs = {})
#   %addcmul_4 : Tensor "f32[8, 1280, 18, 18][414720, 324, 18, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.addcmul.default](args = (%view_8, %view_9, %mul_4), kwargs = {})
#   %add_13 : Tensor "f32[8, 1280, 18, 18][414720, 324, 18, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%gelu_4, %addcmul_4), kwargs = {})
#   return %add_13
# SchedulerNodes: [SchedulerNode(name='op92')]

triton_poi_fused_add_addcmul_div_linalg_vector_20 = async_compile.triton('triton_poi_fused_add_addcmul_div_linalg_vector_20', '''
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
    size_hints={'z0': 8, 'y1': 1280, 'x2': 324},
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addcmul_div_linalg_vector_20', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [2], 'axis_names': ['z0', 'y1', 'x2'], 'axis_static_values': (('z0', 8), ('y1', 1280), ('x2', 324)), 'low_dims': {2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addcmul_div_linalg_vector_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, z0_numel, y1_numel, x2_numel, Z0BLOCK, Y1BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr):
    x2_numel = 324
    X2BLOCK_SUB: tl.constexpr = 324
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(1) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[None,:,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            x2 = base_x2[None,None,:]
            tmp0 = tl.load(in_out_ptr0 + (x2 + 324*y1 + 414720*z0), y1_mask & z0_mask)
            tmp1 = tl.load(in_ptr0 + (y1), y1_mask)
            tmp2 = tl.load(in_ptr1 + (y1), y1_mask)
            tmp3 = tl.load(in_ptr2 + (y1 + 1280*z0), y1_mask & z0_mask)
            tmp5 = tl.load(in_ptr3 + (z0), z0_mask)
            tmp4 = tl.sqrt_rn(tmp3)
            tmp6 = tl.full([1, 1, 1], 1280.0, tl.float32)
            tmp7 = (tmp5 / tmp6)
            tmp8 = tl.full([1, 1, 1], 1e-06, tl.float32)
            tmp9 = tmp7 + tmp8
            tmp10 = (tmp4 / tmp9)
            tmp11 = tmp0 * tmp10
            tmp12 = tmp2 * tmp11
            tmp13 = tl.full([1, 1, 1], 1.0, tl.float32)
            tmp14 = tmp13 * tmp12
            tmp15 = tmp1 + tmp14
            tmp16 = tmp0 + tmp15
            tl.store(in_out_ptr0 + (x2 + 324*y1 + 414720*z0), tmp16, y1_mask & z0_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/be/cbe3aj5h4k2qtcofq4h7zb6zhx67kjo4rfwxn5zqtogyjk3ls5qo.py
# Topologically Sorted Source Nodes: [x_63], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_63 => add_14
# Graph fragment:
#   %convolution_17 : Tensor "f32[8, 320, 18, 18][103680, 324, 18, 1]npu:0" = PlaceHolder[target=convolution_17]
#   %convolution_14 : Tensor "f32[8, 320, 18, 18][103680, 324, 18, 1]npu:0" = PlaceHolder[target=convolution_14]
#   %add_14 : Tensor "f32[8, 320, 18, 18][103680, 324, 18, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %convolution_14), kwargs = {})
#   return %add_14
# SchedulerNodes: [SchedulerNode(name='op95')]

triton_poi_fused_add_21 = async_compile.triton('triton_poi_fused_add_21', '''
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
    size_hints={'t0': 8, 'z1': 320, 'y2': 18, 'x3': 18},
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 't0_numel': 'i32', 'z1_numel': 'i32', 'y2_numel': 'i32', 'x3_numel': 'i32', 'T0BLOCK': 'i32', 'Z1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_21', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [2, 3], 'axis_names': ['t0', 'z1', 'y2', 'x3'], 'axis_static_values': (('t0', 8), ('z1', 320), ('y2', 18), ('x3', 18)), 'low_dims': {3}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('T0BLOCK', 'Z1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_21(in_out_ptr0, in_ptr0, t0_numel, z1_numel, y2_numel, x3_numel, T0BLOCK, Z1BLOCK, T0BLOCK_SUB : tl.constexpr, Z1BLOCK_SUB : tl.constexpr):
    y2_numel = 18
    Y2BLOCK_SUB: tl.constexpr = 18
    x3_numel = 18
    X3BLOCK_SUB: tl.constexpr = 18
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
            tmp0 = tl.load(in_out_ptr0 + (x3 + 18*y2 + 324*z1 + 103680*t0), t0_mask & z1_mask)
            tmp1 = tl.load(in_ptr0 + (x3 + 18*y2 + 324*z1 + 103680*t0), t0_mask & z1_mask)
            tmp2 = tmp0 + tmp1
            tl.store(in_out_ptr0 + (x3 + 18*y2 + 324*z1 + 103680*t0), tmp2, t0_mask & z1_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/35/c35qtabpy2wq35d47lnrcwe5aahnghvtvrnl56wryzcludfiyg6u.py
# Topologically Sorted Source Nodes: [x_140, x_141, x_142], Original ATen: [aten.add, aten.permute, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_140 => add_35
#   x_141 => permute_30
#   x_142 => native_layer_norm_15
# Graph fragment:
#   %convolution_38 : Tensor "f32[8, 320, 18, 18][103680, 324, 18, 1]npu:0" = PlaceHolder[target=convolution_38]
#   %add_32 : Tensor "f32[8, 320, 18, 18][103680, 324, 18, 1]npu:0" = PlaceHolder[target=add_32]
#   %buf207 : Tensor "f32[8, 18, 18, 1][324, 18, 1, 2592]npu:0" = PlaceHolder[target=buf207]
#   %buf208 : Tensor "f32[8, 18, 18, 1][324, 18, 1, 2592]npu:0" = PlaceHolder[target=buf208]
#   %add_35 : Tensor "f32[8, 320, 18, 18][103680, 324, 18, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_38, %add_32), kwargs = {})
#   %permute_30 : Tensor "f32[8, 18, 18, 320][103680, 18, 1, 324]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_35, [0, 2, 3, 1]), kwargs = {})
#   %native_layer_norm_15 : [num_users=1] = call_function[target=torch.ops.aten.native_layer_norm.default](args = (%permute_30, [320], %arg133_1, %arg134_1, 1e-06), kwargs = {})
#   return %buf207,%buf208,%buf209
# SchedulerNodes: [SchedulerNode(name='op207'), SchedulerNode(name='op208'), SchedulerNode(name='op209')]

triton_per_fused_add_native_layer_norm_permute_22 = async_compile.triton('triton_per_fused_add_native_layer_norm_permute_22', '''
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
    size_hints={'z0': 8, 'y1': 18, 'x2': 18, 'r3': 320},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'r3_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_permute_22', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [], 'axis_names': ['z0', 'y1', 'x2', 'r3'], 'axis_static_values': (('z0', 8), ('y1', 18), ('x2', 18), ('r3', 320)), 'low_dims': {2}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_permute_22(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, z0_numel, y1_numel, x2_numel, r3_numel, Z0BLOCK, Y1BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr):
    R3BLOCK_SUB: tl.constexpr = 320
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(1) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (x2_numel + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    base_r3= tl.arange(0, R3BLOCK_SUB)
    loops_r3 = (r3_numel + R3BLOCK_SUB - 1) // R3BLOCK_SUB
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[None,None,:,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            for loop_x2 in range(loops_x2):
                x2 = (loop_x2 * X2BLOCK_SUB) + base_x2[None,None,None,:]
                x2_mask = x2 < x2_numel
                r3 = base_r3[None,:,None,None]
                r3_mask = True
                tmp0 = tl.load(in_ptr0 + (x2 + 18*y1 + 324*r3 + 103680*z0), x2_mask & y1_mask & z0_mask, other=0.0)
                tmp1 = tl.load(in_ptr1 + (x2 + 18*y1 + 324*r3 + 103680*z0), x2_mask & y1_mask & z0_mask, other=0.0)
                tmp2 = tmp0 + tmp1
                tmp4 = tl.where(tl.broadcast_to((z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, 1, Y1BLOCK_SUB, 1]) & x2_mask.reshape([1, 1, 1, X2BLOCK_SUB])), [Z0BLOCK_SUB, R3BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB]), tmp2, 0)
                tmp5 = tl.sum(tmp4, 1).reshape(Z0BLOCK_SUB, 1, Y1BLOCK_SUB, X2BLOCK_SUB)
                tmp6 = tl.full([1, 1, 1, 1], 320.0, tl.float32)
                tmp7 = (tmp5 / tmp6)
                tmp8 = tmp2 - tmp7
                tmp9 = tmp8 * tmp8
                tmp11 = tl.where(tl.broadcast_to((z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, 1, Y1BLOCK_SUB, 1]) & x2_mask.reshape([1, 1, 1, X2BLOCK_SUB])), [Z0BLOCK_SUB, R3BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB]), tmp9, 0)
                tmp12 = tl.sum(tmp11, 1).reshape(Z0BLOCK_SUB, 1, Y1BLOCK_SUB, X2BLOCK_SUB)
                tl.store(in_out_ptr0 + (x2 + 18*y1 + 324*z0), tmp7, x2_mask & y1_mask & z0_mask)
                tl.store(out_ptr0 + (x2 + 18*y1 + 324*z0 ), tmp12, x2_mask & y1_mask & z0_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/6h/c6h3esk7jgortxx5ltp2ndry4w4fnpmhwybs42snmybw24c2cps6.py
# Topologically Sorted Source Nodes: [x_140, x_141, x_142], Original ATen: [aten.add, aten.permute, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_140 => add_35
#   x_141 => permute_30
#   x_142 => native_layer_norm_15
# Graph fragment:
#   %convolution_38 : Tensor "f32[8, 320, 18, 18][103680, 324, 18, 1]npu:0" = PlaceHolder[target=convolution_38]
#   %add_32 : Tensor "f32[8, 320, 18, 18][103680, 324, 18, 1]npu:0" = PlaceHolder[target=add_32]
#   %buf208 : Tensor "f32[8, 18, 18, 1][324, 18, 1, 2592]npu:0" = PlaceHolder[target=buf208]
#   %buf209 : Tensor "f32[8, 18, 18, 1][324, 18, 1, 2592]npu:0" = PlaceHolder[target=buf209]
#   %arg133_1 : Tensor "f32[320][1]npu:0" = PlaceHolder[target=arg133_1]
#   %arg134_1 : Tensor "f32[320][1]npu:0" = PlaceHolder[target=arg134_1]
#   %add_35 : Tensor "f32[8, 320, 18, 18][103680, 324, 18, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_38, %add_32), kwargs = {})
#   %permute_30 : Tensor "f32[8, 18, 18, 320][103680, 18, 1, 324]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_35, [0, 2, 3, 1]), kwargs = {})
#   %native_layer_norm_15 : [num_users=1] = call_function[target=torch.ops.aten.native_layer_norm.default](args = (%permute_30, [320], %arg133_1, %arg134_1, 1e-06), kwargs = {})
#   return %getitem_45
# SchedulerNodes: [SchedulerNode(name='op210')]

triton_poi_fused_add_native_layer_norm_permute_23 = async_compile.triton('triton_poi_fused_add_native_layer_norm_permute_23', '''
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
    size_hints={'y1': 8, 'y0': 320, 'x2': 324},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'y1_numel': 'i32', 'y0_numel': 'i32', 'x2_numel': 'i32', 'Y1BLOCK': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_permute_23', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [2], 'axis_names': ['y1', 'y0', 'x2'], 'axis_static_values': (('y1', 8), ('y0', 320), ('x2', 324)), 'low_dims': {1, 2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Y1BLOCK', 'Y0BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_permute_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, y1_numel, y0_numel, x2_numel, Y1BLOCK, Y0BLOCK, Y1BLOCK_SUB : tl.constexpr, Y0BLOCK_SUB : tl.constexpr):
    x2_numel = 324
    X2BLOCK_SUB: tl.constexpr = 324
    y1_offset = tl.program_id(0) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    y0_offset = tl.program_id(1) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    for loop_y1 in range(loops_y1):
        y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[:,None,None]
        y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
        for loop_y0 in range(loops_y0):
            y0_2 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[None,None,:]
            y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[None,:,None]
            y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
            y0_2_mask = y0_2 < min(Y0BLOCK+y0_offset, y0_numel)
            x2_1 = base_x2[None,:,None]
            x2 = base_x2[None,None,:]
            tmp0 = tl.load(in_ptr0 + (x2 + 324*y0 + 103680*y1), y0_mask & y1_mask)
            tmp1 = tl.load(in_ptr1 + (x2 + 324*y0 + 103680*y1), y0_mask & y1_mask)
            tmp3 = tl.load(in_ptr2 + (x2 + 324*y1), y1_mask)
            tmp5 = tl.load(in_ptr3 + (x2 + 324*y1), y1_mask)
            tmp12 = tl.load(in_ptr4 + (y0), y0_mask)
            tmp14 = tl.load(in_ptr5 + (y0), y0_mask)
            tmp2 = tmp0 + tmp1
            tmp4 = tmp2 - tmp3
            tmp6 = tl.full([1, 1, 1], 320.0, tl.float32)
            tmp7 = (tmp5 / tmp6)
            tmp8 = tl.full([1, 1, 1], 1e-06, tl.float32)
            tmp9 = tmp7 + tmp8
            tmp10 = libdevice.rsqrt(tmp9)
            tmp11 = tmp4 * tmp10
            tmp13 = tmp11 * tmp12
            tmp15 = tmp13 + tmp14
            tl.store(out_ptr0 + (y0_2 + 320*x2_1 + 103680*y1), tmp15.permute([0, 2, 1]), y0_2_mask & y1_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/li/clib5zqr2kdumf7yxnh6djout2hj6vw7qj5bdmvvghemt4pi4fl2.py
# Topologically Sorted Source Nodes: [x_145, x_146], Original ATen: [aten.permute, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_145 => permute_32
#   x_146 => native_layer_norm_16
# Graph fragment:
#   %convolution_40 : Tensor "f32[8, 640, 9, 9][51840, 81, 9, 1]npu:0" = PlaceHolder[target=convolution_40]
#   %buf215 : Tensor "f32[8, 9, 9, 1][81, 9, 1, 648]npu:0" = PlaceHolder[target=buf215]
#   %buf216 : Tensor "f32[8, 9, 9, 1][81, 9, 1, 648]npu:0" = PlaceHolder[target=buf216]
#   %permute_32 : Tensor "f32[8, 9, 9, 640][51840, 9, 1, 81]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convolution_40, [0, 2, 3, 1]), kwargs = {})
#   %native_layer_norm_16 : [num_users=1] = call_function[target=torch.ops.aten.native_layer_norm.default](args = (%permute_32, [640], %arg139_1, %arg140_1, 1e-06), kwargs = {})
#   return %buf215,%buf216,%buf217
# SchedulerNodes: [SchedulerNode(name='op215'), SchedulerNode(name='op216'), SchedulerNode(name='op217')]

triton_per_fused_native_layer_norm_permute_24 = async_compile.triton('triton_per_fused_native_layer_norm_permute_24', '''
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
    size_hints={'z0': 8, 'y1': 9, 'x2': 9, 'r3': 640},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'r3_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32', 'X2BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_permute_24', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1, 2], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [], 'axis_names': ['z0', 'y1', 'x2', 'r3'], 'axis_static_values': (('z0', 8), ('y1', 9), ('x2', 9), ('r3', 640)), 'low_dims': {2}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK', 'X2BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_native_layer_norm_permute_24(in_out_ptr0, in_ptr0, out_ptr0, z0_numel, y1_numel, x2_numel, r3_numel, Z0BLOCK, Y1BLOCK, X2BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr):
    R3BLOCK_SUB: tl.constexpr = 640
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(1) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    x2_offset = tl.program_id(2) * X2BLOCK
    base_x2= tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (X2BLOCK + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    base_r3= tl.arange(0, R3BLOCK_SUB)
    loops_r3 = (r3_numel + R3BLOCK_SUB - 1) // R3BLOCK_SUB
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[None,None,:,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            for loop_x2 in range(loops_x2):
                x2 = x2_offset + (loop_x2 * X2BLOCK_SUB) + base_x2[None,None,None,:]
                x2_mask = x2 < min(X2BLOCK+x2_offset, x2_numel)
                r3 = base_r3[None,:,None,None]
                r3_mask = True
                tmp0 = tl.load(in_ptr0 + (x2 + 9*y1 + 81*r3 + 51840*z0), x2_mask & y1_mask & z0_mask, other=0.0)
                tmp2 = tl.where(tl.broadcast_to((z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, 1, Y1BLOCK_SUB, 1]) & x2_mask.reshape([1, 1, 1, X2BLOCK_SUB])), [Z0BLOCK_SUB, R3BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB]), tmp0, 0)
                tmp3 = tl.sum(tmp2, 1).reshape(Z0BLOCK_SUB, 1, Y1BLOCK_SUB, X2BLOCK_SUB)
                tmp4 = tl.full([1, 1, 1, 1], 640.0, tl.float32)
                tmp5 = (tmp3 / tmp4)
                tmp6 = tmp0 - tmp5
                tmp7 = tmp6 * tmp6
                tmp9 = tl.where(tl.broadcast_to((z0_mask.reshape([Z0BLOCK_SUB, 1, 1, 1]) & y1_mask.reshape([1, 1, Y1BLOCK_SUB, 1]) & x2_mask.reshape([1, 1, 1, X2BLOCK_SUB])), [Z0BLOCK_SUB, R3BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB]), tmp7, 0)
                tmp10 = tl.sum(tmp9, 1).reshape(Z0BLOCK_SUB, 1, Y1BLOCK_SUB, X2BLOCK_SUB)
                tl.store(in_out_ptr0 + (x2 + 9*y1 + 81*z0), tmp5, x2_mask & y1_mask & z0_mask)
                tl.store(out_ptr0 + (x2 + 9*y1 + 81*z0 ), tmp10, x2_mask & y1_mask & z0_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/ag/cag3ddwkphpsuzlbr46p2pa56rybing2caaw5mf2h2fflachq5nk.py
# Topologically Sorted Source Nodes: [x_145, x_146], Original ATen: [aten.permute, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_145 => permute_32
#   x_146 => native_layer_norm_16
# Graph fragment:
#   %convolution_40 : Tensor "f32[8, 640, 9, 9][51840, 81, 9, 1]npu:0" = PlaceHolder[target=convolution_40]
#   %buf216 : Tensor "f32[8, 9, 9, 1][81, 9, 1, 648]npu:0" = PlaceHolder[target=buf216]
#   %buf217 : Tensor "f32[8, 9, 9, 1][81, 9, 1, 648]npu:0" = PlaceHolder[target=buf217]
#   %arg139_1 : Tensor "f32[640][1]npu:0" = PlaceHolder[target=arg139_1]
#   %arg140_1 : Tensor "f32[640][1]npu:0" = PlaceHolder[target=arg140_1]
#   %permute_32 : Tensor "f32[8, 9, 9, 640][51840, 9, 1, 81]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convolution_40, [0, 2, 3, 1]), kwargs = {})
#   %native_layer_norm_16 : [num_users=1] = call_function[target=torch.ops.aten.native_layer_norm.default](args = (%permute_32, [640], %arg139_1, %arg140_1, 1e-06), kwargs = {})
#   return %getitem_48
# SchedulerNodes: [SchedulerNode(name='op218')]

triton_poi_fused_native_layer_norm_permute_25 = async_compile.triton('triton_poi_fused_native_layer_norm_permute_25', '''
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
    size_hints={'y1': 8, 'y0': 81, 'x2': 640},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'y1_numel': 'i32', 'y0_numel': 'i32', 'x2_numel': 'i32', 'Y1BLOCK': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_permute_25', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [2], 'axis_names': ['y1', 'y0', 'x2'], 'axis_static_values': (('y1', 8), ('y0', 81), ('x2', 640)), 'low_dims': {1, 2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Y1BLOCK', 'Y0BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_permute_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, y1_numel, y0_numel, x2_numel, Y1BLOCK, Y0BLOCK, Y1BLOCK_SUB : tl.constexpr, Y0BLOCK_SUB : tl.constexpr):
    x2_numel = 640
    X2BLOCK_SUB: tl.constexpr = 640
    y1_offset = tl.program_id(0) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    y0_offset = tl.program_id(1) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    for loop_y1 in range(loops_y1):
        y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[:,None,None]
        y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
        for loop_y0 in range(loops_y0):
            y0_1 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[None,:,None]
            y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[None,None,:]
            y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
            y0_1_mask = y0_1 < min(Y0BLOCK+y0_offset, y0_numel)
            x2_2 = base_x2[None,None,:]
            x2 = base_x2[None,:,None]
            tmp0 = tl.load(in_ptr0 + (y0 + 81*x2 + 51840*y1), y0_mask & y1_mask)
            tmp1 = tl.load(in_ptr1 + (y0 + 81*y1), y0_mask & y1_mask)
            tmp3 = tl.load(in_ptr2 + (y0 + 81*y1), y0_mask & y1_mask)
            tmp10 = tl.load(in_ptr3 + (x2), None)
            tmp12 = tl.load(in_ptr4 + (x2), None)
            tmp2 = tmp0 - tmp1
            tmp4 = tl.full([1, 1, 1], 640.0, tl.float32)
            tmp5 = (tmp3 / tmp4)
            tmp6 = tl.full([1, 1, 1], 1e-06, tl.float32)
            tmp7 = tmp5 + tmp6
            tmp8 = libdevice.rsqrt(tmp7)
            tmp9 = tmp2 * tmp8
            tmp11 = tmp9 * tmp10
            tmp13 = tmp11 + tmp12
            tl.store(out_ptr0 + (x2_2 + 640*y0_1 + 51840*y1), tmp13.permute([0, 2, 1]), y0_1_mask & y1_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/a7/ca7sl67vdh2v3acv4d4brxfig52lwzyqnqmqpfgtitglnructdar.py
# Topologically Sorted Source Nodes: [x_g_12], Original ATen: [aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   x_g_12 => pow_25, sum_13
# Graph fragment:
#   %gelu_12 : Tensor "f32[8, 2560, 9, 9][207360, 81, 9, 1]npu:0" = PlaceHolder[target=gelu_12]
#   %pow_25 : Tensor "f32[8, 2560, 9, 9][207360, 81, 9, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%gelu_12, 2), kwargs = {})
#   %sum_13 : Tensor "f32[8, 2560, 1, 1][2560, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_25, [2, 3], True), kwargs = {})
#   return %sum_13
# SchedulerNodes: [SchedulerNode(name='op223')]

triton_per_fused_linalg_vector_norm_26 = async_compile.triton('triton_per_fused_linalg_vector_norm_26', '''
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
    size_hints={'y0': 8, 'x1': 2560, 'r2': 81},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'r2_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_linalg_vector_norm_26', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['y0', 'x1', 'r2'], 'axis_static_values': (('y0', 8), ('x1', 2560), ('r2', 81)), 'low_dims': {2}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_linalg_vector_norm_26(in_ptr0, out_ptr0, y0_numel, x1_numel, r2_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    R2BLOCK_SUB: tl.constexpr = 81
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
            tmp0 = tl.load(in_ptr0 + (r2 + 81*x1 + 207360*y0), x1_mask & y0_mask, other=0.0)
            tmp1 = tmp0 * tmp0
            tmp3 = tl.where(tl.broadcast_to((y0_mask.reshape([Y0BLOCK_SUB, 1, 1]) & x1_mask.reshape([1, X1BLOCK_SUB, 1])), [Y0BLOCK_SUB, X1BLOCK_SUB, R2BLOCK_SUB]), tmp1, 0)
            tmp4 = tl.sum(tmp3, 2).reshape(Y0BLOCK_SUB, X1BLOCK_SUB, 1)
            tl.store(out_ptr0 + (x1 + 2560*y0 ), tmp4, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/3x/c3xm4cv6vqynlrw4fz5kfvnuwfn4wypoctajbfgtrduktrmbi3my.py
# Topologically Sorted Source Nodes: [x_g_12, mean_12], Original ATen: [aten.linalg_vector_norm, aten.mean]
# Source node to ATen node mapping:
#   mean_12 => mean_12
#   x_g_12 => pow_26
# Graph fragment:
#   %sum_13 : Tensor "f32[8, 2560, 1, 1][2560, 1, 20480, 20480]npu:0" = PlaceHolder[target=sum_13]
#   %pow_26 : Tensor "f32[8, 2560, 1, 1][2560, 1, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_13, 0.5), kwargs = {})
#   %mean_12 : Tensor "f32[8, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_26, [1], True), kwargs = {})
#   return %buf224
# SchedulerNodes: [SchedulerNode(name='op224')]

triton_per_fused_linalg_vector_norm_mean_27 = async_compile.triton('triton_per_fused_linalg_vector_norm_mean_27', '''
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
    size_hints={'x0': 8, 'r1': 2560},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_linalg_vector_norm_mean_27', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 8), ('r1', 2560)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_linalg_vector_norm_mean_27(in_ptr0, out_ptr0, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 2560
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
        tmp0 = tl.load(in_ptr0 + (r1 + 2560*x0), x0_mask, other=0.0)
        tmp1 = tl.sqrt_rn(tmp0)
        tmp3 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp1, 0)
        tmp4 = tl.sum(tmp3, 1).reshape(X0BLOCK_SUB, 1)
        tl.store(out_ptr0 + (x0 ), tmp4, x0_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/no/cnoznl326ibt55afmuu3kzfmrfilrinvpohoelwphk7zonwhgope.py
# Topologically Sorted Source Nodes: [view_24, view_25, x_g_12, mean_12, add_36, x_n_12, mul_12, addcmul_12, x_151], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul]
# Source node to ATen node mapping:
#   add_36 => add_36
#   addcmul_12 => addcmul_12
#   mean_12 => mean_12
#   mul_12 => mul_12
#   view_24 => view_24
#   view_25 => view_25
#   x_151 => add_37
#   x_g_12 => pow_26
#   x_n_12 => div_12
# Graph fragment:
#   %gelu_12 : Tensor "f32[8, 2560, 9, 9][207360, 81, 9, 1]npu:0" = PlaceHolder[target=gelu_12]
#   %arg143_1 : Tensor "f32[2560][1]npu:0" = PlaceHolder[target=arg143_1]
#   %arg144_1 : Tensor "f32[2560][1]npu:0" = PlaceHolder[target=arg144_1]
#   %sum_13 : Tensor "f32[8, 2560, 1, 1][2560, 1, 20480, 20480]npu:0" = PlaceHolder[target=sum_13]
#   %buf224 : Tensor "f32[8, 1, 1, 1][1, 8, 8, 8]npu:0" = PlaceHolder[target=buf224]
#   %view_24 : Tensor "f32[1, 2560, 1, 1][2560, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg143_1, [1, -1, 1, 1]), kwargs = {})
#   %view_25 : Tensor "f32[1, 2560, 1, 1][2560, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg144_1, [1, -1, 1, 1]), kwargs = {})
#   %pow_26 : Tensor "f32[8, 2560, 1, 1][2560, 1, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_13, 0.5), kwargs = {})
#   %mean_12 : Tensor "f32[8, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_26, [1], True), kwargs = {})
#   %add_36 : Tensor "f32[8, 1, 1, 1][1, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_12, 1e-06), kwargs = {})
#   %div_12 : Tensor "f32[8, 2560, 1, 1][2560, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%pow_26, %add_36), kwargs = {})
#   %mul_12 : Tensor "f32[8, 2560, 9, 9][207360, 81, 9, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gelu_12, %div_12), kwargs = {})
#   %addcmul_12 : Tensor "f32[8, 2560, 9, 9][207360, 81, 9, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.addcmul.default](args = (%view_24, %view_25, %mul_12), kwargs = {})
#   %add_37 : Tensor "f32[8, 2560, 9, 9][207360, 81, 9, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%gelu_12, %addcmul_12), kwargs = {})
#   return %add_37
# SchedulerNodes: [SchedulerNode(name='op225')]

triton_poi_fused_add_addcmul_div_linalg_vector_28 = async_compile.triton('triton_poi_fused_add_addcmul_div_linalg_vector_28', '''
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
    size_hints={'z0': 8, 'y1': 2560, 'x2': 81},
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addcmul_div_linalg_vector_28', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [2], 'axis_names': ['z0', 'y1', 'x2'], 'axis_static_values': (('z0', 8), ('y1', 2560), ('x2', 81)), 'low_dims': {2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addcmul_div_linalg_vector_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, z0_numel, y1_numel, x2_numel, Z0BLOCK, Y1BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr):
    x2_numel = 81
    X2BLOCK_SUB: tl.constexpr = 81
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(1) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[None,:,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            x2 = base_x2[None,None,:]
            tmp0 = tl.load(in_out_ptr0 + (x2 + 81*y1 + 207360*z0), y1_mask & z0_mask)
            tmp1 = tl.load(in_ptr0 + (y1), y1_mask)
            tmp2 = tl.load(in_ptr1 + (y1), y1_mask)
            tmp3 = tl.load(in_ptr2 + (y1 + 2560*z0), y1_mask & z0_mask)
            tmp5 = tl.load(in_ptr3 + (z0), z0_mask)
            tmp4 = tl.sqrt_rn(tmp3)
            tmp6 = tl.full([1, 1, 1], 2560.0, tl.float32)
            tmp7 = (tmp5 / tmp6)
            tmp8 = tl.full([1, 1, 1], 1e-06, tl.float32)
            tmp9 = tmp7 + tmp8
            tmp10 = (tmp4 / tmp9)
            tmp11 = tmp0 * tmp10
            tmp12 = tmp2 * tmp11
            tmp13 = tl.full([1, 1, 1], 1.0, tl.float32)
            tmp14 = tmp13 * tmp12
            tmp15 = tmp1 + tmp14
            tmp16 = tmp0 + tmp15
            tl.store(in_out_ptr0 + (x2 + 81*y1 + 207360*z0), tmp16, y1_mask & z0_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/7d/c7dg3hu4dntdaropynx5neki225zlib4zioklul6ktl723ouch5v.py
# Topologically Sorted Source Nodes: [x_154], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_154 => add_38
# Graph fragment:
#   %convolution_42 : Tensor "f32[8, 640, 9, 9][51840, 81, 9, 1]npu:0" = PlaceHolder[target=convolution_42]
#   %convolution_39 : Tensor "f32[8, 640, 9, 9][51840, 81, 9, 1]npu:0" = PlaceHolder[target=convolution_39]
#   %add_38 : Tensor "f32[8, 640, 9, 9][51840, 81, 9, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_42, %convolution_39), kwargs = {})
#   return %add_38
# SchedulerNodes: [SchedulerNode(name='op228')]

triton_poi_fused_add_29 = async_compile.triton('triton_poi_fused_add_29', '''
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
    size_hints={'t0': 8, 'z1': 640, 'y2': 9, 'x3': 9},
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 't0_numel': 'i32', 'z1_numel': 'i32', 'y2_numel': 'i32', 'x3_numel': 'i32', 'T0BLOCK': 'i32', 'Z1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_29', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [2, 3], 'axis_names': ['t0', 'z1', 'y2', 'x3'], 'axis_static_values': (('t0', 8), ('z1', 640), ('y2', 9), ('x3', 9)), 'low_dims': {3}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('T0BLOCK', 'Z1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_29(in_out_ptr0, in_ptr0, t0_numel, z1_numel, y2_numel, x3_numel, T0BLOCK, Z1BLOCK, T0BLOCK_SUB : tl.constexpr, Z1BLOCK_SUB : tl.constexpr):
    y2_numel = 9
    Y2BLOCK_SUB: tl.constexpr = 9
    x3_numel = 9
    X3BLOCK_SUB: tl.constexpr = 9
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
            tmp0 = tl.load(in_out_ptr0 + (x3 + 9*y2 + 81*z1 + 51840*t0), t0_mask & z1_mask)
            tmp1 = tl.load(in_ptr0 + (x3 + 9*y2 + 81*z1 + 51840*t0), t0_mask & z1_mask)
            tmp2 = tmp0 + tmp1
            tl.store(in_out_ptr0 + (x3 + 9*y2 + 81*z1 + 51840*t0), tmp2, t0_mask & z1_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/wf/cwfi2mclso5o45loj6mykk6r3oibic2ldcl673zfmvuyut7jbu3o.py
# Topologically Sorted Source Nodes: [x_165, x_166], Original ATen: [aten.add, aten.mean]
# Source node to ATen node mapping:
#   x_165 => add_41
#   x_166 => mean_14
# Graph fragment:
#   %convolution_45 : Tensor "f32[8, 640, 9, 9][51840, 81, 9, 1]npu:0" = PlaceHolder[target=convolution_45]
#   %add_38 : Tensor "f32[8, 640, 9, 9][51840, 81, 9, 1]npu:0" = PlaceHolder[target=add_38]
#   %add_41 : Tensor "f32[8, 640, 9, 9][51840, 81, 9, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_45, %add_38), kwargs = {})
#   %mean_14 : Tensor "f32[8, 640, 1, 1][640, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_41, [-1, -2], True), kwargs = {})
#   return %buf244
# SchedulerNodes: [SchedulerNode(name='op244')]

triton_per_fused_add_mean_30 = async_compile.triton('triton_per_fused_add_mean_30', '''
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
    size_hints={'y0': 8, 'x1': 640, 'r2': 81},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'r2_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_30', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['y0', 'x1', 'r2'], 'axis_static_values': (('y0', 8), ('x1', 640), ('r2', 81)), 'low_dims': {2}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_add_mean_30(in_ptr0, in_ptr1, out_ptr0, y0_numel, x1_numel, r2_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    R2BLOCK_SUB: tl.constexpr = 81
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
            tmp0 = tl.load(in_ptr0 + (r2 + 81*x1 + 51840*y0), x1_mask & y0_mask, other=0.0)
            tmp1 = tl.load(in_ptr1 + (r2 + 81*x1 + 51840*y0), x1_mask & y0_mask, other=0.0)
            tmp2 = tmp0 + tmp1
            tmp4 = tl.where(tl.broadcast_to((y0_mask.reshape([Y0BLOCK_SUB, 1, 1]) & x1_mask.reshape([1, X1BLOCK_SUB, 1])), [Y0BLOCK_SUB, X1BLOCK_SUB, R2BLOCK_SUB]), tmp2, 0)
            tmp5 = tl.sum(tmp4, 2).reshape(Y0BLOCK_SUB, X1BLOCK_SUB, 1)
            tl.store(out_ptr0 + (x1 + 640*y0 ), tmp5, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/convnextv2_nano.fcmae_ft_in22k_in1k/v7/cv7srlmjpqyxauvmgubeqlkkpzrbgrhosyroiolfpkys6ojfaty3.py
# Topologically Sorted Source Nodes: [x_165, x_166, x_167, x_168], Original ATen: [aten.add, aten.mean, aten.permute, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_165 => add_41
#   x_166 => mean_14
#   x_167 => permute_36
#   x_168 => native_layer_norm_18
# Graph fragment:
#   %buf244 : Tensor "f32[8, 640, 1, 1][640, 1, 5120, 5120]npu:0" = PlaceHolder[target=buf244]
#   %buf245 : Tensor "f32[8, 1, 1, 1][1, 8, 8, 8]npu:0" = PlaceHolder[target=buf245]
#   %buf246 : Tensor "f32[8, 1, 1, 1][1, 8, 8, 8]npu:0" = PlaceHolder[target=buf246]
#   %buf247 : Tensor "f32[8, 1, 1, 1][1, 8, 8, 8]npu:0" = PlaceHolder[target=buf247]
#   %arg157_1 : Tensor "f32[640][1]npu:0" = PlaceHolder[target=arg157_1]
#   %arg158_1 : Tensor "f32[640][1]npu:0" = PlaceHolder[target=arg158_1]
#   %add_41 : Tensor "f32[8, 640, 9, 9][51840, 81, 9, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_45, %add_38), kwargs = {})
#   %mean_14 : Tensor "f32[8, 640, 1, 1][640, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_41, [-1, -2], True), kwargs = {})
#   %permute_36 : Tensor "f32[8, 1, 1, 640][640, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mean_14, [0, 2, 3, 1]), kwargs = {})
#   %native_layer_norm_18 : [num_users=1] = call_function[target=torch.ops.aten.native_layer_norm.default](args = (%permute_36, [640], %arg157_1, %arg158_1, 1e-06), kwargs = {})
#   return %buf245,%buf246,%buf247,%getitem_54
# SchedulerNodes: [SchedulerNode(name='op245'), SchedulerNode(name='op246'), SchedulerNode(name='op247'), SchedulerNode(name='op248')]

triton_per_fused_add_mean_native_layer_norm_pe_31 = async_compile.triton('triton_per_fused_add_mean_native_layer_norm_pe_31', '''
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
    size_hints={'x0': 8, 'r1': 640},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_native_layer_norm_pe_31', 'mutated_arg_names': ['in_out_ptr1'], 'backend_hash': '<redacted>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 8), ('r1', 640)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': True, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_add_mean_native_layer_norm_pe_31(in_out_ptr1, in_ptr0, in_ptr1, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 640
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
        tmp0 = tl.load(in_out_ptr1 + (r1 + 640*x0), x0_mask, other=0.0)
        tmp1 = tl.full([1, 1], 81.0, tl.float32)
        tmp2 = (tmp0 / tmp1)
        tmp4 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp2, 0)
        tmp5 = tl.sum(tmp4, 1).reshape(X0BLOCK_SUB, 1)
        tmp6 = tl.full([1, 1], 640.0, tl.float32)
        tmp7 = (tmp5 / tmp6)
        tmp8 = tmp2 - tmp7
        tmp9 = tmp8 * tmp8
        tmp11 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp9, 0)
        tmp12 = tl.sum(tmp11, 1).reshape(X0BLOCK_SUB, 1)
        tmp13 = (tmp12 / tmp6)
        tmp14 = tl.full([1, 1], 1e-06, tl.float32)
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp8 * tmp16
        tmp18 = tl.load(in_ptr0 + (r1), None)
        tmp19 = tmp17 * tmp18
        tmp20 = tl.load(in_ptr1 + (r1), None)
        tmp21 = tmp19 + tmp20
        tl.store(in_out_ptr1 + (r1 + 640*x0), tmp21, x0_mask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg2_1 = copy_if_misaligned(arg2_1)
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:1
            buf0 = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1)
            del arg0_1
            del arg1_1
            del arg2_1
            buf1 = buf0
            assert_size_stride(buf1, (8, 80, 72, 72), (414720, 5184, 72, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf1, 16, 'torch.ops.aten.convolution.default')
            del buf0
            buf2 = empty_strided((8, 72, 72, 1), (5184, 72, 1, 41472), device='npu', dtype=torch.float32)
            buf3 = buf2; del buf2  # reuse
            buf4 = empty_strided((8, 72, 72, 1), (5184, 72, 1, 41472), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_permute_0.run(buf3, buf1, buf4, 8, 72, 72, 80, stream=raw_stream0)
            buf5 = empty_strided((8, 72, 72, 80), (414720, 5760, 80, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_native_layer_norm_permute_1.run(buf1, buf3, buf4, arg3_1, arg4_1, buf5, 8, 5184, 80, stream=raw_stream0)
            del arg3_1
            del arg4_1
            del buf1
            del buf3
            del buf4
            # Topologically Sorted Source Nodes: [x, x_1, x_2, x_3], Original ATen: [aten.permute, aten.native_layer_norm, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:2
            buf6 = torch.ops.aten.convolution.default(reinterpret_tensor(buf5, (8, 80, 72, 72), (414720, 1, 5760, 80), 0), arg5_1, arg6_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 80)
            del arg5_1
            del arg6_1
            buf7 = buf6
            assert_size_stride(buf7, (8, 80, 72, 72), (414720, 5184, 72, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf7, 16, 'torch.ops.aten.convolution.default')
            del buf6
            buf8 = empty_strided((8, 72, 72, 1), (5184, 72, 1, 41472), device='npu', dtype=torch.float32)
            buf9 = buf8; del buf8  # reuse
            buf10 = empty_strided((8, 72, 72, 1), (5184, 72, 1, 41472), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_permute_0.run(buf9, buf7, buf10, 8, 72, 72, 80, stream=raw_stream0)
            buf11 = empty_strided((8, 72, 72, 80), (414720, 5760, 80, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_native_layer_norm_permute_1.run(buf7, buf9, buf10, arg7_1, arg8_1, buf11, 8, 5184, 80, stream=raw_stream0)
            del arg7_1
            del arg8_1
            del buf10
            del buf7
            del buf9
            # Topologically Sorted Source Nodes: [x_4, x_5, x_6, x_7], Original ATen: [aten.permute, aten.native_layer_norm, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:3
            buf12 = torch.ops.aten.convolution.default(reinterpret_tensor(buf11, (8, 80, 72, 72), (414720, 1, 5760, 80), 0), arg9_1, arg10_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg10_1
            del arg9_1
            del buf11
            buf13 = buf12
            assert_size_stride(buf13, (8, 320, 72, 72), (1658880, 5184, 72, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf13, 16, 'torch.ops.aten.convolution.default')
            del buf12
            # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.gelu]
            # [Provenance debug handles] torch.ops.aten.gelu.default:4
            buf14 = torch.ops.aten.gelu.default(buf13)
            del buf13
            buf15 = buf14
            assert_size_stride(buf15, (8, 320, 72, 72), (1658880, 5184, 72, 1), 'torch.ops.aten.gelu.default')
            assert_alignment(buf15, 16, 'torch.ops.aten.gelu.default')
            del buf14
            buf16 = empty_strided((8, 320, 1, 1), (320, 1, 2560, 2560), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g], Original ATen: [aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_linalg_vector_norm_2.run(buf15, buf16, 8, 320, 5184, stream=raw_stream0)
            buf17 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g, mean], Original ATen: [aten.linalg_vector_norm, aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_mean_3.run(buf16, buf17, 8, 320, stream=raw_stream0)
            buf18 = buf15; del buf15  # reuse
            # Topologically Sorted Source Nodes: [view, view_1, x_g, mean, add, x_n, mul, addcmul, x_10], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addcmul_div_linalg_vector_4.run(buf18, arg11_1, arg12_1, buf16, buf17, 8, 320, 5184, stream=raw_stream0)
            del arg11_1
            del arg12_1
            del buf16
            del buf17
            # Topologically Sorted Source Nodes: [view, view_1, x_g, mean, add, x_n, mul, addcmul, x_10, x_11], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:5
            buf19 = torch.ops.aten.convolution.default(buf18, arg13_1, arg14_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg13_1
            del arg14_1
            del buf18
            buf20 = buf19
            assert_size_stride(buf20, (8, 80, 72, 72), (414720, 5184, 72, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf20, 16, 'torch.ops.aten.convolution.default')
            del buf19
            buf21 = buf20; del buf20  # reuse
            # Topologically Sorted Source Nodes: [x, x_1, x_2, x_13], Original ATen: [aten.permute, aten.native_layer_norm, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_native_layer_norm_permute_5.run(buf21, buf5, 8, 80, 5184, stream=raw_stream0)
            del buf5
            # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:6
            buf22 = torch.ops.aten.convolution.default(buf21, arg15_1, arg16_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 80)
            del arg15_1
            del arg16_1
            buf23 = buf22
            assert_size_stride(buf23, (8, 80, 72, 72), (414720, 5184, 72, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf23, 16, 'torch.ops.aten.convolution.default')
            del buf22
            buf24 = empty_strided((8, 72, 72, 1), (5184, 72, 1, 41472), device='npu', dtype=torch.float32)
            buf25 = buf24; del buf24  # reuse
            buf26 = empty_strided((8, 72, 72, 1), (5184, 72, 1, 41472), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_15, x_16], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_permute_0.run(buf25, buf23, buf26, 8, 72, 72, 80, stream=raw_stream0)
            buf27 = empty_strided((8, 72, 72, 80), (414720, 5760, 80, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_15, x_16], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_native_layer_norm_permute_1.run(buf23, buf25, buf26, arg17_1, arg18_1, buf27, 8, 5184, 80, stream=raw_stream0)
            del arg17_1
            del arg18_1
            del buf23
            del buf25
            del buf26
            # Topologically Sorted Source Nodes: [x_15, x_16, x_17, x_18], Original ATen: [aten.permute, aten.native_layer_norm, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:7
            buf28 = torch.ops.aten.convolution.default(reinterpret_tensor(buf27, (8, 80, 72, 72), (414720, 1, 5760, 80), 0), arg19_1, arg20_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg19_1
            del arg20_1
            del buf27
            buf29 = buf28
            assert_size_stride(buf29, (8, 320, 72, 72), (1658880, 5184, 72, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf29, 16, 'torch.ops.aten.convolution.default')
            del buf28
            # Topologically Sorted Source Nodes: [x_19], Original ATen: [aten.gelu]
            # [Provenance debug handles] torch.ops.aten.gelu.default:8
            buf30 = torch.ops.aten.gelu.default(buf29)
            del buf29
            buf31 = buf30
            assert_size_stride(buf31, (8, 320, 72, 72), (1658880, 5184, 72, 1), 'torch.ops.aten.gelu.default')
            assert_alignment(buf31, 16, 'torch.ops.aten.gelu.default')
            del buf30
            buf32 = empty_strided((8, 320, 1, 1), (320, 1, 2560, 2560), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_1], Original ATen: [aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_linalg_vector_norm_2.run(buf31, buf32, 8, 320, 5184, stream=raw_stream0)
            buf33 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_1, mean_1], Original ATen: [aten.linalg_vector_norm, aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_mean_3.run(buf32, buf33, 8, 320, stream=raw_stream0)
            buf34 = buf31; del buf31  # reuse
            # Topologically Sorted Source Nodes: [view_2, view_3, x_g_1, mean_1, add_3, x_n_1, mul_1, addcmul_1, x_21], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addcmul_div_linalg_vector_4.run(buf34, arg21_1, arg22_1, buf32, buf33, 8, 320, 5184, stream=raw_stream0)
            del arg21_1
            del arg22_1
            del buf32
            del buf33
            # Topologically Sorted Source Nodes: [view_2, view_3, x_g_1, mean_1, add_3, x_n_1, mul_1, addcmul_1, x_21, x_22], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:9
            buf35 = torch.ops.aten.convolution.default(buf34, arg23_1, arg24_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg23_1
            del arg24_1
            del buf34
            buf36 = buf35
            assert_size_stride(buf36, (8, 80, 72, 72), (414720, 5184, 72, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf36, 16, 'torch.ops.aten.convolution.default')
            del buf35
            buf37 = empty_strided((8, 72, 72, 1), (5184, 72, 1, 41472), device='npu', dtype=torch.float32)
            buf38 = buf37; del buf37  # reuse
            buf39 = empty_strided((8, 72, 72, 1), (5184, 72, 1, 41472), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_24, x_25, x_26], Original ATen: [aten.add, aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_add_native_layer_norm_permute_6.run(buf38, buf36, buf21, buf39, 8, 72, 72, 80, stream=raw_stream0)
            buf40 = empty_strided((8, 72, 72, 80), (414720, 5760, 80, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_24, x_25, x_26], Original ATen: [aten.add, aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_native_layer_norm_permute_7.run(buf36, buf21, buf38, buf39, arg25_1, arg26_1, buf40, 8, 80, 5184, stream=raw_stream0)
            del arg25_1
            del arg26_1
            del buf21
            del buf36
            del buf38
            del buf39
            # Topologically Sorted Source Nodes: [x_24, x_25, x_26, x_27, input_2], Original ATen: [aten.add, aten.permute, aten.native_layer_norm, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:10
            buf41 = torch.ops.aten.convolution.default(reinterpret_tensor(buf40, (8, 80, 72, 72), (414720, 1, 5760, 80), 0), arg27_1, arg28_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
            del arg27_1
            del arg28_1
            del buf40
            buf42 = buf41
            assert_size_stride(buf42, (8, 160, 36, 36), (207360, 1296, 36, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf42, 16, 'torch.ops.aten.convolution.default')
            del buf41
            # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:11
            buf43 = torch.ops.aten.convolution.default(buf42, arg29_1, arg30_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 160)
            del arg29_1
            del arg30_1
            buf44 = buf43
            assert_size_stride(buf44, (8, 160, 36, 36), (207360, 1296, 36, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf44, 16, 'torch.ops.aten.convolution.default')
            del buf43
            buf45 = empty_strided((8, 36, 36, 1), (1296, 36, 1, 10368), device='npu', dtype=torch.float32)
            buf46 = buf45; del buf45  # reuse
            buf47 = empty_strided((8, 36, 36, 1), (1296, 36, 1, 10368), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_29, x_30], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_permute_8.run(buf46, buf44, buf47, 8, 36, 36, 160, stream=raw_stream0)
            buf48 = empty_strided((8, 36, 36, 160), (207360, 5760, 160, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_29, x_30], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_native_layer_norm_permute_9.run(buf44, buf46, buf47, arg31_1, arg32_1, buf48, 8, 1296, 160, stream=raw_stream0)
            del arg31_1
            del arg32_1
            del buf44
            del buf46
            del buf47
            # Topologically Sorted Source Nodes: [x_29, x_30, x_31, x_32], Original ATen: [aten.permute, aten.native_layer_norm, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:12
            buf49 = torch.ops.aten.convolution.default(reinterpret_tensor(buf48, (8, 160, 36, 36), (207360, 1, 5760, 160), 0), arg33_1, arg34_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg33_1
            del arg34_1
            del buf48
            buf50 = buf49
            assert_size_stride(buf50, (8, 640, 36, 36), (829440, 1296, 36, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf50, 16, 'torch.ops.aten.convolution.default')
            del buf49
            # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.gelu]
            # [Provenance debug handles] torch.ops.aten.gelu.default:13
            buf51 = torch.ops.aten.gelu.default(buf50)
            del buf50
            buf52 = buf51
            assert_size_stride(buf52, (8, 640, 36, 36), (829440, 1296, 36, 1), 'torch.ops.aten.gelu.default')
            assert_alignment(buf52, 16, 'torch.ops.aten.gelu.default')
            del buf51
            buf53 = empty_strided((8, 640, 1, 1), (640, 1, 5120, 5120), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_2], Original ATen: [aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_10.run(buf52, buf53, 8, 640, 1296, stream=raw_stream0)
            buf54 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_2, mean_2], Original ATen: [aten.linalg_vector_norm, aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_mean_11.run(buf53, buf54, 8, 640, stream=raw_stream0)
            buf55 = buf52; del buf52  # reuse
            # Topologically Sorted Source Nodes: [view_4, view_5, x_g_2, mean_2, add_6, x_n_2, mul_2, addcmul_2, x_35], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addcmul_div_linalg_vector_12.run(buf55, arg35_1, arg36_1, buf53, buf54, 8, 640, 1296, stream=raw_stream0)
            del arg35_1
            del arg36_1
            del buf53
            del buf54
            # Topologically Sorted Source Nodes: [view_4, view_5, x_g_2, mean_2, add_6, x_n_2, mul_2, addcmul_2, x_35, x_36], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:14
            buf56 = torch.ops.aten.convolution.default(buf55, arg37_1, arg38_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg37_1
            del arg38_1
            del buf55
            buf57 = buf56
            assert_size_stride(buf57, (8, 160, 36, 36), (207360, 1296, 36, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf57, 16, 'torch.ops.aten.convolution.default')
            del buf56
            buf58 = buf57; del buf57  # reuse
            # Topologically Sorted Source Nodes: [x_38], Original ATen: [aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_13.run(buf58, buf42, 8, 160, 36, 36, stream=raw_stream0)
            del buf42
            # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:15
            buf59 = torch.ops.aten.convolution.default(buf58, arg39_1, arg40_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 160)
            del arg39_1
            del arg40_1
            buf60 = buf59
            assert_size_stride(buf60, (8, 160, 36, 36), (207360, 1296, 36, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf60, 16, 'torch.ops.aten.convolution.default')
            del buf59
            buf61 = empty_strided((8, 36, 36, 1), (1296, 36, 1, 10368), device='npu', dtype=torch.float32)
            buf62 = buf61; del buf61  # reuse
            buf63 = empty_strided((8, 36, 36, 1), (1296, 36, 1, 10368), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_40, x_41], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_permute_8.run(buf62, buf60, buf63, 8, 36, 36, 160, stream=raw_stream0)
            buf64 = empty_strided((8, 36, 36, 160), (207360, 5760, 160, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_40, x_41], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_native_layer_norm_permute_9.run(buf60, buf62, buf63, arg41_1, arg42_1, buf64, 8, 1296, 160, stream=raw_stream0)
            del arg41_1
            del arg42_1
            del buf60
            del buf62
            del buf63
            # Topologically Sorted Source Nodes: [x_40, x_41, x_42, x_43], Original ATen: [aten.permute, aten.native_layer_norm, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:16
            buf65 = torch.ops.aten.convolution.default(reinterpret_tensor(buf64, (8, 160, 36, 36), (207360, 1, 5760, 160), 0), arg43_1, arg44_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg43_1
            del arg44_1
            del buf64
            buf66 = buf65
            assert_size_stride(buf66, (8, 640, 36, 36), (829440, 1296, 36, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf66, 16, 'torch.ops.aten.convolution.default')
            del buf65
            # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten.gelu]
            # [Provenance debug handles] torch.ops.aten.gelu.default:17
            buf67 = torch.ops.aten.gelu.default(buf66)
            del buf66
            buf68 = buf67
            assert_size_stride(buf68, (8, 640, 36, 36), (829440, 1296, 36, 1), 'torch.ops.aten.gelu.default')
            assert_alignment(buf68, 16, 'torch.ops.aten.gelu.default')
            del buf67
            buf69 = empty_strided((8, 640, 1, 1), (640, 1, 5120, 5120), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_3], Original ATen: [aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_10.run(buf68, buf69, 8, 640, 1296, stream=raw_stream0)
            buf70 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_3, mean_3], Original ATen: [aten.linalg_vector_norm, aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_mean_11.run(buf69, buf70, 8, 640, stream=raw_stream0)
            buf71 = buf68; del buf68  # reuse
            # Topologically Sorted Source Nodes: [view_6, view_7, x_g_3, mean_3, add_9, x_n_3, mul_3, addcmul_3, x_46], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addcmul_div_linalg_vector_12.run(buf71, arg45_1, arg46_1, buf69, buf70, 8, 640, 1296, stream=raw_stream0)
            del arg45_1
            del arg46_1
            del buf69
            del buf70
            # Topologically Sorted Source Nodes: [view_6, view_7, x_g_3, mean_3, add_9, x_n_3, mul_3, addcmul_3, x_46, x_47], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:18
            buf72 = torch.ops.aten.convolution.default(buf71, arg47_1, arg48_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg47_1
            del arg48_1
            del buf71
            buf73 = buf72
            assert_size_stride(buf73, (8, 160, 36, 36), (207360, 1296, 36, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf73, 16, 'torch.ops.aten.convolution.default')
            del buf72
            buf74 = empty_strided((8, 36, 36, 1), (1296, 36, 1, 10368), device='npu', dtype=torch.float32)
            buf75 = buf74; del buf74  # reuse
            buf76 = empty_strided((8, 36, 36, 1), (1296, 36, 1, 10368), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_49, x_50, x_51], Original ATen: [aten.add, aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_add_native_layer_norm_permute_14.run(buf75, buf73, buf58, buf76, 8, 36, 36, 160, stream=raw_stream0)
            buf77 = empty_strided((8, 36, 36, 160), (207360, 5760, 160, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_49, x_50, x_51], Original ATen: [aten.add, aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_native_layer_norm_permute_15.run(buf73, buf58, buf75, buf76, arg49_1, arg50_1, buf77, 8, 160, 1296, stream=raw_stream0)
            del arg49_1
            del arg50_1
            del buf58
            del buf73
            del buf75
            del buf76
            # Topologically Sorted Source Nodes: [x_49, x_50, x_51, x_52, input_3], Original ATen: [aten.add, aten.permute, aten.native_layer_norm, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:19
            buf78 = torch.ops.aten.convolution.default(reinterpret_tensor(buf77, (8, 160, 36, 36), (207360, 1, 5760, 160), 0), arg51_1, arg52_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
            del arg51_1
            del arg52_1
            del buf77
            buf79 = buf78
            assert_size_stride(buf79, (8, 320, 18, 18), (103680, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf79, 16, 'torch.ops.aten.convolution.default')
            del buf78
            # Topologically Sorted Source Nodes: [x_53], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:20
            buf80 = torch.ops.aten.convolution.default(buf79, arg53_1, arg54_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320)
            del arg53_1
            del arg54_1
            buf81 = buf80
            assert_size_stride(buf81, (8, 320, 18, 18), (103680, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf81, 16, 'torch.ops.aten.convolution.default')
            del buf80
            buf82 = empty_strided((8, 18, 18, 1), (324, 18, 1, 2592), device='npu', dtype=torch.float32)
            buf83 = buf82; del buf82  # reuse
            buf84 = empty_strided((8, 18, 18, 1), (324, 18, 1, 2592), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_54, x_55], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_permute_16.run(buf83, buf81, buf84, 8, 18, 18, 320, stream=raw_stream0)
            buf85 = empty_strided((8, 18, 18, 320), (103680, 5760, 320, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_54, x_55], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_native_layer_norm_permute_17.run(buf81, buf83, buf84, arg55_1, arg56_1, buf85, 8, 324, 320, stream=raw_stream0)
            del arg55_1
            del arg56_1
            del buf81
            del buf83
            del buf84
            # Topologically Sorted Source Nodes: [x_54, x_55, x_56, x_57], Original ATen: [aten.permute, aten.native_layer_norm, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:21
            buf86 = torch.ops.aten.convolution.default(reinterpret_tensor(buf85, (8, 320, 18, 18), (103680, 1, 5760, 320), 0), arg57_1, arg58_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg57_1
            del arg58_1
            del buf85
            buf87 = buf86
            assert_size_stride(buf87, (8, 1280, 18, 18), (414720, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf87, 16, 'torch.ops.aten.convolution.default')
            del buf86
            # Topologically Sorted Source Nodes: [x_58], Original ATen: [aten.gelu]
            # [Provenance debug handles] torch.ops.aten.gelu.default:22
            buf88 = torch.ops.aten.gelu.default(buf87)
            del buf87
            buf89 = buf88
            assert_size_stride(buf89, (8, 1280, 18, 18), (414720, 324, 18, 1), 'torch.ops.aten.gelu.default')
            assert_alignment(buf89, 16, 'torch.ops.aten.gelu.default')
            del buf88
            buf90 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_4], Original ATen: [aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_18.run(buf89, buf90, 8, 1280, 324, stream=raw_stream0)
            buf91 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_4, mean_4], Original ATen: [aten.linalg_vector_norm, aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_mean_19.run(buf90, buf91, 8, 1280, stream=raw_stream0)
            buf92 = buf89; del buf89  # reuse
            # Topologically Sorted Source Nodes: [view_8, view_9, x_g_4, mean_4, add_12, x_n_4, mul_4, addcmul_4, x_60], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addcmul_div_linalg_vector_20.run(buf92, arg59_1, arg60_1, buf90, buf91, 8, 1280, 324, stream=raw_stream0)
            del arg59_1
            del arg60_1
            del buf90
            del buf91
            # Topologically Sorted Source Nodes: [view_8, view_9, x_g_4, mean_4, add_12, x_n_4, mul_4, addcmul_4, x_60, x_61], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:23
            buf93 = torch.ops.aten.convolution.default(buf92, arg61_1, arg62_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg61_1
            del arg62_1
            del buf92
            buf94 = buf93
            assert_size_stride(buf94, (8, 320, 18, 18), (103680, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf94, 16, 'torch.ops.aten.convolution.default')
            del buf93
            buf95 = buf94; del buf94  # reuse
            # Topologically Sorted Source Nodes: [x_63], Original ATen: [aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_21.run(buf95, buf79, 8, 320, 18, 18, stream=raw_stream0)
            del buf79
            # Topologically Sorted Source Nodes: [x_64], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:24
            buf96 = torch.ops.aten.convolution.default(buf95, arg63_1, arg64_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320)
            del arg63_1
            del arg64_1
            buf97 = buf96
            assert_size_stride(buf97, (8, 320, 18, 18), (103680, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf97, 16, 'torch.ops.aten.convolution.default')
            del buf96
            buf98 = empty_strided((8, 18, 18, 1), (324, 18, 1, 2592), device='npu', dtype=torch.float32)
            buf99 = buf98; del buf98  # reuse
            buf100 = empty_strided((8, 18, 18, 1), (324, 18, 1, 2592), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_65, x_66], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_permute_16.run(buf99, buf97, buf100, 8, 18, 18, 320, stream=raw_stream0)
            buf101 = empty_strided((8, 18, 18, 320), (103680, 5760, 320, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_65, x_66], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_native_layer_norm_permute_17.run(buf97, buf99, buf100, arg65_1, arg66_1, buf101, 8, 324, 320, stream=raw_stream0)
            del arg65_1
            del arg66_1
            del buf100
            del buf97
            del buf99
            # Topologically Sorted Source Nodes: [x_65, x_66, x_67, x_68], Original ATen: [aten.permute, aten.native_layer_norm, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:25
            buf102 = torch.ops.aten.convolution.default(reinterpret_tensor(buf101, (8, 320, 18, 18), (103680, 1, 5760, 320), 0), arg67_1, arg68_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg67_1
            del arg68_1
            del buf101
            buf103 = buf102
            assert_size_stride(buf103, (8, 1280, 18, 18), (414720, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf103, 16, 'torch.ops.aten.convolution.default')
            del buf102
            # Topologically Sorted Source Nodes: [x_69], Original ATen: [aten.gelu]
            # [Provenance debug handles] torch.ops.aten.gelu.default:26
            buf104 = torch.ops.aten.gelu.default(buf103)
            del buf103
            buf105 = buf104
            assert_size_stride(buf105, (8, 1280, 18, 18), (414720, 324, 18, 1), 'torch.ops.aten.gelu.default')
            assert_alignment(buf105, 16, 'torch.ops.aten.gelu.default')
            del buf104
            buf106 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_5], Original ATen: [aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_18.run(buf105, buf106, 8, 1280, 324, stream=raw_stream0)
            buf107 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_5, mean_5], Original ATen: [aten.linalg_vector_norm, aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_mean_19.run(buf106, buf107, 8, 1280, stream=raw_stream0)
            buf108 = buf105; del buf105  # reuse
            # Topologically Sorted Source Nodes: [view_10, view_11, x_g_5, mean_5, add_15, x_n_5, mul_5, addcmul_5, x_71], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addcmul_div_linalg_vector_20.run(buf108, arg69_1, arg70_1, buf106, buf107, 8, 1280, 324, stream=raw_stream0)
            del arg69_1
            del arg70_1
            del buf106
            del buf107
            # Topologically Sorted Source Nodes: [view_10, view_11, x_g_5, mean_5, add_15, x_n_5, mul_5, addcmul_5, x_71, x_72], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:27
            buf109 = torch.ops.aten.convolution.default(buf108, arg71_1, arg72_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg71_1
            del arg72_1
            del buf108
            buf110 = buf109
            assert_size_stride(buf110, (8, 320, 18, 18), (103680, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf110, 16, 'torch.ops.aten.convolution.default')
            del buf109
            buf111 = buf110; del buf110  # reuse
            # Topologically Sorted Source Nodes: [x_74], Original ATen: [aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_21.run(buf111, buf95, 8, 320, 18, 18, stream=raw_stream0)
            del buf95
            # Topologically Sorted Source Nodes: [x_75], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:28
            buf112 = torch.ops.aten.convolution.default(buf111, arg73_1, arg74_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320)
            del arg73_1
            del arg74_1
            buf113 = buf112
            assert_size_stride(buf113, (8, 320, 18, 18), (103680, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf113, 16, 'torch.ops.aten.convolution.default')
            del buf112
            buf114 = empty_strided((8, 18, 18, 1), (324, 18, 1, 2592), device='npu', dtype=torch.float32)
            buf115 = buf114; del buf114  # reuse
            buf116 = empty_strided((8, 18, 18, 1), (324, 18, 1, 2592), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_76, x_77], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_permute_16.run(buf115, buf113, buf116, 8, 18, 18, 320, stream=raw_stream0)
            buf117 = empty_strided((8, 18, 18, 320), (103680, 5760, 320, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_76, x_77], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_native_layer_norm_permute_17.run(buf113, buf115, buf116, arg75_1, arg76_1, buf117, 8, 324, 320, stream=raw_stream0)
            del arg75_1
            del arg76_1
            del buf113
            del buf115
            del buf116
            # Topologically Sorted Source Nodes: [x_76, x_77, x_78, x_79], Original ATen: [aten.permute, aten.native_layer_norm, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:29
            buf118 = torch.ops.aten.convolution.default(reinterpret_tensor(buf117, (8, 320, 18, 18), (103680, 1, 5760, 320), 0), arg77_1, arg78_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg77_1
            del arg78_1
            del buf117
            buf119 = buf118
            assert_size_stride(buf119, (8, 1280, 18, 18), (414720, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf119, 16, 'torch.ops.aten.convolution.default')
            del buf118
            # Topologically Sorted Source Nodes: [x_80], Original ATen: [aten.gelu]
            # [Provenance debug handles] torch.ops.aten.gelu.default:30
            buf120 = torch.ops.aten.gelu.default(buf119)
            del buf119
            buf121 = buf120
            assert_size_stride(buf121, (8, 1280, 18, 18), (414720, 324, 18, 1), 'torch.ops.aten.gelu.default')
            assert_alignment(buf121, 16, 'torch.ops.aten.gelu.default')
            del buf120
            buf122 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_6], Original ATen: [aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_18.run(buf121, buf122, 8, 1280, 324, stream=raw_stream0)
            buf123 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_6, mean_6], Original ATen: [aten.linalg_vector_norm, aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_mean_19.run(buf122, buf123, 8, 1280, stream=raw_stream0)
            buf124 = buf121; del buf121  # reuse
            # Topologically Sorted Source Nodes: [view_12, view_13, x_g_6, mean_6, add_18, x_n_6, mul_6, addcmul_6, x_82], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addcmul_div_linalg_vector_20.run(buf124, arg79_1, arg80_1, buf122, buf123, 8, 1280, 324, stream=raw_stream0)
            del arg79_1
            del arg80_1
            del buf122
            del buf123
            # Topologically Sorted Source Nodes: [view_12, view_13, x_g_6, mean_6, add_18, x_n_6, mul_6, addcmul_6, x_82, x_83], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:31
            buf125 = torch.ops.aten.convolution.default(buf124, arg81_1, arg82_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg81_1
            del arg82_1
            del buf124
            buf126 = buf125
            assert_size_stride(buf126, (8, 320, 18, 18), (103680, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf126, 16, 'torch.ops.aten.convolution.default')
            del buf125
            buf127 = buf126; del buf126  # reuse
            # Topologically Sorted Source Nodes: [x_85], Original ATen: [aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_21.run(buf127, buf111, 8, 320, 18, 18, stream=raw_stream0)
            del buf111
            # Topologically Sorted Source Nodes: [x_86], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:32
            buf128 = torch.ops.aten.convolution.default(buf127, arg83_1, arg84_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320)
            del arg83_1
            del arg84_1
            buf129 = buf128
            assert_size_stride(buf129, (8, 320, 18, 18), (103680, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf129, 16, 'torch.ops.aten.convolution.default')
            del buf128
            buf130 = empty_strided((8, 18, 18, 1), (324, 18, 1, 2592), device='npu', dtype=torch.float32)
            buf131 = buf130; del buf130  # reuse
            buf132 = empty_strided((8, 18, 18, 1), (324, 18, 1, 2592), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_87, x_88], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_permute_16.run(buf131, buf129, buf132, 8, 18, 18, 320, stream=raw_stream0)
            buf133 = empty_strided((8, 18, 18, 320), (103680, 5760, 320, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_87, x_88], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_native_layer_norm_permute_17.run(buf129, buf131, buf132, arg85_1, arg86_1, buf133, 8, 324, 320, stream=raw_stream0)
            del arg85_1
            del arg86_1
            del buf129
            del buf131
            del buf132
            # Topologically Sorted Source Nodes: [x_87, x_88, x_89, x_90], Original ATen: [aten.permute, aten.native_layer_norm, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:33
            buf134 = torch.ops.aten.convolution.default(reinterpret_tensor(buf133, (8, 320, 18, 18), (103680, 1, 5760, 320), 0), arg87_1, arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg87_1
            del arg88_1
            del buf133
            buf135 = buf134
            assert_size_stride(buf135, (8, 1280, 18, 18), (414720, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf135, 16, 'torch.ops.aten.convolution.default')
            del buf134
            # Topologically Sorted Source Nodes: [x_91], Original ATen: [aten.gelu]
            # [Provenance debug handles] torch.ops.aten.gelu.default:34
            buf136 = torch.ops.aten.gelu.default(buf135)
            del buf135
            buf137 = buf136
            assert_size_stride(buf137, (8, 1280, 18, 18), (414720, 324, 18, 1), 'torch.ops.aten.gelu.default')
            assert_alignment(buf137, 16, 'torch.ops.aten.gelu.default')
            del buf136
            buf138 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_7], Original ATen: [aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_18.run(buf137, buf138, 8, 1280, 324, stream=raw_stream0)
            buf139 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_7, mean_7], Original ATen: [aten.linalg_vector_norm, aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_mean_19.run(buf138, buf139, 8, 1280, stream=raw_stream0)
            buf140 = buf137; del buf137  # reuse
            # Topologically Sorted Source Nodes: [view_14, view_15, x_g_7, mean_7, add_21, x_n_7, mul_7, addcmul_7, x_93], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addcmul_div_linalg_vector_20.run(buf140, arg89_1, arg90_1, buf138, buf139, 8, 1280, 324, stream=raw_stream0)
            del arg89_1
            del arg90_1
            del buf138
            del buf139
            # Topologically Sorted Source Nodes: [view_14, view_15, x_g_7, mean_7, add_21, x_n_7, mul_7, addcmul_7, x_93, x_94], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:35
            buf141 = torch.ops.aten.convolution.default(buf140, arg91_1, arg92_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg91_1
            del arg92_1
            del buf140
            buf142 = buf141
            assert_size_stride(buf142, (8, 320, 18, 18), (103680, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf142, 16, 'torch.ops.aten.convolution.default')
            del buf141
            buf143 = buf142; del buf142  # reuse
            # Topologically Sorted Source Nodes: [x_96], Original ATen: [aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_21.run(buf143, buf127, 8, 320, 18, 18, stream=raw_stream0)
            del buf127
            # Topologically Sorted Source Nodes: [x_97], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:36
            buf144 = torch.ops.aten.convolution.default(buf143, arg93_1, arg94_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320)
            del arg93_1
            del arg94_1
            buf145 = buf144
            assert_size_stride(buf145, (8, 320, 18, 18), (103680, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf145, 16, 'torch.ops.aten.convolution.default')
            del buf144
            buf146 = empty_strided((8, 18, 18, 1), (324, 18, 1, 2592), device='npu', dtype=torch.float32)
            buf147 = buf146; del buf146  # reuse
            buf148 = empty_strided((8, 18, 18, 1), (324, 18, 1, 2592), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_98, x_99], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_permute_16.run(buf147, buf145, buf148, 8, 18, 18, 320, stream=raw_stream0)
            buf149 = empty_strided((8, 18, 18, 320), (103680, 5760, 320, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_98, x_99], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_native_layer_norm_permute_17.run(buf145, buf147, buf148, arg95_1, arg96_1, buf149, 8, 324, 320, stream=raw_stream0)
            del arg95_1
            del arg96_1
            del buf145
            del buf147
            del buf148
            # Topologically Sorted Source Nodes: [x_98, x_99, x_100, x_101], Original ATen: [aten.permute, aten.native_layer_norm, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:37
            buf150 = torch.ops.aten.convolution.default(reinterpret_tensor(buf149, (8, 320, 18, 18), (103680, 1, 5760, 320), 0), arg97_1, arg98_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg97_1
            del arg98_1
            del buf149
            buf151 = buf150
            assert_size_stride(buf151, (8, 1280, 18, 18), (414720, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf151, 16, 'torch.ops.aten.convolution.default')
            del buf150
            # Topologically Sorted Source Nodes: [x_102], Original ATen: [aten.gelu]
            # [Provenance debug handles] torch.ops.aten.gelu.default:38
            buf152 = torch.ops.aten.gelu.default(buf151)
            del buf151
            buf153 = buf152
            assert_size_stride(buf153, (8, 1280, 18, 18), (414720, 324, 18, 1), 'torch.ops.aten.gelu.default')
            assert_alignment(buf153, 16, 'torch.ops.aten.gelu.default')
            del buf152
            buf154 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_8], Original ATen: [aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_18.run(buf153, buf154, 8, 1280, 324, stream=raw_stream0)
            buf155 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_8, mean_8], Original ATen: [aten.linalg_vector_norm, aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_mean_19.run(buf154, buf155, 8, 1280, stream=raw_stream0)
            buf156 = buf153; del buf153  # reuse
            # Topologically Sorted Source Nodes: [view_16, view_17, x_g_8, mean_8, add_24, x_n_8, mul_8, addcmul_8, x_104], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addcmul_div_linalg_vector_20.run(buf156, arg99_1, arg100_1, buf154, buf155, 8, 1280, 324, stream=raw_stream0)
            del arg100_1
            del arg99_1
            del buf154
            del buf155
            # Topologically Sorted Source Nodes: [view_16, view_17, x_g_8, mean_8, add_24, x_n_8, mul_8, addcmul_8, x_104, x_105], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:39
            buf157 = torch.ops.aten.convolution.default(buf156, arg101_1, arg102_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg101_1
            del arg102_1
            del buf156
            buf158 = buf157
            assert_size_stride(buf158, (8, 320, 18, 18), (103680, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf158, 16, 'torch.ops.aten.convolution.default')
            del buf157
            buf159 = buf158; del buf158  # reuse
            # Topologically Sorted Source Nodes: [x_107], Original ATen: [aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_21.run(buf159, buf143, 8, 320, 18, 18, stream=raw_stream0)
            del buf143
            # Topologically Sorted Source Nodes: [x_108], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:40
            buf160 = torch.ops.aten.convolution.default(buf159, arg103_1, arg104_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320)
            del arg103_1
            del arg104_1
            buf161 = buf160
            assert_size_stride(buf161, (8, 320, 18, 18), (103680, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf161, 16, 'torch.ops.aten.convolution.default')
            del buf160
            buf162 = empty_strided((8, 18, 18, 1), (324, 18, 1, 2592), device='npu', dtype=torch.float32)
            buf163 = buf162; del buf162  # reuse
            buf164 = empty_strided((8, 18, 18, 1), (324, 18, 1, 2592), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_109, x_110], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_permute_16.run(buf163, buf161, buf164, 8, 18, 18, 320, stream=raw_stream0)
            buf165 = empty_strided((8, 18, 18, 320), (103680, 5760, 320, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_109, x_110], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_native_layer_norm_permute_17.run(buf161, buf163, buf164, arg105_1, arg106_1, buf165, 8, 324, 320, stream=raw_stream0)
            del arg105_1
            del arg106_1
            del buf161
            del buf163
            del buf164
            # Topologically Sorted Source Nodes: [x_109, x_110, x_111, x_112], Original ATen: [aten.permute, aten.native_layer_norm, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:41
            buf166 = torch.ops.aten.convolution.default(reinterpret_tensor(buf165, (8, 320, 18, 18), (103680, 1, 5760, 320), 0), arg107_1, arg108_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg107_1
            del arg108_1
            del buf165
            buf167 = buf166
            assert_size_stride(buf167, (8, 1280, 18, 18), (414720, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf167, 16, 'torch.ops.aten.convolution.default')
            del buf166
            # Topologically Sorted Source Nodes: [x_113], Original ATen: [aten.gelu]
            # [Provenance debug handles] torch.ops.aten.gelu.default:42
            buf168 = torch.ops.aten.gelu.default(buf167)
            del buf167
            buf169 = buf168
            assert_size_stride(buf169, (8, 1280, 18, 18), (414720, 324, 18, 1), 'torch.ops.aten.gelu.default')
            assert_alignment(buf169, 16, 'torch.ops.aten.gelu.default')
            del buf168
            buf170 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_9], Original ATen: [aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_18.run(buf169, buf170, 8, 1280, 324, stream=raw_stream0)
            buf171 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_9, mean_9], Original ATen: [aten.linalg_vector_norm, aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_mean_19.run(buf170, buf171, 8, 1280, stream=raw_stream0)
            buf172 = buf169; del buf169  # reuse
            # Topologically Sorted Source Nodes: [view_18, view_19, x_g_9, mean_9, add_27, x_n_9, mul_9, addcmul_9, x_115], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addcmul_div_linalg_vector_20.run(buf172, arg109_1, arg110_1, buf170, buf171, 8, 1280, 324, stream=raw_stream0)
            del arg109_1
            del arg110_1
            del buf170
            del buf171
            # Topologically Sorted Source Nodes: [view_18, view_19, x_g_9, mean_9, add_27, x_n_9, mul_9, addcmul_9, x_115, x_116], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:43
            buf173 = torch.ops.aten.convolution.default(buf172, arg111_1, arg112_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg111_1
            del arg112_1
            del buf172
            buf174 = buf173
            assert_size_stride(buf174, (8, 320, 18, 18), (103680, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf174, 16, 'torch.ops.aten.convolution.default')
            del buf173
            buf175 = buf174; del buf174  # reuse
            # Topologically Sorted Source Nodes: [x_118], Original ATen: [aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_21.run(buf175, buf159, 8, 320, 18, 18, stream=raw_stream0)
            del buf159
            # Topologically Sorted Source Nodes: [x_119], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:44
            buf176 = torch.ops.aten.convolution.default(buf175, arg113_1, arg114_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320)
            del arg113_1
            del arg114_1
            buf177 = buf176
            assert_size_stride(buf177, (8, 320, 18, 18), (103680, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf177, 16, 'torch.ops.aten.convolution.default')
            del buf176
            buf178 = empty_strided((8, 18, 18, 1), (324, 18, 1, 2592), device='npu', dtype=torch.float32)
            buf179 = buf178; del buf178  # reuse
            buf180 = empty_strided((8, 18, 18, 1), (324, 18, 1, 2592), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_120, x_121], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_permute_16.run(buf179, buf177, buf180, 8, 18, 18, 320, stream=raw_stream0)
            buf181 = empty_strided((8, 18, 18, 320), (103680, 5760, 320, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_120, x_121], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_native_layer_norm_permute_17.run(buf177, buf179, buf180, arg115_1, arg116_1, buf181, 8, 324, 320, stream=raw_stream0)
            del arg115_1
            del arg116_1
            del buf177
            del buf179
            del buf180
            # Topologically Sorted Source Nodes: [x_120, x_121, x_122, x_123], Original ATen: [aten.permute, aten.native_layer_norm, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:45
            buf182 = torch.ops.aten.convolution.default(reinterpret_tensor(buf181, (8, 320, 18, 18), (103680, 1, 5760, 320), 0), arg117_1, arg118_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg117_1
            del arg118_1
            del buf181
            buf183 = buf182
            assert_size_stride(buf183, (8, 1280, 18, 18), (414720, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf183, 16, 'torch.ops.aten.convolution.default')
            del buf182
            # Topologically Sorted Source Nodes: [x_124], Original ATen: [aten.gelu]
            # [Provenance debug handles] torch.ops.aten.gelu.default:46
            buf184 = torch.ops.aten.gelu.default(buf183)
            del buf183
            buf185 = buf184
            assert_size_stride(buf185, (8, 1280, 18, 18), (414720, 324, 18, 1), 'torch.ops.aten.gelu.default')
            assert_alignment(buf185, 16, 'torch.ops.aten.gelu.default')
            del buf184
            buf186 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_10], Original ATen: [aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_18.run(buf185, buf186, 8, 1280, 324, stream=raw_stream0)
            buf187 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_10, mean_10], Original ATen: [aten.linalg_vector_norm, aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_mean_19.run(buf186, buf187, 8, 1280, stream=raw_stream0)
            buf188 = buf185; del buf185  # reuse
            # Topologically Sorted Source Nodes: [view_20, view_21, x_g_10, mean_10, add_30, x_n_10, mul_10, addcmul_10, x_126], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addcmul_div_linalg_vector_20.run(buf188, arg119_1, arg120_1, buf186, buf187, 8, 1280, 324, stream=raw_stream0)
            del arg119_1
            del arg120_1
            del buf186
            del buf187
            # Topologically Sorted Source Nodes: [view_20, view_21, x_g_10, mean_10, add_30, x_n_10, mul_10, addcmul_10, x_126, x_127], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:47
            buf189 = torch.ops.aten.convolution.default(buf188, arg121_1, arg122_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg121_1
            del arg122_1
            del buf188
            buf190 = buf189
            assert_size_stride(buf190, (8, 320, 18, 18), (103680, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf190, 16, 'torch.ops.aten.convolution.default')
            del buf189
            buf191 = buf190; del buf190  # reuse
            # Topologically Sorted Source Nodes: [x_129], Original ATen: [aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_21.run(buf191, buf175, 8, 320, 18, 18, stream=raw_stream0)
            del buf175
            # Topologically Sorted Source Nodes: [x_130], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:48
            buf192 = torch.ops.aten.convolution.default(buf191, arg123_1, arg124_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 320)
            del arg123_1
            del arg124_1
            buf193 = buf192
            assert_size_stride(buf193, (8, 320, 18, 18), (103680, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf193, 16, 'torch.ops.aten.convolution.default')
            del buf192
            buf194 = empty_strided((8, 18, 18, 1), (324, 18, 1, 2592), device='npu', dtype=torch.float32)
            buf195 = buf194; del buf194  # reuse
            buf196 = empty_strided((8, 18, 18, 1), (324, 18, 1, 2592), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_131, x_132], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_permute_16.run(buf195, buf193, buf196, 8, 18, 18, 320, stream=raw_stream0)
            buf197 = empty_strided((8, 18, 18, 320), (103680, 5760, 320, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_131, x_132], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_native_layer_norm_permute_17.run(buf193, buf195, buf196, arg125_1, arg126_1, buf197, 8, 324, 320, stream=raw_stream0)
            del arg125_1
            del arg126_1
            del buf193
            del buf195
            del buf196
            # Topologically Sorted Source Nodes: [x_131, x_132, x_133, x_134], Original ATen: [aten.permute, aten.native_layer_norm, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:49
            buf198 = torch.ops.aten.convolution.default(reinterpret_tensor(buf197, (8, 320, 18, 18), (103680, 1, 5760, 320), 0), arg127_1, arg128_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg127_1
            del arg128_1
            del buf197
            buf199 = buf198
            assert_size_stride(buf199, (8, 1280, 18, 18), (414720, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf199, 16, 'torch.ops.aten.convolution.default')
            del buf198
            # Topologically Sorted Source Nodes: [x_135], Original ATen: [aten.gelu]
            # [Provenance debug handles] torch.ops.aten.gelu.default:50
            buf200 = torch.ops.aten.gelu.default(buf199)
            del buf199
            buf201 = buf200
            assert_size_stride(buf201, (8, 1280, 18, 18), (414720, 324, 18, 1), 'torch.ops.aten.gelu.default')
            assert_alignment(buf201, 16, 'torch.ops.aten.gelu.default')
            del buf200
            buf202 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_11], Original ATen: [aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_18.run(buf201, buf202, 8, 1280, 324, stream=raw_stream0)
            buf203 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_11, mean_11], Original ATen: [aten.linalg_vector_norm, aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_mean_19.run(buf202, buf203, 8, 1280, stream=raw_stream0)
            buf204 = buf201; del buf201  # reuse
            # Topologically Sorted Source Nodes: [view_22, view_23, x_g_11, mean_11, add_33, x_n_11, mul_11, addcmul_11, x_137], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addcmul_div_linalg_vector_20.run(buf204, arg129_1, arg130_1, buf202, buf203, 8, 1280, 324, stream=raw_stream0)
            del arg129_1
            del arg130_1
            del buf202
            del buf203
            # Topologically Sorted Source Nodes: [view_22, view_23, x_g_11, mean_11, add_33, x_n_11, mul_11, addcmul_11, x_137, x_138], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:51
            buf205 = torch.ops.aten.convolution.default(buf204, arg131_1, arg132_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg131_1
            del arg132_1
            del buf204
            buf206 = buf205
            assert_size_stride(buf206, (8, 320, 18, 18), (103680, 324, 18, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf206, 16, 'torch.ops.aten.convolution.default')
            del buf205
            buf207 = empty_strided((8, 18, 18, 1), (324, 18, 1, 2592), device='npu', dtype=torch.float32)
            buf208 = buf207; del buf207  # reuse
            buf209 = empty_strided((8, 18, 18, 1), (324, 18, 1, 2592), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_140, x_141, x_142], Original ATen: [aten.add, aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_add_native_layer_norm_permute_22.run(buf208, buf206, buf191, buf209, 8, 18, 18, 320, stream=raw_stream0)
            buf210 = empty_strided((8, 18, 18, 320), (103680, 5760, 320, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_140, x_141, x_142], Original ATen: [aten.add, aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_native_layer_norm_permute_23.run(buf206, buf191, buf208, buf209, arg133_1, arg134_1, buf210, 8, 320, 324, stream=raw_stream0)
            del arg133_1
            del arg134_1
            del buf191
            del buf206
            del buf208
            del buf209
            # Topologically Sorted Source Nodes: [x_140, x_141, x_142, x_143, input_4], Original ATen: [aten.add, aten.permute, aten.native_layer_norm, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:52
            buf211 = torch.ops.aten.convolution.default(reinterpret_tensor(buf210, (8, 320, 18, 18), (103680, 1, 5760, 320), 0), arg135_1, arg136_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
            del arg135_1
            del arg136_1
            del buf210
            buf212 = buf211
            assert_size_stride(buf212, (8, 640, 9, 9), (51840, 81, 9, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf212, 16, 'torch.ops.aten.convolution.default')
            del buf211
            # Topologically Sorted Source Nodes: [x_144], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:53
            buf213 = torch.ops.aten.convolution.default(buf212, arg137_1, arg138_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 640)
            del arg137_1
            del arg138_1
            buf214 = buf213
            assert_size_stride(buf214, (8, 640, 9, 9), (51840, 81, 9, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf214, 16, 'torch.ops.aten.convolution.default')
            del buf213
            buf215 = empty_strided((8, 9, 9, 1), (81, 9, 1, 648), device='npu', dtype=torch.float32)
            buf216 = buf215; del buf215  # reuse
            buf217 = empty_strided((8, 9, 9, 1), (81, 9, 1, 648), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_145, x_146], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_permute_24.run(buf216, buf214, buf217, 8, 9, 9, 640, stream=raw_stream0)
            buf218 = empty_strided((8, 9, 9, 640), (51840, 5760, 640, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_145, x_146], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_native_layer_norm_permute_25.run(buf214, buf216, buf217, arg139_1, arg140_1, buf218, 8, 81, 640, stream=raw_stream0)
            del arg139_1
            del arg140_1
            del buf214
            del buf216
            del buf217
            # Topologically Sorted Source Nodes: [x_145, x_146, x_147, x_148], Original ATen: [aten.permute, aten.native_layer_norm, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:54
            buf219 = torch.ops.aten.convolution.default(reinterpret_tensor(buf218, (8, 640, 9, 9), (51840, 1, 5760, 640), 0), arg141_1, arg142_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg141_1
            del arg142_1
            del buf218
            buf220 = buf219
            assert_size_stride(buf220, (8, 2560, 9, 9), (207360, 81, 9, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf220, 16, 'torch.ops.aten.convolution.default')
            del buf219
            # Topologically Sorted Source Nodes: [x_149], Original ATen: [aten.gelu]
            # [Provenance debug handles] torch.ops.aten.gelu.default:55
            buf221 = torch.ops.aten.gelu.default(buf220)
            del buf220
            buf222 = buf221
            assert_size_stride(buf222, (8, 2560, 9, 9), (207360, 81, 9, 1), 'torch.ops.aten.gelu.default')
            assert_alignment(buf222, 16, 'torch.ops.aten.gelu.default')
            del buf221
            buf223 = empty_strided((8, 2560, 1, 1), (2560, 1, 20480, 20480), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_12], Original ATen: [aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_26.run(buf222, buf223, 8, 2560, 81, stream=raw_stream0)
            buf224 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_12, mean_12], Original ATen: [aten.linalg_vector_norm, aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_mean_27.run(buf223, buf224, 8, 2560, stream=raw_stream0)
            buf225 = buf222; del buf222  # reuse
            # Topologically Sorted Source Nodes: [view_24, view_25, x_g_12, mean_12, add_36, x_n_12, mul_12, addcmul_12, x_151], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addcmul_div_linalg_vector_28.run(buf225, arg143_1, arg144_1, buf223, buf224, 8, 2560, 81, stream=raw_stream0)
            del arg143_1
            del arg144_1
            del buf223
            del buf224
            # Topologically Sorted Source Nodes: [view_24, view_25, x_g_12, mean_12, add_36, x_n_12, mul_12, addcmul_12, x_151, x_152], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:56
            buf226 = torch.ops.aten.convolution.default(buf225, arg145_1, arg146_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg145_1
            del arg146_1
            del buf225
            buf227 = buf226
            assert_size_stride(buf227, (8, 640, 9, 9), (51840, 81, 9, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf227, 16, 'torch.ops.aten.convolution.default')
            del buf226
            buf228 = buf227; del buf227  # reuse
            # Topologically Sorted Source Nodes: [x_154], Original ATen: [aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_29.run(buf228, buf212, 8, 640, 9, 9, stream=raw_stream0)
            del buf212
            # Topologically Sorted Source Nodes: [x_155], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:57
            buf229 = torch.ops.aten.convolution.default(buf228, arg147_1, arg148_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 640)
            del arg147_1
            del arg148_1
            buf230 = buf229
            assert_size_stride(buf230, (8, 640, 9, 9), (51840, 81, 9, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf230, 16, 'torch.ops.aten.convolution.default')
            del buf229
            buf231 = empty_strided((8, 9, 9, 1), (81, 9, 1, 648), device='npu', dtype=torch.float32)
            buf232 = buf231; del buf231  # reuse
            buf233 = empty_strided((8, 9, 9, 1), (81, 9, 1, 648), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_156, x_157], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_permute_24.run(buf232, buf230, buf233, 8, 9, 9, 640, stream=raw_stream0)
            buf234 = empty_strided((8, 9, 9, 640), (51840, 5760, 640, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_156, x_157], Original ATen: [aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_native_layer_norm_permute_25.run(buf230, buf232, buf233, arg149_1, arg150_1, buf234, 8, 81, 640, stream=raw_stream0)
            del arg149_1
            del arg150_1
            del buf230
            del buf232
            del buf233
            # Topologically Sorted Source Nodes: [x_156, x_157, x_158, x_159], Original ATen: [aten.permute, aten.native_layer_norm, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:58
            buf235 = torch.ops.aten.convolution.default(reinterpret_tensor(buf234, (8, 640, 9, 9), (51840, 1, 5760, 640), 0), arg151_1, arg152_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg151_1
            del arg152_1
            del buf234
            buf236 = buf235
            assert_size_stride(buf236, (8, 2560, 9, 9), (207360, 81, 9, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf236, 16, 'torch.ops.aten.convolution.default')
            del buf235
            # Topologically Sorted Source Nodes: [x_160], Original ATen: [aten.gelu]
            # [Provenance debug handles] torch.ops.aten.gelu.default:59
            buf237 = torch.ops.aten.gelu.default(buf236)
            del buf236
            buf238 = buf237
            assert_size_stride(buf238, (8, 2560, 9, 9), (207360, 81, 9, 1), 'torch.ops.aten.gelu.default')
            assert_alignment(buf238, 16, 'torch.ops.aten.gelu.default')
            del buf237
            buf239 = empty_strided((8, 2560, 1, 1), (2560, 1, 20480, 20480), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_13], Original ATen: [aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_26.run(buf238, buf239, 8, 2560, 81, stream=raw_stream0)
            buf240 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_g_13, mean_13], Original ATen: [aten.linalg_vector_norm, aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_mean_27.run(buf239, buf240, 8, 2560, stream=raw_stream0)
            buf241 = buf238; del buf238  # reuse
            # Topologically Sorted Source Nodes: [view_26, view_27, x_g_13, mean_13, add_39, x_n_13, mul_13, addcmul_13, x_162], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addcmul_div_linalg_vector_28.run(buf241, arg153_1, arg154_1, buf239, buf240, 8, 2560, 81, stream=raw_stream0)
            del arg153_1
            del arg154_1
            del buf239
            del buf240
            # Topologically Sorted Source Nodes: [view_26, view_27, x_g_13, mean_13, add_39, x_n_13, mul_13, addcmul_13, x_162, x_163], Original ATen: [aten.view, aten.linalg_vector_norm, aten.mean, aten.add, aten.div, aten.mul, aten.addcmul, aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:60
            buf242 = torch.ops.aten.convolution.default(buf241, arg155_1, arg156_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg155_1
            del arg156_1
            del buf241
            buf243 = buf242
            assert_size_stride(buf243, (8, 640, 9, 9), (51840, 81, 9, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf243, 16, 'torch.ops.aten.convolution.default')
            del buf242
            buf244 = empty_strided((8, 640, 1, 1), (640, 1, 5120, 5120), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [x_165, x_166], Original ATen: [aten.add, aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_add_mean_30.run(buf243, buf228, buf244, 8, 640, 81, stream=raw_stream0)
            del buf228
            del buf243
            buf248 = reinterpret_tensor(buf244, (8, 1, 1, 640), (640, 1, 5120, 1), 0); del buf244  # reuse
            # Topologically Sorted Source Nodes: [x_165, x_166, x_167, x_168], Original ATen: [aten.add, aten.mean, aten.permute, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_add_mean_native_layer_norm_pe_31.run(buf248, arg157_1, arg158_1, 8, 640, stream=raw_stream0)
            del arg157_1
            del arg158_1
            # Topologically Sorted Source Nodes: [x_165, x_166, x_167, x_168, x_169, x_170, x_172], Original ATen: [aten.add, aten.mean, aten.permute, aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] torch.ops.aten.addmm.default:61
            buf249 = torch.ops.aten.addmm.default(arg160_1, reinterpret_tensor(buf248, (8, 640), (640, 1), 0), reinterpret_tensor(arg159_1, (640, 1000), (1, 640), 0), alpha=1, beta=1)
            del arg159_1
            del arg160_1
            del buf248
            buf250 = buf249
            assert_size_stride(buf250, (8, 1000), (1000, 1), 'torch.ops.aten.addmm.default')
            assert_alignment(buf250, 16, 'torch.ops.aten.addmm.default')
            del buf249
        return (buf250, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((80, 3, 4, 4), (48, 16, 4, 1), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((80, ), (1, ), device='npu:0', dtype=torch.float32)
    arg2_1 = rand_strided((8, 3, 288, 288), (248832, 82944, 288, 1), device='npu:0', dtype=torch.float32)
    arg3_1 = rand_strided((80, ), (1, ), device='npu:0', dtype=torch.float32)
    arg4_1 = rand_strided((80, ), (1, ), device='npu:0', dtype=torch.float32)
    arg5_1 = rand_strided((80, 1, 7, 7), (49, 49, 7, 1), device='npu:0', dtype=torch.float32)
    arg6_1 = rand_strided((80, ), (1, ), device='npu:0', dtype=torch.float32)
    arg7_1 = rand_strided((80, ), (1, ), device='npu:0', dtype=torch.float32)
    arg8_1 = rand_strided((80, ), (1, ), device='npu:0', dtype=torch.float32)
    arg9_1 = rand_strided((320, 80, 1, 1), (80, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg10_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg11_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg12_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg13_1 = rand_strided((80, 320, 1, 1), (320, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg14_1 = rand_strided((80, ), (1, ), device='npu:0', dtype=torch.float32)
    arg15_1 = rand_strided((80, 1, 7, 7), (49, 49, 7, 1), device='npu:0', dtype=torch.float32)
    arg16_1 = rand_strided((80, ), (1, ), device='npu:0', dtype=torch.float32)
    arg17_1 = rand_strided((80, ), (1, ), device='npu:0', dtype=torch.float32)
    arg18_1 = rand_strided((80, ), (1, ), device='npu:0', dtype=torch.float32)
    arg19_1 = rand_strided((320, 80, 1, 1), (80, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg20_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg21_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg22_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg23_1 = rand_strided((80, 320, 1, 1), (320, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg24_1 = rand_strided((80, ), (1, ), device='npu:0', dtype=torch.float32)
    arg25_1 = rand_strided((80, ), (1, ), device='npu:0', dtype=torch.float32)
    arg26_1 = rand_strided((80, ), (1, ), device='npu:0', dtype=torch.float32)
    arg27_1 = rand_strided((160, 80, 2, 2), (320, 4, 2, 1), device='npu:0', dtype=torch.float32)
    arg28_1 = rand_strided((160, ), (1, ), device='npu:0', dtype=torch.float32)
    arg29_1 = rand_strided((160, 1, 7, 7), (49, 49, 7, 1), device='npu:0', dtype=torch.float32)
    arg30_1 = rand_strided((160, ), (1, ), device='npu:0', dtype=torch.float32)
    arg31_1 = rand_strided((160, ), (1, ), device='npu:0', dtype=torch.float32)
    arg32_1 = rand_strided((160, ), (1, ), device='npu:0', dtype=torch.float32)
    arg33_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg34_1 = rand_strided((640, ), (1, ), device='npu:0', dtype=torch.float32)
    arg35_1 = rand_strided((640, ), (1, ), device='npu:0', dtype=torch.float32)
    arg36_1 = rand_strided((640, ), (1, ), device='npu:0', dtype=torch.float32)
    arg37_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg38_1 = rand_strided((160, ), (1, ), device='npu:0', dtype=torch.float32)
    arg39_1 = rand_strided((160, 1, 7, 7), (49, 49, 7, 1), device='npu:0', dtype=torch.float32)
    arg40_1 = rand_strided((160, ), (1, ), device='npu:0', dtype=torch.float32)
    arg41_1 = rand_strided((160, ), (1, ), device='npu:0', dtype=torch.float32)
    arg42_1 = rand_strided((160, ), (1, ), device='npu:0', dtype=torch.float32)
    arg43_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg44_1 = rand_strided((640, ), (1, ), device='npu:0', dtype=torch.float32)
    arg45_1 = rand_strided((640, ), (1, ), device='npu:0', dtype=torch.float32)
    arg46_1 = rand_strided((640, ), (1, ), device='npu:0', dtype=torch.float32)
    arg47_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg48_1 = rand_strided((160, ), (1, ), device='npu:0', dtype=torch.float32)
    arg49_1 = rand_strided((160, ), (1, ), device='npu:0', dtype=torch.float32)
    arg50_1 = rand_strided((160, ), (1, ), device='npu:0', dtype=torch.float32)
    arg51_1 = rand_strided((320, 160, 2, 2), (640, 4, 2, 1), device='npu:0', dtype=torch.float32)
    arg52_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg53_1 = rand_strided((320, 1, 7, 7), (49, 49, 7, 1), device='npu:0', dtype=torch.float32)
    arg54_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg55_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg56_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg57_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg58_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg59_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg60_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg61_1 = rand_strided((320, 1280, 1, 1), (1280, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg62_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg63_1 = rand_strided((320, 1, 7, 7), (49, 49, 7, 1), device='npu:0', dtype=torch.float32)
    arg64_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg65_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg66_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg67_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg68_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg69_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg70_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg71_1 = rand_strided((320, 1280, 1, 1), (1280, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg72_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg73_1 = rand_strided((320, 1, 7, 7), (49, 49, 7, 1), device='npu:0', dtype=torch.float32)
    arg74_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg75_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg76_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg77_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg78_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg79_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg80_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg81_1 = rand_strided((320, 1280, 1, 1), (1280, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg82_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg83_1 = rand_strided((320, 1, 7, 7), (49, 49, 7, 1), device='npu:0', dtype=torch.float32)
    arg84_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg85_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg86_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg87_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg88_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg89_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg90_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg91_1 = rand_strided((320, 1280, 1, 1), (1280, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg92_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg93_1 = rand_strided((320, 1, 7, 7), (49, 49, 7, 1), device='npu:0', dtype=torch.float32)
    arg94_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg95_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg96_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg97_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg98_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg99_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg100_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg101_1 = rand_strided((320, 1280, 1, 1), (1280, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg102_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg103_1 = rand_strided((320, 1, 7, 7), (49, 49, 7, 1), device='npu:0', dtype=torch.float32)
    arg104_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg105_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg106_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg107_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg108_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg109_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg110_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg111_1 = rand_strided((320, 1280, 1, 1), (1280, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg112_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg113_1 = rand_strided((320, 1, 7, 7), (49, 49, 7, 1), device='npu:0', dtype=torch.float32)
    arg114_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg115_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg116_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg117_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg118_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg119_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg120_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg121_1 = rand_strided((320, 1280, 1, 1), (1280, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg122_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg123_1 = rand_strided((320, 1, 7, 7), (49, 49, 7, 1), device='npu:0', dtype=torch.float32)
    arg124_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg125_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg126_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg127_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg128_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg129_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg130_1 = rand_strided((1280, ), (1, ), device='npu:0', dtype=torch.float32)
    arg131_1 = rand_strided((320, 1280, 1, 1), (1280, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg132_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg133_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg134_1 = rand_strided((320, ), (1, ), device='npu:0', dtype=torch.float32)
    arg135_1 = rand_strided((640, 320, 2, 2), (1280, 4, 2, 1), device='npu:0', dtype=torch.float32)
    arg136_1 = rand_strided((640, ), (1, ), device='npu:0', dtype=torch.float32)
    arg137_1 = rand_strided((640, 1, 7, 7), (49, 49, 7, 1), device='npu:0', dtype=torch.float32)
    arg138_1 = rand_strided((640, ), (1, ), device='npu:0', dtype=torch.float32)
    arg139_1 = rand_strided((640, ), (1, ), device='npu:0', dtype=torch.float32)
    arg140_1 = rand_strided((640, ), (1, ), device='npu:0', dtype=torch.float32)
    arg141_1 = rand_strided((2560, 640, 1, 1), (640, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg142_1 = rand_strided((2560, ), (1, ), device='npu:0', dtype=torch.float32)
    arg143_1 = rand_strided((2560, ), (1, ), device='npu:0', dtype=torch.float32)
    arg144_1 = rand_strided((2560, ), (1, ), device='npu:0', dtype=torch.float32)
    arg145_1 = rand_strided((640, 2560, 1, 1), (2560, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg146_1 = rand_strided((640, ), (1, ), device='npu:0', dtype=torch.float32)
    arg147_1 = rand_strided((640, 1, 7, 7), (49, 49, 7, 1), device='npu:0', dtype=torch.float32)
    arg148_1 = rand_strided((640, ), (1, ), device='npu:0', dtype=torch.float32)
    arg149_1 = rand_strided((640, ), (1, ), device='npu:0', dtype=torch.float32)
    arg150_1 = rand_strided((640, ), (1, ), device='npu:0', dtype=torch.float32)
    arg151_1 = rand_strided((2560, 640, 1, 1), (640, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg152_1 = rand_strided((2560, ), (1, ), device='npu:0', dtype=torch.float32)
    arg153_1 = rand_strided((2560, ), (1, ), device='npu:0', dtype=torch.float32)
    arg154_1 = rand_strided((2560, ), (1, ), device='npu:0', dtype=torch.float32)
    arg155_1 = rand_strided((640, 2560, 1, 1), (2560, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg156_1 = rand_strided((640, ), (1, ), device='npu:0', dtype=torch.float32)
    arg157_1 = rand_strided((640, ), (1, ), device='npu:0', dtype=torch.float32)
    arg158_1 = rand_strided((640, ), (1, ), device='npu:0', dtype=torch.float32)
    arg159_1 = rand_strided((1000, 640), (640, 1), device='npu:0', dtype=torch.float32)
    arg160_1 = rand_strided((1000, ), (1, ), device='npu:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('convnextv2_nano.fcmae_ft_in22k_in1k', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
