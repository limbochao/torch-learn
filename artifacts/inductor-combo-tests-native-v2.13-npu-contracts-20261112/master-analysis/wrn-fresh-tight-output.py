# AOT ID: ['1_inference']
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


# kernel path: /remote/PTA_213/combo-master-analysis-20261113/wrn-fresh/tmp2r1ign12/mo/cmobkxtk7bag6xzf2morvihhpfa3ltf3ejycwz4bt636noin6dge.py
# Topologically Sorted Source Nodes: [batch_norm, relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm => add, add_1, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   relu => relu
# Graph fragment:
#   %convolution : Tensor "f32[1, 64, 112, 112][802816, 12544, 112, 1]npu:0" = PlaceHolder[target=convolution]
#   %arg2_1 : Tensor "f32[64][1]npu:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[64][1]npu:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[64][1]npu:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[64][1]npu:0" = PlaceHolder[target=arg5_1]
#   %unsqueeze : Tensor "f32[64, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg2_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[64, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[1, 64, 112, 112][802816, 12544, 112, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[64][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[64][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[64][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[64, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%reciprocal, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[64, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[1, 64, 112, 112][802816, 12544, 112, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[64, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[64, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[1, 64, 112, 112][802816, 12544, 112, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[64, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[64, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[1, 64, 112, 112][802816, 12544, 112, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : Tensor "f32[1, 64, 112, 112][802816, 12544, 112, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
#   return %relu
# SchedulerNodes: [SchedulerNode(name='op2')]

triton_poi_fused__native_batch_norm_legit_no_t_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_t_0', '''
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
    size_hints={'y0': 64, 'x1': 12544}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_t_0', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 64), ('x1', 12544)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_t_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, y0_numel, x1_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
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
            tmp0 = tl.load(in_out_ptr0 + (x1 + 12544*y0), x1_mask & y0_mask)
            tmp1 = tl.load(in_ptr0 + (y0), y0_mask)
            tmp3 = tl.load(in_ptr1 + (y0), y0_mask)
            tmp10 = tl.load(in_ptr2 + (y0), y0_mask)
            tmp12 = tl.load(in_ptr3 + (y0), y0_mask)
            tmp2 = tmp0 - tmp1
            tmp4 = tl.full([1, 1], 1e-05, tl.float32)
            tmp5 = tmp3 + tmp4
            tmp6 = tl.sqrt_rn(tmp5)
            tmp7 = tl.full([1, 1], 1.0, tl.float32)
            tmp8 = (tmp7 / tmp6)
            tmp9 = tmp2 * tmp8
            tmp11 = tmp9 * tmp10
            tmp13 = tmp11 + tmp12
            tmp14 = tl.full([1, 1], 0, tl.int32)
            tmp15 = triton_helpers.maximum(tmp14, tmp13)
            tl.store(in_out_ptr0 + (x1 + 12544*y0), tmp15, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: /remote/PTA_213/combo-master-analysis-20261113/wrn-fresh/tmp2r1ign12/cs/ccstivmgzd2rao5vzc3xfgxrisedfou55gfeipnqjnz4mlcld3qn.py
# Topologically Sorted Source Nodes: [batch_norm_1, out], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_1 => add_2, add_3, mul_4, mul_5, reciprocal_1, sqrt_1, sub_1, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_8, unsqueeze_9
#   out => relu_1
# Graph fragment:
#   %convolution_1 : Tensor "f32[1, 128, 56, 56][401408, 3136, 56, 1]npu:0" = PlaceHolder[target=convolution_1]
#   %arg7_1 : Tensor "f32[128][1]npu:0" = PlaceHolder[target=arg7_1]
#   %arg8_1 : Tensor "f32[128][1]npu:0" = PlaceHolder[target=arg8_1]
#   %arg9_1 : Tensor "f32[128][1]npu:0" = PlaceHolder[target=arg9_1]
#   %arg10_1 : Tensor "f32[128][1]npu:0" = PlaceHolder[target=arg10_1]
#   %unsqueeze_8 : Tensor "f32[128, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg7_1, -1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[128, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, -1), kwargs = {})
#   %sub_1 : Tensor "f32[1, 128, 56, 56][401408, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %add_2 : Tensor "f32[128][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, 1e-05), kwargs = {})
#   %sqrt_1 : Tensor "f32[128][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : Tensor "f32[128][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[128, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%reciprocal_1, -1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[128, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, -1), kwargs = {})
#   %mul_4 : Tensor "f32[1, 128, 56, 56][401408, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[128, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg9_1, -1), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[128, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, -1), kwargs = {})
#   %mul_5 : Tensor "f32[1, 128, 56, 56][401408, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[128, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg10_1, -1), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[128, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, -1), kwargs = {})
#   %add_3 : Tensor "f32[1, 128, 56, 56][401408, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : Tensor "f32[1, 128, 56, 56][401408, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
#   return %relu_1
# SchedulerNodes: [SchedulerNode(name='op8')]

triton_poi_fused__native_batch_norm_legit_no_t_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_t_1', '''
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
    size_hints={'y0': 128, 'x1': 3136}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_t_1', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 128), ('x1', 3136)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_t_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, y0_numel, x1_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
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
            tmp0 = tl.load(in_out_ptr0 + (x1 + 3136*y0), x1_mask & y0_mask)
            tmp1 = tl.load(in_ptr0 + (y0), y0_mask)
            tmp3 = tl.load(in_ptr1 + (y0), y0_mask)
            tmp10 = tl.load(in_ptr2 + (y0), y0_mask)
            tmp12 = tl.load(in_ptr3 + (y0), y0_mask)
            tmp2 = tmp0 - tmp1
            tmp4 = tl.full([1, 1], 1e-05, tl.float32)
            tmp5 = tmp3 + tmp4
            tmp6 = tl.sqrt_rn(tmp5)
            tmp7 = tl.full([1, 1], 1.0, tl.float32)
            tmp8 = (tmp7 / tmp6)
            tmp9 = tmp2 * tmp8
            tmp11 = tmp9 * tmp10
            tmp13 = tmp11 + tmp12
            tmp14 = tl.full([1, 1], 0, tl.int32)
            tmp15 = triton_helpers.maximum(tmp14, tmp13)
            tl.store(in_out_ptr0 + (x1 + 3136*y0), tmp15, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: /remote/PTA_213/combo-master-analysis-20261113/wrn-fresh/tmp2r1ign12/br/cbrmm6xpixyz7gslxsn4lxmvklf2u45hcm6d2c6zvyc42ugnklca.py
# Topologically Sorted Source Nodes: [out_2, input_2, add], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add => add_10
#   input_2 => add_8, add_9, mul_13, mul_14, reciprocal_4, sqrt_4, sub_4, unsqueeze_32, unsqueeze_33, unsqueeze_34, unsqueeze_35, unsqueeze_36, unsqueeze_37, unsqueeze_38, unsqueeze_39
#   out_2 => add_6, add_7, mul_10, mul_11, reciprocal_3, sqrt_3, sub_3, unsqueeze_24, unsqueeze_25, unsqueeze_26, unsqueeze_27, unsqueeze_28, unsqueeze_29, unsqueeze_30, unsqueeze_31
# Graph fragment:
#   %convolution_3 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0" = PlaceHolder[target=convolution_3]
#   %arg17_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg17_1]
#   %arg18_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg18_1]
#   %arg19_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg19_1]
#   %arg20_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg20_1]
#   %convolution_4 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0" = PlaceHolder[target=convolution_4]
#   %arg22_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg22_1]
#   %arg23_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg23_1]
#   %arg24_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg24_1]
#   %arg25_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg25_1]
#   %unsqueeze_24 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg17_1, -1), kwargs = {})
#   %unsqueeze_25 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_24, -1), kwargs = {})
#   %sub_3 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %add_6 : Tensor "f32[512][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg18_1, 1e-05), kwargs = {})
#   %sqrt_3 : Tensor "f32[512][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_6,), kwargs = {})
#   %reciprocal_3 : Tensor "f32[512][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_3,), kwargs = {})
#   %unsqueeze_26 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%reciprocal_3, -1), kwargs = {})
#   %unsqueeze_27 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_26, -1), kwargs = {})
#   %mul_10 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %unsqueeze_28 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg19_1, -1), kwargs = {})
#   %unsqueeze_29 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_28, -1), kwargs = {})
#   %mul_11 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %unsqueeze_30 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg20_1, -1), kwargs = {})
#   %unsqueeze_31 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_30, -1), kwargs = {})
#   %add_7 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %unsqueeze_32 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg22_1, -1), kwargs = {})
#   %unsqueeze_33 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_32, -1), kwargs = {})
#   %sub_4 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %add_8 : Tensor "f32[512][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg23_1, 1e-05), kwargs = {})
#   %sqrt_4 : Tensor "f32[512][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_8,), kwargs = {})
#   %reciprocal_4 : Tensor "f32[512][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_4,), kwargs = {})
#   %unsqueeze_34 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%reciprocal_4, -1), kwargs = {})
#   %unsqueeze_35 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_34, -1), kwargs = {})
#   %mul_13 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %unsqueeze_36 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg24_1, -1), kwargs = {})
#   %unsqueeze_37 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_36, -1), kwargs = {})
#   %mul_14 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %unsqueeze_38 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg25_1, -1), kwargs = {})
#   %unsqueeze_39 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_38, -1), kwargs = {})
#   %add_9 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %add_10 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %add_9), kwargs = {})
#   return %add_10
# SchedulerNodes: [SchedulerNode(name='op16')]

triton_poi_fused__native_batch_norm_legit_no_t_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_t_2', '''
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
    size_hints={'z0': 512, 'y1': 56, 'x2': 56}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'Y1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_t_2', 'mutated_arg_names': [], 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'split_axis': [1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [2], 'axis_names': ['z0', 'y1', 'x2'], 'axis_static_values': (('z0', 512), ('y1', 56), ('x2', 56)), 'low_dims': {0, 2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y1BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_t_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, z0_numel, y1_numel, x2_numel, Y1BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr):
    x2_numel = 56
    X2BLOCK_SUB: tl.constexpr = 56
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (z0_numel + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(0) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    for loop_z0 in range(loops_z0):
        z0_2 = (loop_z0 * Z0BLOCK_SUB) + base_z0[None,None,:]
        z0 = (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None]
        z0_mask = z0 < z0_numel
        z0_2_mask = z0_2 < z0_numel
        for loop_y1 in range(loops_y1):
            y1_0 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[:,None,None]
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[None,:,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            y1_0_mask = y1_0 < min(Y1BLOCK+y1_offset, y1_numel)
            x2_1 = base_x2[None,:,None]
            x2 = base_x2[None,None,:]
            tmp0 = tl.load(in_ptr0 + (x2 + 56*y1 + 3136*z0), y1_mask & z0_mask)
            tmp1 = tl.load(in_ptr1 + (z0), z0_mask)
            tmp3 = tl.load(in_ptr2 + (z0), z0_mask)
            tmp10 = tl.load(in_ptr3 + (z0), z0_mask)
            tmp12 = tl.load(in_ptr4 + (z0), z0_mask)
            tmp14 = tl.load(in_ptr5 + (x2 + 56*y1 + 3136*z0), y1_mask & z0_mask)
            tmp15 = tl.load(in_ptr6 + (z0), z0_mask)
            tmp17 = tl.load(in_ptr7 + (z0), z0_mask)
            tmp22 = tl.load(in_ptr8 + (z0), z0_mask)
            tmp24 = tl.load(in_ptr9 + (z0), z0_mask)
            tmp2 = tmp0 - tmp1
            tmp4 = tl.full([1, 1, 1], 1e-05, tl.float32)
            tmp5 = tmp3 + tmp4
            tmp6 = tl.sqrt_rn(tmp5)
            tmp7 = tl.full([1, 1, 1], 1.0, tl.float32)
            tmp8 = (tmp7 / tmp6)
            tmp9 = tmp2 * tmp8
            tmp11 = tmp9 * tmp10
            tmp13 = tmp11 + tmp12
            tmp16 = tmp14 - tmp15
            tmp18 = tmp17 + tmp4
            tmp19 = tl.sqrt_rn(tmp18)
            tmp20 = (tmp7 / tmp19)
            tmp21 = tmp16 * tmp20
            tmp23 = tmp21 * tmp22
            tmp25 = tmp23 + tmp24
            tmp26 = tmp13 + tmp25
            tl.store(out_ptr0 + (z0_2 + 512*x2_1 + 28672*y1_0), tmp26.permute([1, 2, 0]), y1_0_mask & z0_2_mask)
''', device_str='npu')


# kernel path: /remote/PTA_213/combo-master-analysis-20261113/wrn-fresh/tmp2r1ign12/v7/cv7b7cmz7wfb2u7n6dlkoqh77bjt4tqdk2ssrwxw2w6awnw273tr.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   input_3 => relu_3
# Graph fragment:
#   %add_10 : Tensor "f32[1, 512, 56, 56][1605632, 1, 28672, 512]npu:0" = PlaceHolder[target=add_10]
#   %relu_3 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_10,), kwargs = {})
#   return %relu_3
# SchedulerNodes: [SchedulerNode(name='op17')]

triton_poi_fused_relu_3 = async_compile.triton('triton_poi_fused_relu_3', '''
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
    size_hints={'z0': 56, 'y1': 56, 'x2': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'Z0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'split_axis': [0], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [2], 'axis_names': ['z0', 'y1', 'x2'], 'axis_static_values': (('z0', 56), ('y1', 56), ('x2', 512)), 'low_dims': {2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Z0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_3(in_out_ptr0, z0_numel, y1_numel, x2_numel, Z0BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr):
    x2_numel = 512
    X2BLOCK_SUB: tl.constexpr = 512
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (y1_numel + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = (loop_y1 * Y1BLOCK_SUB) + base_y1[None,:,None]
            y1_mask = y1 < y1_numel
            x2 = base_x2[None,None,:]
            tmp0 = tl.load(in_out_ptr0 + (x2 + 512*y1 + 28672*z0), y1_mask & z0_mask)
            tmp1 = tl.full([1, 1, 1], 0, tl.int32)
            tmp2 = triton_helpers.maximum(tmp1, tmp0)
            tl.store(in_out_ptr0 + (x2 + 512*y1 + 28672*z0), tmp2, y1_mask & z0_mask)
''', device_str='npu')


# kernel path: /remote/PTA_213/combo-master-analysis-20261113/wrn-fresh/tmp2r1ign12/zk/czktwlv2p2hczzcdtliqpeqsskpyyslmrpnq72uandqzqmwkobfk.py
# Topologically Sorted Source Nodes: [out_5, add_1, input_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_1 => add_17
#   input_4 => relu_6
#   out_5 => add_15, add_16, mul_22, mul_23, reciprocal_7, sqrt_7, sub_7, unsqueeze_56, unsqueeze_57, unsqueeze_58, unsqueeze_59, unsqueeze_60, unsqueeze_61, unsqueeze_62, unsqueeze_63
# Graph fragment:
#   %convolution_7 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0" = PlaceHolder[target=convolution_7]
#   %arg37_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg37_1]
#   %arg38_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg38_1]
#   %arg39_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg39_1]
#   %arg40_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg40_1]
#   %relu_3 : Tensor "f32[1, 512, 56, 56][1605632, 1, 28672, 512]npu:0" = PlaceHolder[target=relu_3]
#   %unsqueeze_56 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg37_1, -1), kwargs = {})
#   %unsqueeze_57 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_56, -1), kwargs = {})
#   %sub_7 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %add_15 : Tensor "f32[512][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg38_1, 1e-05), kwargs = {})
#   %sqrt_7 : Tensor "f32[512][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_15,), kwargs = {})
#   %reciprocal_7 : Tensor "f32[512][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_7,), kwargs = {})
#   %unsqueeze_58 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%reciprocal_7, -1), kwargs = {})
#   %unsqueeze_59 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_58, -1), kwargs = {})
#   %mul_22 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %unsqueeze_60 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg39_1, -1), kwargs = {})
#   %unsqueeze_61 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_60, -1), kwargs = {})
#   %mul_23 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %unsqueeze_62 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg40_1, -1), kwargs = {})
#   %unsqueeze_63 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_62, -1), kwargs = {})
#   %add_16 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %add_17 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %relu_3), kwargs = {})
#   %relu_6 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_17,), kwargs = {})
#   return %relu_6
# SchedulerNodes: [SchedulerNode(name='op26')]

triton_poi_fused__native_batch_norm_legit_no_t_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_t_4', '''
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
    size_hints={'y0': 512, 'x1': 3136}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_t_4', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 512), ('x1', 3136)), 'low_dims': {0, 1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_t_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
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
            tmp0 = tl.load(in_out_ptr0 + (x1 + 3136*y0), x1_mask & y0_mask)
            tmp1 = tl.load(in_ptr0 + (y0), y0_mask)
            tmp3 = tl.load(in_ptr1 + (y0), y0_mask)
            tmp10 = tl.load(in_ptr2 + (y0), y0_mask)
            tmp12 = tl.load(in_ptr3 + (y0), y0_mask)
            tmp14 = tl.load(in_ptr4 + (y0 + 512*x1), x1_mask & y0_mask)
            tmp2 = tmp0 - tmp1
            tmp4 = tl.full([1, 1], 1e-05, tl.float32)
            tmp5 = tmp3 + tmp4
            tmp6 = tl.sqrt_rn(tmp5)
            tmp7 = tl.full([1, 1], 1.0, tl.float32)
            tmp8 = (tmp7 / tmp6)
            tmp9 = tmp2 * tmp8
            tmp11 = tmp9 * tmp10
            tmp13 = tmp11 + tmp12
            tmp15 = tmp13 + tmp14
            tmp16 = tl.full([1, 1], 0, tl.int32)
            tmp17 = triton_helpers.maximum(tmp16, tmp15)
            tl.store(in_out_ptr0 + (x1 + 3136*y0), tmp17, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: /remote/PTA_213/combo-master-analysis-20261113/wrn-fresh/tmp2r1ign12/kr/ckri3rb5272tabqx3h57cvvno2jmtmrrb2wxnea5nzaul522gvga.py
# Topologically Sorted Source Nodes: [out_8, add_2, input_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_2 => add_24
#   input_5 => relu_9
#   out_8 => add_22, add_23, mul_31, mul_32, reciprocal_10, sqrt_10, sub_10, unsqueeze_80, unsqueeze_81, unsqueeze_82, unsqueeze_83, unsqueeze_84, unsqueeze_85, unsqueeze_86, unsqueeze_87
# Graph fragment:
#   %convolution_10 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0" = PlaceHolder[target=convolution_10]
#   %arg52_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg52_1]
#   %arg53_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg53_1]
#   %arg54_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg54_1]
#   %arg55_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg55_1]
#   %relu_6 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0" = PlaceHolder[target=relu_6]
#   %unsqueeze_80 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg52_1, -1), kwargs = {})
#   %unsqueeze_81 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_80, -1), kwargs = {})
#   %sub_10 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_81), kwargs = {})
#   %add_22 : Tensor "f32[512][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg53_1, 1e-05), kwargs = {})
#   %sqrt_10 : Tensor "f32[512][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_22,), kwargs = {})
#   %reciprocal_10 : Tensor "f32[512][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_10,), kwargs = {})
#   %unsqueeze_82 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%reciprocal_10, -1), kwargs = {})
#   %unsqueeze_83 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_82, -1), kwargs = {})
#   %mul_31 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %unsqueeze_84 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg54_1, -1), kwargs = {})
#   %unsqueeze_85 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_84, -1), kwargs = {})
#   %mul_32 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_85), kwargs = {})
#   %unsqueeze_86 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg55_1, -1), kwargs = {})
#   %unsqueeze_87 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_86, -1), kwargs = {})
#   %add_23 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_87), kwargs = {})
#   %add_24 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_23, %relu_6), kwargs = {})
#   %relu_9 : Tensor "f32[1, 512, 56, 56][1605632, 3136, 56, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_24,), kwargs = {})
#   return %relu_9
# SchedulerNodes: [SchedulerNode(name='op35')]

triton_poi_fused__native_batch_norm_legit_no_t_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_t_5', '''
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
    size_hints={'y0': 512, 'x1': 3136}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_t_5', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 512), ('x1', 3136)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_t_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
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
            tmp0 = tl.load(in_out_ptr0 + (x1 + 3136*y0), x1_mask & y0_mask)
            tmp1 = tl.load(in_ptr0 + (y0), y0_mask)
            tmp3 = tl.load(in_ptr1 + (y0), y0_mask)
            tmp10 = tl.load(in_ptr2 + (y0), y0_mask)
            tmp12 = tl.load(in_ptr3 + (y0), y0_mask)
            tmp14 = tl.load(in_ptr4 + (x1 + 3136*y0), x1_mask & y0_mask)
            tmp2 = tmp0 - tmp1
            tmp4 = tl.full([1, 1], 1e-05, tl.float32)
            tmp5 = tmp3 + tmp4
            tmp6 = tl.sqrt_rn(tmp5)
            tmp7 = tl.full([1, 1], 1.0, tl.float32)
            tmp8 = (tmp7 / tmp6)
            tmp9 = tmp2 * tmp8
            tmp11 = tmp9 * tmp10
            tmp13 = tmp11 + tmp12
            tmp15 = tmp13 + tmp14
            tmp16 = tl.full([1, 1], 0, tl.int32)
            tmp17 = triton_helpers.maximum(tmp16, tmp15)
            tl.store(in_out_ptr0 + (x1 + 3136*y0), tmp17, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: /remote/PTA_213/combo-master-analysis-20261113/wrn-fresh/tmp2r1ign12/hh/chh2sez4ihghoajfvrc7wnn5fdfs4w4mfdofj77hfmmfbzfkpyxt.py
# Topologically Sorted Source Nodes: [batch_norm_11, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_11 => add_25, add_26, mul_34, mul_35, reciprocal_11, sqrt_11, sub_11, unsqueeze_88, unsqueeze_89, unsqueeze_90, unsqueeze_91, unsqueeze_92, unsqueeze_93, unsqueeze_94, unsqueeze_95
#   out_9 => relu_10
# Graph fragment:
#   %convolution_11 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]npu:0" = PlaceHolder[target=convolution_11]
#   %arg57_1 : Tensor "f32[256][1]npu:0" = PlaceHolder[target=arg57_1]
#   %arg58_1 : Tensor "f32[256][1]npu:0" = PlaceHolder[target=arg58_1]
#   %arg59_1 : Tensor "f32[256][1]npu:0" = PlaceHolder[target=arg59_1]
#   %arg60_1 : Tensor "f32[256][1]npu:0" = PlaceHolder[target=arg60_1]
#   %unsqueeze_88 : Tensor "f32[256, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg57_1, -1), kwargs = {})
#   %unsqueeze_89 : Tensor "f32[256, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_88, -1), kwargs = {})
#   %sub_11 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %add_25 : Tensor "f32[256][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg58_1, 1e-05), kwargs = {})
#   %sqrt_11 : Tensor "f32[256][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_25,), kwargs = {})
#   %reciprocal_11 : Tensor "f32[256][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_11,), kwargs = {})
#   %unsqueeze_90 : Tensor "f32[256, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%reciprocal_11, -1), kwargs = {})
#   %unsqueeze_91 : Tensor "f32[256, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_90, -1), kwargs = {})
#   %mul_34 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %unsqueeze_92 : Tensor "f32[256, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg59_1, -1), kwargs = {})
#   %unsqueeze_93 : Tensor "f32[256, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_92, -1), kwargs = {})
#   %mul_35 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %unsqueeze_94 : Tensor "f32[256, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg60_1, -1), kwargs = {})
#   %unsqueeze_95 : Tensor "f32[256, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_94, -1), kwargs = {})
#   %add_26 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
#   %relu_10 : Tensor "f32[1, 256, 56, 56][802816, 3136, 56, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_26,), kwargs = {})
#   return %relu_10
# SchedulerNodes: [SchedulerNode(name='op38')]

triton_poi_fused__native_batch_norm_legit_no_t_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_t_6', '''
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
    size_hints={'y0': 256, 'x1': 3136}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_t_6', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 256), ('x1', 3136)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_t_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
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
            tmp0 = tl.load(in_out_ptr0 + (x1 + 3136*y0), x1_mask & y0_mask)
            tmp1 = tl.load(in_ptr0 + (y0), y0_mask)
            tmp3 = tl.load(in_ptr1 + (y0), y0_mask)
            tmp10 = tl.load(in_ptr2 + (y0), y0_mask)
            tmp12 = tl.load(in_ptr3 + (y0), y0_mask)
            tmp2 = tmp0 - tmp1
            tmp4 = tl.full([1, 1], 1e-05, tl.float32)
            tmp5 = tmp3 + tmp4
            tmp6 = tl.sqrt_rn(tmp5)
            tmp7 = tl.full([1, 1], 1.0, tl.float32)
            tmp8 = (tmp7 / tmp6)
            tmp9 = tmp2 * tmp8
            tmp11 = tmp9 * tmp10
            tmp13 = tmp11 + tmp12
            tmp14 = tl.full([1, 1], 0, tl.int32)
            tmp15 = triton_helpers.maximum(tmp14, tmp13)
            tl.store(in_out_ptr0 + (x1 + 3136*y0), tmp15, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: /remote/PTA_213/combo-master-analysis-20261113/wrn-fresh/tmp2r1ign12/pi/cpixqx5qcvgrlrmnk3e7dtuwdesm3e7634uuaxnxsyupnyj5uy2a.py
# Topologically Sorted Source Nodes: [batch_norm_12, out_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_12 => add_27, add_28, mul_37, mul_38, reciprocal_12, sqrt_12, sub_12, unsqueeze_100, unsqueeze_101, unsqueeze_102, unsqueeze_103, unsqueeze_96, unsqueeze_97, unsqueeze_98, unsqueeze_99
#   out_10 => relu_11
# Graph fragment:
#   %convolution_12 : Tensor "f32[1, 256, 28, 28][200704, 784, 28, 1]npu:0" = PlaceHolder[target=convolution_12]
#   %arg62_1 : Tensor "f32[256][1]npu:0" = PlaceHolder[target=arg62_1]
#   %arg63_1 : Tensor "f32[256][1]npu:0" = PlaceHolder[target=arg63_1]
#   %arg64_1 : Tensor "f32[256][1]npu:0" = PlaceHolder[target=arg64_1]
#   %arg65_1 : Tensor "f32[256][1]npu:0" = PlaceHolder[target=arg65_1]
#   %unsqueeze_96 : Tensor "f32[256, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg62_1, -1), kwargs = {})
#   %unsqueeze_97 : Tensor "f32[256, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_96, -1), kwargs = {})
#   %sub_12 : Tensor "f32[1, 256, 28, 28][200704, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_97), kwargs = {})
#   %add_27 : Tensor "f32[256][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg63_1, 1e-05), kwargs = {})
#   %sqrt_12 : Tensor "f32[256][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_27,), kwargs = {})
#   %reciprocal_12 : Tensor "f32[256][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_12,), kwargs = {})
#   %unsqueeze_98 : Tensor "f32[256, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%reciprocal_12, -1), kwargs = {})
#   %unsqueeze_99 : Tensor "f32[256, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_98, -1), kwargs = {})
#   %mul_37 : Tensor "f32[1, 256, 28, 28][200704, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %unsqueeze_100 : Tensor "f32[256, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg64_1, -1), kwargs = {})
#   %unsqueeze_101 : Tensor "f32[256, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_100, -1), kwargs = {})
#   %mul_38 : Tensor "f32[1, 256, 28, 28][200704, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %unsqueeze_102 : Tensor "f32[256, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg65_1, -1), kwargs = {})
#   %unsqueeze_103 : Tensor "f32[256, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_102, -1), kwargs = {})
#   %add_28 : Tensor "f32[1, 256, 28, 28][200704, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %relu_11 : Tensor "f32[1, 256, 28, 28][200704, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_28,), kwargs = {})
#   return %relu_11
# SchedulerNodes: [SchedulerNode(name='op41')]

triton_poi_fused__native_batch_norm_legit_no_t_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_t_7', '''
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
    size_hints={'y0': 256, 'x1': 784}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_t_7', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 256), ('x1', 784)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_t_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp0 = tl.load(in_out_ptr0 + (x1 + 784*y0), y0_mask)
        tmp1 = tl.load(in_ptr0 + (y0), y0_mask)
        tmp3 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp10 = tl.load(in_ptr2 + (y0), y0_mask)
        tmp12 = tl.load(in_ptr3 + (y0), y0_mask)
        tmp2 = tmp0 - tmp1
        tmp4 = tl.full([1, 1], 1e-05, tl.float32)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.sqrt_rn(tmp5)
        tmp7 = tl.full([1, 1], 1.0, tl.float32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = tmp2 * tmp8
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tl.full([1, 1], 0, tl.int32)
        tmp15 = triton_helpers.maximum(tmp14, tmp13)
        tl.store(in_out_ptr0 + (x1 + 784*y0), tmp15, y0_mask)
''', device_str='npu')


# kernel path: /remote/PTA_213/combo-master-analysis-20261113/wrn-fresh/tmp2r1ign12/cs/ccs4avlj6utudqauekiapvd4bakeun26tico5gzjflrwpkk3ziqm.py
# Topologically Sorted Source Nodes: [out_11, input_7, add_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_3 => add_33
#   input_7 => add_31, add_32, mul_43, mul_44, reciprocal_14, sqrt_14, sub_14, unsqueeze_112, unsqueeze_113, unsqueeze_114, unsqueeze_115, unsqueeze_116, unsqueeze_117, unsqueeze_118, unsqueeze_119
#   out_11 => add_29, add_30, mul_40, mul_41, reciprocal_13, sqrt_13, sub_13, unsqueeze_104, unsqueeze_105, unsqueeze_106, unsqueeze_107, unsqueeze_108, unsqueeze_109, unsqueeze_110, unsqueeze_111
# Graph fragment:
#   %convolution_13 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0" = PlaceHolder[target=convolution_13]
#   %arg67_1 : Tensor "f32[1024][1]npu:0" = PlaceHolder[target=arg67_1]
#   %arg68_1 : Tensor "f32[1024][1]npu:0" = PlaceHolder[target=arg68_1]
#   %arg69_1 : Tensor "f32[1024][1]npu:0" = PlaceHolder[target=arg69_1]
#   %arg70_1 : Tensor "f32[1024][1]npu:0" = PlaceHolder[target=arg70_1]
#   %convolution_14 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0" = PlaceHolder[target=convolution_14]
#   %arg72_1 : Tensor "f32[1024][1]npu:0" = PlaceHolder[target=arg72_1]
#   %arg73_1 : Tensor "f32[1024][1]npu:0" = PlaceHolder[target=arg73_1]
#   %arg74_1 : Tensor "f32[1024][1]npu:0" = PlaceHolder[target=arg74_1]
#   %arg75_1 : Tensor "f32[1024][1]npu:0" = PlaceHolder[target=arg75_1]
#   %unsqueeze_104 : Tensor "f32[1024, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg67_1, -1), kwargs = {})
#   %unsqueeze_105 : Tensor "f32[1024, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_104, -1), kwargs = {})
#   %sub_13 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_105), kwargs = {})
#   %add_29 : Tensor "f32[1024][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg68_1, 1e-05), kwargs = {})
#   %sqrt_13 : Tensor "f32[1024][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_29,), kwargs = {})
#   %reciprocal_13 : Tensor "f32[1024][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_13,), kwargs = {})
#   %unsqueeze_106 : Tensor "f32[1024, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%reciprocal_13, -1), kwargs = {})
#   %unsqueeze_107 : Tensor "f32[1024, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_106, -1), kwargs = {})
#   %mul_40 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %unsqueeze_108 : Tensor "f32[1024, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg69_1, -1), kwargs = {})
#   %unsqueeze_109 : Tensor "f32[1024, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_108, -1), kwargs = {})
#   %mul_41 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_109), kwargs = {})
#   %unsqueeze_110 : Tensor "f32[1024, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg70_1, -1), kwargs = {})
#   %unsqueeze_111 : Tensor "f32[1024, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_110, -1), kwargs = {})
#   %add_30 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_111), kwargs = {})
#   %unsqueeze_112 : Tensor "f32[1024, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg72_1, -1), kwargs = {})
#   %unsqueeze_113 : Tensor "f32[1024, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_112, -1), kwargs = {})
#   %sub_14 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_14, %unsqueeze_113), kwargs = {})
#   %add_31 : Tensor "f32[1024][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg73_1, 1e-05), kwargs = {})
#   %sqrt_14 : Tensor "f32[1024][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_31,), kwargs = {})
#   %reciprocal_14 : Tensor "f32[1024][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_14,), kwargs = {})
#   %unsqueeze_114 : Tensor "f32[1024, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%reciprocal_14, -1), kwargs = {})
#   %unsqueeze_115 : Tensor "f32[1024, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_114, -1), kwargs = {})
#   %mul_43 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %unsqueeze_116 : Tensor "f32[1024, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg74_1, -1), kwargs = {})
#   %unsqueeze_117 : Tensor "f32[1024, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_116, -1), kwargs = {})
#   %mul_44 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_117), kwargs = {})
#   %unsqueeze_118 : Tensor "f32[1024, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg75_1, -1), kwargs = {})
#   %unsqueeze_119 : Tensor "f32[1024, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_118, -1), kwargs = {})
#   %add_32 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_119), kwargs = {})
#   %add_33 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_30, %add_32), kwargs = {})
#   return %add_33
# SchedulerNodes: [SchedulerNode(name='op46')]

triton_poi_fused__native_batch_norm_legit_no_t_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_t_8', '''
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
    size_hints={'z0': 1024, 'y1': 28, 'x2': 28}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_t_8', 'mutated_arg_names': [], 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [2], 'axis_names': ['z0', 'y1', 'x2'], 'axis_static_values': (('z0', 1024), ('y1', 28), ('x2', 28)), 'low_dims': {0, 2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_t_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, z0_numel, y1_numel, x2_numel, Z0BLOCK, Y1BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr):
    x2_numel = 28
    X2BLOCK_SUB: tl.constexpr = 28
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(1) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    for loop_z0 in range(loops_z0):
        z0_2 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[None,None,:]
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        z0_2_mask = z0_2 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1_0 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[:,None,None]
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[None,:,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            y1_0_mask = y1_0 < min(Y1BLOCK+y1_offset, y1_numel)
            x2_1 = base_x2[None,:,None]
            x2 = base_x2[None,None,:]
            tmp0 = tl.load(in_ptr0 + (x2 + 28*y1 + 784*z0), y1_mask & z0_mask)
            tmp1 = tl.load(in_ptr1 + (z0), z0_mask)
            tmp3 = tl.load(in_ptr2 + (z0), z0_mask)
            tmp10 = tl.load(in_ptr3 + (z0), z0_mask)
            tmp12 = tl.load(in_ptr4 + (z0), z0_mask)
            tmp14 = tl.load(in_ptr5 + (x2 + 28*y1 + 784*z0), y1_mask & z0_mask)
            tmp15 = tl.load(in_ptr6 + (z0), z0_mask)
            tmp17 = tl.load(in_ptr7 + (z0), z0_mask)
            tmp22 = tl.load(in_ptr8 + (z0), z0_mask)
            tmp24 = tl.load(in_ptr9 + (z0), z0_mask)
            tmp2 = tmp0 - tmp1
            tmp4 = tl.full([1, 1, 1], 1e-05, tl.float32)
            tmp5 = tmp3 + tmp4
            tmp6 = tl.sqrt_rn(tmp5)
            tmp7 = tl.full([1, 1, 1], 1.0, tl.float32)
            tmp8 = (tmp7 / tmp6)
            tmp9 = tmp2 * tmp8
            tmp11 = tmp9 * tmp10
            tmp13 = tmp11 + tmp12
            tmp16 = tmp14 - tmp15
            tmp18 = tmp17 + tmp4
            tmp19 = tl.sqrt_rn(tmp18)
            tmp20 = (tmp7 / tmp19)
            tmp21 = tmp16 * tmp20
            tmp23 = tmp21 * tmp22
            tmp25 = tmp23 + tmp24
            tmp26 = tmp13 + tmp25
            tl.store(out_ptr0 + (z0_2 + 1024*x2_1 + 28672*y1_0), tmp26.permute([1, 2, 0]), y1_0_mask & z0_2_mask)
''', device_str='npu')


# kernel path: /remote/PTA_213/combo-master-analysis-20261113/wrn-fresh/tmp2r1ign12/2o/c2ohylsl4mqbcabnjktv7g7tbnpzlc3dnxl26o6lkxtnbakeqlxl.py
# Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   input_8 => relu_12
# Graph fragment:
#   %add_33 : Tensor "f32[1, 1024, 28, 28][802816, 1, 28672, 1024]npu:0" = PlaceHolder[target=add_33]
#   %relu_12 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_33,), kwargs = {})
#   return %relu_12
# SchedulerNodes: [SchedulerNode(name='op47')]

triton_poi_fused_relu_9 = async_compile.triton('triton_poi_fused_relu_9', '''
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
    size_hints={'z0': 28, 'y1': 28, 'x2': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [2], 'axis_names': ['z0', 'y1', 'x2'], 'axis_static_values': (('z0', 28), ('y1', 28), ('x2', 1024)), 'low_dims': {2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_9(in_out_ptr0, z0_numel, y1_numel, x2_numel, Z0BLOCK, Y1BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr):
    x2_numel = 1024
    X2BLOCK_SUB: tl.constexpr = 1024
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
            tmp0 = tl.load(in_out_ptr0 + (x2 + 1024*y1 + 28672*z0), y1_mask & z0_mask)
            tmp1 = tl.full([1, 1, 1], 0, tl.int32)
            tmp2 = triton_helpers.maximum(tmp1, tmp0)
            tl.store(in_out_ptr0 + (x2 + 1024*y1 + 28672*z0), tmp2, y1_mask & z0_mask)
''', device_str='npu')


# kernel path: /remote/PTA_213/combo-master-analysis-20261113/wrn-fresh/tmp2r1ign12/si/csi64el5yfwmkwsmhnn5othj25sfp2zgybrunao3p6og6fi3vn6z.py
# Topologically Sorted Source Nodes: [out_14, add_4, input_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_4 => add_40
#   input_9 => relu_15
#   out_14 => add_38, add_39, mul_52, mul_53, reciprocal_17, sqrt_17, sub_17, unsqueeze_136, unsqueeze_137, unsqueeze_138, unsqueeze_139, unsqueeze_140, unsqueeze_141, unsqueeze_142, unsqueeze_143
# Graph fragment:
#   %convolution_17 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0" = PlaceHolder[target=convolution_17]
#   %arg87_1 : Tensor "f32[1024][1]npu:0" = PlaceHolder[target=arg87_1]
#   %arg88_1 : Tensor "f32[1024][1]npu:0" = PlaceHolder[target=arg88_1]
#   %arg89_1 : Tensor "f32[1024][1]npu:0" = PlaceHolder[target=arg89_1]
#   %arg90_1 : Tensor "f32[1024][1]npu:0" = PlaceHolder[target=arg90_1]
#   %relu_12 : Tensor "f32[1, 1024, 28, 28][802816, 1, 28672, 1024]npu:0" = PlaceHolder[target=relu_12]
#   %unsqueeze_136 : Tensor "f32[1024, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg87_1, -1), kwargs = {})
#   %unsqueeze_137 : Tensor "f32[1024, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_136, -1), kwargs = {})
#   %sub_17 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_137), kwargs = {})
#   %add_38 : Tensor "f32[1024][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg88_1, 1e-05), kwargs = {})
#   %sqrt_17 : Tensor "f32[1024][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_38,), kwargs = {})
#   %reciprocal_17 : Tensor "f32[1024][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_17,), kwargs = {})
#   %unsqueeze_138 : Tensor "f32[1024, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%reciprocal_17, -1), kwargs = {})
#   %unsqueeze_139 : Tensor "f32[1024, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_138, -1), kwargs = {})
#   %mul_52 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_139), kwargs = {})
#   %unsqueeze_140 : Tensor "f32[1024, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg89_1, -1), kwargs = {})
#   %unsqueeze_141 : Tensor "f32[1024, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_140, -1), kwargs = {})
#   %mul_53 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %unsqueeze_141), kwargs = {})
#   %unsqueeze_142 : Tensor "f32[1024, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg90_1, -1), kwargs = {})
#   %unsqueeze_143 : Tensor "f32[1024, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_142, -1), kwargs = {})
#   %add_39 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %unsqueeze_143), kwargs = {})
#   %add_40 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_39, %relu_12), kwargs = {})
#   %relu_15 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_40,), kwargs = {})
#   return %relu_15
# SchedulerNodes: [SchedulerNode(name='op56')]

triton_poi_fused__native_batch_norm_legit_no_t_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_t_10', '''
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
    size_hints={'y0': 1024, 'x1': 784}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_t_10', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 1024), ('x1', 784)), 'low_dims': {0, 1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_t_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp0 = tl.load(in_out_ptr0 + (x1 + 784*y0), y0_mask)
        tmp1 = tl.load(in_ptr0 + (y0), y0_mask)
        tmp3 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp10 = tl.load(in_ptr2 + (y0), y0_mask)
        tmp12 = tl.load(in_ptr3 + (y0), y0_mask)
        tmp14 = tl.load(in_ptr4 + (y0 + 1024*x1), y0_mask)
        tmp2 = tmp0 - tmp1
        tmp4 = tl.full([1, 1], 1e-05, tl.float32)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.sqrt_rn(tmp5)
        tmp7 = tl.full([1, 1], 1.0, tl.float32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = tmp2 * tmp8
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tl.full([1, 1], 0, tl.int32)
        tmp17 = triton_helpers.maximum(tmp16, tmp15)
        tl.store(in_out_ptr0 + (x1 + 784*y0), tmp17, y0_mask)
''', device_str='npu')


# kernel path: /remote/PTA_213/combo-master-analysis-20261113/wrn-fresh/tmp2r1ign12/yo/cyobonczxk64wrbsght4fnelnc6rbsnhi7nyu6hrx7qjls4k27gd.py
# Topologically Sorted Source Nodes: [out_17, add_5, input_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_5 => add_47
#   input_10 => relu_18
#   out_17 => add_45, add_46, mul_61, mul_62, reciprocal_20, sqrt_20, sub_20, unsqueeze_160, unsqueeze_161, unsqueeze_162, unsqueeze_163, unsqueeze_164, unsqueeze_165, unsqueeze_166, unsqueeze_167
# Graph fragment:
#   %convolution_20 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0" = PlaceHolder[target=convolution_20]
#   %arg102_1 : Tensor "f32[1024][1]npu:0" = PlaceHolder[target=arg102_1]
#   %arg103_1 : Tensor "f32[1024][1]npu:0" = PlaceHolder[target=arg103_1]
#   %arg104_1 : Tensor "f32[1024][1]npu:0" = PlaceHolder[target=arg104_1]
#   %arg105_1 : Tensor "f32[1024][1]npu:0" = PlaceHolder[target=arg105_1]
#   %relu_15 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0" = PlaceHolder[target=relu_15]
#   %unsqueeze_160 : Tensor "f32[1024, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg102_1, -1), kwargs = {})
#   %unsqueeze_161 : Tensor "f32[1024, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_160, -1), kwargs = {})
#   %sub_20 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_20, %unsqueeze_161), kwargs = {})
#   %add_45 : Tensor "f32[1024][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg103_1, 1e-05), kwargs = {})
#   %sqrt_20 : Tensor "f32[1024][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_45,), kwargs = {})
#   %reciprocal_20 : Tensor "f32[1024][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_20,), kwargs = {})
#   %unsqueeze_162 : Tensor "f32[1024, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%reciprocal_20, -1), kwargs = {})
#   %unsqueeze_163 : Tensor "f32[1024, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_162, -1), kwargs = {})
#   %mul_61 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %unsqueeze_164 : Tensor "f32[1024, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg104_1, -1), kwargs = {})
#   %unsqueeze_165 : Tensor "f32[1024, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_164, -1), kwargs = {})
#   %mul_62 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_165), kwargs = {})
#   %unsqueeze_166 : Tensor "f32[1024, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg105_1, -1), kwargs = {})
#   %unsqueeze_167 : Tensor "f32[1024, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_166, -1), kwargs = {})
#   %add_46 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_167), kwargs = {})
#   %add_47 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_46, %relu_15), kwargs = {})
#   %relu_18 : Tensor "f32[1, 1024, 28, 28][802816, 784, 28, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_47,), kwargs = {})
#   return %relu_18
# SchedulerNodes: [SchedulerNode(name='op65')]

triton_poi_fused__native_batch_norm_legit_no_t_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_t_11', '''
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
    size_hints={'y0': 1024, 'x1': 784}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_t_11', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 1024), ('x1', 784)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_t_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp0 = tl.load(in_out_ptr0 + (x1 + 784*y0), y0_mask)
        tmp1 = tl.load(in_ptr0 + (y0), y0_mask)
        tmp3 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp10 = tl.load(in_ptr2 + (y0), y0_mask)
        tmp12 = tl.load(in_ptr3 + (y0), y0_mask)
        tmp14 = tl.load(in_ptr4 + (x1 + 784*y0), y0_mask)
        tmp2 = tmp0 - tmp1
        tmp4 = tl.full([1, 1], 1e-05, tl.float32)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.sqrt_rn(tmp5)
        tmp7 = tl.full([1, 1], 1.0, tl.float32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = tmp2 * tmp8
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tl.full([1, 1], 0, tl.int32)
        tmp17 = triton_helpers.maximum(tmp16, tmp15)
        tl.store(in_out_ptr0 + (x1 + 784*y0), tmp17, y0_mask)
''', device_str='npu')


# kernel path: /remote/PTA_213/combo-master-analysis-20261113/wrn-fresh/tmp2r1ign12/ir/cirtdnt4v5xpdm7v6dwdikdmnsq7t3bqau5ue7lvdrfzuirwqeas.py
# Topologically Sorted Source Nodes: [batch_norm_24, out_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_24 => add_55, add_56, mul_73, mul_74, reciprocal_24, sqrt_24, sub_24, unsqueeze_192, unsqueeze_193, unsqueeze_194, unsqueeze_195, unsqueeze_196, unsqueeze_197, unsqueeze_198, unsqueeze_199
#   out_21 => relu_22
# Graph fragment:
#   %convolution_24 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]npu:0" = PlaceHolder[target=convolution_24]
#   %arg122_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg122_1]
#   %arg123_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg123_1]
#   %arg124_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg124_1]
#   %arg125_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg125_1]
#   %unsqueeze_192 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg122_1, -1), kwargs = {})
#   %unsqueeze_193 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_192, -1), kwargs = {})
#   %sub_24 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_193), kwargs = {})
#   %add_55 : Tensor "f32[512][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg123_1, 1e-05), kwargs = {})
#   %sqrt_24 : Tensor "f32[512][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_55,), kwargs = {})
#   %reciprocal_24 : Tensor "f32[512][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_24,), kwargs = {})
#   %unsqueeze_194 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%reciprocal_24, -1), kwargs = {})
#   %unsqueeze_195 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_194, -1), kwargs = {})
#   %mul_73 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %unsqueeze_196 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg124_1, -1), kwargs = {})
#   %unsqueeze_197 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_196, -1), kwargs = {})
#   %mul_74 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_197), kwargs = {})
#   %unsqueeze_198 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg125_1, -1), kwargs = {})
#   %unsqueeze_199 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_198, -1), kwargs = {})
#   %add_56 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_199), kwargs = {})
#   %relu_22 : Tensor "f32[1, 512, 28, 28][401408, 784, 28, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_56,), kwargs = {})
#   return %relu_22
# SchedulerNodes: [SchedulerNode(name='op77')]

triton_poi_fused__native_batch_norm_legit_no_t_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_t_12', '''
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
    size_hints={'y0': 512, 'x1': 784}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_t_12', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 512), ('x1', 784)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_t_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp0 = tl.load(in_out_ptr0 + (x1 + 784*y0), y0_mask)
        tmp1 = tl.load(in_ptr0 + (y0), y0_mask)
        tmp3 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp10 = tl.load(in_ptr2 + (y0), y0_mask)
        tmp12 = tl.load(in_ptr3 + (y0), y0_mask)
        tmp2 = tmp0 - tmp1
        tmp4 = tl.full([1, 1], 1e-05, tl.float32)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.sqrt_rn(tmp5)
        tmp7 = tl.full([1, 1], 1.0, tl.float32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = tmp2 * tmp8
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tl.full([1, 1], 0, tl.int32)
        tmp15 = triton_helpers.maximum(tmp14, tmp13)
        tl.store(in_out_ptr0 + (x1 + 784*y0), tmp15, y0_mask)
''', device_str='npu')


# kernel path: /remote/PTA_213/combo-master-analysis-20261113/wrn-fresh/tmp2r1ign12/v4/cv4pqp75ygf3m6vxnfdp7bcy64sunh22fo6nhxndco6xcoeczptu.py
# Topologically Sorted Source Nodes: [batch_norm_25, out_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_25 => add_57, add_58, mul_76, mul_77, reciprocal_25, sqrt_25, sub_25, unsqueeze_200, unsqueeze_201, unsqueeze_202, unsqueeze_203, unsqueeze_204, unsqueeze_205, unsqueeze_206, unsqueeze_207
#   out_22 => relu_23
# Graph fragment:
#   %convolution_25 : Tensor "f32[1, 512, 14, 14][100352, 196, 14, 1]npu:0" = PlaceHolder[target=convolution_25]
#   %arg127_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg127_1]
#   %arg128_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg128_1]
#   %arg129_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg129_1]
#   %arg130_1 : Tensor "f32[512][1]npu:0" = PlaceHolder[target=arg130_1]
#   %unsqueeze_200 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg127_1, -1), kwargs = {})
#   %unsqueeze_201 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_200, -1), kwargs = {})
#   %sub_25 : Tensor "f32[1, 512, 14, 14][100352, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_201), kwargs = {})
#   %add_57 : Tensor "f32[512][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg128_1, 1e-05), kwargs = {})
#   %sqrt_25 : Tensor "f32[512][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_57,), kwargs = {})
#   %reciprocal_25 : Tensor "f32[512][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_25,), kwargs = {})
#   %unsqueeze_202 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%reciprocal_25, -1), kwargs = {})
#   %unsqueeze_203 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_202, -1), kwargs = {})
#   %mul_76 : Tensor "f32[1, 512, 14, 14][100352, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_203), kwargs = {})
#   %unsqueeze_204 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg129_1, -1), kwargs = {})
#   %unsqueeze_205 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_204, -1), kwargs = {})
#   %mul_77 : Tensor "f32[1, 512, 14, 14][100352, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, %unsqueeze_205), kwargs = {})
#   %unsqueeze_206 : Tensor "f32[512, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg130_1, -1), kwargs = {})
#   %unsqueeze_207 : Tensor "f32[512, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_206, -1), kwargs = {})
#   %add_58 : Tensor "f32[1, 512, 14, 14][100352, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_77, %unsqueeze_207), kwargs = {})
#   %relu_23 : Tensor "f32[1, 512, 14, 14][100352, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_58,), kwargs = {})
#   return %relu_23
# SchedulerNodes: [SchedulerNode(name='op80')]

triton_poi_fused__native_batch_norm_legit_no_t_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_t_13', '''
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
    size_hints={'y0': 512, 'x1': 196}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_t_13', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 512), ('x1', 196)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_t_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp0 = tl.load(in_out_ptr0 + (x1 + 196*y0), y0_mask)
        tmp1 = tl.load(in_ptr0 + (y0), y0_mask)
        tmp3 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp10 = tl.load(in_ptr2 + (y0), y0_mask)
        tmp12 = tl.load(in_ptr3 + (y0), y0_mask)
        tmp2 = tmp0 - tmp1
        tmp4 = tl.full([1, 1], 1e-05, tl.float32)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.sqrt_rn(tmp5)
        tmp7 = tl.full([1, 1], 1.0, tl.float32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = tmp2 * tmp8
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tl.full([1, 1], 0, tl.int32)
        tmp15 = triton_helpers.maximum(tmp14, tmp13)
        tl.store(in_out_ptr0 + (x1 + 196*y0), tmp15, y0_mask)
''', device_str='npu')


# kernel path: /remote/PTA_213/combo-master-analysis-20261113/wrn-fresh/tmp2r1ign12/44/c443t63q7y62kbna55arvpweqizbq7znutzohveyqu2g5zepxpv7.py
# Topologically Sorted Source Nodes: [out_23, input_13, add_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_7 => add_63
#   input_13 => add_61, add_62, mul_82, mul_83, reciprocal_27, sqrt_27, sub_27, unsqueeze_216, unsqueeze_217, unsqueeze_218, unsqueeze_219, unsqueeze_220, unsqueeze_221, unsqueeze_222, unsqueeze_223
#   out_23 => add_59, add_60, mul_79, mul_80, reciprocal_26, sqrt_26, sub_26, unsqueeze_208, unsqueeze_209, unsqueeze_210, unsqueeze_211, unsqueeze_212, unsqueeze_213, unsqueeze_214, unsqueeze_215
# Graph fragment:
#   %convolution_26 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0" = PlaceHolder[target=convolution_26]
#   %arg132_1 : Tensor "f32[2048][1]npu:0" = PlaceHolder[target=arg132_1]
#   %arg133_1 : Tensor "f32[2048][1]npu:0" = PlaceHolder[target=arg133_1]
#   %arg134_1 : Tensor "f32[2048][1]npu:0" = PlaceHolder[target=arg134_1]
#   %arg135_1 : Tensor "f32[2048][1]npu:0" = PlaceHolder[target=arg135_1]
#   %convolution_27 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0" = PlaceHolder[target=convolution_27]
#   %arg137_1 : Tensor "f32[2048][1]npu:0" = PlaceHolder[target=arg137_1]
#   %arg138_1 : Tensor "f32[2048][1]npu:0" = PlaceHolder[target=arg138_1]
#   %arg139_1 : Tensor "f32[2048][1]npu:0" = PlaceHolder[target=arg139_1]
#   %arg140_1 : Tensor "f32[2048][1]npu:0" = PlaceHolder[target=arg140_1]
#   %unsqueeze_208 : Tensor "f32[2048, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg132_1, -1), kwargs = {})
#   %unsqueeze_209 : Tensor "f32[2048, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_208, -1), kwargs = {})
#   %sub_26 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_209), kwargs = {})
#   %add_59 : Tensor "f32[2048][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg133_1, 1e-05), kwargs = {})
#   %sqrt_26 : Tensor "f32[2048][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_59,), kwargs = {})
#   %reciprocal_26 : Tensor "f32[2048][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_26,), kwargs = {})
#   %unsqueeze_210 : Tensor "f32[2048, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%reciprocal_26, -1), kwargs = {})
#   %unsqueeze_211 : Tensor "f32[2048, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_210, -1), kwargs = {})
#   %mul_79 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_211), kwargs = {})
#   %unsqueeze_212 : Tensor "f32[2048, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg134_1, -1), kwargs = {})
#   %unsqueeze_213 : Tensor "f32[2048, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_212, -1), kwargs = {})
#   %mul_80 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %unsqueeze_213), kwargs = {})
#   %unsqueeze_214 : Tensor "f32[2048, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg135_1, -1), kwargs = {})
#   %unsqueeze_215 : Tensor "f32[2048, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_214, -1), kwargs = {})
#   %add_60 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %unsqueeze_215), kwargs = {})
#   %unsqueeze_216 : Tensor "f32[2048, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg137_1, -1), kwargs = {})
#   %unsqueeze_217 : Tensor "f32[2048, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_216, -1), kwargs = {})
#   %sub_27 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_27, %unsqueeze_217), kwargs = {})
#   %add_61 : Tensor "f32[2048][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg138_1, 1e-05), kwargs = {})
#   %sqrt_27 : Tensor "f32[2048][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_61,), kwargs = {})
#   %reciprocal_27 : Tensor "f32[2048][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_27,), kwargs = {})
#   %unsqueeze_218 : Tensor "f32[2048, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%reciprocal_27, -1), kwargs = {})
#   %unsqueeze_219 : Tensor "f32[2048, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_218, -1), kwargs = {})
#   %mul_82 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %unsqueeze_219), kwargs = {})
#   %unsqueeze_220 : Tensor "f32[2048, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg139_1, -1), kwargs = {})
#   %unsqueeze_221 : Tensor "f32[2048, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_220, -1), kwargs = {})
#   %mul_83 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, %unsqueeze_221), kwargs = {})
#   %unsqueeze_222 : Tensor "f32[2048, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg140_1, -1), kwargs = {})
#   %unsqueeze_223 : Tensor "f32[2048, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_222, -1), kwargs = {})
#   %add_62 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_83, %unsqueeze_223), kwargs = {})
#   %add_63 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_60, %add_62), kwargs = {})
#   return %add_63
# SchedulerNodes: [SchedulerNode(name='op85')]

triton_poi_fused__native_batch_norm_legit_no_t_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_t_14', '''
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
    size_hints={'z0': 2048, 'y1': 14, 'x2': 14}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_t_14', 'mutated_arg_names': [], 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [2], 'axis_names': ['z0', 'y1', 'x2'], 'axis_static_values': (('z0', 2048), ('y1', 14), ('x2', 14)), 'low_dims': {0, 2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_t_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, z0_numel, y1_numel, x2_numel, Z0BLOCK, Y1BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr):
    x2_numel = 14
    X2BLOCK_SUB: tl.constexpr = 14
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(1) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    for loop_z0 in range(loops_z0):
        z0_2 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[None,None,:]
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        z0_2_mask = z0_2 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1_0 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[:,None,None]
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[None,:,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            y1_0_mask = y1_0 < min(Y1BLOCK+y1_offset, y1_numel)
            x2_1 = base_x2[None,:,None]
            x2 = base_x2[None,None,:]
            tmp0 = tl.load(in_ptr0 + (x2 + 14*y1 + 196*z0), y1_mask & z0_mask)
            tmp1 = tl.load(in_ptr1 + (z0), z0_mask)
            tmp3 = tl.load(in_ptr2 + (z0), z0_mask)
            tmp10 = tl.load(in_ptr3 + (z0), z0_mask)
            tmp12 = tl.load(in_ptr4 + (z0), z0_mask)
            tmp14 = tl.load(in_ptr5 + (x2 + 14*y1 + 196*z0), y1_mask & z0_mask)
            tmp15 = tl.load(in_ptr6 + (z0), z0_mask)
            tmp17 = tl.load(in_ptr7 + (z0), z0_mask)
            tmp22 = tl.load(in_ptr8 + (z0), z0_mask)
            tmp24 = tl.load(in_ptr9 + (z0), z0_mask)
            tmp2 = tmp0 - tmp1
            tmp4 = tl.full([1, 1, 1], 1e-05, tl.float32)
            tmp5 = tmp3 + tmp4
            tmp6 = tl.sqrt_rn(tmp5)
            tmp7 = tl.full([1, 1, 1], 1.0, tl.float32)
            tmp8 = (tmp7 / tmp6)
            tmp9 = tmp2 * tmp8
            tmp11 = tmp9 * tmp10
            tmp13 = tmp11 + tmp12
            tmp16 = tmp14 - tmp15
            tmp18 = tmp17 + tmp4
            tmp19 = tl.sqrt_rn(tmp18)
            tmp20 = (tmp7 / tmp19)
            tmp21 = tmp16 * tmp20
            tmp23 = tmp21 * tmp22
            tmp25 = tmp23 + tmp24
            tmp26 = tmp13 + tmp25
            tl.store(out_ptr0 + (z0_2 + 2048*x2_1 + 28672*y1_0), tmp26.permute([1, 2, 0]), y1_0_mask & z0_2_mask)
''', device_str='npu')


# kernel path: /remote/PTA_213/combo-master-analysis-20261113/wrn-fresh/tmp2r1ign12/lj/cljswbei2dtgpovxlgdbbjvulijdf4nnfoovr5xosp3dm43up2tx.py
# Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   input_14 => relu_24
# Graph fragment:
#   %add_63 : Tensor "f32[1, 2048, 14, 14][401408, 1, 28672, 2048]npu:0" = PlaceHolder[target=add_63]
#   %relu_24 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_63,), kwargs = {})
#   return %relu_24
# SchedulerNodes: [SchedulerNode(name='op86')]

triton_poi_fused_relu_15 = async_compile.triton('triton_poi_fused_relu_15', '''
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
    size_hints={'z0': 14, 'y1': 14, 'x2': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['z0', 'y1', 'x2'], 'axis_static_values': (('z0', 14), ('y1', 14), ('x2', 2048)), 'low_dims': {2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_15(in_out_ptr0, z0_numel, y1_numel, x2_numel, Z0BLOCK, Y1BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr):
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
                tmp0 = tl.load(in_out_ptr0 + (x2 + 2048*y1 + 28672*z0), x2_mask & y1_mask & z0_mask)
                tmp1 = tl.full([1, 1, 1], 0, tl.int32)
                tmp2 = triton_helpers.maximum(tmp1, tmp0)
                tl.store(in_out_ptr0 + (x2 + 2048*y1 + 28672*z0), tmp2, x2_mask & y1_mask & z0_mask)
''', device_str='npu')


# kernel path: /remote/PTA_213/combo-master-analysis-20261113/wrn-fresh/tmp2r1ign12/w7/cw7g3ic2g7sbs4lktu6x7jinqsbmxktkmjm57liwadk2bfcm3btm.py
# Topologically Sorted Source Nodes: [out_26, add_8, input_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_8 => add_70
#   input_15 => relu_27
#   out_26 => add_68, add_69, mul_91, mul_92, reciprocal_30, sqrt_30, sub_30, unsqueeze_240, unsqueeze_241, unsqueeze_242, unsqueeze_243, unsqueeze_244, unsqueeze_245, unsqueeze_246, unsqueeze_247
# Graph fragment:
#   %convolution_30 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0" = PlaceHolder[target=convolution_30]
#   %arg152_1 : Tensor "f32[2048][1]npu:0" = PlaceHolder[target=arg152_1]
#   %arg153_1 : Tensor "f32[2048][1]npu:0" = PlaceHolder[target=arg153_1]
#   %arg154_1 : Tensor "f32[2048][1]npu:0" = PlaceHolder[target=arg154_1]
#   %arg155_1 : Tensor "f32[2048][1]npu:0" = PlaceHolder[target=arg155_1]
#   %relu_24 : Tensor "f32[1, 2048, 14, 14][401408, 1, 28672, 2048]npu:0" = PlaceHolder[target=relu_24]
#   %unsqueeze_240 : Tensor "f32[2048, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg152_1, -1), kwargs = {})
#   %unsqueeze_241 : Tensor "f32[2048, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_240, -1), kwargs = {})
#   %sub_30 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_30, %unsqueeze_241), kwargs = {})
#   %add_68 : Tensor "f32[2048][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg153_1, 1e-05), kwargs = {})
#   %sqrt_30 : Tensor "f32[2048][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_68,), kwargs = {})
#   %reciprocal_30 : Tensor "f32[2048][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_30,), kwargs = {})
#   %unsqueeze_242 : Tensor "f32[2048, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%reciprocal_30, -1), kwargs = {})
#   %unsqueeze_243 : Tensor "f32[2048, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_242, -1), kwargs = {})
#   %mul_91 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %unsqueeze_243), kwargs = {})
#   %unsqueeze_244 : Tensor "f32[2048, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg154_1, -1), kwargs = {})
#   %unsqueeze_245 : Tensor "f32[2048, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_244, -1), kwargs = {})
#   %mul_92 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_91, %unsqueeze_245), kwargs = {})
#   %unsqueeze_246 : Tensor "f32[2048, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg155_1, -1), kwargs = {})
#   %unsqueeze_247 : Tensor "f32[2048, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_246, -1), kwargs = {})
#   %add_69 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_92, %unsqueeze_247), kwargs = {})
#   %add_70 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_69, %relu_24), kwargs = {})
#   %relu_27 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_70,), kwargs = {})
#   return %relu_27
# SchedulerNodes: [SchedulerNode(name='op95')]

triton_poi_fused__native_batch_norm_legit_no_t_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_t_16', '''
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
    size_hints={'y0': 2048, 'x1': 196}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_t_16', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 2048), ('x1', 196)), 'low_dims': {0, 1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_t_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp0 = tl.load(in_out_ptr0 + (x1 + 196*y0), y0_mask)
        tmp1 = tl.load(in_ptr0 + (y0), y0_mask)
        tmp3 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp10 = tl.load(in_ptr2 + (y0), y0_mask)
        tmp12 = tl.load(in_ptr3 + (y0), y0_mask)
        tmp14 = tl.load(in_ptr4 + (y0 + 2048*x1), y0_mask)
        tmp2 = tmp0 - tmp1
        tmp4 = tl.full([1, 1], 1e-05, tl.float32)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.sqrt_rn(tmp5)
        tmp7 = tl.full([1, 1], 1.0, tl.float32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = tmp2 * tmp8
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tl.full([1, 1], 0, tl.int32)
        tmp17 = triton_helpers.maximum(tmp16, tmp15)
        tl.store(in_out_ptr0 + (x1 + 196*y0), tmp17, y0_mask)
''', device_str='npu')


# kernel path: /remote/PTA_213/combo-master-analysis-20261113/wrn-fresh/tmp2r1ign12/qq/cqqq4r4r7fhpsbaigto7yzvz7ce6h32qnfga3pzuv2ti5ofw22dk.py
# Topologically Sorted Source Nodes: [out_29, add_9, input_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_9 => add_77
#   input_16 => relu_30
#   out_29 => add_75, add_76, mul_100, mul_101, reciprocal_33, sqrt_33, sub_33, unsqueeze_264, unsqueeze_265, unsqueeze_266, unsqueeze_267, unsqueeze_268, unsqueeze_269, unsqueeze_270, unsqueeze_271
# Graph fragment:
#   %convolution_33 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0" = PlaceHolder[target=convolution_33]
#   %arg167_1 : Tensor "f32[2048][1]npu:0" = PlaceHolder[target=arg167_1]
#   %arg168_1 : Tensor "f32[2048][1]npu:0" = PlaceHolder[target=arg168_1]
#   %arg169_1 : Tensor "f32[2048][1]npu:0" = PlaceHolder[target=arg169_1]
#   %arg170_1 : Tensor "f32[2048][1]npu:0" = PlaceHolder[target=arg170_1]
#   %relu_27 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0" = PlaceHolder[target=relu_27]
#   %unsqueeze_264 : Tensor "f32[2048, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg167_1, -1), kwargs = {})
#   %unsqueeze_265 : Tensor "f32[2048, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_264, -1), kwargs = {})
#   %sub_33 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_33, %unsqueeze_265), kwargs = {})
#   %add_75 : Tensor "f32[2048][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg168_1, 1e-05), kwargs = {})
#   %sqrt_33 : Tensor "f32[2048][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_75,), kwargs = {})
#   %reciprocal_33 : Tensor "f32[2048][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_33,), kwargs = {})
#   %unsqueeze_266 : Tensor "f32[2048, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%reciprocal_33, -1), kwargs = {})
#   %unsqueeze_267 : Tensor "f32[2048, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_266, -1), kwargs = {})
#   %mul_100 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %unsqueeze_267), kwargs = {})
#   %unsqueeze_268 : Tensor "f32[2048, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg169_1, -1), kwargs = {})
#   %unsqueeze_269 : Tensor "f32[2048, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_268, -1), kwargs = {})
#   %mul_101 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_100, %unsqueeze_269), kwargs = {})
#   %unsqueeze_270 : Tensor "f32[2048, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg170_1, -1), kwargs = {})
#   %unsqueeze_271 : Tensor "f32[2048, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_270, -1), kwargs = {})
#   %add_76 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_101, %unsqueeze_271), kwargs = {})
#   %add_77 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_76, %relu_27), kwargs = {})
#   %relu_30 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_77,), kwargs = {})
#   return %relu_30
# SchedulerNodes: [SchedulerNode(name='op104')]

triton_poi_fused__native_batch_norm_legit_no_t_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_t_17', '''
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
    size_hints={'y0': 2048, 'x1': 196}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_t_17', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 2048), ('x1', 196)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_t_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
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
        tmp0 = tl.load(in_out_ptr0 + (x1 + 196*y0), y0_mask)
        tmp1 = tl.load(in_ptr0 + (y0), y0_mask)
        tmp3 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp10 = tl.load(in_ptr2 + (y0), y0_mask)
        tmp12 = tl.load(in_ptr3 + (y0), y0_mask)
        tmp14 = tl.load(in_ptr4 + (x1 + 196*y0), y0_mask)
        tmp2 = tmp0 - tmp1
        tmp4 = tl.full([1, 1], 1e-05, tl.float32)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.sqrt_rn(tmp5)
        tmp7 = tl.full([1, 1], 1.0, tl.float32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = tmp2 * tmp8
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tl.full([1, 1], 0, tl.int32)
        tmp17 = triton_helpers.maximum(tmp16, tmp15)
        tl.store(in_out_ptr0 + (x1 + 196*y0), tmp17, y0_mask)
''', device_str='npu')


# kernel path: /remote/PTA_213/combo-master-analysis-20261113/wrn-fresh/tmp2r1ign12/l6/cl67aajc6smog4zy7zyzschwfwp3gnrvdzyi4aapp5yus4p4z7o3.py
# Topologically Sorted Source Nodes: [out_80, add_26, input_33, adaptive_avg_pool2d], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   adaptive_avg_pool2d => mean
#   add_26 => add_196
#   input_33 => relu_81
#   out_80 => add_194, add_195, mul_253, mul_254, reciprocal_84, sqrt_84, sub_84, unsqueeze_672, unsqueeze_673, unsqueeze_674, unsqueeze_675, unsqueeze_676, unsqueeze_677, unsqueeze_678, unsqueeze_679
# Graph fragment:
#   %convolution_84 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0" = PlaceHolder[target=convolution_84]
#   %arg422_1 : Tensor "f32[2048][1]npu:0" = PlaceHolder[target=arg422_1]
#   %arg423_1 : Tensor "f32[2048][1]npu:0" = PlaceHolder[target=arg423_1]
#   %arg424_1 : Tensor "f32[2048][1]npu:0" = PlaceHolder[target=arg424_1]
#   %arg425_1 : Tensor "f32[2048][1]npu:0" = PlaceHolder[target=arg425_1]
#   %relu_78 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0" = PlaceHolder[target=relu_78]
#   %buf257 : Tensor "f32[1, 2048, 1, 1][2048, 1, 2048, 2048]npu:0" = PlaceHolder[target=buf257]
#   %unsqueeze_672 : Tensor "f32[2048, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg422_1, -1), kwargs = {})
#   %unsqueeze_673 : Tensor "f32[2048, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_672, -1), kwargs = {})
#   %sub_84 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_84, %unsqueeze_673), kwargs = {})
#   %add_194 : Tensor "f32[2048][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg423_1, 1e-05), kwargs = {})
#   %sqrt_84 : Tensor "f32[2048][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_194,), kwargs = {})
#   %reciprocal_84 : Tensor "f32[2048][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_84,), kwargs = {})
#   %unsqueeze_674 : Tensor "f32[2048, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%reciprocal_84, -1), kwargs = {})
#   %unsqueeze_675 : Tensor "f32[2048, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_674, -1), kwargs = {})
#   %mul_253 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_84, %unsqueeze_675), kwargs = {})
#   %unsqueeze_676 : Tensor "f32[2048, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg424_1, -1), kwargs = {})
#   %unsqueeze_677 : Tensor "f32[2048, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_676, -1), kwargs = {})
#   %mul_254 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_253, %unsqueeze_677), kwargs = {})
#   %unsqueeze_678 : Tensor "f32[2048, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg425_1, -1), kwargs = {})
#   %unsqueeze_679 : Tensor "f32[2048, 1, 1][1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_678, -1), kwargs = {})
#   %add_195 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_254, %unsqueeze_679), kwargs = {})
#   %add_196 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_195, %relu_78), kwargs = {})
#   %relu_81 : Tensor "f32[1, 2048, 14, 14][401408, 196, 14, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_196,), kwargs = {})
#   %mean : Tensor "f32[1, 2048, 1, 1][2048, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_81, [-1, -2], True), kwargs = {})
#   return %buf257,%mean
# SchedulerNodes: [SchedulerNode(name='op257'), SchedulerNode(name='op258')]

triton_per_fused__native_batch_norm_legit_no_t_18 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_t_18', '''
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
    size_hints={'x0': 2048, 'r1': 196},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_t_18', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 2048), ('r1', 196)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_t_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 196
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
        tmp0 = tl.load(in_ptr0 + (r1 + 196*x0), x0_mask, other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp3 = tl.load(in_ptr2 + (x0), x0_mask)
        tmp10 = tl.load(in_ptr3 + (x0), x0_mask)
        tmp12 = tl.load(in_ptr4 + (x0), x0_mask)
        tmp14 = tl.load(in_ptr5 + (r1 + 196*x0), x0_mask, other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = tl.full([1, 1], 1e-05, tl.float32)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.sqrt_rn(tmp5)
        tmp7 = tl.full([1, 1], 1.0, tl.float32)
        tmp8 = (tmp7 / tmp6)
        tmp9 = tmp2 * tmp8
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tl.full([1, 1], 0, tl.int32)
        tmp17 = triton_helpers.maximum(tmp16, tmp15)
        tmp19 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp17, 0)
        tmp20 = tl.sum(tmp19, 1).reshape(X0BLOCK_SUB, 1)
        tmp21 = tl.full([1, 1], 196.0, tl.float32)
        tmp22 = (tmp20 / tmp21)
        tl.store(in_out_ptr0 + (x0), tmp22, x0_mask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg1_1 = copy_if_misaligned(arg1_1)
            # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
            buf0 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)
            del arg0_1
            del arg1_1
            buf1 = buf0
            assert_size_stride(buf1, (1, 64, 112, 112), (802816, 12544, 112, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf1, 16, 'torch.ops.aten.convolution.default')
            del buf0
            buf2 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [batch_norm, relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_0.run(buf2, arg2_1, arg3_1, arg4_1, arg5_1, 64, 12544, stream=raw_stream0)
            del arg2_1
            del arg3_1
            del arg4_1
            del arg5_1
            # Topologically Sorted Source Nodes: [batch_norm, relu, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
            buf3 = torch.ops.aten.max_pool2d_with_indices.default(buf2, [3, 3], [2, 2], [1, 1])
            del buf2
            buf4 = buf3[0]
            assert_size_stride(buf4, (1, 64, 56, 56), (200704, 3136, 56, 1), 'torch.ops.aten.max_pool2d_with_indices.default')
            assert_alignment(buf4, 16, 'torch.ops.aten.max_pool2d_with_indices.default')
            del buf3
            # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
            buf6 = torch.ops.aten.convolution.default(buf4, arg6_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg6_1
            buf7 = buf6
            assert_size_stride(buf7, (1, 128, 56, 56), (401408, 3136, 56, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf7, 16, 'torch.ops.aten.convolution.default')
            del buf6
            buf8 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_1, out], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_1.run(buf8, arg7_1, arg8_1, arg9_1, arg10_1, 128, 3136, stream=raw_stream0)
            del arg10_1
            del arg7_1
            del arg8_1
            del arg9_1
            # Topologically Sorted Source Nodes: [batch_norm_1, out, conv2d_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf9 = torch.ops.aten.convolution.default(buf8, arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg11_1
            del buf8
            buf10 = buf9
            assert_size_stride(buf10, (1, 128, 56, 56), (401408, 3136, 56, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf10, 16, 'torch.ops.aten.convolution.default')
            del buf9
            buf11 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_2, out_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_1.run(buf11, arg12_1, arg13_1, arg14_1, arg15_1, 128, 3136, stream=raw_stream0)
            del arg12_1
            del arg13_1
            del arg14_1
            del arg15_1
            # Topologically Sorted Source Nodes: [batch_norm_2, out_1, conv2d_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf12 = torch.ops.aten.convolution.default(buf11, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg16_1
            del buf11
            buf13 = buf12
            assert_size_stride(buf13, (1, 512, 56, 56), (1605632, 3136, 56, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf13, 16, 'torch.ops.aten.convolution.default')
            del buf12
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            buf14 = torch.ops.aten.convolution.default(buf4, arg21_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg21_1
            del buf4
            buf15 = buf14
            assert_size_stride(buf15, (1, 512, 56, 56), (1605632, 3136, 56, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf15, 16, 'torch.ops.aten.convolution.default')
            del buf14
            buf16 = empty_strided_npu((1, 512, 56, 56), (1605632, 1, 28672, 512), torch.float32)
            # Topologically Sorted Source Nodes: [out_2, input_2, add], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_2.run(buf13, arg17_1, arg18_1, arg19_1, arg20_1, buf15, arg22_1, arg23_1, arg24_1, arg25_1, buf16, 512, 56, 56, stream=raw_stream0)
            del arg17_1
            del arg18_1
            del arg19_1
            del arg20_1
            del arg22_1
            del arg23_1
            del arg24_1
            del arg25_1
            del buf13
            del buf15
            buf17 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_relu_3.run(buf17, 56, 56, 512, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_3, conv2d_5], Original ATen: [aten.relu, aten.convolution]
            buf18 = torch.ops.aten.convolution.default(buf17, arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg26_1
            buf19 = buf18
            assert_size_stride(buf19, (1, 128, 56, 56), (401408, 3136, 56, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf19, 16, 'torch.ops.aten.convolution.default')
            del buf18
            buf20 = buf19; del buf19  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_5, out_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_1.run(buf20, arg27_1, arg28_1, arg29_1, arg30_1, 128, 3136, stream=raw_stream0)
            del arg27_1
            del arg28_1
            del arg29_1
            del arg30_1
            # Topologically Sorted Source Nodes: [batch_norm_5, out_3, conv2d_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf21 = torch.ops.aten.convolution.default(buf20, arg31_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg31_1
            del buf20
            buf22 = buf21
            assert_size_stride(buf22, (1, 128, 56, 56), (401408, 3136, 56, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf22, 16, 'torch.ops.aten.convolution.default')
            del buf21
            buf23 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_6, out_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_1.run(buf23, arg32_1, arg33_1, arg34_1, arg35_1, 128, 3136, stream=raw_stream0)
            del arg32_1
            del arg33_1
            del arg34_1
            del arg35_1
            # Topologically Sorted Source Nodes: [batch_norm_6, out_4, conv2d_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf24 = torch.ops.aten.convolution.default(buf23, arg36_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg36_1
            del buf23
            buf25 = buf24
            assert_size_stride(buf25, (1, 512, 56, 56), (1605632, 3136, 56, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf25, 16, 'torch.ops.aten.convolution.default')
            del buf24
            buf26 = buf25; del buf25  # reuse
            # Topologically Sorted Source Nodes: [out_5, add_1, input_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_4.run(buf26, arg37_1, arg38_1, arg39_1, arg40_1, buf17, 512, 3136, stream=raw_stream0)
            del arg37_1
            del arg38_1
            del arg39_1
            del arg40_1
            del buf17
            # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
            buf27 = torch.ops.aten.convolution.default(buf26, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg41_1
            buf28 = buf27
            assert_size_stride(buf28, (1, 128, 56, 56), (401408, 3136, 56, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf28, 16, 'torch.ops.aten.convolution.default')
            del buf27
            buf29 = buf28; del buf28  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_8, out_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_1.run(buf29, arg42_1, arg43_1, arg44_1, arg45_1, 128, 3136, stream=raw_stream0)
            del arg42_1
            del arg43_1
            del arg44_1
            del arg45_1
            # Topologically Sorted Source Nodes: [batch_norm_8, out_6, conv2d_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf30 = torch.ops.aten.convolution.default(buf29, arg46_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg46_1
            del buf29
            buf31 = buf30
            assert_size_stride(buf31, (1, 128, 56, 56), (401408, 3136, 56, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf31, 16, 'torch.ops.aten.convolution.default')
            del buf30
            buf32 = buf31; del buf31  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_9, out_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_1.run(buf32, arg47_1, arg48_1, arg49_1, arg50_1, 128, 3136, stream=raw_stream0)
            del arg47_1
            del arg48_1
            del arg49_1
            del arg50_1
            # Topologically Sorted Source Nodes: [batch_norm_9, out_7, conv2d_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf33 = torch.ops.aten.convolution.default(buf32, arg51_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg51_1
            del buf32
            buf34 = buf33
            assert_size_stride(buf34, (1, 512, 56, 56), (1605632, 3136, 56, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf34, 16, 'torch.ops.aten.convolution.default')
            del buf33
            buf35 = buf34; del buf34  # reuse
            # Topologically Sorted Source Nodes: [out_8, add_2, input_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_5.run(buf35, arg52_1, arg53_1, arg54_1, arg55_1, buf26, 512, 3136, stream=raw_stream0)
            del arg52_1
            del arg53_1
            del arg54_1
            del arg55_1
            del buf26
            # Topologically Sorted Source Nodes: [conv2d_11], Original ATen: [aten.convolution]
            buf36 = torch.ops.aten.convolution.default(buf35, arg56_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg56_1
            buf37 = buf36
            assert_size_stride(buf37, (1, 256, 56, 56), (802816, 3136, 56, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf37, 16, 'torch.ops.aten.convolution.default')
            del buf36
            buf38 = buf37; del buf37  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_11, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_6.run(buf38, arg57_1, arg58_1, arg59_1, arg60_1, 256, 3136, stream=raw_stream0)
            del arg57_1
            del arg58_1
            del arg59_1
            del arg60_1
            # Topologically Sorted Source Nodes: [batch_norm_11, out_9, conv2d_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf39 = torch.ops.aten.convolution.default(buf38, arg61_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
            del arg61_1
            buf40 = buf39
            assert_size_stride(buf40, (1, 256, 28, 28), (200704, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf40, 16, 'torch.ops.aten.convolution.default')
            del buf39
            buf41 = buf40; del buf40  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_12, out_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_7.run(buf41, arg62_1, arg63_1, arg64_1, arg65_1, 256, 784, stream=raw_stream0)
            del arg62_1
            del arg63_1
            del arg64_1
            del arg65_1
            # Topologically Sorted Source Nodes: [batch_norm_12, out_10, conv2d_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf42 = torch.ops.aten.convolution.default(buf41, arg66_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg66_1
            del buf41
            buf43 = buf42
            assert_size_stride(buf43, (1, 1024, 28, 28), (802816, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf43, 16, 'torch.ops.aten.convolution.default')
            del buf42
            # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
            buf44 = torch.ops.aten.convolution.default(buf35, arg71_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
            del arg71_1
            del buf35
            buf45 = buf44
            assert_size_stride(buf45, (1, 1024, 28, 28), (802816, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf45, 16, 'torch.ops.aten.convolution.default')
            del buf44
            buf46 = reinterpret_tensor(buf38, (1, 1024, 28, 28), (802816, 1, 28672, 1024), 0); del buf38  # reuse
            # Topologically Sorted Source Nodes: [out_11, input_7, add_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_8.run(buf43, arg67_1, arg68_1, arg69_1, arg70_1, buf45, arg72_1, arg73_1, arg74_1, arg75_1, buf46, 1024, 28, 28, stream=raw_stream0)
            del arg67_1
            del arg68_1
            del arg69_1
            del arg70_1
            del arg72_1
            del arg73_1
            del arg74_1
            del arg75_1
            del buf43
            del buf45
            buf47 = buf46; del buf46  # reuse
            # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_relu_9.run(buf47, 28, 28, 1024, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_8, conv2d_15], Original ATen: [aten.relu, aten.convolution]
            buf48 = torch.ops.aten.convolution.default(buf47, arg76_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg76_1
            buf49 = buf48
            assert_size_stride(buf49, (1, 256, 28, 28), (200704, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf49, 16, 'torch.ops.aten.convolution.default')
            del buf48
            buf50 = buf49; del buf49  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_15, out_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_7.run(buf50, arg77_1, arg78_1, arg79_1, arg80_1, 256, 784, stream=raw_stream0)
            del arg77_1
            del arg78_1
            del arg79_1
            del arg80_1
            # Topologically Sorted Source Nodes: [batch_norm_15, out_12, conv2d_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf51 = torch.ops.aten.convolution.default(buf50, arg81_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg81_1
            del buf50
            buf52 = buf51
            assert_size_stride(buf52, (1, 256, 28, 28), (200704, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf52, 16, 'torch.ops.aten.convolution.default')
            del buf51
            buf53 = buf52; del buf52  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_16, out_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_7.run(buf53, arg82_1, arg83_1, arg84_1, arg85_1, 256, 784, stream=raw_stream0)
            del arg82_1
            del arg83_1
            del arg84_1
            del arg85_1
            # Topologically Sorted Source Nodes: [batch_norm_16, out_13, conv2d_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf54 = torch.ops.aten.convolution.default(buf53, arg86_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg86_1
            del buf53
            buf55 = buf54
            assert_size_stride(buf55, (1, 1024, 28, 28), (802816, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf55, 16, 'torch.ops.aten.convolution.default')
            del buf54
            buf56 = buf55; del buf55  # reuse
            # Topologically Sorted Source Nodes: [out_14, add_4, input_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_10.run(buf56, arg87_1, arg88_1, arg89_1, arg90_1, buf47, 1024, 784, stream=raw_stream0)
            del arg87_1
            del arg88_1
            del arg89_1
            del arg90_1
            del buf47
            # Topologically Sorted Source Nodes: [conv2d_18], Original ATen: [aten.convolution]
            buf57 = torch.ops.aten.convolution.default(buf56, arg91_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg91_1
            buf58 = buf57
            assert_size_stride(buf58, (1, 256, 28, 28), (200704, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf58, 16, 'torch.ops.aten.convolution.default')
            del buf57
            buf59 = buf58; del buf58  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_18, out_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_7.run(buf59, arg92_1, arg93_1, arg94_1, arg95_1, 256, 784, stream=raw_stream0)
            del arg92_1
            del arg93_1
            del arg94_1
            del arg95_1
            # Topologically Sorted Source Nodes: [batch_norm_18, out_15, conv2d_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf60 = torch.ops.aten.convolution.default(buf59, arg96_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg96_1
            del buf59
            buf61 = buf60
            assert_size_stride(buf61, (1, 256, 28, 28), (200704, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf61, 16, 'torch.ops.aten.convolution.default')
            del buf60
            buf62 = buf61; del buf61  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_19, out_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_7.run(buf62, arg97_1, arg98_1, arg99_1, arg100_1, 256, 784, stream=raw_stream0)
            del arg100_1
            del arg97_1
            del arg98_1
            del arg99_1
            # Topologically Sorted Source Nodes: [batch_norm_19, out_16, conv2d_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf63 = torch.ops.aten.convolution.default(buf62, arg101_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg101_1
            del buf62
            buf64 = buf63
            assert_size_stride(buf64, (1, 1024, 28, 28), (802816, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf64, 16, 'torch.ops.aten.convolution.default')
            del buf63
            buf65 = buf64; del buf64  # reuse
            # Topologically Sorted Source Nodes: [out_17, add_5, input_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_11.run(buf65, arg102_1, arg103_1, arg104_1, arg105_1, buf56, 1024, 784, stream=raw_stream0)
            del arg102_1
            del arg103_1
            del arg104_1
            del arg105_1
            del buf56
            # Topologically Sorted Source Nodes: [conv2d_21], Original ATen: [aten.convolution]
            buf66 = torch.ops.aten.convolution.default(buf65, arg106_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg106_1
            buf67 = buf66
            assert_size_stride(buf67, (1, 256, 28, 28), (200704, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf67, 16, 'torch.ops.aten.convolution.default')
            del buf66
            buf68 = buf67; del buf67  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_21, out_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_7.run(buf68, arg107_1, arg108_1, arg109_1, arg110_1, 256, 784, stream=raw_stream0)
            del arg107_1
            del arg108_1
            del arg109_1
            del arg110_1
            # Topologically Sorted Source Nodes: [batch_norm_21, out_18, conv2d_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf69 = torch.ops.aten.convolution.default(buf68, arg111_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg111_1
            del buf68
            buf70 = buf69
            assert_size_stride(buf70, (1, 256, 28, 28), (200704, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf70, 16, 'torch.ops.aten.convolution.default')
            del buf69
            buf71 = buf70; del buf70  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_22, out_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_7.run(buf71, arg112_1, arg113_1, arg114_1, arg115_1, 256, 784, stream=raw_stream0)
            del arg112_1
            del arg113_1
            del arg114_1
            del arg115_1
            # Topologically Sorted Source Nodes: [batch_norm_22, out_19, conv2d_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf72 = torch.ops.aten.convolution.default(buf71, arg116_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg116_1
            del buf71
            buf73 = buf72
            assert_size_stride(buf73, (1, 1024, 28, 28), (802816, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf73, 16, 'torch.ops.aten.convolution.default')
            del buf72
            buf74 = buf73; del buf73  # reuse
            # Topologically Sorted Source Nodes: [out_20, add_6, input_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_11.run(buf74, arg117_1, arg118_1, arg119_1, arg120_1, buf65, 1024, 784, stream=raw_stream0)
            del arg117_1
            del arg118_1
            del arg119_1
            del arg120_1
            del buf65
            # Topologically Sorted Source Nodes: [conv2d_24], Original ATen: [aten.convolution]
            buf75 = torch.ops.aten.convolution.default(buf74, arg121_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg121_1
            buf76 = buf75
            assert_size_stride(buf76, (1, 512, 28, 28), (401408, 784, 28, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf76, 16, 'torch.ops.aten.convolution.default')
            del buf75
            buf77 = buf76; del buf76  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_24, out_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_12.run(buf77, arg122_1, arg123_1, arg124_1, arg125_1, 512, 784, stream=raw_stream0)
            del arg122_1
            del arg123_1
            del arg124_1
            del arg125_1
            # Topologically Sorted Source Nodes: [batch_norm_24, out_21, conv2d_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf78 = torch.ops.aten.convolution.default(buf77, arg126_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
            del arg126_1
            buf79 = buf78
            assert_size_stride(buf79, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf79, 16, 'torch.ops.aten.convolution.default')
            del buf78
            buf80 = buf79; del buf79  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_25, out_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf80, arg127_1, arg128_1, arg129_1, arg130_1, 512, 196, stream=raw_stream0)
            del arg127_1
            del arg128_1
            del arg129_1
            del arg130_1
            # Topologically Sorted Source Nodes: [batch_norm_25, out_22, conv2d_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf81 = torch.ops.aten.convolution.default(buf80, arg131_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg131_1
            del buf80
            buf82 = buf81
            assert_size_stride(buf82, (1, 2048, 14, 14), (401408, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf82, 16, 'torch.ops.aten.convolution.default')
            del buf81
            # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
            buf83 = torch.ops.aten.convolution.default(buf74, arg136_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
            del arg136_1
            del buf74
            buf84 = buf83
            assert_size_stride(buf84, (1, 2048, 14, 14), (401408, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf84, 16, 'torch.ops.aten.convolution.default')
            del buf83
            buf85 = reinterpret_tensor(buf77, (1, 2048, 14, 14), (401408, 1, 28672, 2048), 0); del buf77  # reuse
            # Topologically Sorted Source Nodes: [out_23, input_13, add_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_14.run(buf82, arg132_1, arg133_1, arg134_1, arg135_1, buf84, arg137_1, arg138_1, arg139_1, arg140_1, buf85, 2048, 14, 14, stream=raw_stream0)
            del arg132_1
            del arg133_1
            del arg134_1
            del arg135_1
            del arg137_1
            del arg138_1
            del arg139_1
            del arg140_1
            del buf82
            del buf84
            buf86 = buf85; del buf85  # reuse
            # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_relu_15.run(buf86, 14, 14, 2048, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_14, conv2d_28], Original ATen: [aten.relu, aten.convolution]
            buf87 = torch.ops.aten.convolution.default(buf86, arg141_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg141_1
            buf88 = buf87
            assert_size_stride(buf88, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf88, 16, 'torch.ops.aten.convolution.default')
            del buf87
            buf89 = buf88; del buf88  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_28, out_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf89, arg142_1, arg143_1, arg144_1, arg145_1, 512, 196, stream=raw_stream0)
            del arg142_1
            del arg143_1
            del arg144_1
            del arg145_1
            # Topologically Sorted Source Nodes: [batch_norm_28, out_24, conv2d_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf90 = torch.ops.aten.convolution.default(buf89, arg146_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg146_1
            del buf89
            buf91 = buf90
            assert_size_stride(buf91, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf91, 16, 'torch.ops.aten.convolution.default')
            del buf90
            buf92 = buf91; del buf91  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_29, out_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf92, arg147_1, arg148_1, arg149_1, arg150_1, 512, 196, stream=raw_stream0)
            del arg147_1
            del arg148_1
            del arg149_1
            del arg150_1
            # Topologically Sorted Source Nodes: [batch_norm_29, out_25, conv2d_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf93 = torch.ops.aten.convolution.default(buf92, arg151_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg151_1
            del buf92
            buf94 = buf93
            assert_size_stride(buf94, (1, 2048, 14, 14), (401408, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf94, 16, 'torch.ops.aten.convolution.default')
            del buf93
            buf95 = buf94; del buf94  # reuse
            # Topologically Sorted Source Nodes: [out_26, add_8, input_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_16.run(buf95, arg152_1, arg153_1, arg154_1, arg155_1, buf86, 2048, 196, stream=raw_stream0)
            del arg152_1
            del arg153_1
            del arg154_1
            del arg155_1
            del buf86
            # Topologically Sorted Source Nodes: [conv2d_31], Original ATen: [aten.convolution]
            buf96 = torch.ops.aten.convolution.default(buf95, arg156_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg156_1
            buf97 = buf96
            assert_size_stride(buf97, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf97, 16, 'torch.ops.aten.convolution.default')
            del buf96
            buf98 = buf97; del buf97  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_31, out_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf98, arg157_1, arg158_1, arg159_1, arg160_1, 512, 196, stream=raw_stream0)
            del arg157_1
            del arg158_1
            del arg159_1
            del arg160_1
            # Topologically Sorted Source Nodes: [batch_norm_31, out_27, conv2d_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf99 = torch.ops.aten.convolution.default(buf98, arg161_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg161_1
            del buf98
            buf100 = buf99
            assert_size_stride(buf100, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf100, 16, 'torch.ops.aten.convolution.default')
            del buf99
            buf101 = buf100; del buf100  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_32, out_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf101, arg162_1, arg163_1, arg164_1, arg165_1, 512, 196, stream=raw_stream0)
            del arg162_1
            del arg163_1
            del arg164_1
            del arg165_1
            # Topologically Sorted Source Nodes: [batch_norm_32, out_28, conv2d_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf102 = torch.ops.aten.convolution.default(buf101, arg166_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg166_1
            del buf101
            buf103 = buf102
            assert_size_stride(buf103, (1, 2048, 14, 14), (401408, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf103, 16, 'torch.ops.aten.convolution.default')
            del buf102
            buf104 = buf103; del buf103  # reuse
            # Topologically Sorted Source Nodes: [out_29, add_9, input_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_17.run(buf104, arg167_1, arg168_1, arg169_1, arg170_1, buf95, 2048, 196, stream=raw_stream0)
            del arg167_1
            del arg168_1
            del arg169_1
            del arg170_1
            del buf95
            # Topologically Sorted Source Nodes: [conv2d_34], Original ATen: [aten.convolution]
            buf105 = torch.ops.aten.convolution.default(buf104, arg171_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg171_1
            buf106 = buf105
            assert_size_stride(buf106, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf106, 16, 'torch.ops.aten.convolution.default')
            del buf105
            buf107 = buf106; del buf106  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_34, out_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf107, arg172_1, arg173_1, arg174_1, arg175_1, 512, 196, stream=raw_stream0)
            del arg172_1
            del arg173_1
            del arg174_1
            del arg175_1
            # Topologically Sorted Source Nodes: [batch_norm_34, out_30, conv2d_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf108 = torch.ops.aten.convolution.default(buf107, arg176_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg176_1
            del buf107
            buf109 = buf108
            assert_size_stride(buf109, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf109, 16, 'torch.ops.aten.convolution.default')
            del buf108
            buf110 = buf109; del buf109  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_35, out_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf110, arg177_1, arg178_1, arg179_1, arg180_1, 512, 196, stream=raw_stream0)
            del arg177_1
            del arg178_1
            del arg179_1
            del arg180_1
            # Topologically Sorted Source Nodes: [batch_norm_35, out_31, conv2d_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf111 = torch.ops.aten.convolution.default(buf110, arg181_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg181_1
            del buf110
            buf112 = buf111
            assert_size_stride(buf112, (1, 2048, 14, 14), (401408, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf112, 16, 'torch.ops.aten.convolution.default')
            del buf111
            buf113 = buf112; del buf112  # reuse
            # Topologically Sorted Source Nodes: [out_32, add_10, input_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_17.run(buf113, arg182_1, arg183_1, arg184_1, arg185_1, buf104, 2048, 196, stream=raw_stream0)
            del arg182_1
            del arg183_1
            del arg184_1
            del arg185_1
            del buf104
            # Topologically Sorted Source Nodes: [conv2d_37], Original ATen: [aten.convolution]
            buf114 = torch.ops.aten.convolution.default(buf113, arg186_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg186_1
            buf115 = buf114
            assert_size_stride(buf115, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf115, 16, 'torch.ops.aten.convolution.default')
            del buf114
            buf116 = buf115; del buf115  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_37, out_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf116, arg187_1, arg188_1, arg189_1, arg190_1, 512, 196, stream=raw_stream0)
            del arg187_1
            del arg188_1
            del arg189_1
            del arg190_1
            # Topologically Sorted Source Nodes: [batch_norm_37, out_33, conv2d_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf117 = torch.ops.aten.convolution.default(buf116, arg191_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg191_1
            del buf116
            buf118 = buf117
            assert_size_stride(buf118, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf118, 16, 'torch.ops.aten.convolution.default')
            del buf117
            buf119 = buf118; del buf118  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_38, out_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf119, arg192_1, arg193_1, arg194_1, arg195_1, 512, 196, stream=raw_stream0)
            del arg192_1
            del arg193_1
            del arg194_1
            del arg195_1
            # Topologically Sorted Source Nodes: [batch_norm_38, out_34, conv2d_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf120 = torch.ops.aten.convolution.default(buf119, arg196_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg196_1
            del buf119
            buf121 = buf120
            assert_size_stride(buf121, (1, 2048, 14, 14), (401408, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf121, 16, 'torch.ops.aten.convolution.default')
            del buf120
            buf122 = buf121; del buf121  # reuse
            # Topologically Sorted Source Nodes: [out_35, add_11, input_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_17.run(buf122, arg197_1, arg198_1, arg199_1, arg200_1, buf113, 2048, 196, stream=raw_stream0)
            del arg197_1
            del arg198_1
            del arg199_1
            del arg200_1
            del buf113
            # Topologically Sorted Source Nodes: [conv2d_40], Original ATen: [aten.convolution]
            buf123 = torch.ops.aten.convolution.default(buf122, arg201_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg201_1
            buf124 = buf123
            assert_size_stride(buf124, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf124, 16, 'torch.ops.aten.convolution.default')
            del buf123
            buf125 = buf124; del buf124  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_40, out_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf125, arg202_1, arg203_1, arg204_1, arg205_1, 512, 196, stream=raw_stream0)
            del arg202_1
            del arg203_1
            del arg204_1
            del arg205_1
            # Topologically Sorted Source Nodes: [batch_norm_40, out_36, conv2d_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf126 = torch.ops.aten.convolution.default(buf125, arg206_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg206_1
            del buf125
            buf127 = buf126
            assert_size_stride(buf127, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf127, 16, 'torch.ops.aten.convolution.default')
            del buf126
            buf128 = buf127; del buf127  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_41, out_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf128, arg207_1, arg208_1, arg209_1, arg210_1, 512, 196, stream=raw_stream0)
            del arg207_1
            del arg208_1
            del arg209_1
            del arg210_1
            # Topologically Sorted Source Nodes: [batch_norm_41, out_37, conv2d_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf129 = torch.ops.aten.convolution.default(buf128, arg211_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg211_1
            del buf128
            buf130 = buf129
            assert_size_stride(buf130, (1, 2048, 14, 14), (401408, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf130, 16, 'torch.ops.aten.convolution.default')
            del buf129
            buf131 = buf130; del buf130  # reuse
            # Topologically Sorted Source Nodes: [out_38, add_12, input_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_17.run(buf131, arg212_1, arg213_1, arg214_1, arg215_1, buf122, 2048, 196, stream=raw_stream0)
            del arg212_1
            del arg213_1
            del arg214_1
            del arg215_1
            del buf122
            # Topologically Sorted Source Nodes: [conv2d_43], Original ATen: [aten.convolution]
            buf132 = torch.ops.aten.convolution.default(buf131, arg216_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg216_1
            buf133 = buf132
            assert_size_stride(buf133, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf133, 16, 'torch.ops.aten.convolution.default')
            del buf132
            buf134 = buf133; del buf133  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_43, out_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf134, arg217_1, arg218_1, arg219_1, arg220_1, 512, 196, stream=raw_stream0)
            del arg217_1
            del arg218_1
            del arg219_1
            del arg220_1
            # Topologically Sorted Source Nodes: [batch_norm_43, out_39, conv2d_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf135 = torch.ops.aten.convolution.default(buf134, arg221_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg221_1
            del buf134
            buf136 = buf135
            assert_size_stride(buf136, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf136, 16, 'torch.ops.aten.convolution.default')
            del buf135
            buf137 = buf136; del buf136  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_44, out_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf137, arg222_1, arg223_1, arg224_1, arg225_1, 512, 196, stream=raw_stream0)
            del arg222_1
            del arg223_1
            del arg224_1
            del arg225_1
            # Topologically Sorted Source Nodes: [batch_norm_44, out_40, conv2d_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf138 = torch.ops.aten.convolution.default(buf137, arg226_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg226_1
            del buf137
            buf139 = buf138
            assert_size_stride(buf139, (1, 2048, 14, 14), (401408, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf139, 16, 'torch.ops.aten.convolution.default')
            del buf138
            buf140 = buf139; del buf139  # reuse
            # Topologically Sorted Source Nodes: [out_41, add_13, input_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_17.run(buf140, arg227_1, arg228_1, arg229_1, arg230_1, buf131, 2048, 196, stream=raw_stream0)
            del arg227_1
            del arg228_1
            del arg229_1
            del arg230_1
            del buf131
            # Topologically Sorted Source Nodes: [conv2d_46], Original ATen: [aten.convolution]
            buf141 = torch.ops.aten.convolution.default(buf140, arg231_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg231_1
            buf142 = buf141
            assert_size_stride(buf142, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf142, 16, 'torch.ops.aten.convolution.default')
            del buf141
            buf143 = buf142; del buf142  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_46, out_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf143, arg232_1, arg233_1, arg234_1, arg235_1, 512, 196, stream=raw_stream0)
            del arg232_1
            del arg233_1
            del arg234_1
            del arg235_1
            # Topologically Sorted Source Nodes: [batch_norm_46, out_42, conv2d_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf144 = torch.ops.aten.convolution.default(buf143, arg236_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg236_1
            del buf143
            buf145 = buf144
            assert_size_stride(buf145, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf145, 16, 'torch.ops.aten.convolution.default')
            del buf144
            buf146 = buf145; del buf145  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_47, out_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf146, arg237_1, arg238_1, arg239_1, arg240_1, 512, 196, stream=raw_stream0)
            del arg237_1
            del arg238_1
            del arg239_1
            del arg240_1
            # Topologically Sorted Source Nodes: [batch_norm_47, out_43, conv2d_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf147 = torch.ops.aten.convolution.default(buf146, arg241_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg241_1
            del buf146
            buf148 = buf147
            assert_size_stride(buf148, (1, 2048, 14, 14), (401408, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf148, 16, 'torch.ops.aten.convolution.default')
            del buf147
            buf149 = buf148; del buf148  # reuse
            # Topologically Sorted Source Nodes: [out_44, add_14, input_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_17.run(buf149, arg242_1, arg243_1, arg244_1, arg245_1, buf140, 2048, 196, stream=raw_stream0)
            del arg242_1
            del arg243_1
            del arg244_1
            del arg245_1
            del buf140
            # Topologically Sorted Source Nodes: [conv2d_49], Original ATen: [aten.convolution]
            buf150 = torch.ops.aten.convolution.default(buf149, arg246_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg246_1
            buf151 = buf150
            assert_size_stride(buf151, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf151, 16, 'torch.ops.aten.convolution.default')
            del buf150
            buf152 = buf151; del buf151  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_49, out_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf152, arg247_1, arg248_1, arg249_1, arg250_1, 512, 196, stream=raw_stream0)
            del arg247_1
            del arg248_1
            del arg249_1
            del arg250_1
            # Topologically Sorted Source Nodes: [batch_norm_49, out_45, conv2d_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf153 = torch.ops.aten.convolution.default(buf152, arg251_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg251_1
            del buf152
            buf154 = buf153
            assert_size_stride(buf154, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf154, 16, 'torch.ops.aten.convolution.default')
            del buf153
            buf155 = buf154; del buf154  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_50, out_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf155, arg252_1, arg253_1, arg254_1, arg255_1, 512, 196, stream=raw_stream0)
            del arg252_1
            del arg253_1
            del arg254_1
            del arg255_1
            # Topologically Sorted Source Nodes: [batch_norm_50, out_46, conv2d_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf156 = torch.ops.aten.convolution.default(buf155, arg256_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg256_1
            del buf155
            buf157 = buf156
            assert_size_stride(buf157, (1, 2048, 14, 14), (401408, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf157, 16, 'torch.ops.aten.convolution.default')
            del buf156
            buf158 = buf157; del buf157  # reuse
            # Topologically Sorted Source Nodes: [out_47, add_15, input_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_17.run(buf158, arg257_1, arg258_1, arg259_1, arg260_1, buf149, 2048, 196, stream=raw_stream0)
            del arg257_1
            del arg258_1
            del arg259_1
            del arg260_1
            del buf149
            # Topologically Sorted Source Nodes: [conv2d_52], Original ATen: [aten.convolution]
            buf159 = torch.ops.aten.convolution.default(buf158, arg261_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg261_1
            buf160 = buf159
            assert_size_stride(buf160, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf160, 16, 'torch.ops.aten.convolution.default')
            del buf159
            buf161 = buf160; del buf160  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_52, out_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf161, arg262_1, arg263_1, arg264_1, arg265_1, 512, 196, stream=raw_stream0)
            del arg262_1
            del arg263_1
            del arg264_1
            del arg265_1
            # Topologically Sorted Source Nodes: [batch_norm_52, out_48, conv2d_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf162 = torch.ops.aten.convolution.default(buf161, arg266_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg266_1
            del buf161
            buf163 = buf162
            assert_size_stride(buf163, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf163, 16, 'torch.ops.aten.convolution.default')
            del buf162
            buf164 = buf163; del buf163  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_53, out_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf164, arg267_1, arg268_1, arg269_1, arg270_1, 512, 196, stream=raw_stream0)
            del arg267_1
            del arg268_1
            del arg269_1
            del arg270_1
            # Topologically Sorted Source Nodes: [batch_norm_53, out_49, conv2d_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf165 = torch.ops.aten.convolution.default(buf164, arg271_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg271_1
            del buf164
            buf166 = buf165
            assert_size_stride(buf166, (1, 2048, 14, 14), (401408, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf166, 16, 'torch.ops.aten.convolution.default')
            del buf165
            buf167 = buf166; del buf166  # reuse
            # Topologically Sorted Source Nodes: [out_50, add_16, input_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_17.run(buf167, arg272_1, arg273_1, arg274_1, arg275_1, buf158, 2048, 196, stream=raw_stream0)
            del arg272_1
            del arg273_1
            del arg274_1
            del arg275_1
            del buf158
            # Topologically Sorted Source Nodes: [conv2d_55], Original ATen: [aten.convolution]
            buf168 = torch.ops.aten.convolution.default(buf167, arg276_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg276_1
            buf169 = buf168
            assert_size_stride(buf169, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf169, 16, 'torch.ops.aten.convolution.default')
            del buf168
            buf170 = buf169; del buf169  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_55, out_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf170, arg277_1, arg278_1, arg279_1, arg280_1, 512, 196, stream=raw_stream0)
            del arg277_1
            del arg278_1
            del arg279_1
            del arg280_1
            # Topologically Sorted Source Nodes: [batch_norm_55, out_51, conv2d_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf171 = torch.ops.aten.convolution.default(buf170, arg281_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg281_1
            del buf170
            buf172 = buf171
            assert_size_stride(buf172, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf172, 16, 'torch.ops.aten.convolution.default')
            del buf171
            buf173 = buf172; del buf172  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_56, out_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf173, arg282_1, arg283_1, arg284_1, arg285_1, 512, 196, stream=raw_stream0)
            del arg282_1
            del arg283_1
            del arg284_1
            del arg285_1
            # Topologically Sorted Source Nodes: [batch_norm_56, out_52, conv2d_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf174 = torch.ops.aten.convolution.default(buf173, arg286_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg286_1
            del buf173
            buf175 = buf174
            assert_size_stride(buf175, (1, 2048, 14, 14), (401408, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf175, 16, 'torch.ops.aten.convolution.default')
            del buf174
            buf176 = buf175; del buf175  # reuse
            # Topologically Sorted Source Nodes: [out_53, add_17, input_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_17.run(buf176, arg287_1, arg288_1, arg289_1, arg290_1, buf167, 2048, 196, stream=raw_stream0)
            del arg287_1
            del arg288_1
            del arg289_1
            del arg290_1
            del buf167
            # Topologically Sorted Source Nodes: [conv2d_58], Original ATen: [aten.convolution]
            buf177 = torch.ops.aten.convolution.default(buf176, arg291_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg291_1
            buf178 = buf177
            assert_size_stride(buf178, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf178, 16, 'torch.ops.aten.convolution.default')
            del buf177
            buf179 = buf178; del buf178  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_58, out_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf179, arg292_1, arg293_1, arg294_1, arg295_1, 512, 196, stream=raw_stream0)
            del arg292_1
            del arg293_1
            del arg294_1
            del arg295_1
            # Topologically Sorted Source Nodes: [batch_norm_58, out_54, conv2d_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf180 = torch.ops.aten.convolution.default(buf179, arg296_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg296_1
            del buf179
            buf181 = buf180
            assert_size_stride(buf181, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf181, 16, 'torch.ops.aten.convolution.default')
            del buf180
            buf182 = buf181; del buf181  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_59, out_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf182, arg297_1, arg298_1, arg299_1, arg300_1, 512, 196, stream=raw_stream0)
            del arg297_1
            del arg298_1
            del arg299_1
            del arg300_1
            # Topologically Sorted Source Nodes: [batch_norm_59, out_55, conv2d_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf183 = torch.ops.aten.convolution.default(buf182, arg301_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg301_1
            del buf182
            buf184 = buf183
            assert_size_stride(buf184, (1, 2048, 14, 14), (401408, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf184, 16, 'torch.ops.aten.convolution.default')
            del buf183
            buf185 = buf184; del buf184  # reuse
            # Topologically Sorted Source Nodes: [out_56, add_18, input_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_17.run(buf185, arg302_1, arg303_1, arg304_1, arg305_1, buf176, 2048, 196, stream=raw_stream0)
            del arg302_1
            del arg303_1
            del arg304_1
            del arg305_1
            del buf176
            # Topologically Sorted Source Nodes: [conv2d_61], Original ATen: [aten.convolution]
            buf186 = torch.ops.aten.convolution.default(buf185, arg306_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg306_1
            buf187 = buf186
            assert_size_stride(buf187, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf187, 16, 'torch.ops.aten.convolution.default')
            del buf186
            buf188 = buf187; del buf187  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_61, out_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf188, arg307_1, arg308_1, arg309_1, arg310_1, 512, 196, stream=raw_stream0)
            del arg307_1
            del arg308_1
            del arg309_1
            del arg310_1
            # Topologically Sorted Source Nodes: [batch_norm_61, out_57, conv2d_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf189 = torch.ops.aten.convolution.default(buf188, arg311_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg311_1
            del buf188
            buf190 = buf189
            assert_size_stride(buf190, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf190, 16, 'torch.ops.aten.convolution.default')
            del buf189
            buf191 = buf190; del buf190  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_62, out_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf191, arg312_1, arg313_1, arg314_1, arg315_1, 512, 196, stream=raw_stream0)
            del arg312_1
            del arg313_1
            del arg314_1
            del arg315_1
            # Topologically Sorted Source Nodes: [batch_norm_62, out_58, conv2d_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf192 = torch.ops.aten.convolution.default(buf191, arg316_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg316_1
            del buf191
            buf193 = buf192
            assert_size_stride(buf193, (1, 2048, 14, 14), (401408, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf193, 16, 'torch.ops.aten.convolution.default')
            del buf192
            buf194 = buf193; del buf193  # reuse
            # Topologically Sorted Source Nodes: [out_59, add_19, input_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_17.run(buf194, arg317_1, arg318_1, arg319_1, arg320_1, buf185, 2048, 196, stream=raw_stream0)
            del arg317_1
            del arg318_1
            del arg319_1
            del arg320_1
            del buf185
            # Topologically Sorted Source Nodes: [conv2d_64], Original ATen: [aten.convolution]
            buf195 = torch.ops.aten.convolution.default(buf194, arg321_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg321_1
            buf196 = buf195
            assert_size_stride(buf196, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf196, 16, 'torch.ops.aten.convolution.default')
            del buf195
            buf197 = buf196; del buf196  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_64, out_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf197, arg322_1, arg323_1, arg324_1, arg325_1, 512, 196, stream=raw_stream0)
            del arg322_1
            del arg323_1
            del arg324_1
            del arg325_1
            # Topologically Sorted Source Nodes: [batch_norm_64, out_60, conv2d_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf198 = torch.ops.aten.convolution.default(buf197, arg326_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg326_1
            del buf197
            buf199 = buf198
            assert_size_stride(buf199, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf199, 16, 'torch.ops.aten.convolution.default')
            del buf198
            buf200 = buf199; del buf199  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_65, out_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf200, arg327_1, arg328_1, arg329_1, arg330_1, 512, 196, stream=raw_stream0)
            del arg327_1
            del arg328_1
            del arg329_1
            del arg330_1
            # Topologically Sorted Source Nodes: [batch_norm_65, out_61, conv2d_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf201 = torch.ops.aten.convolution.default(buf200, arg331_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg331_1
            del buf200
            buf202 = buf201
            assert_size_stride(buf202, (1, 2048, 14, 14), (401408, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf202, 16, 'torch.ops.aten.convolution.default')
            del buf201
            buf203 = buf202; del buf202  # reuse
            # Topologically Sorted Source Nodes: [out_62, add_20, input_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_17.run(buf203, arg332_1, arg333_1, arg334_1, arg335_1, buf194, 2048, 196, stream=raw_stream0)
            del arg332_1
            del arg333_1
            del arg334_1
            del arg335_1
            del buf194
            # Topologically Sorted Source Nodes: [conv2d_67], Original ATen: [aten.convolution]
            buf204 = torch.ops.aten.convolution.default(buf203, arg336_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg336_1
            buf205 = buf204
            assert_size_stride(buf205, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf205, 16, 'torch.ops.aten.convolution.default')
            del buf204
            buf206 = buf205; del buf205  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_67, out_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf206, arg337_1, arg338_1, arg339_1, arg340_1, 512, 196, stream=raw_stream0)
            del arg337_1
            del arg338_1
            del arg339_1
            del arg340_1
            # Topologically Sorted Source Nodes: [batch_norm_67, out_63, conv2d_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf207 = torch.ops.aten.convolution.default(buf206, arg341_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg341_1
            del buf206
            buf208 = buf207
            assert_size_stride(buf208, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf208, 16, 'torch.ops.aten.convolution.default')
            del buf207
            buf209 = buf208; del buf208  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_68, out_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf209, arg342_1, arg343_1, arg344_1, arg345_1, 512, 196, stream=raw_stream0)
            del arg342_1
            del arg343_1
            del arg344_1
            del arg345_1
            # Topologically Sorted Source Nodes: [batch_norm_68, out_64, conv2d_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf210 = torch.ops.aten.convolution.default(buf209, arg346_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg346_1
            del buf209
            buf211 = buf210
            assert_size_stride(buf211, (1, 2048, 14, 14), (401408, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf211, 16, 'torch.ops.aten.convolution.default')
            del buf210
            buf212 = buf211; del buf211  # reuse
            # Topologically Sorted Source Nodes: [out_65, add_21, input_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_17.run(buf212, arg347_1, arg348_1, arg349_1, arg350_1, buf203, 2048, 196, stream=raw_stream0)
            del arg347_1
            del arg348_1
            del arg349_1
            del arg350_1
            del buf203
            # Topologically Sorted Source Nodes: [conv2d_70], Original ATen: [aten.convolution]
            buf213 = torch.ops.aten.convolution.default(buf212, arg351_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg351_1
            buf214 = buf213
            assert_size_stride(buf214, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf214, 16, 'torch.ops.aten.convolution.default')
            del buf213
            buf215 = buf214; del buf214  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_70, out_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf215, arg352_1, arg353_1, arg354_1, arg355_1, 512, 196, stream=raw_stream0)
            del arg352_1
            del arg353_1
            del arg354_1
            del arg355_1
            # Topologically Sorted Source Nodes: [batch_norm_70, out_66, conv2d_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf216 = torch.ops.aten.convolution.default(buf215, arg356_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg356_1
            del buf215
            buf217 = buf216
            assert_size_stride(buf217, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf217, 16, 'torch.ops.aten.convolution.default')
            del buf216
            buf218 = buf217; del buf217  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_71, out_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf218, arg357_1, arg358_1, arg359_1, arg360_1, 512, 196, stream=raw_stream0)
            del arg357_1
            del arg358_1
            del arg359_1
            del arg360_1
            # Topologically Sorted Source Nodes: [batch_norm_71, out_67, conv2d_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf219 = torch.ops.aten.convolution.default(buf218, arg361_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg361_1
            del buf218
            buf220 = buf219
            assert_size_stride(buf220, (1, 2048, 14, 14), (401408, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf220, 16, 'torch.ops.aten.convolution.default')
            del buf219
            buf221 = buf220; del buf220  # reuse
            # Topologically Sorted Source Nodes: [out_68, add_22, input_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_17.run(buf221, arg362_1, arg363_1, arg364_1, arg365_1, buf212, 2048, 196, stream=raw_stream0)
            del arg362_1
            del arg363_1
            del arg364_1
            del arg365_1
            del buf212
            # Topologically Sorted Source Nodes: [conv2d_73], Original ATen: [aten.convolution]
            buf222 = torch.ops.aten.convolution.default(buf221, arg366_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg366_1
            buf223 = buf222
            assert_size_stride(buf223, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf223, 16, 'torch.ops.aten.convolution.default')
            del buf222
            buf224 = buf223; del buf223  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_73, out_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf224, arg367_1, arg368_1, arg369_1, arg370_1, 512, 196, stream=raw_stream0)
            del arg367_1
            del arg368_1
            del arg369_1
            del arg370_1
            # Topologically Sorted Source Nodes: [batch_norm_73, out_69, conv2d_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf225 = torch.ops.aten.convolution.default(buf224, arg371_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg371_1
            del buf224
            buf226 = buf225
            assert_size_stride(buf226, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf226, 16, 'torch.ops.aten.convolution.default')
            del buf225
            buf227 = buf226; del buf226  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_74, out_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf227, arg372_1, arg373_1, arg374_1, arg375_1, 512, 196, stream=raw_stream0)
            del arg372_1
            del arg373_1
            del arg374_1
            del arg375_1
            # Topologically Sorted Source Nodes: [batch_norm_74, out_70, conv2d_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf228 = torch.ops.aten.convolution.default(buf227, arg376_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg376_1
            del buf227
            buf229 = buf228
            assert_size_stride(buf229, (1, 2048, 14, 14), (401408, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf229, 16, 'torch.ops.aten.convolution.default')
            del buf228
            buf230 = buf229; del buf229  # reuse
            # Topologically Sorted Source Nodes: [out_71, add_23, input_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_17.run(buf230, arg377_1, arg378_1, arg379_1, arg380_1, buf221, 2048, 196, stream=raw_stream0)
            del arg377_1
            del arg378_1
            del arg379_1
            del arg380_1
            del buf221
            # Topologically Sorted Source Nodes: [conv2d_76], Original ATen: [aten.convolution]
            buf231 = torch.ops.aten.convolution.default(buf230, arg381_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg381_1
            buf232 = buf231
            assert_size_stride(buf232, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf232, 16, 'torch.ops.aten.convolution.default')
            del buf231
            buf233 = buf232; del buf232  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_76, out_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf233, arg382_1, arg383_1, arg384_1, arg385_1, 512, 196, stream=raw_stream0)
            del arg382_1
            del arg383_1
            del arg384_1
            del arg385_1
            # Topologically Sorted Source Nodes: [batch_norm_76, out_72, conv2d_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf234 = torch.ops.aten.convolution.default(buf233, arg386_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg386_1
            del buf233
            buf235 = buf234
            assert_size_stride(buf235, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf235, 16, 'torch.ops.aten.convolution.default')
            del buf234
            buf236 = buf235; del buf235  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_77, out_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf236, arg387_1, arg388_1, arg389_1, arg390_1, 512, 196, stream=raw_stream0)
            del arg387_1
            del arg388_1
            del arg389_1
            del arg390_1
            # Topologically Sorted Source Nodes: [batch_norm_77, out_73, conv2d_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf237 = torch.ops.aten.convolution.default(buf236, arg391_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg391_1
            del buf236
            buf238 = buf237
            assert_size_stride(buf238, (1, 2048, 14, 14), (401408, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf238, 16, 'torch.ops.aten.convolution.default')
            del buf237
            buf239 = buf238; del buf238  # reuse
            # Topologically Sorted Source Nodes: [out_74, add_24, input_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_17.run(buf239, arg392_1, arg393_1, arg394_1, arg395_1, buf230, 2048, 196, stream=raw_stream0)
            del arg392_1
            del arg393_1
            del arg394_1
            del arg395_1
            del buf230
            # Topologically Sorted Source Nodes: [conv2d_79], Original ATen: [aten.convolution]
            buf240 = torch.ops.aten.convolution.default(buf239, arg396_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg396_1
            buf241 = buf240
            assert_size_stride(buf241, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf241, 16, 'torch.ops.aten.convolution.default')
            del buf240
            buf242 = buf241; del buf241  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_79, out_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf242, arg397_1, arg398_1, arg399_1, arg400_1, 512, 196, stream=raw_stream0)
            del arg397_1
            del arg398_1
            del arg399_1
            del arg400_1
            # Topologically Sorted Source Nodes: [batch_norm_79, out_75, conv2d_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf243 = torch.ops.aten.convolution.default(buf242, arg401_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg401_1
            del buf242
            buf244 = buf243
            assert_size_stride(buf244, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf244, 16, 'torch.ops.aten.convolution.default')
            del buf243
            buf245 = buf244; del buf244  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_80, out_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf245, arg402_1, arg403_1, arg404_1, arg405_1, 512, 196, stream=raw_stream0)
            del arg402_1
            del arg403_1
            del arg404_1
            del arg405_1
            # Topologically Sorted Source Nodes: [batch_norm_80, out_76, conv2d_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf246 = torch.ops.aten.convolution.default(buf245, arg406_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg406_1
            del buf245
            buf247 = buf246
            assert_size_stride(buf247, (1, 2048, 14, 14), (401408, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf247, 16, 'torch.ops.aten.convolution.default')
            del buf246
            buf248 = buf247; del buf247  # reuse
            # Topologically Sorted Source Nodes: [out_77, add_25, input_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_17.run(buf248, arg407_1, arg408_1, arg409_1, arg410_1, buf239, 2048, 196, stream=raw_stream0)
            del arg407_1
            del arg408_1
            del arg409_1
            del arg410_1
            del buf239
            # Topologically Sorted Source Nodes: [conv2d_82], Original ATen: [aten.convolution]
            buf249 = torch.ops.aten.convolution.default(buf248, arg411_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg411_1
            buf250 = buf249
            assert_size_stride(buf250, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf250, 16, 'torch.ops.aten.convolution.default')
            del buf249
            buf251 = buf250; del buf250  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_82, out_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf251, arg412_1, arg413_1, arg414_1, arg415_1, 512, 196, stream=raw_stream0)
            del arg412_1
            del arg413_1
            del arg414_1
            del arg415_1
            # Topologically Sorted Source Nodes: [batch_norm_82, out_78, conv2d_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf252 = torch.ops.aten.convolution.default(buf251, arg416_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            del arg416_1
            del buf251
            buf253 = buf252
            assert_size_stride(buf253, (1, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf253, 16, 'torch.ops.aten.convolution.default')
            del buf252
            buf254 = buf253; del buf253  # reuse
            # Topologically Sorted Source Nodes: [batch_norm_83, out_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_t_13.run(buf254, arg417_1, arg418_1, arg419_1, arg420_1, 512, 196, stream=raw_stream0)
            del arg417_1
            del arg418_1
            del arg419_1
            del arg420_1
            # Topologically Sorted Source Nodes: [batch_norm_83, out_79, conv2d_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            buf255 = torch.ops.aten.convolution.default(buf254, arg421_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            del arg421_1
            del buf254
            buf256 = buf255
            assert_size_stride(buf256, (1, 2048, 14, 14), (401408, 196, 14, 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf256, 16, 'torch.ops.aten.convolution.default')
            del buf255
            buf257 = empty_strided_npu((1, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
            buf258 = buf257; del buf257  # reuse
            # Topologically Sorted Source Nodes: [out_80, add_26, input_33, adaptive_avg_pool2d], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_no_t_18.run(buf258, buf256, arg422_1, arg423_1, arg424_1, arg425_1, buf248, 2048, 196, stream=raw_stream0)
            del arg422_1
            del arg423_1
            del arg424_1
            del arg425_1
            del buf248
            del buf256
            # Topologically Sorted Source Nodes: [out_80, add_26, input_33, adaptive_avg_pool2d, flatten, linear], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean, aten.view, aten.t, aten.addmm]
            buf259 = torch.ops.aten.addmm.default(arg427_1, reinterpret_tensor(buf258, (1, 2048), (0, 1), 0), reinterpret_tensor(arg426_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1)
            del arg426_1
            del arg427_1
            del buf258
            buf260 = buf259
            assert_size_stride(buf260, (1, 1000), (1000, 1), 'torch.ops.aten.addmm.default')
            assert_alignment(buf260, 16, 'torch.ops.aten.addmm.default')
            del buf259
        return (buf260, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 3, 224, 224), (150528, 50176, 224, 1), device='npu:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg16_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg17_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg18_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg19_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg20_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg21_1 = rand_strided((512, 64, 1, 1), (64, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg22_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg23_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg24_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg27_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg30_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg34_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg35_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg36_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg37_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg38_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg40_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg41_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg42_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg43_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg48_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg49_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg50_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg53_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg54_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg55_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg61_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg62_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg63_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg64_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg65_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg66_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg67_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg68_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg69_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg70_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg71_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg72_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg73_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg74_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg75_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg76_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg78_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg79_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg80_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg82_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg83_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg86_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg87_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg88_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg89_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg90_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg91_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg95_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg96_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg97_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg98_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg99_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg100_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg101_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg102_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg103_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg104_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg105_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg106_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg107_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg108_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg109_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg110_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg112_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg113_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg114_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg115_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg118_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg119_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg120_1 = rand_strided((1024, ), (1, ), device='npu:0', dtype=torch.float32)
    arg121_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg122_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg123_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg124_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg125_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg126_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg127_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg128_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg129_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg130_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg131_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg132_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg133_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg134_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg135_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg136_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg137_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg138_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg139_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg140_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg141_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg142_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg143_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg144_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg145_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg146_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg147_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg148_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg149_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg150_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg151_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg152_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg153_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg154_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg155_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg156_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg157_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg158_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg159_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg160_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg161_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg162_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg163_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg164_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg165_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg166_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg167_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg168_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg169_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg170_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg171_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg172_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg173_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg174_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg175_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg176_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg177_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg178_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg179_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg180_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg181_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg182_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg183_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg184_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg185_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg186_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg187_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg188_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg189_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg190_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg191_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg192_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg193_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg194_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg195_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg196_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg197_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg198_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg199_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg200_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg201_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg202_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg203_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg204_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg205_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg206_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg207_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg208_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg209_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg210_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg211_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg212_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg213_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg214_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg215_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg216_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg217_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg218_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg219_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg220_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg221_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg222_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg223_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg224_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg225_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg226_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg227_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg228_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg229_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg230_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg231_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg232_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg233_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg234_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg235_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg236_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg237_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg238_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg239_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg240_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg241_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg242_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg243_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg244_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg245_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg246_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg247_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg248_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg249_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg250_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg251_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg252_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg253_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg254_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg255_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg256_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg257_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg258_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg259_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg260_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg261_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg262_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg263_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg264_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg265_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg266_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg267_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg268_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg269_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg270_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg271_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg272_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg273_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg274_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg275_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg276_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg277_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg278_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg279_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg280_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg281_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg282_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg283_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg284_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg285_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg286_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg287_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg288_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg289_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg290_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg291_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg292_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg293_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg294_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg295_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg296_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg297_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg298_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg299_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg300_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg301_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg302_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg303_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg304_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg305_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg306_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg307_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg308_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg309_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg310_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg311_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg312_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg313_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg314_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg315_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg316_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg317_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg318_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg319_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg320_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg321_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg322_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg323_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg324_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg325_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg326_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg327_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg328_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg329_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg330_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg331_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg332_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg333_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg334_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg335_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg336_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg337_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg338_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg339_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg340_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg341_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg342_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg343_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg344_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg345_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg346_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg347_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg348_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg349_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg350_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg351_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg352_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg353_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg354_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg355_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg356_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg357_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg358_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg359_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg360_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg361_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg362_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg363_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg364_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg365_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg366_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg367_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg368_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg369_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg370_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg371_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg372_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg373_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg374_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg375_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg376_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg377_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg378_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg379_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg380_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg381_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg382_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg383_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg384_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg385_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg386_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg387_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg388_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg389_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg390_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg391_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg392_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg393_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg394_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg395_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg396_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg397_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg398_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg399_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg400_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg401_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg402_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg403_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg404_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg405_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg406_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg407_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg408_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg409_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg410_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg411_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg412_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg413_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg414_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg415_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg416_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='npu:0', dtype=torch.float32)
    arg417_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg418_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg419_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg420_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    arg421_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='npu:0', dtype=torch.float32)
    arg422_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg423_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg424_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg425_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg426_1 = rand_strided((1000, 2048), (2048, 1), device='npu:0', dtype=torch.float32)
    arg427_1 = rand_strided((1000, ), (1, ), device='npu:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
