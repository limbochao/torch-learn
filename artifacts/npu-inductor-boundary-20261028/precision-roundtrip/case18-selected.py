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
from torch_npu._C import _npu_getCurrentRawStream as get_raw_stream
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch_npu._C import _npu_getCurrentRawStream as get_raw_stream
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


# kernel path: <test-repo>/boundary_20261028/layout_wrapper_v3/tmp1n8dexr9/yl/cyljg4lkhdc74vmxyb6cau5oqcp7pv5c3kmlvjn2mvksm3axkc56.py
# Topologically Sorted Source Nodes: [catted_b, catted_b_add, catted_bf16, fused], Original ATen: [aten.cat, aten.add, aten._to_copy, aten.sub]
# Source node to ATen node mapping:
#   catted_b => cat
#   catted_b_add => add
#   catted_bf16 => convert_element_type
#   fused => sub
# Graph fragment:
#   %arg0_1 : Tensor "f32[32, 256, 1490][381440, 1490, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[32, 256, 1490][381440, 1490, 1]npu:0" = PlaceHolder[target=arg1_1]
#   %convert_element_type : Tensor "bf16[32, 512, 1490][762880, 1490, 1]npu:0" = PlaceHolder[target=convert_element_type]
#   %arg3_1 : Tensor "bf16[32, 512, 1490][762880, 1490, 1]npu:0" = PlaceHolder[target=arg3_1]
#   %cat : Tensor "f32[32, 512, 1490][762880, 1490, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%arg0_1, %arg1_1], 1), kwargs = {})
#   %add : Tensor "f32[32, 512, 1490][762880, 1490, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat, 1.0), kwargs = {})
#   %convert_element_type : Tensor "bf16[32, 512, 1490][762880, 1490, 1]npu:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.bfloat16), kwargs = {})
#   %sub : Tensor "bf16[32, 512, 1490][762880, 1490, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %arg3_1), kwargs = {})
#   return %convert_element_type,%sub
# SchedulerNodes: [SchedulerNode(name='op0'), SchedulerNode(name='op4')]

triton_poi_fused__to_copy_add_cat_sub_0 = async_compile.triton('triton_poi_fused__to_copy_add_cat_sub_0', '''
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
    size_hints={'z0': 32, 'y1': 512, 'x2': 1490}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_cat_sub_0', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [2], 'axis_names': ['z0', 'y1', 'x2'], 'axis_static_values': (('z0', 32), ('y1', 512), ('x2', 1490)), 'low_dims': {2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.bfloat16, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_cat_sub_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, z0_numel, y1_numel, x2_numel, Z0BLOCK, Y1BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr):
    x2_numel = 1490
    X2BLOCK_SUB: tl.constexpr = 2048
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
            x2_mask = x2 < x2_numel
            tmp17 = tl.load(in_ptr2 + (x2 + 1490*y1 + 762880*z0), x2_mask & y1_mask & z0_mask).to(tl.float32)
            tmp0 = y1
            tmp1 = tl.full([1, 1, 1], 0, tl.int64)
            tmp2 = tmp0 >= tmp1
            tmp3 = (tmp0).to(tl.int64)
            tmp4 = tl.full([1, 1, 1], 256, tl.int64)
            tmp5 = tmp3 < tmp4
            tmp6 = tmp5 & y1_mask
            tmp7 = tl.load(in_ptr0 + (x2 + 1490*(y1) + 381440*z0), tmp6 & x2_mask & z0_mask, other=0.0)
            tmp8 = tmp0 >= tmp4
            tmp9 = tl.full([1, 1, 1], 512, tl.int64)
            tmp10 = tmp0 < tmp9
            tmp11 = tmp8 & y1_mask
            tmp12 = tl.load(in_ptr1 + (x2 + 1490*((-256) + y1) + 381440*z0), tmp11 & x2_mask & z0_mask, other=0.0)
            tmp13 = tl.where(tmp5, tmp7, tmp12)
            tmp14 = tl.full([1, 1, 1], 1.0, tl.float32)
            tmp15 = tmp13 + tmp14
            tmp16 = tmp15.to(tl.float32)
            tmp18 = tmp16 - tmp17
            tl.store(out_ptr0 + (x2 + 1490*y1 + 762880*z0), tmp16, x2_mask & y1_mask & z0_mask)
            tl.store(out_ptr1 + (x2 + 1490*y1 + 762880*z0), tmp18, x2_mask & y1_mask & z0_mask)
''', device_str='npu')


# kernel path: <test-repo>/boundary_20261028/layout_wrapper_v3/tmp1n8dexr9/be/cbezlkpqkwhcy4duuv3siayaydqpiwodwl7b5nf4k4kdletd3zc5.py
# Topologically Sorted Source Nodes: [to_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   to_1 => convert_element_type_1
# Graph fragment:
#   %arg2_1 : Tensor "f32[32, 512, 512][262144, 512, 1]npu:0" = PlaceHolder[target=arg2_1]
#   %convert_element_type_1 : Tensor "bf16[32, 512, 512][262144, 512, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg2_1, torch.bfloat16), kwargs = {})
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
    size_hints={'z0': 32, 'y1': 512, 'x2': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [2], 'axis_names': ['z0', 'y1', 'x2'], 'axis_static_values': (('z0', 32), ('y1', 512), ('x2', 512)), 'low_dims': {2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.bfloat16, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, z0_numel, y1_numel, x2_numel, Z0BLOCK, Y1BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr):
    x2_numel = 512
    X2BLOCK_SUB: tl.constexpr = 512
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
            tmp0 = tl.load(in_ptr0 + (x2 + 512*y1 + 262144*z0), y1_mask & z0_mask)
            tmp1 = tmp0.to(tl.float32)
            tl.store(out_ptr0 + (x2 + 512*y1 + 262144*z0), tmp1, y1_mask & z0_mask)
''', device_str='npu')


# kernel path: <test-repo>/boundary_20261028/layout_wrapper_v3/tmp1n8dexr9/ka/cka2bpynj2pmojrgqr32hbs3aeczg6tqc6z5nfike5tqztqqgnir.py
# Topologically Sorted Source Nodes: [to_1, bmm1, bmm1_fp32], Original ATen: [aten._to_copy, aten.bmm]
# Source node to ATen node mapping:
#   bmm1 => bmm
#   bmm1_fp32 => convert_element_type_4
#   to_1 => convert_element_type_1
# Graph fragment:
#   %convert_element_type_1 : Tensor "bf16[32, 512, 512][262144, 512, 1]npu:0" = PlaceHolder[target=convert_element_type_1]
#   %convert_element_type : Tensor "bf16[32, 512, 1490][762880, 1490, 1]npu:0" = PlaceHolder[target=convert_element_type]
#   %bmm : Tensor "bf16[32, 512, 1490][762880, 1490, 1]npu:0" = PlaceHolder[target=bmm]
#   %convert_element_type_1 : Tensor "bf16[32, 512, 512][262144, 512, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg2_1, torch.bfloat16), kwargs = {})
#   %bmm : Tensor "bf16[32, 512, 1490][762880, 1490, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.bmm.default](args = (%convert_element_type_1, %convert_element_type), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[32, 512, 1490][762880, 1490, 1]npu:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bmm, torch.float32), kwargs = {})
#   return %bmm,%convert_element_type_4
# SchedulerNodes: [SchedulerNode(name='op2'), SchedulerNode(name='op3')]

triton_tem_fused__to_copy_bmm_2 = async_compile.triton('triton_tem_fused__to_copy_bmm_2', '''
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

@triton_heuristics.template(
    num_stages=2,
    num_warps=4,
    triton_meta={'signature': {'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*fp32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'npu_compile_options': {'unit_flag': False, 'enable_hivm_batch_matmul': True}},
    inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_bmm_2', 'backend_hash': '<redacted>', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'BLOCK_M': 128, 'BLOCK_N': 256},
)
@triton.jit
def triton_tem_fused__to_copy_bmm_2(arg_A, arg_B, out_ptr0, out_ptr1):
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 256
    BLOCK_K : tl.constexpr = 256
    GROUP_M : tl.constexpr = 1
    ALLOW_HF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    EVEN_K : tl.constexpr = True
    L1_COPIES : tl.constexpr = 2
    SHIFT_M : tl.constexpr = False
    SHIFT_N : tl.constexpr = True
    MASK_M : tl.constexpr = False
    MASK_N : tl.constexpr = False
    K_MAIN : tl.constexpr = 512
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 512
    N = 1490
    K = 512

    stride_aq = 262144
    stride_am = 512
    stride_ak = 1

    stride_bq = 762880
    stride_bk = 1490
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # A ragged M or N tail cannot be left to the store mask alone: on the last
    # batch the overhanging rows and columns run off the end of the tensor and
    # the MTE reports an out-of-range DDR address (aicore error 95).  Where the
    # extent covers at least one whole tile the last tile is pulled back to end
    # on it, as the persistent template does, and then no load and no store is
    # masked at all.  The pulled-back tile recomputes the rows or columns it now
    # shares with its predecessor and stores the same values into them, which an
    # elementwise epilogue is indifferent to.
    #
    # The clamp is on the scalar base and not on rm: `tl.minimum` on the vector
    # makes the backend materialize the whole 2-D index tensor in UB and every
    # wide tile then fails to fit.  An extent shorter than its own tile has
    # nothing to pull back to and keeps the mask.

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)


    rn = tl.minimum(pid_n * BLOCK_N, N - BLOCK_N) + tl.arange(0, BLOCK_N)





    # batch dimension index — precompute batch offsets before K-loop
    # to avoid redundant multiply inside the hot loop
    idx_q = tl.program_id(1).to(INDEX_DTYPE)
    a_batch_off = idx_q * stride_aq
    b_batch_off = idx_q * stride_bq


    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    # Whole K tiles only.  A ragged K overhangs on its last tile and on no
    # other, so that tile is peeled out below instead of every iteration
    # carrying k_mask: the mask is a compare, a 2-D `and` against m_mask/n_mask
    # and a fill on both operands, and at K=1023 with BLOCK_K=128 seven of the
    # eight iterations would pay all of it to mask nothing.
    for k_start in range(0, K_MAIN, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a_ptrs = A + (rm[:, None] * stride_am + offs_k[None, :] * stride_ak
                      + a_batch_off)
        b_ptrs = B + (offs_k[:, None] * stride_bk + rn[None, :] * stride_bn
                      + b_batch_off)
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        # A second L1 slot per operand, so iteration k+1's MTE2 runs under
        # iteration k's MAC instead of waiting on the same buffer.  The
        # persistent and batched templates both stage their operands this way;
        # this loop was the one leaving the two pipes serialized.
        extension.multibuffer(a, L1_COPIES)
        extension.multibuffer(b, L1_COPIES)

        acc = tl.dot(a, b, acc=acc, out_dtype=ACC_TYPE)



    # rematerialize rm, rn and idx_q to save registers, on the same bases the
    # loads used so a pulled-back tile stores where it read

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)


    rn = tl.minimum(pid_n * BLOCK_N, N - BLOCK_N) + tl.arange(0, BLOCK_N)

    idx_q = tl.program_id(1).to(INDEX_DTYPE)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    # Only an axis that is neither whole nor pulled back still overhangs.  A
    # tile that overhangs nowhere wants no mask rather than a true one: the
    # comparison and its 2-D `and` are per-tile work, and they were being paid
    # even on shapes every axis of which divides.

    mask = None


    # inductor generates a suffix
    xindex = idx_n + 1490*idx_m + 762880*idx_q
    tmp0 = acc.to(tl.bfloat16)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
    tmp2 = tmp1.to(tl.float32)
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp2, mask)
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp2, mask)
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
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg1_1 = copy_if_misaligned(arg1_1)
            arg3_1 = copy_if_misaligned(arg3_1)
            buf0 = empty_strided_npu((32, 512, 1490), (762880, 1490, 1), torch.bfloat16)
            buf4 = empty_strided_npu((32, 512, 1490), (762880, 1490, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [catted_b, catted_b_add, catted_bf16, fused], Original ATen: [aten.cat, aten.add, aten._to_copy, aten.sub]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_sub_0.run(arg0_1, arg1_1, arg3_1, buf0, buf4, 32, 512, 1490, stream=raw_stream0)
            del arg0_1
            del arg1_1
            del arg3_1
            arg2_1 = copy_if_misaligned(arg2_1)
            buf1 = empty_strided_npu((32, 512, 512), (262144, 512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_1], Original ATen: [aten._to_copy]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg2_1, buf1, 32, 512, 512, stream=raw_stream0)
            del arg2_1
            buf2 = empty_strided_npu((32, 512, 1490), (762880, 1490, 1), torch.bfloat16)
            buf3 = empty_strided_npu((32, 512, 1490), (762880, 1490, 1), torch.float32)
            # Topologically Sorted Source Nodes: [to_1, bmm1, bmm1_fp32], Original ATen: [aten._to_copy, aten.bmm]
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_bmm_2.run(buf1, buf0, buf2, buf3, 24, 32, 1, stream=raw_stream0)
            del buf0
            del buf1
        return (buf2, buf3, buf4, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((32, 256, 1490), (381440, 1490, 1), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, 256, 1490), (381440, 1490, 1), device='npu:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, 512, 512), (262144, 512, 1), device='npu:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, 512, 1490), (762880, 1490, 1), device='npu:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
