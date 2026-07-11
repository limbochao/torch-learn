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
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch_npu._C import _npu_getCurrentRawStream as get_raw_stream
import torch_npu
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
from torch_npu._C import _npu_getCurrentRawStream as get_raw_stream



# kernel path: /tmp/torchinductor_root/uf/cufpdnul32obewqaveoaugfl3wc5irtj6jkasvexndfqhy7vnm2i.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   input_1 => add_tensor_13
#   input_2 => relu
# Graph fragment:
#   %arg3_1 : Tensor "f32[6144][1]npu:0" = PlaceHolder[target=arg3_1]
#   %mm_default_13 : Tensor "f32[128, 6144][6144, 1]npu:0" = PlaceHolder[target=mm_default_13]
#   %add_tensor_13 : Tensor "f32[128, 6144][6144, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_13), kwargs = {})
#   %relu : Tensor "f32[128, 6144][6144, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_13,), kwargs = {})
#   return %relu
# SchedulerNodes: [SchedulerNode(name='op1')]

triton_poi_fused_addmm_relu_0 = async_compile.triton('triton_poi_fused_addmm_relu_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

import torch
import torch_npu
from torch_npu._inductor.runtime import triton_heuristics as triton_heuristics
from torch_npu._inductor.runtime import triton_helpers
from torch_npu._inductor.runtime.triton_helpers import libdevice, extension, math as tl_math

@triton_heuristics.pointwise(
    size_hints={'y0': 128, 'x1': 6144}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_0(in_out_ptr0, in_ptr0, y0_numel, x1_numel, Y0BLOCK : tl.constexpr, X1BLOCK : tl.constexpr, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
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
            tmp0 = tl.load(in_ptr0 + (x1), x1_mask)
            tmp1 = tl.load(in_out_ptr0 + (x1 + 6144*y0), x1_mask & y0_mask)
            tmp2 = tmp0 + tmp1
            tmp3 = tl.full([1, 1], 0, tl.int32)
            tmp4 = tl.maximum(tmp3, tmp2, tl.PropagateNan.ALL)
            tl.store(in_out_ptr0 + (x1 + 6144*y0), tmp4, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: /tmp/torchinductor_root/qo/cqohtq3df7vuhu33tdj5usfndd35yqothgkgtdcjszyzz24lvfpv.py
# Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   input_13 => add_tensor_7
#   input_14 => relu_6
# Graph fragment:
#   %arg16_1 : Tensor "f32[3072][1]npu:0" = PlaceHolder[target=arg16_1]
#   %mm_default_7 : Tensor "f32[128, 3072][3072, 1]npu:0" = PlaceHolder[target=mm_default_7]
#   %add_tensor_7 : Tensor "f32[128, 3072][3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg16_1, %mm_default_7), kwargs = {})
#   %relu_6 : Tensor "f32[128, 3072][3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_7,), kwargs = {})
#   return %relu_6
# SchedulerNodes: [SchedulerNode(name='op13')]

triton_poi_fused_addmm_relu_1 = async_compile.triton('triton_poi_fused_addmm_relu_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

import torch
import torch_npu
from torch_npu._inductor.runtime import triton_heuristics as triton_heuristics
from torch_npu._inductor.runtime import triton_helpers
from torch_npu._inductor.runtime.triton_helpers import libdevice, extension, math as tl_math

@triton_heuristics.pointwise(
    size_hints={'y0': 128, 'x1': 3072}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_1(in_out_ptr0, in_ptr0, y0_numel, x1_numel, Y0BLOCK : tl.constexpr, X1BLOCK : tl.constexpr, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
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
            tmp0 = tl.load(in_ptr0 + (x1), x1_mask)
            tmp1 = tl.load(in_out_ptr0 + (x1 + 3072*y0), x1_mask & y0_mask)
            tmp2 = tmp0 + tmp1
            tmp3 = tl.full([1, 1], 0, tl.int32)
            tmp4 = tl.maximum(tmp3, tmp2, tl.PropagateNan.ALL)
            tl.store(in_out_ptr0 + (x1 + 3072*y0), tmp4, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: /tmp/torchinductor_root/35/c353tiin5p7z4mtqpxuhfgx6iqq3qsejd2bdx7llcaf27qkgf2xw.py
# Topologically Sorted Source Nodes: [input_19, input_20], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   input_19 => add_tensor_4
#   input_20 => relu_9
# Graph fragment:
#   %arg22_1 : Tensor "f32[1536][1]npu:0" = PlaceHolder[target=arg22_1]
#   %mm_default_4 : Tensor "f32[128, 1536][1536, 1]npu:0" = PlaceHolder[target=mm_default_4]
#   %add_tensor_4 : Tensor "f32[128, 1536][1536, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg22_1, %mm_default_4), kwargs = {})
#   %relu_9 : Tensor "f32[128, 1536][1536, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_4,), kwargs = {})
#   return %relu_9
# SchedulerNodes: [SchedulerNode(name='op19')]

triton_poi_fused_addmm_relu_2 = async_compile.triton('triton_poi_fused_addmm_relu_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

import torch
import torch_npu
from torch_npu._inductor.runtime import triton_heuristics as triton_heuristics
from torch_npu._inductor.runtime import triton_helpers
from torch_npu._inductor.runtime.triton_helpers import libdevice, extension, math as tl_math

@triton_heuristics.pointwise(
    size_hints={'y0': 128, 'x1': 1536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_2(in_out_ptr0, in_ptr0, y0_numel, x1_numel, Y0BLOCK : tl.constexpr, X1BLOCK : tl.constexpr, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
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
            tmp0 = tl.load(in_ptr0 + (x1), x1_mask)
            tmp1 = tl.load(in_out_ptr0 + (x1 + 1536*y0), x1_mask & y0_mask)
            tmp2 = tmp0 + tmp1
            tmp3 = tl.full([1, 1], 0, tl.int32)
            tmp4 = tl.maximum(tmp3, tmp2, tl.PropagateNan.ALL)
            tl.store(in_out_ptr0 + (x1 + 1536*y0), tmp4, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: /tmp/torchinductor_root/oq/coq4ygf44rrlqkxnok6rns2rx7qcl3557fuws2issfv2bzujrigx.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %arg0_1 : Tensor "i64[128, 1000][1000, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[5000, 8][8, 1]npu:0" = PlaceHolder[target=arg1_1]
#   %reshape_default : Tensor "i64[128, 50, 20][1000, 20, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [128, 50, 20]), kwargs = {})
#   %embedding_default : Tensor "f32[128, 50, 20, 8][8000, 160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %reshape_default), kwargs = {})
#   %sum_dim_int_list : Tensor "f32[128, 50, 8][400, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_default, [2]), kwargs = {})
#   return %sum_dim_int_list
# SchedulerNodes: [SchedulerNode(name='op25')]

triton_red_fused_3 = async_compile.triton('triton_red_fused_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

import torch
import torch_npu
from torch_npu._inductor.runtime import triton_heuristics as triton_heuristics
from torch_npu._inductor.runtime import triton_helpers
from torch_npu._inductor.runtime.triton_helpers import libdevice, extension, math as tl_math

@triton_heuristics.reduction(
    size_hints={'z0': 128, 'y1': 50, 'x2': 8, 'r3': 20},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'r3_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_3', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2, 3], 'no_loop_axis': [2], 'axis_names': ['z0', 'y1', 'x2', 'r3'], 'low_dims': {2}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_only', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False}
)
@triton.jit
def triton_red_fused_3(in_ptr0, in_ptr1, out_ptr0, z0_numel, y1_numel, x2_numel, r3_numel, Z0BLOCK : tl.constexpr, Y1BLOCK : tl.constexpr, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr, R3BLOCK_SUB : tl.constexpr):
    x2_numel = 8
    X2BLOCK_SUB: tl.constexpr = 8
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(1) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    base_r3= tl.arange(0, R3BLOCK_SUB)
    loops_r3 = (r3_numel + R3BLOCK_SUB - 1) // R3BLOCK_SUB
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[None,:,None,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            x2 = base_x2[None,None,None,:]
            x2_mask = x2 < x2_numel
            _tmp8 = tl.full([Z0BLOCK_SUB, Y1BLOCK_SUB, R3BLOCK_SUB, X2BLOCK_SUB], 0, tl.float32)
            for loop_r3 in range(loops_r3):
                r3 = (loop_r3 * R3BLOCK_SUB) + base_r3[None,None,:,None]
                r3_mask = r3 < r3_numel
                tmp0 = tl.load(in_ptr0 + (r3 + 20*y1 + 1000*z0), r3_mask & y1_mask & z0_mask, other=0.0)
                tmp1 = tl.full([Z0BLOCK_SUB, Y1BLOCK_SUB, R3BLOCK_SUB, X2BLOCK_SUB], 5000, tl.int32)
                tmp2 = tmp0 + tmp1
                tmp3 = tmp0 < 0
                tmp4 = tl.where(tmp3, tmp2, tmp0)
                tl.device_assert(((0 <= tmp4) & (tmp4 < 5000)) | ~(r3_mask & y1_mask & z0_mask), "index out of bounds: 0 <= tmp4 < 5000")
                tmp6 = tl.load(in_ptr1 + (x2 + 8*tmp4), r3_mask & x2_mask & y1_mask & z0_mask)
                tmp7 = tl.reshape(tmp6, [Z0BLOCK_SUB, Y1BLOCK_SUB, R3BLOCK_SUB, X2BLOCK_SUB])
                tmp9 = _tmp8 + tmp7
                _tmp8 = tl.where((r3_mask & x2_mask & y1_mask & z0_mask).reshape([Z0BLOCK_SUB, Y1BLOCK_SUB, R3BLOCK_SUB, X2BLOCK_SUB]), tmp9, _tmp8)
            tmp8 = tl.sum(_tmp8, 2).reshape(Z0BLOCK_SUB, Y1BLOCK_SUB, 1, X2BLOCK_SUB)
            tl.store(out_ptr0 + (x2 + 8*y1 + 400*z0 ), tmp8, x2_mask & y1_mask & z0_mask)
''', device_str='npu')


# kernel path: /tmp/torchinductor_root/og/cogl4rnbbedvdpbici5rxjkh3qd6xmvrcynzq6i3ip5melkwzhbl.py
# Topologically Sorted Source Nodes: [upper_tri_mask, zeros_like, input_25, input_26, activations, activations_1, concat], Original ATen: [aten.triu, aten.zeros_like, aten.addmm, aten.relu, aten.where, aten.view, aten.cat]
# Source node to ATen node mapping:
#   activations => where_1
#   activations_1 => view_3
#   concat => cat_2
#   input_25 => add_tensor_1
#   input_26 => relu_12
#   upper_tri_mask => iota, unsqueeze_1, unsqueeze_2
#   zeros_like => full_default_2
# Graph fragment:
#   %arg28_1 : Tensor "f32[8][1]npu:0" = PlaceHolder[target=arg28_1]
#   %mm_default_1 : Tensor "f32[128, 8][8, 1]npu:0" = PlaceHolder[target=mm_default_1]
#   %relu_12 : Tensor "f32[128, 8][408, 1]npu:0" = PlaceHolder[target=relu_12]
#   %iota : Tensor "i32[408][1]npu:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (408,), kwargs = {start: 0, step: 1, dtype: torch.int32, device: npu:0, requires_grad: False})
#   %unsqueeze_1 : Tensor "i32[1, 408][408, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, -2), kwargs = {})
#   %unsqueeze_2 : Tensor "i32[408, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, -1), kwargs = {})
#   %ge_tensor : Tensor "b8[408, 408][408, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%unsqueeze_1, %unsqueeze_2), kwargs = {})
#   %full_default_2 : Tensor "f32[128, 408, 408][166464, 408, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 408, 408], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %add_tensor_1 : Tensor "f32[128, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg28_1, %mm_default_1), kwargs = {})
#   %relu_12 : Tensor "f32[128, 8][8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_1,), kwargs = {})
#   %where_1 : Tensor "f32[128, 408, 408][166464, 408, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_tensor, %full_default_2, %bmm), kwargs = {})
#   %view_3 : Tensor "f32[128, 166464][166464, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%where_1, [128, 166464]), kwargs = {})
#   %cat_2 : Tensor "f32[128, 166472][166472, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_12, %view_3], -1), kwargs = {})
#   return %relu_12,%buf30
# SchedulerNodes: [SchedulerNode(name='op26'), SchedulerNode(name='op30')]

triton_poi_fused_addmm_cat_relu_triu_view_wher_4 = async_compile.triton('triton_poi_fused_addmm_cat_relu_triu_view_wher_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

import torch
import torch_npu
from torch_npu._inductor.runtime import triton_heuristics as triton_heuristics
from torch_npu._inductor.runtime import triton_helpers
from torch_npu._inductor.runtime.triton_helpers import libdevice, extension, math as tl_math

@triton_heuristics.pointwise(
    size_hints={'y0': 128, 'x1': 8}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_cat_relu_triu_view_wher_4', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_cat_relu_triu_view_wher_4(in_ptr0, in_ptr1, out_ptr0, out_ptr1, y0_numel, x1_numel, Y0BLOCK : tl.constexpr, X1BLOCK : tl.constexpr, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
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
            tmp0 = tl.load(in_ptr0 + (x1), x1_mask)
            tmp1 = tl.load(in_ptr1 + (x1 + 8*y0), x1_mask & y0_mask)
            tmp2 = tmp0 + tmp1
            tmp3 = tl.full([1, 1], 0, tl.int32)
            tmp4 = tl.maximum(tmp3, tmp2, tl.PropagateNan.ALL)
            tl.store(out_ptr0 + (x1 + 408*y0), tmp4, x1_mask & y0_mask)
            tl.store(out_ptr1 + (x1 + 166472*y0), tmp4, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: /tmp/torchinductor_root/eb/ceb2uohdnsudpryh2rgcqtirqg23lrqg2yudi43sed66nsvb7xyg.py
# Topologically Sorted Source Nodes: [input_25, input_26], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   input_25 => add_tensor_1
#   input_26 => relu_12
# Graph fragment:
#   %sum_dim_int_list : Tensor "f32[128, 50, 8][400, 8, 1]npu:0" = PlaceHolder[target=sum_dim_int_list]
#   %add_tensor_1 : Tensor "f32[128, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg28_1, %mm_default_1), kwargs = {})
#   %relu_12 : Tensor "f32[128, 8][8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_1,), kwargs = {})
#   %reshape_default_1 : Tensor "f32[128, 400][400, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_dim_int_list, [128, 400]), kwargs = {})
#   %cat_1_1 : Tensor "f32[128, 408][408, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_12, %reshape_default_1], -1), kwargs = {})
#   return %buf27
# SchedulerNodes: [SchedulerNode(name='op27')]

triton_poi_fused_addmm_relu_5 = async_compile.triton('triton_poi_fused_addmm_relu_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

import torch
import torch_npu
from torch_npu._inductor.runtime import triton_heuristics as triton_heuristics
from torch_npu._inductor.runtime import triton_helpers
from torch_npu._inductor.runtime.triton_helpers import libdevice, extension, math as tl_math

@triton_heuristics.pointwise(
    size_hints={'y0': 128, 'x1': 400}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_5', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_5(in_ptr0, out_ptr0, y0_numel, x1_numel, Y0BLOCK : tl.constexpr, X1BLOCK : tl.constexpr, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
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
            tmp0 = tl.load(in_ptr0 + (x1 + 400*y0), x1_mask & y0_mask)
            tl.store(out_ptr0 + (x1 + 408*y0), tmp0, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: /tmp/torchinductor_root/ko/ckobvw2h33ei7qtwdkj3hdil4le4vnkkdzewontrmsjzkjmp6b7q.py
# Topologically Sorted Source Nodes: [upper_tri_mask, zeros_like, activations, activations_1, concat], Original ATen: [aten.triu, aten.zeros_like, aten.where, aten.view, aten.cat]
# Source node to ATen node mapping:
#   activations => where_1
#   activations_1 => view_3
#   concat => cat_2
#   upper_tri_mask => iota, unsqueeze_1, unsqueeze_2
#   zeros_like => full_default_2
# Graph fragment:
#   %bmm : Tensor "f32[128, 408, 408][166464, 408, 1]npu:0" = PlaceHolder[target=bmm]
#   %iota : Tensor "i32[408][1]npu:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (408,), kwargs = {start: 0, step: 1, dtype: torch.int32, device: npu:0, requires_grad: False})
#   %unsqueeze_1 : Tensor "i32[1, 408][408, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, -2), kwargs = {})
#   %unsqueeze_2 : Tensor "i32[408, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, -1), kwargs = {})
#   %ge_tensor : Tensor "b8[408, 408][408, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%unsqueeze_1, %unsqueeze_2), kwargs = {})
#   %full_default_2 : Tensor "f32[128, 408, 408][166464, 408, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 408, 408], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_1 : Tensor "f32[128, 408, 408][166464, 408, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_tensor, %full_default_2, %bmm), kwargs = {})
#   %view_3 : Tensor "f32[128, 166464][166464, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%where_1, [128, 166464]), kwargs = {})
#   %cat_2 : Tensor "f32[128, 166472][166472, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_12, %view_3], -1), kwargs = {})
#   return %buf31
# SchedulerNodes: [SchedulerNode(name='op31')]

triton_poi_fused_cat_triu_view_where_zeros_lik_6 = async_compile.triton('triton_poi_fused_cat_triu_view_where_zeros_lik_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

import torch
import torch_npu
from torch_npu._inductor.runtime import triton_heuristics as triton_heuristics
from torch_npu._inductor.runtime import triton_helpers
from torch_npu._inductor.runtime.triton_helpers import libdevice, extension, math as tl_math

@triton_heuristics.pointwise(
    size_hints={'y0': 128, 'x2': 408, 'x3': 408},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x2_numel': 'i32', 'x3_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_triu_view_where_zeros_lik_6', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [2], 'axis_names': ['y0', 'x2', 'x3'], 'low_dims': {2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_triu_view_where_zeros_lik_6(in_ptr0, out_ptr0, y0_numel, x2_numel, x3_numel, Y0BLOCK : tl.constexpr, X2BLOCK : tl.constexpr, Y0BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr):
    x3_numel = 408
    X3BLOCK_SUB: tl.constexpr = 408
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    x2_offset = tl.program_id(1) * X2BLOCK
    base_x2= tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (X2BLOCK + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    base_x3= tl.arange(0, X3BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        for loop_x2 in range(loops_x2):
            x2 = x2_offset + (loop_x2 * X2BLOCK_SUB) + base_x2[None,:,None]
            x2_mask = x2 < min(X2BLOCK+x2_offset, x2_numel)
            x3 = base_x3[None,None,:]
            tmp3 = tl.load(in_ptr0 + (x3 + 408*x2 + 166464*y0), x2_mask & y0_mask)
            tmp0 = x3
            tmp1 = x2
            tmp2 = tmp0 >= tmp1
            tmp4 = 0.0
            tmp5 = tl.where(tmp2, tmp4, tmp3)
            tl.store(out_ptr0 + (x3 + 408*x2 + 166472*y0), tmp5, x2_mask & y0_mask)
''', device_str='npu')


# kernel path: /tmp/torchinductor_root/mi/cmiaiiu6cwqjlg667v35pujasmgm5izsawcdurcahm76xgwvw6gz.py
# Topologically Sorted Source Nodes: [input_27, sigmoid], Original ATen: [aten.addmm, aten.sigmoid]
# Source node to ATen node mapping:
#   input_27 => add_tensor
#   sigmoid => sigmoid
# Graph fragment:
#   %arg30_1 : Tensor "f32[1][1]npu:0" = PlaceHolder[target=arg30_1]
#   %mm_default : Tensor "f32[128, 1][1, 1]npu:0" = PlaceHolder[target=mm_default]
#   %add_tensor : Tensor "f32[128, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg30_1, %mm_default), kwargs = {})
#   %sigmoid : Tensor "f32[128, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor,), kwargs = {})
#   return %sigmoid
# SchedulerNodes: [SchedulerNode(name='op34')]

triton_poi_fused_addmm_sigmoid_7 = async_compile.triton('triton_poi_fused_addmm_sigmoid_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

import torch
import torch_npu
from torch_npu._inductor.runtime import triton_heuristics as triton_heuristics
from torch_npu._inductor.runtime import triton_helpers
from torch_npu._inductor.runtime.triton_helpers import libdevice, extension, math as tl_math

@triton_heuristics.pointwise(
    size_hints={'x0': 128},
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'x0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_sigmoid_7', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['x0'], 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_sigmoid_7(in_out_ptr0, in_ptr0, x0_numel, X0BLOCK : tl.constexpr, X0BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        tmp0 = tl.load(in_ptr0 + tl.arange(0,1) +  (0))
        tmp1 = tl.reshape(tmp0, [1 ])
        tmp2 = tl.load(in_out_ptr0 + (x0), x0_mask)
        tmp3 = tmp1 + tmp2
        tmp4 = tl.sigmoid(tmp3)
        tl.store(in_out_ptr0 + (x0), tmp4, x0_mask)
''', device_str='npu')

def partition_0(args):
    arg4_1, arg2_1, arg3_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg0_1, arg1_1, arg28_1, arg29_1, arg30_1 = args
    args.clear()
    with torch.npu.utils.device(0):
        torch.npu.set_device(0)
        buf0 = empty_strided((128, 6144), (6144, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(arg4_1, reinterpret_tensor(arg2_1, (1000, 6144), (1, 1000), 0), out=buf0)
        del arg2_1
        del arg4_1
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_0.run(buf1, arg3_1, 128, 6144, stream=stream0)
        del arg3_1
        buf2 = empty_strided((128, 6144), (6144, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf1, reinterpret_tensor(arg5_1, (6144, 6144), (1, 6144), 0), out=buf2)
        del arg5_1
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_0.run(buf3, arg6_1, 128, 6144, stream=stream0)
        del arg6_1
        buf4 = empty_strided((128, 6144), (6144, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_3, input_4, input_5], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf3, reinterpret_tensor(arg7_1, (6144, 6144), (1, 6144), 0), out=buf4)
        del arg7_1
        del buf3
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_0.run(buf5, arg8_1, 128, 6144, stream=stream0)
        del arg8_1
        buf6 = empty_strided((128, 6144), (6144, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_5, input_6, input_7], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf5, reinterpret_tensor(arg9_1, (6144, 6144), (1, 6144), 0), out=buf6)
        del arg9_1
        del buf5
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_0.run(buf7, arg10_1, 128, 6144, stream=stream0)
        del arg10_1
        buf8 = empty_strided((128, 6144), (6144, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_7, input_8, input_9], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf7, reinterpret_tensor(arg11_1, (6144, 6144), (1, 6144), 0), out=buf8)
        del arg11_1
        del buf7
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [input_9, input_10], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_0.run(buf9, arg12_1, 128, 6144, stream=stream0)
        del arg12_1
        buf10 = empty_strided((128, 6144), (6144, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_9, input_10, input_11], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf9, reinterpret_tensor(arg13_1, (6144, 6144), (1, 6144), 0), out=buf10)
        del arg13_1
        del buf9
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_0.run(buf11, arg14_1, 128, 6144, stream=stream0)
        del arg14_1
        buf12 = empty_strided((128, 3072), (3072, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_11, input_12, input_13], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf11, reinterpret_tensor(arg15_1, (6144, 3072), (1, 6144), 0), out=buf12)
        del arg15_1
        del buf11
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_1.run(buf13, arg16_1, 128, 3072, stream=stream0)
        del arg16_1
        buf14 = empty_strided((128, 3072), (3072, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf13, reinterpret_tensor(arg17_1, (3072, 3072), (1, 3072), 0), out=buf14)
        del arg17_1
        del buf13
        buf15 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [input_15, input_16], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_1.run(buf15, arg18_1, 128, 3072, stream=stream0)
        del arg18_1
        buf16 = empty_strided((128, 3072), (3072, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_15, input_16, input_17], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf15, reinterpret_tensor(arg19_1, (3072, 3072), (1, 3072), 0), out=buf16)
        del arg19_1
        del buf15
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [input_17, input_18], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_1.run(buf17, arg20_1, 128, 3072, stream=stream0)
        del arg20_1
        buf18 = empty_strided((128, 1536), (1536, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_17, input_18, input_19], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf17, reinterpret_tensor(arg21_1, (3072, 1536), (1, 3072), 0), out=buf18)
        del arg21_1
        del buf17
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [input_19, input_20], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_2.run(buf19, arg22_1, 128, 1536, stream=stream0)
        del arg22_1
        buf20 = empty_strided((128, 1536), (1536, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_19, input_20, input_21], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf19, reinterpret_tensor(arg23_1, (1536, 1536), (1, 1536), 0), out=buf20)
        del arg23_1
        del buf19
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [input_21, input_22], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_2.run(buf21, arg24_1, 128, 1536, stream=stream0)
        del arg24_1
        buf22 = empty_strided((128, 1536), (1536, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_21, input_22, input_23], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf21, reinterpret_tensor(arg25_1, (1536, 1536), (1, 1536), 0), out=buf22)
        del arg25_1
        del buf21
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [input_23, input_24], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_2.run(buf23, arg26_1, 128, 1536, stream=stream0)
        del arg26_1
        buf24 = empty_strided((128, 8), (8, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_23, input_24, input_25], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf23, reinterpret_tensor(arg27_1, (1536, 8), (1, 1536), 0), out=buf24)
        del arg27_1
        del buf23
        buf25 = empty_strided((128, 50, 8), (400, 8, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_red_fused_3.run(arg0_1, arg1_1, buf25, 128, 50, 8, 20, stream=stream0)
        del arg0_1
        del arg1_1
        buf28 = empty_strided((128, 408), (408, 1), device='npu', dtype=torch.float32)
        buf26 = reinterpret_tensor(buf28, (128, 8), (408, 1), 0)  # alias
        buf32 = empty_strided((128, 166472), (166472, 1), device='npu', dtype=torch.float32)
        buf30 = reinterpret_tensor(buf32, (128, 8), (166472, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [upper_tri_mask, zeros_like, input_25, input_26, activations, activations_1, concat], Original ATen: [aten.triu, aten.zeros_like, aten.addmm, aten.relu, aten.where, aten.view, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_cat_relu_triu_view_wher_4.run(arg28_1, buf24, buf26, buf30, 128, 8, stream=stream0)
        del arg28_1
        del buf24
        buf27 = reinterpret_tensor(buf28, (128, 400), (408, 1), 8)  # alias
        # Topologically Sorted Source Nodes: [input_25, input_26], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_5.run(buf25, buf27, 128, 400, stream=stream0)
        del buf25
        del buf26
        del buf27
        buf29 = empty_strided((128, 408, 408), (166464, 408, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [inputs, transpose, xactions], Original ATen: [aten.unsqueeze, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf28, (128, 408, 1), (408, 1, 1), 0), reinterpret_tensor(buf28, (128, 1, 408), (408, 1, 1), 0), out=buf29)
        del buf28
        buf31 = reinterpret_tensor(buf32, (128, 166464), (166472, 1), 8)  # alias
        # Topologically Sorted Source Nodes: [upper_tri_mask, zeros_like, activations, activations_1, concat], Original ATen: [aten.triu, aten.zeros_like, aten.where, aten.view, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_triu_view_where_zeros_lik_6.run(buf29, buf31, 128, 408, 408, stream=stream0)
        del buf29
        del buf30
        del buf31
        buf33 = empty_strided((128, 1), (1, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf32, reinterpret_tensor(arg29_1, (166472, 1), (1, 166472), 0), out=buf33)
        del arg29_1
        del buf32
        buf34 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [input_27, sigmoid], Original ATen: [aten.addmm, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_sigmoid_7.run(buf34, arg30_1, 128, stream=stream0)
        del arg30_1
    return (buf34, )


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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1 = args
        args.clear()
        partition0_args = [arg4_1, arg2_1, arg3_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg0_1, arg1_1, arg28_1, arg29_1, arg30_1]
        del arg4_1, arg2_1, arg3_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg0_1, arg1_1, arg28_1, arg29_1, arg30_1
        (buf34,) = self.partitions[0](partition0_args)
        del partition0_args
        return (reinterpret_tensor(buf34, (128, ), (1, ), 0), )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((128, 1000), (1000, 1), device='npu:0', dtype=torch.int64)
    arg1_1 = rand_strided((5000, 8), (8, 1), device='npu:0', dtype=torch.float32)
    arg2_1 = rand_strided((6144, 1000), (1000, 1), device='npu:0', dtype=torch.float32)
    arg3_1 = rand_strided((6144, ), (1, ), device='npu:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, 1000), (1000, 1), device='npu:0', dtype=torch.float32)
    arg5_1 = rand_strided((6144, 6144), (6144, 1), device='npu:0', dtype=torch.float32)
    arg6_1 = rand_strided((6144, ), (1, ), device='npu:0', dtype=torch.float32)
    arg7_1 = rand_strided((6144, 6144), (6144, 1), device='npu:0', dtype=torch.float32)
    arg8_1 = rand_strided((6144, ), (1, ), device='npu:0', dtype=torch.float32)
    arg9_1 = rand_strided((6144, 6144), (6144, 1), device='npu:0', dtype=torch.float32)
    arg10_1 = rand_strided((6144, ), (1, ), device='npu:0', dtype=torch.float32)
    arg11_1 = rand_strided((6144, 6144), (6144, 1), device='npu:0', dtype=torch.float32)
    arg12_1 = rand_strided((6144, ), (1, ), device='npu:0', dtype=torch.float32)
    arg13_1 = rand_strided((6144, 6144), (6144, 1), device='npu:0', dtype=torch.float32)
    arg14_1 = rand_strided((6144, ), (1, ), device='npu:0', dtype=torch.float32)
    arg15_1 = rand_strided((3072, 6144), (6144, 1), device='npu:0', dtype=torch.float32)
    arg16_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg17_1 = rand_strided((3072, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg18_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg19_1 = rand_strided((3072, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg20_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg21_1 = rand_strided((1536, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg22_1 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    arg23_1 = rand_strided((1536, 1536), (1536, 1), device='npu:0', dtype=torch.float32)
    arg24_1 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    arg25_1 = rand_strided((1536, 1536), (1536, 1), device='npu:0', dtype=torch.float32)
    arg26_1 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    arg27_1 = rand_strided((8, 1536), (1536, 1), device='npu:0', dtype=torch.float32)
    arg28_1 = rand_strided((8, ), (1, ), device='npu:0', dtype=torch.float32)
    arg29_1 = rand_strided((1, 166472), (166472, 1), device='npu:0', dtype=torch.float32)
    arg30_1 = rand_strided((1, ), (1, ), device='npu:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
