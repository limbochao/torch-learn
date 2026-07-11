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



# kernel path: /tmp/torchinductor_root/gi/cgio5tvd7clvh3bdglvxgn55pmzogplf7t6ecowf563ygvutzyq4.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   input_1 => add_tensor_13
#   input_2 => relu
# Graph fragment:
#   %arg4_1 : Tensor "f32[6144][1]npu:0" = PlaceHolder[target=arg4_1]
#   %mm_default_13 : Tensor "f32[s20, 6144][6144, 1]npu:0" = PlaceHolder[target=mm_default_13]
#   %add_tensor_13 : Tensor "f32[s20, 6144][6144, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, %mm_default_13), kwargs = {})
#   %relu : Tensor "f32[s20, 6144][6144, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_13,), kwargs = {})
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_only', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False},
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


# kernel path: /tmp/torchinductor_root/d2/cd2oqkmqzznvlgckdme47d6dhzazswu35lc25bvnu4vsxfzsr3gk.py
# Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   input_13 => add_tensor_7
#   input_14 => relu_6
# Graph fragment:
#   %arg17_1 : Tensor "f32[3072][1]npu:0" = PlaceHolder[target=arg17_1]
#   %mm_default_7 : Tensor "f32[s20, 3072][3072, 1]npu:0" = PlaceHolder[target=mm_default_7]
#   %add_tensor_7 : Tensor "f32[s20, 3072][3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg17_1, %mm_default_7), kwargs = {})
#   %relu_6 : Tensor "f32[s20, 3072][3072, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_7,), kwargs = {})
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_only', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False},
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


# kernel path: /tmp/torchinductor_root/tm/ctmynld2hbxqfx6hcjk6cdaygzt77mitwoewdsupyqi7vnrdb3qi.py
# Topologically Sorted Source Nodes: [input_19, input_20], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   input_19 => add_tensor_4
#   input_20 => relu_9
# Graph fragment:
#   %arg23_1 : Tensor "f32[1536][1]npu:0" = PlaceHolder[target=arg23_1]
#   %mm_default_4 : Tensor "f32[s20, 1536][1536, 1]npu:0" = PlaceHolder[target=mm_default_4]
#   %add_tensor_4 : Tensor "f32[s20, 1536][1536, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg23_1, %mm_default_4), kwargs = {})
#   %relu_9 : Tensor "f32[s20, 1536][1536, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_4,), kwargs = {})
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_only', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False},
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


# kernel path: /tmp/torchinductor_root/7p/c7pj7krqad7q3xrlsjeu5cw7zp26q23gu54l24cy3ipvosmp4bve.py
# Topologically Sorted Source Nodes: [getitem_2, emb, group_sum, getitem_3, emb_1, group_sum_1, getitem_4, emb_2, group_sum_2, getitem_5, emb_3, group_sum_3, getitem_6, emb_4, group_sum_4, getitem_7, emb_5, group_sum_5, getitem_8, emb_6, group_sum_6, getitem_9, emb_7, group_sum_7, getitem_10, emb_8, group_sum_8, getitem_11, emb_9, group_sum_9, getitem_12, emb_10, group_sum_10, getitem_13, emb_11, group_sum_11, getitem_14, emb_12, group_sum_12, getitem_15, emb_13, group_sum_13, getitem_16, emb_14, group_sum_14, getitem_17, emb_15, group_sum_15, getitem_18, emb_16, group_sum_16, getitem_19, emb_17, group_sum_17, getitem_20, emb_18, group_sum_18, getitem_21, emb_19, group_sum_19, getitem_22, emb_20, group_sum_20], Original ATen: [aten.slice, aten.embedding, aten.sum]
# Source node to ATen node mapping:
#   emb => embedding
#   emb_1 => embedding_1
#   emb_10 => embedding_10
#   emb_11 => embedding_11
#   emb_12 => embedding_12
#   emb_13 => embedding_13
#   emb_14 => embedding_14
#   emb_15 => embedding_15
#   emb_16 => embedding_16
#   emb_17 => embedding_17
#   emb_18 => embedding_18
#   emb_19 => embedding_19
#   emb_2 => embedding_2
#   emb_20 => embedding_20
#   emb_3 => embedding_3
#   emb_4 => embedding_4
#   emb_5 => embedding_5
#   emb_6 => embedding_6
#   emb_7 => embedding_7
#   emb_8 => embedding_8
#   emb_9 => embedding_9
#   getitem_10 => slice_18
#   getitem_11 => slice_20
#   getitem_12 => slice_22
#   getitem_13 => slice_24
#   getitem_14 => slice_26
#   getitem_15 => slice_28
#   getitem_16 => slice_30
#   getitem_17 => slice_32
#   getitem_18 => slice_34
#   getitem_19 => slice_36
#   getitem_2 => slice_2
#   getitem_20 => slice_38
#   getitem_21 => slice_40
#   getitem_22 => slice_42
#   getitem_3 => slice_4
#   getitem_4 => slice_6
#   getitem_5 => slice_8
#   getitem_6 => slice_10
#   getitem_7 => slice_12
#   getitem_8 => slice_14
#   getitem_9 => slice_16
#   group_sum => sum_1
#   group_sum_1 => sum_2
#   group_sum_10 => sum_11
#   group_sum_11 => sum_12
#   group_sum_12 => sum_13
#   group_sum_13 => sum_14
#   group_sum_14 => sum_15
#   group_sum_15 => sum_16
#   group_sum_16 => sum_17
#   group_sum_17 => sum_18
#   group_sum_18 => sum_19
#   group_sum_19 => sum_20
#   group_sum_2 => sum_3
#   group_sum_20 => sum_21
#   group_sum_3 => sum_4
#   group_sum_4 => sum_5
#   group_sum_5 => sum_6
#   group_sum_6 => sum_7
#   group_sum_7 => sum_8
#   group_sum_8 => sum_9
#   group_sum_9 => sum_10
# Graph fragment:
#   %arg1_1 : Tensor "i64[s20, 1000][1000, 1]npu:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[5000, 8][8, 1]npu:0" = PlaceHolder[target=arg2_1]
#   %slice_2 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 0, 20), kwargs = {})
#   %embedding : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_2), kwargs = {})
#   %sum_1 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding, [1]), kwargs = {})
#   %slice_4 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 20, 40), kwargs = {})
#   %embedding_1 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_4), kwargs = {})
#   %sum_2 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_1, [1]), kwargs = {})
#   %slice_6 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 40, 60), kwargs = {})
#   %embedding_2 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_6), kwargs = {})
#   %sum_3 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_2, [1]), kwargs = {})
#   %slice_8 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 60, 80), kwargs = {})
#   %embedding_3 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_8), kwargs = {})
#   %sum_4 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_3, [1]), kwargs = {})
#   %slice_10 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 80, 100), kwargs = {})
#   %embedding_4 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_10), kwargs = {})
#   %sum_5 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_4, [1]), kwargs = {})
#   %slice_12 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 100, 120), kwargs = {})
#   %embedding_5 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_12), kwargs = {})
#   %sum_6 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_5, [1]), kwargs = {})
#   %slice_14 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 120, 140), kwargs = {})
#   %embedding_6 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_14), kwargs = {})
#   %sum_7 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_6, [1]), kwargs = {})
#   %slice_16 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 140, 160), kwargs = {})
#   %embedding_7 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_16), kwargs = {})
#   %sum_8 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_7, [1]), kwargs = {})
#   %slice_18 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 160, 180), kwargs = {})
#   %embedding_8 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_18), kwargs = {})
#   %sum_9 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_8, [1]), kwargs = {})
#   %slice_20 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 180, 200), kwargs = {})
#   %embedding_9 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_20), kwargs = {})
#   %sum_10 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_9, [1]), kwargs = {})
#   %slice_22 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 200, 220), kwargs = {})
#   %embedding_10 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_22), kwargs = {})
#   %sum_11 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_10, [1]), kwargs = {})
#   %slice_24 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 220, 240), kwargs = {})
#   %embedding_11 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_24), kwargs = {})
#   %sum_12 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_11, [1]), kwargs = {})
#   %slice_26 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 240, 260), kwargs = {})
#   %embedding_12 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_26), kwargs = {})
#   %sum_13 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_12, [1]), kwargs = {})
#   %slice_28 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 260, 280), kwargs = {})
#   %embedding_13 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_28), kwargs = {})
#   %sum_14 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_13, [1]), kwargs = {})
#   %slice_30 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 280, 300), kwargs = {})
#   %embedding_14 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_30), kwargs = {})
#   %sum_15 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_14, [1]), kwargs = {})
#   %slice_32 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 300, 320), kwargs = {})
#   %embedding_15 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_32), kwargs = {})
#   %sum_16 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_15, [1]), kwargs = {})
#   %slice_34 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 320, 340), kwargs = {})
#   %embedding_16 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_34), kwargs = {})
#   %sum_17 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_16, [1]), kwargs = {})
#   %slice_36 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 340, 360), kwargs = {})
#   %embedding_17 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_36), kwargs = {})
#   %sum_18 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_17, [1]), kwargs = {})
#   %slice_38 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 360, 380), kwargs = {})
#   %embedding_18 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_38), kwargs = {})
#   %sum_19 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_18, [1]), kwargs = {})
#   %slice_40 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 380, 400), kwargs = {})
#   %embedding_19 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_40), kwargs = {})
#   %sum_20 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_19, [1]), kwargs = {})
#   %slice_42 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 400, 420), kwargs = {})
#   %embedding_20 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_42), kwargs = {})
#   %sum_21 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_20, [1]), kwargs = {})
#   return %sum_1,%sum_2,%sum_3,%sum_4,%sum_5,%sum_6,%sum_7,%sum_8,%sum_9,%sum_10,%sum_11,%sum_12,%sum_13,%sum_14,%sum_15,%sum_16,%sum_17,%sum_18,%sum_19,%sum_20,%sum_21
# SchedulerNodes: [SchedulerNode(name='op25'), SchedulerNode(name='op26'), SchedulerNode(name='op27'), SchedulerNode(name='op28'), SchedulerNode(name='op29'), SchedulerNode(name='op30'), SchedulerNode(name='op31'), SchedulerNode(name='op32'), SchedulerNode(name='op33'), SchedulerNode(name='op34'), SchedulerNode(name='op35'), SchedulerNode(name='op36'), SchedulerNode(name='op37'), SchedulerNode(name='op38'), SchedulerNode(name='op39'), SchedulerNode(name='op40'), SchedulerNode(name='op41'), SchedulerNode(name='op42'), SchedulerNode(name='op43'), SchedulerNode(name='op44'), SchedulerNode(name='op45')]

triton_red_fused_embedding_slice_sum_3 = async_compile.triton('triton_red_fused_embedding_slice_sum_3', '''
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
    size_hints={'y0': 128, 'x1': 8, 'r2': 20},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*fp32', 'out_ptr11': '*fp32', 'out_ptr12': '*fp32', 'out_ptr13': '*fp32', 'out_ptr14': '*fp32', 'out_ptr15': '*fp32', 'out_ptr16': '*fp32', 'out_ptr17': '*fp32', 'out_ptr18': '*fp32', 'out_ptr19': '*fp32', 'out_ptr20': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'r2_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_embedding_slice_sum_3', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['y0', 'x1', 'r2'], 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_only', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False}
)
@triton.jit
def triton_red_fused_embedding_slice_sum_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, y0_numel, x1_numel, r2_numel, Y0BLOCK : tl.constexpr, X1BLOCK : tl.constexpr, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr, R2BLOCK_SUB : tl.constexpr):
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
            x1 = x1_offset + (loop_x1 * X1BLOCK_SUB) + base_x1[None,None,:]
            x1_mask = x1 < min(X1BLOCK+x1_offset, x1_numel)
            _tmp8 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp17 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp26 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp35 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp44 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp53 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp62 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp71 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp80 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp89 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp98 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp107 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp116 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp125 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp134 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp143 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp152 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp161 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp170 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp179 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp188 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            for loop_r2 in range(loops_r2):
                r2 = (loop_r2 * R2BLOCK_SUB) + base_r2[None,:,None]
                r2_mask = r2 < r2_numel
                tmp0 = tl.load(in_ptr0 + (r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp10 = tl.load(in_ptr0 + (20 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp19 = tl.load(in_ptr0 + (40 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp28 = tl.load(in_ptr0 + (60 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp37 = tl.load(in_ptr0 + (80 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp46 = tl.load(in_ptr0 + (100 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp55 = tl.load(in_ptr0 + (120 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp64 = tl.load(in_ptr0 + (140 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp73 = tl.load(in_ptr0 + (160 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp82 = tl.load(in_ptr0 + (180 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp91 = tl.load(in_ptr0 + (200 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp100 = tl.load(in_ptr0 + (220 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp109 = tl.load(in_ptr0 + (240 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp118 = tl.load(in_ptr0 + (260 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp127 = tl.load(in_ptr0 + (280 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp136 = tl.load(in_ptr0 + (300 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp145 = tl.load(in_ptr0 + (320 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp154 = tl.load(in_ptr0 + (340 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp163 = tl.load(in_ptr0 + (360 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp172 = tl.load(in_ptr0 + (380 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp181 = tl.load(in_ptr0 + (400 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp1 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 5000, tl.int32)
                tmp2 = tmp0 + tmp1
                tmp3 = tmp0 < 0
                tmp4 = tl.where(tmp3, tmp2, tmp0)
                tl.device_assert(((0 <= tmp4) & (tmp4 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp4 < 5000")
                tmp6 = tl.load(in_ptr1 + (x1 + 8*tmp4), r2_mask & x1_mask & y0_mask)
                tmp7 = tl.reshape(tmp6, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp9 = _tmp8 + tmp7
                _tmp8 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp9, _tmp8)
                tmp11 = tmp10 + tmp1
                tmp12 = tmp10 < 0
                tmp13 = tl.where(tmp12, tmp11, tmp10)
                tl.device_assert(((0 <= tmp13) & (tmp13 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp13 < 5000")
                tmp15 = tl.load(in_ptr1 + (x1 + 8*tmp13), r2_mask & x1_mask & y0_mask)
                tmp16 = tl.reshape(tmp15, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp18 = _tmp17 + tmp16
                _tmp17 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp18, _tmp17)
                tmp20 = tmp19 + tmp1
                tmp21 = tmp19 < 0
                tmp22 = tl.where(tmp21, tmp20, tmp19)
                tl.device_assert(((0 <= tmp22) & (tmp22 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp22 < 5000")
                tmp24 = tl.load(in_ptr1 + (x1 + 8*tmp22), r2_mask & x1_mask & y0_mask)
                tmp25 = tl.reshape(tmp24, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp27 = _tmp26 + tmp25
                _tmp26 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp27, _tmp26)
                tmp29 = tmp28 + tmp1
                tmp30 = tmp28 < 0
                tmp31 = tl.where(tmp30, tmp29, tmp28)
                tl.device_assert(((0 <= tmp31) & (tmp31 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp31 < 5000")
                tmp33 = tl.load(in_ptr1 + (x1 + 8*tmp31), r2_mask & x1_mask & y0_mask)
                tmp34 = tl.reshape(tmp33, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp36 = _tmp35 + tmp34
                _tmp35 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp36, _tmp35)
                tmp38 = tmp37 + tmp1
                tmp39 = tmp37 < 0
                tmp40 = tl.where(tmp39, tmp38, tmp37)
                tl.device_assert(((0 <= tmp40) & (tmp40 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp40 < 5000")
                tmp42 = tl.load(in_ptr1 + (x1 + 8*tmp40), r2_mask & x1_mask & y0_mask)
                tmp43 = tl.reshape(tmp42, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp45 = _tmp44 + tmp43
                _tmp44 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp45, _tmp44)
                tmp47 = tmp46 + tmp1
                tmp48 = tmp46 < 0
                tmp49 = tl.where(tmp48, tmp47, tmp46)
                tl.device_assert(((0 <= tmp49) & (tmp49 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp49 < 5000")
                tmp51 = tl.load(in_ptr1 + (x1 + 8*tmp49), r2_mask & x1_mask & y0_mask)
                tmp52 = tl.reshape(tmp51, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp54 = _tmp53 + tmp52
                _tmp53 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp54, _tmp53)
                tmp56 = tmp55 + tmp1
                tmp57 = tmp55 < 0
                tmp58 = tl.where(tmp57, tmp56, tmp55)
                tl.device_assert(((0 <= tmp58) & (tmp58 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp58 < 5000")
                tmp60 = tl.load(in_ptr1 + (x1 + 8*tmp58), r2_mask & x1_mask & y0_mask)
                tmp61 = tl.reshape(tmp60, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp63 = _tmp62 + tmp61
                _tmp62 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp63, _tmp62)
                tmp65 = tmp64 + tmp1
                tmp66 = tmp64 < 0
                tmp67 = tl.where(tmp66, tmp65, tmp64)
                tl.device_assert(((0 <= tmp67) & (tmp67 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp67 < 5000")
                tmp69 = tl.load(in_ptr1 + (x1 + 8*tmp67), r2_mask & x1_mask & y0_mask)
                tmp70 = tl.reshape(tmp69, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp72 = _tmp71 + tmp70
                _tmp71 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp72, _tmp71)
                tmp74 = tmp73 + tmp1
                tmp75 = tmp73 < 0
                tmp76 = tl.where(tmp75, tmp74, tmp73)
                tl.device_assert(((0 <= tmp76) & (tmp76 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp76 < 5000")
                tmp78 = tl.load(in_ptr1 + (x1 + 8*tmp76), r2_mask & x1_mask & y0_mask)
                tmp79 = tl.reshape(tmp78, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp81 = _tmp80 + tmp79
                _tmp80 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp81, _tmp80)
                tmp83 = tmp82 + tmp1
                tmp84 = tmp82 < 0
                tmp85 = tl.where(tmp84, tmp83, tmp82)
                tl.device_assert(((0 <= tmp85) & (tmp85 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp85 < 5000")
                tmp87 = tl.load(in_ptr1 + (x1 + 8*tmp85), r2_mask & x1_mask & y0_mask)
                tmp88 = tl.reshape(tmp87, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp90 = _tmp89 + tmp88
                _tmp89 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp90, _tmp89)
                tmp92 = tmp91 + tmp1
                tmp93 = tmp91 < 0
                tmp94 = tl.where(tmp93, tmp92, tmp91)
                tl.device_assert(((0 <= tmp94) & (tmp94 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp94 < 5000")
                tmp96 = tl.load(in_ptr1 + (x1 + 8*tmp94), r2_mask & x1_mask & y0_mask)
                tmp97 = tl.reshape(tmp96, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp99 = _tmp98 + tmp97
                _tmp98 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp99, _tmp98)
                tmp101 = tmp100 + tmp1
                tmp102 = tmp100 < 0
                tmp103 = tl.where(tmp102, tmp101, tmp100)
                tl.device_assert(((0 <= tmp103) & (tmp103 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp103 < 5000")
                tmp105 = tl.load(in_ptr1 + (x1 + 8*tmp103), r2_mask & x1_mask & y0_mask)
                tmp106 = tl.reshape(tmp105, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp108 = _tmp107 + tmp106
                _tmp107 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp108, _tmp107)
                tmp110 = tmp109 + tmp1
                tmp111 = tmp109 < 0
                tmp112 = tl.where(tmp111, tmp110, tmp109)
                tl.device_assert(((0 <= tmp112) & (tmp112 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp112 < 5000")
                tmp114 = tl.load(in_ptr1 + (x1 + 8*tmp112), r2_mask & x1_mask & y0_mask)
                tmp115 = tl.reshape(tmp114, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp117 = _tmp116 + tmp115
                _tmp116 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp117, _tmp116)
                tmp119 = tmp118 + tmp1
                tmp120 = tmp118 < 0
                tmp121 = tl.where(tmp120, tmp119, tmp118)
                tl.device_assert(((0 <= tmp121) & (tmp121 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp121 < 5000")
                tmp123 = tl.load(in_ptr1 + (x1 + 8*tmp121), r2_mask & x1_mask & y0_mask)
                tmp124 = tl.reshape(tmp123, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp126 = _tmp125 + tmp124
                _tmp125 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp126, _tmp125)
                tmp128 = tmp127 + tmp1
                tmp129 = tmp127 < 0
                tmp130 = tl.where(tmp129, tmp128, tmp127)
                tl.device_assert(((0 <= tmp130) & (tmp130 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp130 < 5000")
                tmp132 = tl.load(in_ptr1 + (x1 + 8*tmp130), r2_mask & x1_mask & y0_mask)
                tmp133 = tl.reshape(tmp132, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp135 = _tmp134 + tmp133
                _tmp134 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp135, _tmp134)
                tmp137 = tmp136 + tmp1
                tmp138 = tmp136 < 0
                tmp139 = tl.where(tmp138, tmp137, tmp136)
                tl.device_assert(((0 <= tmp139) & (tmp139 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp139 < 5000")
                tmp141 = tl.load(in_ptr1 + (x1 + 8*tmp139), r2_mask & x1_mask & y0_mask)
                tmp142 = tl.reshape(tmp141, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp144 = _tmp143 + tmp142
                _tmp143 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp144, _tmp143)
                tmp146 = tmp145 + tmp1
                tmp147 = tmp145 < 0
                tmp148 = tl.where(tmp147, tmp146, tmp145)
                tl.device_assert(((0 <= tmp148) & (tmp148 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp148 < 5000")
                tmp150 = tl.load(in_ptr1 + (x1 + 8*tmp148), r2_mask & x1_mask & y0_mask)
                tmp151 = tl.reshape(tmp150, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp153 = _tmp152 + tmp151
                _tmp152 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp153, _tmp152)
                tmp155 = tmp154 + tmp1
                tmp156 = tmp154 < 0
                tmp157 = tl.where(tmp156, tmp155, tmp154)
                tl.device_assert(((0 <= tmp157) & (tmp157 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp157 < 5000")
                tmp159 = tl.load(in_ptr1 + (x1 + 8*tmp157), r2_mask & x1_mask & y0_mask)
                tmp160 = tl.reshape(tmp159, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp162 = _tmp161 + tmp160
                _tmp161 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp162, _tmp161)
                tmp164 = tmp163 + tmp1
                tmp165 = tmp163 < 0
                tmp166 = tl.where(tmp165, tmp164, tmp163)
                tl.device_assert(((0 <= tmp166) & (tmp166 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp166 < 5000")
                tmp168 = tl.load(in_ptr1 + (x1 + 8*tmp166), r2_mask & x1_mask & y0_mask)
                tmp169 = tl.reshape(tmp168, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp171 = _tmp170 + tmp169
                _tmp170 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp171, _tmp170)
                tmp173 = tmp172 + tmp1
                tmp174 = tmp172 < 0
                tmp175 = tl.where(tmp174, tmp173, tmp172)
                tl.device_assert(((0 <= tmp175) & (tmp175 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp175 < 5000")
                tmp177 = tl.load(in_ptr1 + (x1 + 8*tmp175), r2_mask & x1_mask & y0_mask)
                tmp178 = tl.reshape(tmp177, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp180 = _tmp179 + tmp178
                _tmp179 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp180, _tmp179)
                tmp182 = tmp181 + tmp1
                tmp183 = tmp181 < 0
                tmp184 = tl.where(tmp183, tmp182, tmp181)
                tl.device_assert(((0 <= tmp184) & (tmp184 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp184 < 5000")
                tmp186 = tl.load(in_ptr1 + (x1 + 8*tmp184), r2_mask & x1_mask & y0_mask)
                tmp187 = tl.reshape(tmp186, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp189 = _tmp188 + tmp187
                _tmp188 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp189, _tmp188)
            tmp8 = tl.sum(_tmp8, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp17 = tl.sum(_tmp17, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp26 = tl.sum(_tmp26, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp35 = tl.sum(_tmp35, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp44 = tl.sum(_tmp44, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp53 = tl.sum(_tmp53, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp62 = tl.sum(_tmp62, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp71 = tl.sum(_tmp71, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp80 = tl.sum(_tmp80, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp89 = tl.sum(_tmp89, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp98 = tl.sum(_tmp98, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp107 = tl.sum(_tmp107, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp116 = tl.sum(_tmp116, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp125 = tl.sum(_tmp125, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp134 = tl.sum(_tmp134, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp143 = tl.sum(_tmp143, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp152 = tl.sum(_tmp152, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp161 = tl.sum(_tmp161, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp170 = tl.sum(_tmp170, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp179 = tl.sum(_tmp179, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp188 = tl.sum(_tmp188, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tl.store(out_ptr0 + (x1 + 8*y0 ), tmp8, x1_mask & y0_mask)
            tl.store(out_ptr1 + (x1 + 8*y0 ), tmp17, x1_mask & y0_mask)
            tl.store(out_ptr2 + (x1 + 8*y0 ), tmp26, x1_mask & y0_mask)
            tl.store(out_ptr3 + (x1 + 8*y0 ), tmp35, x1_mask & y0_mask)
            tl.store(out_ptr4 + (x1 + 8*y0 ), tmp44, x1_mask & y0_mask)
            tl.store(out_ptr5 + (x1 + 8*y0 ), tmp53, x1_mask & y0_mask)
            tl.store(out_ptr6 + (x1 + 8*y0 ), tmp62, x1_mask & y0_mask)
            tl.store(out_ptr7 + (x1 + 8*y0 ), tmp71, x1_mask & y0_mask)
            tl.store(out_ptr8 + (x1 + 8*y0 ), tmp80, x1_mask & y0_mask)
            tl.store(out_ptr9 + (x1 + 8*y0 ), tmp89, x1_mask & y0_mask)
            tl.store(out_ptr10 + (x1 + 8*y0 ), tmp98, x1_mask & y0_mask)
            tl.store(out_ptr11 + (x1 + 8*y0 ), tmp107, x1_mask & y0_mask)
            tl.store(out_ptr12 + (x1 + 8*y0 ), tmp116, x1_mask & y0_mask)
            tl.store(out_ptr13 + (x1 + 8*y0 ), tmp125, x1_mask & y0_mask)
            tl.store(out_ptr14 + (x1 + 8*y0 ), tmp134, x1_mask & y0_mask)
            tl.store(out_ptr15 + (x1 + 8*y0 ), tmp143, x1_mask & y0_mask)
            tl.store(out_ptr16 + (x1 + 8*y0 ), tmp152, x1_mask & y0_mask)
            tl.store(out_ptr17 + (x1 + 8*y0 ), tmp161, x1_mask & y0_mask)
            tl.store(out_ptr18 + (x1 + 8*y0 ), tmp170, x1_mask & y0_mask)
            tl.store(out_ptr19 + (x1 + 8*y0 ), tmp179, x1_mask & y0_mask)
            tl.store(out_ptr20 + (x1 + 8*y0 ), tmp188, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: /tmp/torchinductor_root/4y/c4yo37g2ipjazqqwykg3negi6i4c3s4mpliyjngc5t4s44bfxcwx.py
# Topologically Sorted Source Nodes: [getitem_23, emb_21, group_sum_21, getitem_24, emb_22, group_sum_22, getitem_25, emb_23, group_sum_23, getitem_26, emb_24, group_sum_24, getitem_27, emb_25, group_sum_25, getitem_28, emb_26, group_sum_26, getitem_29, emb_27, group_sum_27, getitem_30, emb_28, group_sum_28, getitem_31, emb_29, group_sum_29, getitem_32, emb_30, group_sum_30, getitem_33, emb_31, group_sum_31, getitem_34, emb_32, group_sum_32, getitem_35, emb_33, group_sum_33, getitem_36, emb_34, group_sum_34, getitem_37, emb_35, group_sum_35, getitem_38, emb_36, group_sum_36, getitem_39, emb_37, group_sum_37, getitem_40, emb_38, group_sum_38, getitem_41, emb_39, group_sum_39, getitem_42, emb_40, group_sum_40, getitem_43, emb_41, group_sum_41], Original ATen: [aten.slice, aten.embedding, aten.sum]
# Source node to ATen node mapping:
#   emb_21 => embedding_21
#   emb_22 => embedding_22
#   emb_23 => embedding_23
#   emb_24 => embedding_24
#   emb_25 => embedding_25
#   emb_26 => embedding_26
#   emb_27 => embedding_27
#   emb_28 => embedding_28
#   emb_29 => embedding_29
#   emb_30 => embedding_30
#   emb_31 => embedding_31
#   emb_32 => embedding_32
#   emb_33 => embedding_33
#   emb_34 => embedding_34
#   emb_35 => embedding_35
#   emb_36 => embedding_36
#   emb_37 => embedding_37
#   emb_38 => embedding_38
#   emb_39 => embedding_39
#   emb_40 => embedding_40
#   emb_41 => embedding_41
#   getitem_23 => slice_44
#   getitem_24 => slice_46
#   getitem_25 => slice_48
#   getitem_26 => slice_50
#   getitem_27 => slice_52
#   getitem_28 => slice_54
#   getitem_29 => slice_56
#   getitem_30 => slice_58
#   getitem_31 => slice_60
#   getitem_32 => slice_62
#   getitem_33 => slice_64
#   getitem_34 => slice_66
#   getitem_35 => slice_68
#   getitem_36 => slice_70
#   getitem_37 => slice_72
#   getitem_38 => slice_74
#   getitem_39 => slice_76
#   getitem_40 => slice_78
#   getitem_41 => slice_80
#   getitem_42 => slice_82
#   getitem_43 => slice_84
#   group_sum_21 => sum_22
#   group_sum_22 => sum_23
#   group_sum_23 => sum_24
#   group_sum_24 => sum_25
#   group_sum_25 => sum_26
#   group_sum_26 => sum_27
#   group_sum_27 => sum_28
#   group_sum_28 => sum_29
#   group_sum_29 => sum_30
#   group_sum_30 => sum_31
#   group_sum_31 => sum_32
#   group_sum_32 => sum_33
#   group_sum_33 => sum_34
#   group_sum_34 => sum_35
#   group_sum_35 => sum_36
#   group_sum_36 => sum_37
#   group_sum_37 => sum_38
#   group_sum_38 => sum_39
#   group_sum_39 => sum_40
#   group_sum_40 => sum_41
#   group_sum_41 => sum_42
# Graph fragment:
#   %arg1_1 : Tensor "i64[s20, 1000][1000, 1]npu:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[5000, 8][8, 1]npu:0" = PlaceHolder[target=arg2_1]
#   %slice_44 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 420, 440), kwargs = {})
#   %embedding_21 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_44), kwargs = {})
#   %sum_22 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_21, [1]), kwargs = {})
#   %slice_46 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 440, 460), kwargs = {})
#   %embedding_22 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_46), kwargs = {})
#   %sum_23 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_22, [1]), kwargs = {})
#   %slice_48 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 460, 480), kwargs = {})
#   %embedding_23 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_48), kwargs = {})
#   %sum_24 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_23, [1]), kwargs = {})
#   %slice_50 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 480, 500), kwargs = {})
#   %embedding_24 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_50), kwargs = {})
#   %sum_25 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_24, [1]), kwargs = {})
#   %slice_52 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 500, 520), kwargs = {})
#   %embedding_25 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_52), kwargs = {})
#   %sum_26 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_25, [1]), kwargs = {})
#   %slice_54 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 520, 540), kwargs = {})
#   %embedding_26 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_54), kwargs = {})
#   %sum_27 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_26, [1]), kwargs = {})
#   %slice_56 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 540, 560), kwargs = {})
#   %embedding_27 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_56), kwargs = {})
#   %sum_28 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_27, [1]), kwargs = {})
#   %slice_58 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 560, 580), kwargs = {})
#   %embedding_28 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_58), kwargs = {})
#   %sum_29 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_28, [1]), kwargs = {})
#   %slice_60 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 580, 600), kwargs = {})
#   %embedding_29 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_60), kwargs = {})
#   %sum_30 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_29, [1]), kwargs = {})
#   %slice_62 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 600, 620), kwargs = {})
#   %embedding_30 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_62), kwargs = {})
#   %sum_31 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_30, [1]), kwargs = {})
#   %slice_64 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 620, 640), kwargs = {})
#   %embedding_31 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_64), kwargs = {})
#   %sum_32 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_31, [1]), kwargs = {})
#   %slice_66 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 640, 660), kwargs = {})
#   %embedding_32 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_66), kwargs = {})
#   %sum_33 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_32, [1]), kwargs = {})
#   %slice_68 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 660, 680), kwargs = {})
#   %embedding_33 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_68), kwargs = {})
#   %sum_34 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_33, [1]), kwargs = {})
#   %slice_70 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 680, 700), kwargs = {})
#   %embedding_34 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_70), kwargs = {})
#   %sum_35 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_34, [1]), kwargs = {})
#   %slice_72 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 700, 720), kwargs = {})
#   %embedding_35 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_72), kwargs = {})
#   %sum_36 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_35, [1]), kwargs = {})
#   %slice_74 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 720, 740), kwargs = {})
#   %embedding_36 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_74), kwargs = {})
#   %sum_37 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_36, [1]), kwargs = {})
#   %slice_76 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 740, 760), kwargs = {})
#   %embedding_37 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_76), kwargs = {})
#   %sum_38 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_37, [1]), kwargs = {})
#   %slice_78 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 760, 780), kwargs = {})
#   %embedding_38 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_78), kwargs = {})
#   %sum_39 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_38, [1]), kwargs = {})
#   %slice_80 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 780, 800), kwargs = {})
#   %embedding_39 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_80), kwargs = {})
#   %sum_40 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_39, [1]), kwargs = {})
#   %slice_82 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 800, 820), kwargs = {})
#   %embedding_40 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_82), kwargs = {})
#   %sum_41 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_40, [1]), kwargs = {})
#   %slice_84 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 820, 840), kwargs = {})
#   %embedding_41 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_84), kwargs = {})
#   %sum_42 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_41, [1]), kwargs = {})
#   return %sum_22,%sum_23,%sum_24,%sum_25,%sum_26,%sum_27,%sum_28,%sum_29,%sum_30,%sum_31,%sum_32,%sum_33,%sum_34,%sum_35,%sum_36,%sum_37,%sum_38,%sum_39,%sum_40,%sum_41,%sum_42
# SchedulerNodes: [SchedulerNode(name='op46'), SchedulerNode(name='op47'), SchedulerNode(name='op48'), SchedulerNode(name='op49'), SchedulerNode(name='op50'), SchedulerNode(name='op51'), SchedulerNode(name='op52'), SchedulerNode(name='op53'), SchedulerNode(name='op54'), SchedulerNode(name='op55'), SchedulerNode(name='op56'), SchedulerNode(name='op57'), SchedulerNode(name='op58'), SchedulerNode(name='op59'), SchedulerNode(name='op60'), SchedulerNode(name='op61'), SchedulerNode(name='op62'), SchedulerNode(name='op63'), SchedulerNode(name='op64'), SchedulerNode(name='op65'), SchedulerNode(name='op66')]

triton_red_fused_embedding_slice_sum_4 = async_compile.triton('triton_red_fused_embedding_slice_sum_4', '''
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
    size_hints={'y0': 128, 'x1': 8, 'r2': 20},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*fp32', 'out_ptr11': '*fp32', 'out_ptr12': '*fp32', 'out_ptr13': '*fp32', 'out_ptr14': '*fp32', 'out_ptr15': '*fp32', 'out_ptr16': '*fp32', 'out_ptr17': '*fp32', 'out_ptr18': '*fp32', 'out_ptr19': '*fp32', 'out_ptr20': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'r2_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_embedding_slice_sum_4', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['y0', 'x1', 'r2'], 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_only', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False}
)
@triton.jit
def triton_red_fused_embedding_slice_sum_4(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, y0_numel, x1_numel, r2_numel, Y0BLOCK : tl.constexpr, X1BLOCK : tl.constexpr, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr, R2BLOCK_SUB : tl.constexpr):
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
            x1 = x1_offset + (loop_x1 * X1BLOCK_SUB) + base_x1[None,None,:]
            x1_mask = x1 < min(X1BLOCK+x1_offset, x1_numel)
            _tmp8 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp17 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp26 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp35 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp44 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp53 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp62 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp71 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp80 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp89 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp98 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp107 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp116 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp125 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp134 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp143 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp152 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp161 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp170 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp179 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp188 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            for loop_r2 in range(loops_r2):
                r2 = (loop_r2 * R2BLOCK_SUB) + base_r2[None,:,None]
                r2_mask = r2 < r2_numel
                tmp0 = tl.load(in_ptr0 + (420 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp10 = tl.load(in_ptr0 + (440 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp19 = tl.load(in_ptr0 + (460 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp28 = tl.load(in_ptr0 + (480 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp37 = tl.load(in_ptr0 + (500 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp46 = tl.load(in_ptr0 + (520 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp55 = tl.load(in_ptr0 + (540 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp64 = tl.load(in_ptr0 + (560 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp73 = tl.load(in_ptr0 + (580 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp82 = tl.load(in_ptr0 + (600 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp91 = tl.load(in_ptr0 + (620 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp100 = tl.load(in_ptr0 + (640 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp109 = tl.load(in_ptr0 + (660 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp118 = tl.load(in_ptr0 + (680 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp127 = tl.load(in_ptr0 + (700 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp136 = tl.load(in_ptr0 + (720 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp145 = tl.load(in_ptr0 + (740 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp154 = tl.load(in_ptr0 + (760 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp163 = tl.load(in_ptr0 + (780 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp172 = tl.load(in_ptr0 + (800 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp181 = tl.load(in_ptr0 + (820 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp1 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 5000, tl.int32)
                tmp2 = tmp0 + tmp1
                tmp3 = tmp0 < 0
                tmp4 = tl.where(tmp3, tmp2, tmp0)
                tl.device_assert(((0 <= tmp4) & (tmp4 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp4 < 5000")
                tmp6 = tl.load(in_ptr1 + (x1 + 8*tmp4), r2_mask & x1_mask & y0_mask)
                tmp7 = tl.reshape(tmp6, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp9 = _tmp8 + tmp7
                _tmp8 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp9, _tmp8)
                tmp11 = tmp10 + tmp1
                tmp12 = tmp10 < 0
                tmp13 = tl.where(tmp12, tmp11, tmp10)
                tl.device_assert(((0 <= tmp13) & (tmp13 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp13 < 5000")
                tmp15 = tl.load(in_ptr1 + (x1 + 8*tmp13), r2_mask & x1_mask & y0_mask)
                tmp16 = tl.reshape(tmp15, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp18 = _tmp17 + tmp16
                _tmp17 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp18, _tmp17)
                tmp20 = tmp19 + tmp1
                tmp21 = tmp19 < 0
                tmp22 = tl.where(tmp21, tmp20, tmp19)
                tl.device_assert(((0 <= tmp22) & (tmp22 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp22 < 5000")
                tmp24 = tl.load(in_ptr1 + (x1 + 8*tmp22), r2_mask & x1_mask & y0_mask)
                tmp25 = tl.reshape(tmp24, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp27 = _tmp26 + tmp25
                _tmp26 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp27, _tmp26)
                tmp29 = tmp28 + tmp1
                tmp30 = tmp28 < 0
                tmp31 = tl.where(tmp30, tmp29, tmp28)
                tl.device_assert(((0 <= tmp31) & (tmp31 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp31 < 5000")
                tmp33 = tl.load(in_ptr1 + (x1 + 8*tmp31), r2_mask & x1_mask & y0_mask)
                tmp34 = tl.reshape(tmp33, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp36 = _tmp35 + tmp34
                _tmp35 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp36, _tmp35)
                tmp38 = tmp37 + tmp1
                tmp39 = tmp37 < 0
                tmp40 = tl.where(tmp39, tmp38, tmp37)
                tl.device_assert(((0 <= tmp40) & (tmp40 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp40 < 5000")
                tmp42 = tl.load(in_ptr1 + (x1 + 8*tmp40), r2_mask & x1_mask & y0_mask)
                tmp43 = tl.reshape(tmp42, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp45 = _tmp44 + tmp43
                _tmp44 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp45, _tmp44)
                tmp47 = tmp46 + tmp1
                tmp48 = tmp46 < 0
                tmp49 = tl.where(tmp48, tmp47, tmp46)
                tl.device_assert(((0 <= tmp49) & (tmp49 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp49 < 5000")
                tmp51 = tl.load(in_ptr1 + (x1 + 8*tmp49), r2_mask & x1_mask & y0_mask)
                tmp52 = tl.reshape(tmp51, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp54 = _tmp53 + tmp52
                _tmp53 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp54, _tmp53)
                tmp56 = tmp55 + tmp1
                tmp57 = tmp55 < 0
                tmp58 = tl.where(tmp57, tmp56, tmp55)
                tl.device_assert(((0 <= tmp58) & (tmp58 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp58 < 5000")
                tmp60 = tl.load(in_ptr1 + (x1 + 8*tmp58), r2_mask & x1_mask & y0_mask)
                tmp61 = tl.reshape(tmp60, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp63 = _tmp62 + tmp61
                _tmp62 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp63, _tmp62)
                tmp65 = tmp64 + tmp1
                tmp66 = tmp64 < 0
                tmp67 = tl.where(tmp66, tmp65, tmp64)
                tl.device_assert(((0 <= tmp67) & (tmp67 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp67 < 5000")
                tmp69 = tl.load(in_ptr1 + (x1 + 8*tmp67), r2_mask & x1_mask & y0_mask)
                tmp70 = tl.reshape(tmp69, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp72 = _tmp71 + tmp70
                _tmp71 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp72, _tmp71)
                tmp74 = tmp73 + tmp1
                tmp75 = tmp73 < 0
                tmp76 = tl.where(tmp75, tmp74, tmp73)
                tl.device_assert(((0 <= tmp76) & (tmp76 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp76 < 5000")
                tmp78 = tl.load(in_ptr1 + (x1 + 8*tmp76), r2_mask & x1_mask & y0_mask)
                tmp79 = tl.reshape(tmp78, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp81 = _tmp80 + tmp79
                _tmp80 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp81, _tmp80)
                tmp83 = tmp82 + tmp1
                tmp84 = tmp82 < 0
                tmp85 = tl.where(tmp84, tmp83, tmp82)
                tl.device_assert(((0 <= tmp85) & (tmp85 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp85 < 5000")
                tmp87 = tl.load(in_ptr1 + (x1 + 8*tmp85), r2_mask & x1_mask & y0_mask)
                tmp88 = tl.reshape(tmp87, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp90 = _tmp89 + tmp88
                _tmp89 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp90, _tmp89)
                tmp92 = tmp91 + tmp1
                tmp93 = tmp91 < 0
                tmp94 = tl.where(tmp93, tmp92, tmp91)
                tl.device_assert(((0 <= tmp94) & (tmp94 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp94 < 5000")
                tmp96 = tl.load(in_ptr1 + (x1 + 8*tmp94), r2_mask & x1_mask & y0_mask)
                tmp97 = tl.reshape(tmp96, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp99 = _tmp98 + tmp97
                _tmp98 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp99, _tmp98)
                tmp101 = tmp100 + tmp1
                tmp102 = tmp100 < 0
                tmp103 = tl.where(tmp102, tmp101, tmp100)
                tl.device_assert(((0 <= tmp103) & (tmp103 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp103 < 5000")
                tmp105 = tl.load(in_ptr1 + (x1 + 8*tmp103), r2_mask & x1_mask & y0_mask)
                tmp106 = tl.reshape(tmp105, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp108 = _tmp107 + tmp106
                _tmp107 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp108, _tmp107)
                tmp110 = tmp109 + tmp1
                tmp111 = tmp109 < 0
                tmp112 = tl.where(tmp111, tmp110, tmp109)
                tl.device_assert(((0 <= tmp112) & (tmp112 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp112 < 5000")
                tmp114 = tl.load(in_ptr1 + (x1 + 8*tmp112), r2_mask & x1_mask & y0_mask)
                tmp115 = tl.reshape(tmp114, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp117 = _tmp116 + tmp115
                _tmp116 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp117, _tmp116)
                tmp119 = tmp118 + tmp1
                tmp120 = tmp118 < 0
                tmp121 = tl.where(tmp120, tmp119, tmp118)
                tl.device_assert(((0 <= tmp121) & (tmp121 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp121 < 5000")
                tmp123 = tl.load(in_ptr1 + (x1 + 8*tmp121), r2_mask & x1_mask & y0_mask)
                tmp124 = tl.reshape(tmp123, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp126 = _tmp125 + tmp124
                _tmp125 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp126, _tmp125)
                tmp128 = tmp127 + tmp1
                tmp129 = tmp127 < 0
                tmp130 = tl.where(tmp129, tmp128, tmp127)
                tl.device_assert(((0 <= tmp130) & (tmp130 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp130 < 5000")
                tmp132 = tl.load(in_ptr1 + (x1 + 8*tmp130), r2_mask & x1_mask & y0_mask)
                tmp133 = tl.reshape(tmp132, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp135 = _tmp134 + tmp133
                _tmp134 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp135, _tmp134)
                tmp137 = tmp136 + tmp1
                tmp138 = tmp136 < 0
                tmp139 = tl.where(tmp138, tmp137, tmp136)
                tl.device_assert(((0 <= tmp139) & (tmp139 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp139 < 5000")
                tmp141 = tl.load(in_ptr1 + (x1 + 8*tmp139), r2_mask & x1_mask & y0_mask)
                tmp142 = tl.reshape(tmp141, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp144 = _tmp143 + tmp142
                _tmp143 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp144, _tmp143)
                tmp146 = tmp145 + tmp1
                tmp147 = tmp145 < 0
                tmp148 = tl.where(tmp147, tmp146, tmp145)
                tl.device_assert(((0 <= tmp148) & (tmp148 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp148 < 5000")
                tmp150 = tl.load(in_ptr1 + (x1 + 8*tmp148), r2_mask & x1_mask & y0_mask)
                tmp151 = tl.reshape(tmp150, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp153 = _tmp152 + tmp151
                _tmp152 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp153, _tmp152)
                tmp155 = tmp154 + tmp1
                tmp156 = tmp154 < 0
                tmp157 = tl.where(tmp156, tmp155, tmp154)
                tl.device_assert(((0 <= tmp157) & (tmp157 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp157 < 5000")
                tmp159 = tl.load(in_ptr1 + (x1 + 8*tmp157), r2_mask & x1_mask & y0_mask)
                tmp160 = tl.reshape(tmp159, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp162 = _tmp161 + tmp160
                _tmp161 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp162, _tmp161)
                tmp164 = tmp163 + tmp1
                tmp165 = tmp163 < 0
                tmp166 = tl.where(tmp165, tmp164, tmp163)
                tl.device_assert(((0 <= tmp166) & (tmp166 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp166 < 5000")
                tmp168 = tl.load(in_ptr1 + (x1 + 8*tmp166), r2_mask & x1_mask & y0_mask)
                tmp169 = tl.reshape(tmp168, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp171 = _tmp170 + tmp169
                _tmp170 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp171, _tmp170)
                tmp173 = tmp172 + tmp1
                tmp174 = tmp172 < 0
                tmp175 = tl.where(tmp174, tmp173, tmp172)
                tl.device_assert(((0 <= tmp175) & (tmp175 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp175 < 5000")
                tmp177 = tl.load(in_ptr1 + (x1 + 8*tmp175), r2_mask & x1_mask & y0_mask)
                tmp178 = tl.reshape(tmp177, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp180 = _tmp179 + tmp178
                _tmp179 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp180, _tmp179)
                tmp182 = tmp181 + tmp1
                tmp183 = tmp181 < 0
                tmp184 = tl.where(tmp183, tmp182, tmp181)
                tl.device_assert(((0 <= tmp184) & (tmp184 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp184 < 5000")
                tmp186 = tl.load(in_ptr1 + (x1 + 8*tmp184), r2_mask & x1_mask & y0_mask)
                tmp187 = tl.reshape(tmp186, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp189 = _tmp188 + tmp187
                _tmp188 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp189, _tmp188)
            tmp8 = tl.sum(_tmp8, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp17 = tl.sum(_tmp17, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp26 = tl.sum(_tmp26, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp35 = tl.sum(_tmp35, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp44 = tl.sum(_tmp44, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp53 = tl.sum(_tmp53, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp62 = tl.sum(_tmp62, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp71 = tl.sum(_tmp71, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp80 = tl.sum(_tmp80, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp89 = tl.sum(_tmp89, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp98 = tl.sum(_tmp98, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp107 = tl.sum(_tmp107, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp116 = tl.sum(_tmp116, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp125 = tl.sum(_tmp125, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp134 = tl.sum(_tmp134, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp143 = tl.sum(_tmp143, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp152 = tl.sum(_tmp152, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp161 = tl.sum(_tmp161, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp170 = tl.sum(_tmp170, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp179 = tl.sum(_tmp179, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp188 = tl.sum(_tmp188, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tl.store(out_ptr0 + (x1 + 8*y0 ), tmp8, x1_mask & y0_mask)
            tl.store(out_ptr1 + (x1 + 8*y0 ), tmp17, x1_mask & y0_mask)
            tl.store(out_ptr2 + (x1 + 8*y0 ), tmp26, x1_mask & y0_mask)
            tl.store(out_ptr3 + (x1 + 8*y0 ), tmp35, x1_mask & y0_mask)
            tl.store(out_ptr4 + (x1 + 8*y0 ), tmp44, x1_mask & y0_mask)
            tl.store(out_ptr5 + (x1 + 8*y0 ), tmp53, x1_mask & y0_mask)
            tl.store(out_ptr6 + (x1 + 8*y0 ), tmp62, x1_mask & y0_mask)
            tl.store(out_ptr7 + (x1 + 8*y0 ), tmp71, x1_mask & y0_mask)
            tl.store(out_ptr8 + (x1 + 8*y0 ), tmp80, x1_mask & y0_mask)
            tl.store(out_ptr9 + (x1 + 8*y0 ), tmp89, x1_mask & y0_mask)
            tl.store(out_ptr10 + (x1 + 8*y0 ), tmp98, x1_mask & y0_mask)
            tl.store(out_ptr11 + (x1 + 8*y0 ), tmp107, x1_mask & y0_mask)
            tl.store(out_ptr12 + (x1 + 8*y0 ), tmp116, x1_mask & y0_mask)
            tl.store(out_ptr13 + (x1 + 8*y0 ), tmp125, x1_mask & y0_mask)
            tl.store(out_ptr14 + (x1 + 8*y0 ), tmp134, x1_mask & y0_mask)
            tl.store(out_ptr15 + (x1 + 8*y0 ), tmp143, x1_mask & y0_mask)
            tl.store(out_ptr16 + (x1 + 8*y0 ), tmp152, x1_mask & y0_mask)
            tl.store(out_ptr17 + (x1 + 8*y0 ), tmp161, x1_mask & y0_mask)
            tl.store(out_ptr18 + (x1 + 8*y0 ), tmp170, x1_mask & y0_mask)
            tl.store(out_ptr19 + (x1 + 8*y0 ), tmp179, x1_mask & y0_mask)
            tl.store(out_ptr20 + (x1 + 8*y0 ), tmp188, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: /tmp/torchinductor_root/hj/chjfixl6pwl5blimlg2u5dasapxodmfcnrggkuzb4qnwkc2vikiu.py
# Topologically Sorted Source Nodes: [getitem_44, emb_42, group_sum_42, getitem_45, emb_43, group_sum_43, getitem_46, emb_44, group_sum_44, getitem_47, emb_45, group_sum_45, getitem_48, emb_46, group_sum_46, getitem_49, emb_47, group_sum_47, getitem_50, emb_48, group_sum_48, getitem_51, emb_49, group_sum_49], Original ATen: [aten.slice, aten.embedding, aten.sum]
# Source node to ATen node mapping:
#   emb_42 => embedding_42
#   emb_43 => embedding_43
#   emb_44 => embedding_44
#   emb_45 => embedding_45
#   emb_46 => embedding_46
#   emb_47 => embedding_47
#   emb_48 => embedding_48
#   emb_49 => embedding_49
#   getitem_44 => slice_86
#   getitem_45 => slice_88
#   getitem_46 => slice_90
#   getitem_47 => slice_92
#   getitem_48 => slice_94
#   getitem_49 => slice_96
#   getitem_50 => slice_98
#   getitem_51 => slice_100
#   group_sum_42 => sum_43
#   group_sum_43 => sum_44
#   group_sum_44 => sum_45
#   group_sum_45 => sum_46
#   group_sum_46 => sum_47
#   group_sum_47 => sum_48
#   group_sum_48 => sum_49
#   group_sum_49 => sum_50
# Graph fragment:
#   %arg1_1 : Tensor "i64[s20, 1000][1000, 1]npu:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[5000, 8][8, 1]npu:0" = PlaceHolder[target=arg2_1]
#   %slice_86 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 840, 860), kwargs = {})
#   %embedding_42 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_86), kwargs = {})
#   %sum_43 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_42, [1]), kwargs = {})
#   %slice_88 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 860, 880), kwargs = {})
#   %embedding_43 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_88), kwargs = {})
#   %sum_44 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_43, [1]), kwargs = {})
#   %slice_90 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 880, 900), kwargs = {})
#   %embedding_44 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_90), kwargs = {})
#   %sum_45 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_44, [1]), kwargs = {})
#   %slice_92 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 900, 920), kwargs = {})
#   %embedding_45 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_92), kwargs = {})
#   %sum_46 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_45, [1]), kwargs = {})
#   %slice_94 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 920, 940), kwargs = {})
#   %embedding_46 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_94), kwargs = {})
#   %sum_47 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_46, [1]), kwargs = {})
#   %slice_96 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 940, 960), kwargs = {})
#   %embedding_47 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_96), kwargs = {})
#   %sum_48 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_47, [1]), kwargs = {})
#   %slice_98 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 960, 980), kwargs = {})
#   %embedding_48 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_98), kwargs = {})
#   %sum_49 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_48, [1]), kwargs = {})
#   %slice_100 : Tensor "i64[s20, 20][1000, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 980, 1000), kwargs = {})
#   %embedding_49 : Tensor "f32[s20, 20, 8][160, 8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_100), kwargs = {})
#   %sum_50 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%embedding_49, [1]), kwargs = {})
#   return %sum_43,%sum_44,%sum_45,%sum_46,%sum_47,%sum_48,%sum_49,%sum_50
# SchedulerNodes: [SchedulerNode(name='op67'), SchedulerNode(name='op68'), SchedulerNode(name='op69'), SchedulerNode(name='op70'), SchedulerNode(name='op71'), SchedulerNode(name='op72'), SchedulerNode(name='op73'), SchedulerNode(name='op74')]

triton_red_fused_embedding_slice_sum_5 = async_compile.triton('triton_red_fused_embedding_slice_sum_5', '''
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
    size_hints={'y0': 128, 'x1': 8, 'r2': 20},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'r2_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_embedding_slice_sum_5', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['y0', 'x1', 'r2'], 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_only', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False}
)
@triton.jit
def triton_red_fused_embedding_slice_sum_5(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, y0_numel, x1_numel, r2_numel, Y0BLOCK : tl.constexpr, X1BLOCK : tl.constexpr, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr, R2BLOCK_SUB : tl.constexpr):
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
            x1 = x1_offset + (loop_x1 * X1BLOCK_SUB) + base_x1[None,None,:]
            x1_mask = x1 < min(X1BLOCK+x1_offset, x1_numel)
            _tmp8 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp17 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp26 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp35 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp44 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp53 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp62 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            _tmp71 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 0, tl.float32)
            for loop_r2 in range(loops_r2):
                r2 = (loop_r2 * R2BLOCK_SUB) + base_r2[None,:,None]
                r2_mask = r2 < r2_numel
                tmp0 = tl.load(in_ptr0 + (840 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp10 = tl.load(in_ptr0 + (860 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp19 = tl.load(in_ptr0 + (880 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp28 = tl.load(in_ptr0 + (900 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp37 = tl.load(in_ptr0 + (920 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp46 = tl.load(in_ptr0 + (940 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp55 = tl.load(in_ptr0 + (960 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp64 = tl.load(in_ptr0 + (980 + r2 + 1000*y0), r2_mask & y0_mask, other=0.0)
                tmp1 = tl.full([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB], 5000, tl.int32)
                tmp2 = tmp0 + tmp1
                tmp3 = tmp0 < 0
                tmp4 = tl.where(tmp3, tmp2, tmp0)
                tl.device_assert(((0 <= tmp4) & (tmp4 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp4 < 5000")
                tmp6 = tl.load(in_ptr1 + (x1 + 8*tmp4), r2_mask & x1_mask & y0_mask)
                tmp7 = tl.reshape(tmp6, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp9 = _tmp8 + tmp7
                _tmp8 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp9, _tmp8)
                tmp11 = tmp10 + tmp1
                tmp12 = tmp10 < 0
                tmp13 = tl.where(tmp12, tmp11, tmp10)
                tl.device_assert(((0 <= tmp13) & (tmp13 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp13 < 5000")
                tmp15 = tl.load(in_ptr1 + (x1 + 8*tmp13), r2_mask & x1_mask & y0_mask)
                tmp16 = tl.reshape(tmp15, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp18 = _tmp17 + tmp16
                _tmp17 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp18, _tmp17)
                tmp20 = tmp19 + tmp1
                tmp21 = tmp19 < 0
                tmp22 = tl.where(tmp21, tmp20, tmp19)
                tl.device_assert(((0 <= tmp22) & (tmp22 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp22 < 5000")
                tmp24 = tl.load(in_ptr1 + (x1 + 8*tmp22), r2_mask & x1_mask & y0_mask)
                tmp25 = tl.reshape(tmp24, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp27 = _tmp26 + tmp25
                _tmp26 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp27, _tmp26)
                tmp29 = tmp28 + tmp1
                tmp30 = tmp28 < 0
                tmp31 = tl.where(tmp30, tmp29, tmp28)
                tl.device_assert(((0 <= tmp31) & (tmp31 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp31 < 5000")
                tmp33 = tl.load(in_ptr1 + (x1 + 8*tmp31), r2_mask & x1_mask & y0_mask)
                tmp34 = tl.reshape(tmp33, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp36 = _tmp35 + tmp34
                _tmp35 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp36, _tmp35)
                tmp38 = tmp37 + tmp1
                tmp39 = tmp37 < 0
                tmp40 = tl.where(tmp39, tmp38, tmp37)
                tl.device_assert(((0 <= tmp40) & (tmp40 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp40 < 5000")
                tmp42 = tl.load(in_ptr1 + (x1 + 8*tmp40), r2_mask & x1_mask & y0_mask)
                tmp43 = tl.reshape(tmp42, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp45 = _tmp44 + tmp43
                _tmp44 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp45, _tmp44)
                tmp47 = tmp46 + tmp1
                tmp48 = tmp46 < 0
                tmp49 = tl.where(tmp48, tmp47, tmp46)
                tl.device_assert(((0 <= tmp49) & (tmp49 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp49 < 5000")
                tmp51 = tl.load(in_ptr1 + (x1 + 8*tmp49), r2_mask & x1_mask & y0_mask)
                tmp52 = tl.reshape(tmp51, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp54 = _tmp53 + tmp52
                _tmp53 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp54, _tmp53)
                tmp56 = tmp55 + tmp1
                tmp57 = tmp55 < 0
                tmp58 = tl.where(tmp57, tmp56, tmp55)
                tl.device_assert(((0 <= tmp58) & (tmp58 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp58 < 5000")
                tmp60 = tl.load(in_ptr1 + (x1 + 8*tmp58), r2_mask & x1_mask & y0_mask)
                tmp61 = tl.reshape(tmp60, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp63 = _tmp62 + tmp61
                _tmp62 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp63, _tmp62)
                tmp65 = tmp64 + tmp1
                tmp66 = tmp64 < 0
                tmp67 = tl.where(tmp66, tmp65, tmp64)
                tl.device_assert(((0 <= tmp67) & (tmp67 < 5000)) | ~(r2_mask & y0_mask), "index out of bounds: 0 <= tmp67 < 5000")
                tmp69 = tl.load(in_ptr1 + (x1 + 8*tmp67), r2_mask & x1_mask & y0_mask)
                tmp70 = tl.reshape(tmp69, [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB])
                tmp72 = _tmp71 + tmp70
                _tmp71 = tl.where((r2_mask & x1_mask & y0_mask).reshape([Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp72, _tmp71)
            tmp8 = tl.sum(_tmp8, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp17 = tl.sum(_tmp17, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp26 = tl.sum(_tmp26, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp35 = tl.sum(_tmp35, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp44 = tl.sum(_tmp44, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp53 = tl.sum(_tmp53, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp62 = tl.sum(_tmp62, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp71 = tl.sum(_tmp71, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tl.store(out_ptr0 + (x1 + 8*y0 ), tmp8, x1_mask & y0_mask)
            tl.store(out_ptr1 + (x1 + 8*y0 ), tmp17, x1_mask & y0_mask)
            tl.store(out_ptr2 + (x1 + 8*y0 ), tmp26, x1_mask & y0_mask)
            tl.store(out_ptr3 + (x1 + 8*y0 ), tmp35, x1_mask & y0_mask)
            tl.store(out_ptr4 + (x1 + 8*y0 ), tmp44, x1_mask & y0_mask)
            tl.store(out_ptr5 + (x1 + 8*y0 ), tmp53, x1_mask & y0_mask)
            tl.store(out_ptr6 + (x1 + 8*y0 ), tmp62, x1_mask & y0_mask)
            tl.store(out_ptr7 + (x1 + 8*y0 ), tmp71, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: /tmp/torchinductor_root/rw/crwvab6gbbve5uvg2qanvmd72mwnjr26pa3khx7tu4hj2536ti5h.py
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
#   %arg29_1 : Tensor "f32[8][1]npu:0" = PlaceHolder[target=arg29_1]
#   %mm_default_1 : Tensor "f32[s20, 8][8, 1]npu:0" = PlaceHolder[target=mm_default_1]
#   %relu_12 : Tensor "f32[s20, 8][408, 1]npu:0" = PlaceHolder[target=relu_12]
#   %iota : Tensor "i32[408][1]npu:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (408,), kwargs = {start: 0, step: 1, dtype: torch.int32, device: npu:0, requires_grad: False})
#   %unsqueeze_1 : Tensor "i32[1, 408][408, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, -2), kwargs = {})
#   %unsqueeze_2 : Tensor "i32[408, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, -1), kwargs = {})
#   %ge_tensor : Tensor "b8[408, 408][408, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%unsqueeze_1, %unsqueeze_2), kwargs = {})
#   %full_default_2 : Tensor "f32[s20, 408, 408][166464, 408, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%arg0_1, 408, 408], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %add_tensor_1 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg29_1, %mm_default_1), kwargs = {})
#   %relu_12 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_1,), kwargs = {})
#   %where_1 : Tensor "f32[s20, 408, 408][166464, 408, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_tensor, %full_default_2, %bmm), kwargs = {})
#   %view_3 : Tensor "f32[s20, 166464][166464, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%where_1, [%arg0_1, 166464]), kwargs = {})
#   %cat_2 : Tensor "f32[s20, 166472][166472, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_12, %view_3], -1), kwargs = {})
#   return %relu_12,%buf81
# SchedulerNodes: [SchedulerNode(name='op77'), SchedulerNode(name='op81')]

triton_poi_fused_addmm_cat_relu_triu_view_wher_6 = async_compile.triton('triton_poi_fused_addmm_cat_relu_triu_view_wher_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_cat_relu_triu_view_wher_6', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_only', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_cat_relu_triu_view_wher_6(in_ptr0, in_ptr1, out_ptr0, out_ptr1, y0_numel, x1_numel, Y0BLOCK : tl.constexpr, X1BLOCK : tl.constexpr, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
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


# kernel path: /tmp/torchinductor_root/uv/cuvq73ktljy47lzw66hvmif4vremsd3lznrvxzrzifrsda74l5gy.py
# Topologically Sorted Source Nodes: [input_25, input_26, cat_1], Original ATen: [aten.addmm, aten.relu, aten.cat]
# Source node to ATen node mapping:
#   cat_1 => cat_1
#   input_25 => add_tensor_1
#   input_26 => relu_12
# Graph fragment:
#   %cat : Tensor "f32[s20, 400][400, 1]npu:0" = PlaceHolder[target=cat]
#   %add_tensor_1 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg29_1, %mm_default_1), kwargs = {})
#   %relu_12 : Tensor "f32[s20, 8][8, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_1,), kwargs = {})
#   %cat_1 : Tensor "f32[s20, 408][408, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_12, %cat], 1), kwargs = {})
#   return %buf78
# SchedulerNodes: [SchedulerNode(name='op78')]

triton_poi_fused_addmm_cat_relu_7 = async_compile.triton('triton_poi_fused_addmm_cat_relu_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_cat_relu_7', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_only', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_cat_relu_7(in_ptr0, out_ptr0, y0_numel, x1_numel, Y0BLOCK : tl.constexpr, X1BLOCK : tl.constexpr, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
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


# kernel path: /tmp/torchinductor_root/wp/cwpv4x5qju3sjta3fqomfgheiuz4v4cuv7rkg6isgs2digmxwxke.py
# Topologically Sorted Source Nodes: [upper_tri_mask, zeros_like, activations, activations_1, concat], Original ATen: [aten.triu, aten.zeros_like, aten.where, aten.view, aten.cat]
# Source node to ATen node mapping:
#   activations => where_1
#   activations_1 => view_3
#   concat => cat_2
#   upper_tri_mask => iota, unsqueeze_1, unsqueeze_2
#   zeros_like => full_default_2
# Graph fragment:
#   %bmm : Tensor "f32[s20, 408, 408][166464, 408, 1]npu:0" = PlaceHolder[target=bmm]
#   %iota : Tensor "i32[408][1]npu:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (408,), kwargs = {start: 0, step: 1, dtype: torch.int32, device: npu:0, requires_grad: False})
#   %unsqueeze_1 : Tensor "i32[1, 408][408, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, -2), kwargs = {})
#   %unsqueeze_2 : Tensor "i32[408, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, -1), kwargs = {})
#   %ge_tensor : Tensor "b8[408, 408][408, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%unsqueeze_1, %unsqueeze_2), kwargs = {})
#   %full_default_2 : Tensor "f32[s20, 408, 408][166464, 408, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%arg0_1, 408, 408], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_1 : Tensor "f32[s20, 408, 408][166464, 408, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_tensor, %full_default_2, %bmm), kwargs = {})
#   %view_3 : Tensor "f32[s20, 166464][166464, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%where_1, [%arg0_1, 166464]), kwargs = {})
#   %cat_2 : Tensor "f32[s20, 166472][166472, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_12, %view_3], -1), kwargs = {})
#   return %buf82
# SchedulerNodes: [SchedulerNode(name='op82')]

triton_poi_fused_cat_triu_view_where_zeros_lik_8 = async_compile.triton('triton_poi_fused_cat_triu_view_where_zeros_lik_8', '''
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
    size_hints={'y0': 128, 'x1': 166464}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_triu_view_where_zeros_lik_8', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_only', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_triu_view_where_zeros_lik_8(in_ptr0, out_ptr0, y0_numel, x1_numel, Y0BLOCK : tl.constexpr, X1BLOCK : tl.constexpr, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
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
            tmp3 = tl.load(in_ptr0 + (x1 + 166464*y0), x1_mask & y0_mask)
            tmp0 = (x1 % 408)
            tmp1 = x1 // 408
            tmp2 = tmp0 >= tmp1
            tmp4 = 0.0
            tmp5 = tl.where(tmp2, tmp4, tmp3)
            tl.store(out_ptr0 + (x1 + 166472*y0), tmp5, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: /tmp/torchinductor_root/yg/cygxs4bqgkanu6io4xq6btagtf6iu2zeakipe45rdja7kudf7lwe.py
# Topologically Sorted Source Nodes: [input_27, sigmoid], Original ATen: [aten.addmm, aten.sigmoid]
# Source node to ATen node mapping:
#   input_27 => add_tensor
#   sigmoid => sigmoid
# Graph fragment:
#   %arg31_1 : Tensor "f32[1][1]npu:0" = PlaceHolder[target=arg31_1]
#   %mm_default : Tensor "f32[s20, 1][1, 1]npu:0" = PlaceHolder[target=mm_default]
#   %add_tensor : Tensor "f32[s20, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg31_1, %mm_default), kwargs = {})
#   %sigmoid : Tensor "f32[s20, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor,), kwargs = {})
#   return %sigmoid
# SchedulerNodes: [SchedulerNode(name='op85')]

triton_poi_fused_addmm_sigmoid_9 = async_compile.triton('triton_poi_fused_addmm_sigmoid_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'x0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_sigmoid_9', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['x0'], 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_only', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_sigmoid_9(in_out_ptr0, in_ptr0, x0_numel, X0BLOCK : tl.constexpr, X0BLOCK_SUB : tl.constexpr):
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
    arg5_1, arg3_1, arg4_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg1_1, arg2_1, arg29_1, arg30_1, arg31_1, s20 = args
    args.clear()
    s20 = s20
    with torch.npu.utils.device(0):
        torch.npu.set_device(0)
        buf0 = empty_strided((s20, 6144), (6144, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(arg5_1, reinterpret_tensor(arg3_1, (1000, 6144), (1, 1000), 0), out=buf0)
        del arg3_1
        del arg5_1
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_0.run(buf1, arg4_1, s20, 6144, stream=stream0)
        del arg4_1
        buf2 = empty_strided((s20, 6144), (6144, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf1, reinterpret_tensor(arg6_1, (6144, 6144), (1, 6144), 0), out=buf2)
        del arg6_1
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_0.run(buf3, arg7_1, s20, 6144, stream=stream0)
        del arg7_1
        buf4 = empty_strided((s20, 6144), (6144, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_3, input_4, input_5], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf3, reinterpret_tensor(arg8_1, (6144, 6144), (1, 6144), 0), out=buf4)
        del arg8_1
        del buf3
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_0.run(buf5, arg9_1, s20, 6144, stream=stream0)
        del arg9_1
        buf6 = empty_strided((s20, 6144), (6144, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_5, input_6, input_7], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf5, reinterpret_tensor(arg10_1, (6144, 6144), (1, 6144), 0), out=buf6)
        del arg10_1
        del buf5
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_0.run(buf7, arg11_1, s20, 6144, stream=stream0)
        del arg11_1
        buf8 = empty_strided((s20, 6144), (6144, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_7, input_8, input_9], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf7, reinterpret_tensor(arg12_1, (6144, 6144), (1, 6144), 0), out=buf8)
        del arg12_1
        del buf7
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [input_9, input_10], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_0.run(buf9, arg13_1, s20, 6144, stream=stream0)
        del arg13_1
        buf10 = empty_strided((s20, 6144), (6144, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_9, input_10, input_11], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf9, reinterpret_tensor(arg14_1, (6144, 6144), (1, 6144), 0), out=buf10)
        del arg14_1
        del buf9
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_0.run(buf11, arg15_1, s20, 6144, stream=stream0)
        del arg15_1
        buf12 = empty_strided((s20, 3072), (3072, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_11, input_12, input_13], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf11, reinterpret_tensor(arg16_1, (6144, 3072), (1, 6144), 0), out=buf12)
        del arg16_1
        del buf11
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_1.run(buf13, arg17_1, s20, 3072, stream=stream0)
        del arg17_1
        buf14 = empty_strided((s20, 3072), (3072, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf13, reinterpret_tensor(arg18_1, (3072, 3072), (1, 3072), 0), out=buf14)
        del arg18_1
        del buf13
        buf15 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [input_15, input_16], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_1.run(buf15, arg19_1, s20, 3072, stream=stream0)
        del arg19_1
        buf16 = empty_strided((s20, 3072), (3072, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_15, input_16, input_17], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf15, reinterpret_tensor(arg20_1, (3072, 3072), (1, 3072), 0), out=buf16)
        del arg20_1
        del buf15
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [input_17, input_18], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_1.run(buf17, arg21_1, s20, 3072, stream=stream0)
        del arg21_1
        buf18 = empty_strided((s20, 1536), (1536, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_17, input_18, input_19], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf17, reinterpret_tensor(arg22_1, (3072, 1536), (1, 3072), 0), out=buf18)
        del arg22_1
        del buf17
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [input_19, input_20], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_2.run(buf19, arg23_1, s20, 1536, stream=stream0)
        del arg23_1
        buf20 = empty_strided((s20, 1536), (1536, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_19, input_20, input_21], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf19, reinterpret_tensor(arg24_1, (1536, 1536), (1, 1536), 0), out=buf20)
        del arg24_1
        del buf19
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [input_21, input_22], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_2.run(buf21, arg25_1, s20, 1536, stream=stream0)
        del arg25_1
        buf22 = empty_strided((s20, 1536), (1536, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_21, input_22, input_23], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf21, reinterpret_tensor(arg26_1, (1536, 1536), (1, 1536), 0), out=buf22)
        del arg26_1
        del buf21
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [input_23, input_24], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_2.run(buf23, arg27_1, s20, 1536, stream=stream0)
        del arg27_1
        buf24 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_23, input_24, input_25], Original ATen: [aten.addmm, aten.relu, aten.t]
        extern_kernels.mm(buf23, reinterpret_tensor(arg28_1, (1536, 8), (1, 1536), 0), out=buf24)
        del arg28_1
        del buf23
        buf25 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf26 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf27 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf28 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf29 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf30 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf31 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf32 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf33 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf34 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf35 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf36 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf37 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf38 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf39 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf40 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf41 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf42 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf43 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf44 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf45 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [getitem_2, emb, group_sum, getitem_3, emb_1, group_sum_1, getitem_4, emb_2, group_sum_2, getitem_5, emb_3, group_sum_3, getitem_6, emb_4, group_sum_4, getitem_7, emb_5, group_sum_5, getitem_8, emb_6, group_sum_6, getitem_9, emb_7, group_sum_7, getitem_10, emb_8, group_sum_8, getitem_11, emb_9, group_sum_9, getitem_12, emb_10, group_sum_10, getitem_13, emb_11, group_sum_11, getitem_14, emb_12, group_sum_12, getitem_15, emb_13, group_sum_13, getitem_16, emb_14, group_sum_14, getitem_17, emb_15, group_sum_15, getitem_18, emb_16, group_sum_16, getitem_19, emb_17, group_sum_17, getitem_20, emb_18, group_sum_18, getitem_21, emb_19, group_sum_19, getitem_22, emb_20, group_sum_20], Original ATen: [aten.slice, aten.embedding, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_embedding_slice_sum_3.run(arg1_1, arg2_1, buf25, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf34, buf35, buf36, buf37, buf38, buf39, buf40, buf41, buf42, buf43, buf44, buf45, s20, 8, 20, stream=stream0)
        buf46 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf47 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf48 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf49 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf50 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf51 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf52 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf53 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf54 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf55 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf56 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf57 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf58 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf59 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf60 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf61 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf62 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf63 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf64 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf65 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf66 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [getitem_23, emb_21, group_sum_21, getitem_24, emb_22, group_sum_22, getitem_25, emb_23, group_sum_23, getitem_26, emb_24, group_sum_24, getitem_27, emb_25, group_sum_25, getitem_28, emb_26, group_sum_26, getitem_29, emb_27, group_sum_27, getitem_30, emb_28, group_sum_28, getitem_31, emb_29, group_sum_29, getitem_32, emb_30, group_sum_30, getitem_33, emb_31, group_sum_31, getitem_34, emb_32, group_sum_32, getitem_35, emb_33, group_sum_33, getitem_36, emb_34, group_sum_34, getitem_37, emb_35, group_sum_35, getitem_38, emb_36, group_sum_36, getitem_39, emb_37, group_sum_37, getitem_40, emb_38, group_sum_38, getitem_41, emb_39, group_sum_39, getitem_42, emb_40, group_sum_40, getitem_43, emb_41, group_sum_41], Original ATen: [aten.slice, aten.embedding, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_embedding_slice_sum_4.run(arg1_1, arg2_1, buf46, buf47, buf48, buf49, buf50, buf51, buf52, buf53, buf54, buf55, buf56, buf57, buf58, buf59, buf60, buf61, buf62, buf63, buf64, buf65, buf66, s20, 8, 20, stream=stream0)
        buf67 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf68 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf69 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf70 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf71 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf72 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf73 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        buf74 = empty_strided((s20, 8), (8, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [getitem_44, emb_42, group_sum_42, getitem_45, emb_43, group_sum_43, getitem_46, emb_44, group_sum_44, getitem_47, emb_45, group_sum_45, getitem_48, emb_46, group_sum_46, getitem_49, emb_47, group_sum_47, getitem_50, emb_48, group_sum_48, getitem_51, emb_49, group_sum_49], Original ATen: [aten.slice, aten.embedding, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_embedding_slice_sum_5.run(arg1_1, arg2_1, buf67, buf68, buf69, buf70, buf71, buf72, buf73, buf74, s20, 8, 20, stream=stream0)
        del arg1_1
        del arg2_1
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.cat]
        buf75 = torch.ops.aten.cat.default([buf25, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf34, buf35, buf36, buf37, buf38, buf39, buf40, buf41, buf42, buf43, buf44, buf45, buf46, buf47, buf48, buf49, buf50, buf51, buf52, buf53, buf54, buf55, buf56, buf57, buf58, buf59, buf60, buf61, buf62, buf63, buf64, buf65, buf66, buf67, buf68, buf69, buf70, buf71, buf72, buf73, buf74], 1)
        del buf25
        del buf26
        del buf27
        del buf28
        del buf29
        del buf30
        del buf31
        del buf32
        del buf33
        del buf34
        del buf35
        del buf36
        del buf37
        del buf38
        del buf39
        del buf40
        del buf41
        del buf42
        del buf43
        del buf44
        del buf45
        del buf46
        del buf47
        del buf48
        del buf49
        del buf50
        del buf51
        del buf52
        del buf53
        del buf54
        del buf55
        del buf56
        del buf57
        del buf58
        del buf59
        del buf60
        del buf61
        del buf62
        del buf63
        del buf64
        del buf65
        del buf66
        del buf67
        del buf68
        del buf69
        del buf70
        del buf71
        del buf72
        del buf73
        del buf74
        buf76 = buf75
        assert_size_stride(buf76, (s20, 400), (400, 1), 'torch.ops.aten.cat.default')
        assert_alignment(buf76, 16, 'torch.ops.aten.cat.default')
        del buf75
        buf79 = empty_strided((s20, 408), (408, 1), device='npu', dtype=torch.float32)
        buf77 = reinterpret_tensor(buf79, (s20, 8), (408, 1), 0)  # alias
        buf83 = empty_strided((s20, 166472), (166472, 1), device='npu', dtype=torch.float32)
        buf81 = reinterpret_tensor(buf83, (s20, 8), (166472, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [upper_tri_mask, zeros_like, input_25, input_26, activations, activations_1, concat], Original ATen: [aten.triu, aten.zeros_like, aten.addmm, aten.relu, aten.where, aten.view, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_cat_relu_triu_view_wher_6.run(arg29_1, buf24, buf77, buf81, s20, 8, stream=stream0)
        del arg29_1
        del buf24
        buf78 = reinterpret_tensor(buf79, (s20, 400), (408, 1), 8)  # alias
        # Topologically Sorted Source Nodes: [input_25, input_26, cat_1], Original ATen: [aten.addmm, aten.relu, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_cat_relu_7.run(buf76, buf78, s20, 400, stream=stream0)
        del buf76
        del buf77
        del buf78
        buf80 = empty_strided((s20, 408, 408), (166464, 408, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [inputs, transpose, xactions], Original ATen: [aten.unsqueeze, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf79, (s20, 408, 1), (408, 1, 1), 0), reinterpret_tensor(buf79, (s20, 1, 408), (408, 1, 1), 0), out=buf80)
        del buf79
        buf82 = reinterpret_tensor(buf83, (s20, 166464), (166472, 1), 8)  # alias
        # Topologically Sorted Source Nodes: [upper_tri_mask, zeros_like, activations, activations_1, concat], Original ATen: [aten.triu, aten.zeros_like, aten.where, aten.view, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_triu_view_where_zeros_lik_8.run(buf80, buf82, s20, 166464, stream=stream0)
        del buf80
        del buf81
        del buf82
        buf84 = empty_strided((s20, 1), (1, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(buf83, reinterpret_tensor(arg30_1, (166472, 1), (1, 166472), 0), out=buf84)
        del arg30_1
        del buf83
        buf85 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [input_27, sigmoid], Original ATen: [aten.addmm, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_sigmoid_9.run(buf85, arg31_1, s20, stream=stream0)
        del arg31_1
    return (buf85, )


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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1 = args
        args.clear()
        s20 = arg0_1
        partition0_args = [arg5_1, arg3_1, arg4_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg1_1, arg2_1, arg29_1, arg30_1, arg31_1, s20]
        del arg5_1, arg3_1, arg4_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg1_1, arg2_1, arg29_1, arg30_1, arg31_1
        (buf85,) = self.partitions[0](partition0_args)
        del partition0_args
        return (reinterpret_tensor(buf85, (s20, ), (1, ), 0), )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 128
    arg1_1 = rand_strided((128, 1000), (1000, 1), device='npu:0', dtype=torch.int64)
    arg2_1 = rand_strided((5000, 8), (8, 1), device='npu:0', dtype=torch.float32)
    arg3_1 = rand_strided((6144, 1000), (1000, 1), device='npu:0', dtype=torch.float32)
    arg4_1 = rand_strided((6144, ), (1, ), device='npu:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, 1000), (1000, 1), device='npu:0', dtype=torch.float32)
    arg6_1 = rand_strided((6144, 6144), (6144, 1), device='npu:0', dtype=torch.float32)
    arg7_1 = rand_strided((6144, ), (1, ), device='npu:0', dtype=torch.float32)
    arg8_1 = rand_strided((6144, 6144), (6144, 1), device='npu:0', dtype=torch.float32)
    arg9_1 = rand_strided((6144, ), (1, ), device='npu:0', dtype=torch.float32)
    arg10_1 = rand_strided((6144, 6144), (6144, 1), device='npu:0', dtype=torch.float32)
    arg11_1 = rand_strided((6144, ), (1, ), device='npu:0', dtype=torch.float32)
    arg12_1 = rand_strided((6144, 6144), (6144, 1), device='npu:0', dtype=torch.float32)
    arg13_1 = rand_strided((6144, ), (1, ), device='npu:0', dtype=torch.float32)
    arg14_1 = rand_strided((6144, 6144), (6144, 1), device='npu:0', dtype=torch.float32)
    arg15_1 = rand_strided((6144, ), (1, ), device='npu:0', dtype=torch.float32)
    arg16_1 = rand_strided((3072, 6144), (6144, 1), device='npu:0', dtype=torch.float32)
    arg17_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg18_1 = rand_strided((3072, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg19_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg20_1 = rand_strided((3072, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg21_1 = rand_strided((3072, ), (1, ), device='npu:0', dtype=torch.float32)
    arg22_1 = rand_strided((1536, 3072), (3072, 1), device='npu:0', dtype=torch.float32)
    arg23_1 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    arg24_1 = rand_strided((1536, 1536), (1536, 1), device='npu:0', dtype=torch.float32)
    arg25_1 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    arg26_1 = rand_strided((1536, 1536), (1536, 1), device='npu:0', dtype=torch.float32)
    arg27_1 = rand_strided((1536, ), (1, ), device='npu:0', dtype=torch.float32)
    arg28_1 = rand_strided((8, 1536), (1536, 1), device='npu:0', dtype=torch.float32)
    arg29_1 = rand_strided((8, ), (1, ), device='npu:0', dtype=torch.float32)
    arg30_1 = rand_strided((1, 166472), (166472, 1), device='npu:0', dtype=torch.float32)
    arg31_1 = rand_strided((1, ), (1, ), device='npu:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
