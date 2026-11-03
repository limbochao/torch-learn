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


# kernel path: SANITIZED_SOURCE_PATH
# Topologically Sorted Source Nodes: [view_1014, add_368, sum_51, layer_input_92, mul_186, layer_input_93, layer_outputs_253, add_351, mul_259, sum_52], Original ATen: [aten.view, aten.add, aten.sum, aten.mul]
# Source node to ATen node mapping:
#   add_351 => add_351
#   add_368 => add_368
#   layer_input_92 => view_923
#   layer_input_93 => add_348
#   layer_outputs_253 => view_943
#   mul_186 => mul_232
#   mul_259 => mul_259
#   sum_51 => sum_51
#   sum_52 => sum_52
#   view_1014 => view_1014
# Graph fragment:
#   %mul_258 : Tensor "f32[128, 128, 128][16384, 128, 1]npu:0" = PlaceHolder[target=mul_258]
#   %mm_21 : Tensor "f32[16384, 128][128, 1]npu:0" = PlaceHolder[target=mm_21]
#   %addmm_351 : Tensor "f32[16384, 128][128, 1]npu:0" = PlaceHolder[target=addmm_351]
#   %addmm_346 : Tensor "f32[16384, 128][128, 1]npu:0" = PlaceHolder[target=addmm_346]
#   %primals_1070 : Tensor "f32[128][1]npu:0" = PlaceHolder[target=primals_1070]
#   %primals_1071 : Tensor "f32[128][1]npu:0" = PlaceHolder[target=primals_1071]
#   %view_1014 : Tensor "f32[128, 128, 128][16384, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_21, [128, 128, 128]), kwargs = {})
#   %add_368 : Tensor "f32[128, 128, 128][16384, 128, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_258, %view_1014), kwargs = {})
#   %sum_51 : Tensor "f32[1, 1, 128][128, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_368, [0, 1], True), kwargs = {})
#   %view_923 : Tensor "f32[128, 128, 128][16384, 128, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_346, [128, 128, 128]), kwargs = {})
#   %mul_232 : Tensor "f32[128, 128, 128][16384, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_923, %primals_1070), kwargs = {})
#   %add_348 : Tensor "f32[128, 128, 128][16384, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_232, %primals_1071), kwargs = {})
#   %view_943 : Tensor "f32[128, 128, 128][16384, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_351, [128, 128, 128]), kwargs = {})
#   %add_351 : Tensor "f32[128, 128, 128][16384, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_943, %add_348), kwargs = {})
#   %mul_259 : Tensor "f32[128, 128, 128][16384, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_368, %add_351), kwargs = {})
#   %sum_52 : Tensor "f32[1, 1, 128][128, 128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_259, [0, 1], True), kwargs = {})
#   return %sum_51,%sum_52
# SchedulerNodes: [SchedulerNode(name='op62'), SchedulerNode(name='op63')]

triton_red_fused_add_mul_sum_view_13 = async_compile.triton('triton_red_fused_add_mul_sum_view_13', '''
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
    size_hints={'x0': 128, 'r1': 16384},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_view_13', 'mutated_arg_names': [], 'backend_hash': 'SANITIZED_BACKEND_HASH', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 128), ('r1', 16384)), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_red_fused_add_mul_sum_view_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr, R1BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        _tmp4 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        tmp8 = tl.load(in_ptr4 + (x0), x0_mask)
        tmp10 = tl.load(in_ptr5 + (x0), x0_mask)
        _tmp15 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        for loop_r1 in range(loops_r1):
            r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[:,None]
            r1_mask = r1 < r1_numel
            tmp0 = tl.load(in_ptr0 + (x0 + 128*r1), r1_mask & x0_mask, other=0.0)
            tmp1 = tl.load(in_ptr1 + (x0 + 128*r1), r1_mask & x0_mask, other=0.0)
            tmp6 = tl.load(in_ptr2 + (x0 + 128*r1), r1_mask & x0_mask, other=0.0)
            tmp7 = tl.load(in_ptr3 + (x0 + 128*r1), r1_mask & x0_mask, other=0.0)
            tmp2 = tmp0 + tmp1
            tmp3 = tl.reshape(tmp2, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tl.where(tl.reshape((r1_mask & x0_mask), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp5, _tmp4)
            tmp9 = tmp7 * tmp8
            tmp11 = tmp9 + tmp10
            tmp12 = tmp6 + tmp11
            tmp13 = tmp2 * tmp12
            tmp14 = tl.reshape(tmp13, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp16 = _tmp15 + tmp14
            _tmp15 = tl.where(tl.reshape((r1_mask & x0_mask), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp16, _tmp15)
        tmp4 = tl.sum(_tmp4, 0).reshape(1, X0BLOCK_SUB)
        tmp15 = tl.sum(_tmp15, 0).reshape(1, X0BLOCK_SUB)
        tl.store(out_ptr0 + (x0 ), tmp4, x0_mask)
        tl.store(out_ptr1 + (x0 ), tmp15, x0_mask)
''', device_str='npu')
