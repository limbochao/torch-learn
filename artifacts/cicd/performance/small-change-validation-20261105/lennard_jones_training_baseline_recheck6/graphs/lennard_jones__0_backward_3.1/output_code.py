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
#   %tangents_1 : Tensor "f32[4096, 1][1, 1]npu:0" = PlaceHolder[target=tangents_1]
#   %buf1 : Tensor "f32[1, 1, 4096][4096, 4096, 1]npu:0" = PlaceHolder[target=buf1]
#   %sum_1 : Tensor "f32[1, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%tangents_1, [0], True), kwargs = {})
#   return %buf1,%sum_1
# SchedulerNodes: [SchedulerNode(name='op1'), SchedulerNode(name='op2')]

triton_per_fused_sum_0 = async_compile.triton('triton_per_fused_sum_0', '''
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
    size_hints={'r0': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'r0_numel': 'i32', 'R0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_0', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['r0'], 'axis_static_values': (('r0', 4096),), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('R0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_sum_0(in_ptr0, out_ptr1, r0_numel, R0BLOCK):
    R0BLOCK_SUB: tl.constexpr = 4096
    r0_offset = tl.program_id(0) * R0BLOCK
    base_r0= tl.arange(0, R0BLOCK_SUB)
    loops_r0 = (R0BLOCK + R0BLOCK_SUB - 1) // R0BLOCK_SUB
    r0 = base_r0
    r0_mask = True
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp2 = triton_helpers.promote_to_tensor(tl.sum(tmp0, 0))
    tl.store(out_ptr1 + (tl.arange(0,1) ), tmp2, None)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [input_9, permute_5, mul, mul_1, sub, mul_2, sum_2], Original ATen: [aten.t, aten.mm, aten.tanh_backward, aten.sum]
# Source node to ATen node mapping:
#   input_9 => permute_4
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   permute_5 => permute_5
#   sub => sub
#   sum_2 => sum_2
# Graph fragment:
#   %tangents_1 : Tensor "f32[4096, 1][1, 1]npu:0" = PlaceHolder[target=tangents_1]
#   %primals_10 : Tensor "f32[1, 16][16, 1]npu:0" = PlaceHolder[target=primals_10]
#   %tanh_3 : Tensor "f32[4096, 16][16, 1]npu:0" = PlaceHolder[target=tanh_3]
#   %mul_2 : Tensor "f32[4096, 16][16, 1]npu:0" = PlaceHolder[target=mul_2]
#   %permute_4 : Tensor "f32[16, 1][1, 16]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_10, [1, 0]), kwargs = {})
#   %permute_5 : Tensor "f32[1, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_4, [1, 0]), kwargs = {})
#   %mul : Tensor "f32[4096, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %permute_5), kwargs = {})
#   %mul_1 : Tensor "f32[4096, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_3, %tanh_3), kwargs = {})
#   %sub : Tensor "f32[4096, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mul_1), kwargs = {})
#   %mul_2 : Tensor "f32[4096, 16][16, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %sub), kwargs = {})
#   %sum_2 : Tensor "f32[1, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2, [0], True), kwargs = {})
#   return %mul_2,%sum_2
# SchedulerNodes: [SchedulerNode(name='op3'), SchedulerNode(name='op6')]

triton_per_fused_mm_sum_t_tanh_backward_1 = async_compile.triton('triton_per_fused_mm_sum_t_tanh_backward_1', '''
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
    size_hints={'x0': 16, 'r1': 4096},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mm_sum_t_tanh_backward_1', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 16), ('r1', 4096)), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_mm_sum_t_tanh_backward_1(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 4096
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
        tmp0 = tl.load(in_ptr0 + (r1), None)
        tmp1 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp3 = tl.load(in_out_ptr0 + (x0 + 16*r1), x0_mask, other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp3 * tmp3
        tmp5 = tl.full([1, 1], 1.0, tl.float32)
        tmp6 = tmp5 - tmp4
        tmp7 = tmp2 * tmp6
        tmp9 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp7, 0)
        tmp10 = tl.sum(tmp9, 0).reshape(1, X0BLOCK_SUB)
        tl.store(in_out_ptr0 + (x0 + 16*r1), tmp7, x0_mask)
        tl.store(out_ptr0 + (x0 ), tmp10, x0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [mul_3, sub_1, mul_4, sum_3], Original ATen: [aten.tanh_backward, aten.sum]
# Source node to ATen node mapping:
#   mul_3 => mul_3
#   mul_4 => mul_4
#   sub_1 => sub_1
#   sum_3 => sum_3
# Graph fragment:
#   %mm_1 : Tensor "f32[4096, 16][16, 1]npu:0" = PlaceHolder[target=mm_1]
#   %tanh_2 : Tensor "f32[4096, 16][16, 1]npu:0" = PlaceHolder[target=tanh_2]
#   %mul_4 : Tensor "f32[4096, 16][16, 1]npu:0" = PlaceHolder[target=mul_4]
#   %mul_3 : Tensor "f32[4096, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_2, %tanh_2), kwargs = {})
#   %sub_1 : Tensor "f32[4096, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mul_3), kwargs = {})
#   %mul_4 : Tensor "f32[4096, 16][16, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_1, %sub_1), kwargs = {})
#   %sum_3 : Tensor "f32[1, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_4, [0], True), kwargs = {})
#   return %mul_4,%sum_3
# SchedulerNodes: [SchedulerNode(name='op7'), SchedulerNode(name='op10')]

triton_per_fused_sum_tanh_backward_2 = async_compile.triton('triton_per_fused_sum_tanh_backward_2', '''
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
    size_hints={'x0': 16, 'r1': 4096},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_tanh_backward_2', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 16), ('r1', 4096)), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_sum_tanh_backward_2(in_out_ptr0, in_ptr0, out_ptr0, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 4096
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
        tmp0 = tl.load(in_out_ptr0 + (x0 + 16*r1), x0_mask, other=0.0)
        tmp1 = tl.load(in_ptr0 + (x0 + 16*r1), x0_mask, other=0.0)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.full([1, 1], 1.0, tl.float32)
        tmp4 = tmp3 - tmp2
        tmp5 = tmp0 * tmp4
        tmp7 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp5, 0)
        tmp8 = tl.sum(tmp7, 0).reshape(1, X0BLOCK_SUB)
        tl.store(in_out_ptr0 + (x0 + 16*r1), tmp5, x0_mask)
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
        primals_3, primals_4, primals_6, primals_8, primals_10, tanh, tanh_1, tanh_2, tanh_3, tangents_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            tangents_1 = copy_if_misaligned(tangents_1)
            buf0 = empty_strided_npu((1, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_6, permute_8], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:1
            extern_kernels.mm(reinterpret_tensor(tangents_1, (1, 4096), (1, 1), 0), tanh_3, out=buf0)
            buf2 = empty_strided_npu((1, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sum_1], Original ATen: [aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_sum_0.run(tangents_1, buf2, 4096, stream=raw_stream0)
            buf3 = tanh_3; del tanh_3  # reuse
            buf6 = empty_strided_npu((1, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_9, permute_5, mul, mul_1, sub, mul_2, sum_2], Original ATen: [aten.t, aten.mm, aten.tanh_backward, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_mm_sum_t_tanh_backward_1.run(buf3, tangents_1, primals_10, buf6, 16, 4096, stream=raw_stream0)
            del primals_10
            del tangents_1
            buf4 = empty_strided_npu((4096, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_7, permute_9, mm_1], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:2
            extern_kernels.mm(buf3, primals_8, out=buf4)
            del primals_8
            buf5 = empty_strided_npu((16, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_10, permute_12], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:3
            extern_kernels.mm(reinterpret_tensor(buf3, (16, 4096), (1, 16), 0), tanh_2, out=buf5)
            del buf3
            buf7 = buf4; del buf4  # reuse
            buf10 = empty_strided_npu((1, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul_3, sub_1, mul_4, sum_3], Original ATen: [aten.tanh_backward, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_sum_tanh_backward_2.run(buf7, tanh_2, buf10, 16, 4096, stream=raw_stream0)
            del tanh_2
            buf8 = empty_strided_npu((4096, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_5, permute_13, mm_3], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:4
            extern_kernels.mm(buf7, primals_6, out=buf8)
            del primals_6
            buf9 = empty_strided_npu((16, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_14, permute_16], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:5
            extern_kernels.mm(reinterpret_tensor(buf7, (16, 4096), (1, 16), 0), tanh_1, out=buf9)
            del buf7
            buf11 = buf8; del buf8  # reuse
            buf14 = empty_strided_npu((1, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul_5, sub_2, mul_6, sum_4], Original ATen: [aten.tanh_backward, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_sum_tanh_backward_2.run(buf11, tanh_1, buf14, 16, 4096, stream=raw_stream0)
            del tanh_1
            buf12 = empty_strided_npu((4096, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_3, permute_17, mm_5], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:6
            extern_kernels.mm(buf11, primals_4, out=buf12)
            del primals_4
            buf13 = empty_strided_npu((16, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_18, permute_20], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:7
            extern_kernels.mm(reinterpret_tensor(buf11, (16, 4096), (1, 16), 0), tanh, out=buf13)
            del buf11
            buf15 = buf12; del buf12  # reuse
            buf17 = empty_strided_npu((1, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul_7, sub_3, mul_8, sum_5], Original ATen: [aten.tanh_backward, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_sum_tanh_backward_2.run(buf15, tanh, buf17, 16, 4096, stream=raw_stream0)
            del tanh
            buf16 = empty_strided_npu((16, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul_7, sub_3, mul_8, permute_21, permute_23], Original ATen: [aten.tanh_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:8
            extern_kernels.mm(reinterpret_tensor(buf15, (16, 4096), (1, 16), 0), primals_3, out=buf16)
            del buf15
            del primals_3
        return (buf16, reinterpret_tensor(buf17, (16, ), (1, ), 0), None, buf13, reinterpret_tensor(buf14, (16, ), (1, ), 0), buf9, reinterpret_tensor(buf10, (16, ), (1, ), 0), buf5, reinterpret_tensor(buf6, (16, ), (1, ), 0), buf0, reinterpret_tensor(buf2, (1, ), (1, ), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_3 = rand_strided((4096, 1), (1, 1), device='npu:0', dtype=torch.float32)
    primals_4 = rand_strided((16, 16), (16, 1), device='npu:0', dtype=torch.float32)
    primals_6 = rand_strided((16, 16), (16, 1), device='npu:0', dtype=torch.float32)
    primals_8 = rand_strided((16, 16), (16, 1), device='npu:0', dtype=torch.float32)
    primals_10 = rand_strided((1, 16), (16, 1), device='npu:0', dtype=torch.float32)
    tanh = rand_strided((4096, 16), (16, 1), device='npu:0', dtype=torch.float32)
    tanh_1 = rand_strided((4096, 16), (16, 1), device='npu:0', dtype=torch.float32)
    tanh_2 = rand_strided((4096, 16), (16, 1), device='npu:0', dtype=torch.float32)
    tanh_3 = rand_strided((4096, 16), (16, 1), device='npu:0', dtype=torch.float32)
    tangents_1 = rand_strided((4096, 1), (1, 1), device='npu:0', dtype=torch.float32)
    return [primals_3, primals_4, primals_6, primals_8, primals_10, tanh, tanh_1, tanh_2, tanh_3, tangents_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('lennard_jones', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
