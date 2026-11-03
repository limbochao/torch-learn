# AOT ID: ['3_inference']
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
# Topologically Sorted Source Nodes: [a, add_1, a_1, sum_1, v, sum_3], Original ATen: [aten.add, aten.sum, aten.mul]
# Source node to ATen node mapping:
#   a => add
#   a_1 => add_10
#   add_1 => add_5
#   sum_1 => sum_1
#   sum_3 => sum_3
#   v => mul_9
# Graph fragment:
#   %arg1_1 : Tensor "f32[128, 128, s19][128*s19, s19, 1]npu:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[128, 128, s19][128*s19, s19, 1]npu:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[128, 128, s19][128*s19, s19, 1]npu:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[128, 128, s19][128*s19, s19, 1]npu:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[128, 128, s19][128*s19, s19, 1]npu:0" = PlaceHolder[target=arg5_1]
#   %add : Tensor "f32[128, 128, s19][128*s19, s19, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %add_5 : Tensor "f32[128, 128, s19][128*s19, s19, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %arg3_1), kwargs = {})
#   %add_10 : Tensor "f32[128, 128, s19][128*s19, s19, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %arg4_1), kwargs = {})
#   %sum_1 : Tensor "f32[128, 1, s19][s19, s19, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_10, [1], True), kwargs = {})
#   %mul_9 : Tensor "f32[128, 128, s19][128*s19, s19, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %arg5_1), kwargs = {})
#   %sum_3 : Tensor "f32[128, 1, s19][s19, s19, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_9, [1], True), kwargs = {})
#   return %sum_1,%sum_3
# SchedulerNodes: [SchedulerNode(name='op0'), SchedulerNode(name='op2')]

triton_per_fused_add_mul_sum_0 = async_compile.triton('triton_per_fused_add_mul_sum_0', '''
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
    size_hints={'y0': 128, 'x1': 512, 'r2': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i64', 'y0_numel': 'i32', 'x1_numel': 'i32', 'r2_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sum_0', 'mutated_arg_names': [], 'backend_hash': 'SANITIZED_BACKEND_HASH', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['y0', 'x1', 'r2'], 'axis_static_values': (('y0', 128), ('r2', 128)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_add_mul_sum_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ks0, y0_numel, x1_numel, r2_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    R2BLOCK_SUB: tl.constexpr = 128
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
            r2 = base_r2[None,:,None]
            r2_mask = True
            tmp0 = tl.load(in_ptr0 + (x1 + ks0*r2 + 128*ks0*y0), x1_mask & y0_mask, other=0.0)
            tmp1 = tl.load(in_ptr1 + (x1 + ks0*r2 + 128*ks0*y0), x1_mask & y0_mask, other=0.0)
            tmp3 = tl.load(in_ptr2 + (x1 + ks0*r2 + 128*ks0*y0), x1_mask & y0_mask, other=0.0)
            tmp5 = tl.load(in_ptr3 + (x1 + ks0*r2 + 128*ks0*y0), x1_mask & y0_mask, other=0.0)
            tmp10 = tl.load(in_ptr4 + (x1 + ks0*r2 + 128*ks0*y0), x1_mask & y0_mask, other=0.0)
            tmp2 = tmp0 + tmp1
            tmp4 = tmp2 + tmp3
            tmp6 = tmp4 + tmp5
            tmp8 = tl.where(tl.broadcast_to((y0_mask.reshape([Y0BLOCK_SUB, 1, 1]) & x1_mask.reshape([1, 1, X1BLOCK_SUB])), [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp6, 0)
            tmp9 = tl.sum(tmp8, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tmp11 = tmp6 * tmp10
            tmp13 = tl.where(tl.broadcast_to((y0_mask.reshape([Y0BLOCK_SUB, 1, 1]) & x1_mask.reshape([1, 1, X1BLOCK_SUB])), [Y0BLOCK_SUB, R2BLOCK_SUB, X1BLOCK_SUB]), tmp11, 0)
            tmp14 = tl.sum(tmp13, 1).reshape(Y0BLOCK_SUB, 1, X1BLOCK_SUB)
            tl.store(out_ptr0 + (x1 + ks0*y0 ), tmp9, x1_mask & y0_mask)
            tl.store(out_ptr1 + (x1 + ks0*y0 ), tmp14, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: SANITIZED_SOURCE_PATH
# Topologically Sorted Source Nodes: [sum_2], Original ATen: [aten.sum]
# Source node to ATen node mapping:
#   sum_2 => sum_2
# Graph fragment:
#   %sum_1 : Tensor "f32[128, 1, s19][s19, 128*s19, 1]npu:0" = PlaceHolder[target=sum_1]
#   %sum_2 : Tensor "f32[1, 1, s19][s19, s19, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%sum_1, [0], True), kwargs = {})
#   return %sum_2
# SchedulerNodes: [SchedulerNode(name='op1')]

triton_per_fused_sum_1 = async_compile.triton('triton_per_fused_sum_1', '''
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
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_1', 'mutated_arg_names': [], 'backend_hash': 'SANITIZED_BACKEND_HASH', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('r1', 128),), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_sum_1(in_ptr0, out_ptr0, ks0, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + ks0*r1), x0_mask, other=0.0)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 0).reshape(1, X0BLOCK_SUB)
        tl.store(out_ptr0 + (x0 ), tmp3, x0_mask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
        args.clear()
        s43 = arg0_1
        arg1_1_size = arg1_1.size()
        s19 = arg1_1_size[2]
        s36 = s19
        s32 = s19
        s39 = s19
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg1_1 = copy_if_misaligned(arg1_1)
            arg2_1 = copy_if_misaligned(arg2_1)
            arg3_1 = copy_if_misaligned(arg3_1)
            arg4_1 = copy_if_misaligned(arg4_1)
            arg5_1 = copy_if_misaligned(arg5_1)
            buf0 = empty_strided_npu((128, 1, s19), (s19, 128*s19, 1), torch.float32)
            buf2 = empty_strided_npu((128, 1, s19), (s19, 128*s19, 1), torch.float32)
            # Topologically Sorted Source Nodes: [a, add_1, a_1, sum_1, v, sum_3], Original ATen: [aten.add, aten.sum, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_add_mul_sum_0.run(arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, buf0, buf2, s19, 128, s19, 128, stream=raw_stream0)
            del arg1_1
            del arg2_1
            del arg3_1
            del arg4_1
            del arg5_1
            buf1 = empty_strided_npu((1, 1, s19), (s19, s19, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sum_2], Original ATen: [aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_sum_1.run(buf0, buf1, s19, s19, 128, stream=raw_stream0)
            del buf0
            buf3 = empty_strided_npu((1, 1, s19), (s19, s19, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sum_4], Original ATen: [aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_sum_1.run(buf2, buf3, s19, s19, 128, stream=raw_stream0)
            del buf2
        return (buf1, buf3, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = 512
    arg1_1 = rand_strided((128, 128, 512), (65536, 512, 1), device='npu:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, 128, 512), (65536, 512, 1), device='npu:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, 128, 512), (65536, 512, 1), device='npu:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, 128, 512), (65536, 512, 1), device='npu:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, 128, 512), (65536, 512, 1), device='npu:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
