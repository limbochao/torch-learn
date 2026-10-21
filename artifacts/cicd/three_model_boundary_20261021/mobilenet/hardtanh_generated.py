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


# kernel path: <workspace>/three_model_boundary_20261021/mobile_hardtanh_inductor/gn/cgnhzzbsuvvv2qx2i43molz7626ccapojje7mwo53lstvwx4l4wu.py
# Topologically Sorted Source Nodes: [hardtanh_backward_default, sum_1], Original ATen: [aten.hardtanh_backward, aten.sum]
# Source node to ATen node mapping:
#   hardtanh_backward_default => bitwise_or, full_default, where
#   sum_1 => sum_1
# Graph fragment:
#   %arg3_1 : Tensor "f32[s87, s71][s71, 1]npu:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg5_1]
#   %arg2_1 : Tensor "f32[s87, s71][s71, 1]npu:0" = PlaceHolder[target=arg2_1]
#   %le_tensor : Tensor "b8[s87, s71][s71, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%arg3_1, %arg4_1), kwargs = {})
#   %ge_tensor : Tensor "b8[s87, s71][s71, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%arg3_1, %arg5_1), kwargs = {})
#   %bitwise_or : Tensor "b8[s87, s71][s71, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%le_tensor, %ge_tensor), kwargs = {})
#   %full_default : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where : Tensor "f32[s87, s71][s71, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_or, %full_default, %arg2_1), kwargs = {})
#   %sum_1 : Tensor "f32[s71][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%where, [0]), kwargs = {})
#   return %sum_1
# SchedulerNodes: [SchedulerNode(name='op0')]

triton_red_fused_hardtanh_backward_sum_0 = async_compile.triton('triton_red_fused_hardtanh_backward_sum_0', '''
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
    size_hints={'x0': 64, 'r1': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': 'fp64', 'in_ptr2': 'fp64', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_sum_0', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_red_fused_hardtanh_backward_sum_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr, R1BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        tmp1 = in_ptr1
        tmp4 = in_ptr2
        _tmp12 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        for loop_r1 in range(loops_r1):
            r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[:,None]
            r1_mask = r1 < r1_numel
            tmp0 = tl.load(in_ptr0 + (x0 + ks0*r1), r1_mask & x0_mask, other=0.0)
            tmp2 = tmp1.to(tl.float32)
            tmp3 = tmp0 <= tmp2
            tmp5 = tmp4.to(tl.float32)
            tmp6 = tmp0 >= tmp5
            tmp7 = tmp3 | tmp6
            tmp8 = tl.load(in_ptr3 + (x0 + ks0*r1), r1_mask & x0_mask, other=0.0)
            tmp9 = tl.full([1, 1], 0.0, tl.float32)
            tmp10 = tl.where(tmp7, tmp9, tmp8)
            tmp11 = tl.reshape(tmp10, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp13 = _tmp12 + tmp11
            _tmp12 = tl.where(tl.reshape((r1_mask & x0_mask), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp13, _tmp12)
        tmp12 = tl.sum(_tmp12, 0).reshape(1, X0BLOCK_SUB)
        tl.store(out_ptr0 + (x0 ), tmp12, x0_mask)
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
        s87 = arg0_1
        s71 = arg1_1
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg3_1 = copy_if_misaligned(arg3_1)
            arg2_1 = copy_if_misaligned(arg2_1)
            buf0 = empty_strided((s71, ), (1, ), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [hardtanh_backward_default, sum_1], Original ATen: [aten.hardtanh_backward, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_hardtanh_backward_sum_0.run(arg3_1, arg4_1.item(), arg5_1.item(), arg2_1, buf0, s71, s71, s87, stream=raw_stream0)
            del arg2_1
            del arg3_1
            del arg4_1
            del arg5_1
        return (buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = 16
    arg1_1 = 64
    arg2_1 = rand_strided((16, 64), (64, 1), device='npu:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, 64), (64, 1), device='npu:0', dtype=torch.float32)
    arg4_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg5_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
