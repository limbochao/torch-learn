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


# kernel path: <validation>/branch/cache/inductor/55/c55nbkz3c4d2enhtrz3nyjfzaimzo3vaiemtxojrkrufd6ooxd2l.py
# Topologically Sorted Source Nodes: [mul_18, mul_19, sum_1, sub_14, mul_20, mul_21, expand, div, add_25, convolution_backward], Original ATen: [aten.mul, aten.sum, aten.sigmoid_backward, aten.expand, aten.div, aten.add, aten.convolution_backward]
# Source node to ATen node mapping:
#   add_25 => add_25
#   convolution_backward => convolution_backward
#   div => div
#   expand => expand
#   mul_18 => mul_18
#   mul_19 => mul_19
#   mul_20 => mul_20
#   mul_21 => mul_21
#   sub_14 => sub_14
#   sum_1 => sum_1
# Graph fragment:
#   %tangents_1 : Tensor "f32[s4, s14, (((s43 - 1)//32)) + 1, (((s43 - 1)//32)) + 1][s14*((((s43 - 1)//32)) + 1)**2, ((((s43 - 1)//32)) + 1)**2, (((s43 - 1)//32)) + 1, 1]npu:0" = PlaceHolder[target=tangents_1]
#   %convolution : Tensor "f32[s4, s14, (((s43 - 1)//32)) + 1, (((s43 - 1)//32)) + 1][s14*((((s43 - 1)//32)) + 1)**2, ((((s43 - 1)//32)) + 1)**2, (((s43 - 1)//32)) + 1, 1]npu:0" = PlaceHolder[target=convolution]
#   %sigmoid : Tensor "f32[s4, s14, 1, 1][s14, 1, 1, 1]npu:0" = PlaceHolder[target=sigmoid]
#   %sum_1 : Tensor "f32[s4, s14, 1, 1][s14, 1, s14*s4, s14*s4]npu:0" = PlaceHolder[target=sum_1]
#   %mul_18 : Tensor "f32[s4, 16, 7, 7][784, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %convolution), kwargs = {})
#   %mul_19 : Tensor "f32[s4, 16, 7, 7][784, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %sigmoid), kwargs = {})
#   %sum_1 : Tensor "f32[s4, 16, 1, 1][16, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_18, [2, 3], True), kwargs = {})
#   %sub_14 : Tensor "f32[s4, 16, 1, 1][16, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid), kwargs = {})
#   %mul_20 : Tensor "f32[s4, 16, 1, 1][16, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, %sub_14), kwargs = {})
#   %mul_21 : Tensor "f32[s4, 16, 1, 1][16, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_1, %mul_20), kwargs = {})
#   %expand : Tensor "f32[s4, 16, 7, 7][16, 1, 0, 0]npu:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%mul_21, [%primals_1, %primals_2, %sym_size_int_2, %sym_size_int_3]), kwargs = {})
#   %div : Tensor "f32[s4, 16, 7, 7][784, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, %mul_22), kwargs = {})
#   %add_25 : Tensor "f32[s4, 16, 7, 7][784, 49, 7, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %div), kwargs = {})
#   %convolution_backward : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%add_25, %primals_4, %primals_5, [0], [32, 32], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
#   return %sum_1,%buf1
# SchedulerNodes: [SchedulerNode(name='op0'), SchedulerNode(name='op1')]

triton_per_fused_add_convolution_backward_div__0 = async_compile.triton('triton_per_fused_add_convolution_backward_div__0', '''
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
    size_hints={'y0': 2, 'x1': 16, 'r2': 49},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'r2_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_backward_div__0', 'mutated_arg_names': [], 'backend_hash': '<backend-hash>', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['y0', 'x1', 'r2'], 'axis_static_values': (('x1', 16), ('r2', 49)), 'low_dims': {2}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_add_convolution_backward_div__0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, y0_numel, x1_numel, r2_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
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
            tmp0 = tl.load(in_ptr0 + (r2 + 49*x1 + 784*y0), x1_mask & y0_mask, other=0.0)
            tmp1 = tl.load(in_ptr1 + (r2 + 49*x1 + 784*y0), x1_mask & y0_mask, other=0.0)
            tmp6 = tl.load(in_ptr2 + (x1 + 16*y0), x1_mask & y0_mask)
            tmp2 = tmp0 * tmp1
            tmp4 = tl.where(tl.broadcast_to((y0_mask.reshape([Y0BLOCK_SUB, 1, 1]) & x1_mask.reshape([1, X1BLOCK_SUB, 1])), [Y0BLOCK_SUB, X1BLOCK_SUB, R2BLOCK_SUB]), tmp2, 0)
            tmp5 = tl.sum(tmp4, 2).reshape(Y0BLOCK_SUB, X1BLOCK_SUB, 1)
            tmp7 = tmp0 * tmp6
            tmp8 = tl.full([1, 1, 1], 1.0, tl.float32)
            tmp9 = tmp8 - tmp6
            tmp10 = tmp6 * tmp9
            tmp11 = tmp5 * tmp10
            tmp12 = tl.full([1, 1, 1], 49.0, tl.float32)
            tmp13 = (tmp11 / tmp12)
            tmp14 = tmp7 + tmp13
            tl.store(out_ptr1 + (r2 + 49*x1 + 784*y0), tmp14, x1_mask & y0_mask)
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
        primals_1, primals_2, primals_3, primals_4, primals_5, convolution, sigmoid, tangents_1 = args
        args.clear()
        s4 = primals_1
        s14 = primals_2
        s43 = primals_3
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            tangents_1 = copy_if_misaligned(tangents_1)
            buf1 = empty_strided_npu((s4, 16, 7, 7), (784, 49, 7, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul_18, mul_19, sum_1, sub_14, mul_20, mul_21, expand, div, add_25, convolution_backward], Original ATen: [aten.mul, aten.sum, aten.sigmoid_backward, aten.expand, aten.div, aten.add, aten.convolution_backward]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_add_convolution_backward_div__0.run(tangents_1, convolution, sigmoid, buf1, s4, 16, 49, stream=raw_stream0)
            del convolution
            del sigmoid
            del tangents_1
            # Topologically Sorted Source Nodes: [mul_19, sub_14, mul_20, mul_21, expand, div, add_25, convolution_backward], Original ATen: [aten.mul, aten.sigmoid_backward, aten.expand, aten.div, aten.add, aten.convolution_backward]
            # [Provenance debug handles] torch.ops.aten.convolution_backward.default:4
            buf2 = torch.ops.aten.convolution_backward.default(buf1, primals_4, primals_5, [0], (32, 32), (0, 0), (1, 1), False, (0, 0), 1, [True, True, False])
            del buf1
            del primals_4
            del primals_5
            buf3 = buf2[0]
            assert_size_stride(buf3, (s4, 16, 224, 224), (802816, 50176, 224, 1), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf3, 16, 'torch.ops.aten.convolution_backward.default')
            buf4 = buf2[1]
            assert_size_stride(buf4, (16, 16, 1, 1), (16, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf4, 16, 'torch.ops.aten.convolution_backward.default')
            del buf2
        return (None, None, None, buf3, buf4, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = 2
    primals_2 = 16
    primals_3 = 224
    primals_4 = rand_strided((2, 16, 224, 224), (802816, 50176, 224, 1), device='npu:0', dtype=torch.float32)
    primals_5 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='npu:0', dtype=torch.float32)
    convolution = rand_strided((2, 16, 7, 7), (784, 49, 7, 1), device='npu:0', dtype=torch.float32)
    sigmoid = rand_strided((2, 16, 1, 1), (16, 1, 1, 1), device='npu:0', dtype=torch.float32)
    tangents_1 = rand_strided((2, 16, 7, 7), (784, 49, 7, 1), device='npu:0', dtype=torch.float32)
    return [primals_1, primals_2, primals_3, primals_4, primals_5, convolution, sigmoid, tangents_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
