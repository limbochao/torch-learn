# AOT ID: ['0_forward']
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
# Topologically Sorted Source Nodes: [input_1, , input_2], Original ATen: [aten.t, aten.addmm, aten.tanh]
# Source node to ATen node mapping:
#    => add_tensor_3, mul_tensor
#   input_1 => permute
#   input_2 => tanh
# Graph fragment:
#   %primals_2 : Tensor "f32[16][1]npu:0" = PlaceHolder[target=primals_2]
#   %primals_3 : Tensor "f32[4096, 1][1, 1]npu:0" = PlaceHolder[target=primals_3]
#   %primals_1 : Tensor "f32[16, 1][1, 1]npu:0" = PlaceHolder[target=primals_1]
#   %permute : Tensor "f32[1, 16][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_1, [1, 0]), kwargs = {})
#   %mul_tensor : Tensor "f32[4096, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_3, %permute), kwargs = {})
#   %add_tensor_3 : Tensor "f32[4096, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %mul_tensor), kwargs = {})
#   %tanh : Tensor "f32[4096, 16][16, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_tensor_3,), kwargs = {})
#   return %tanh
# SchedulerNodes: [SchedulerNode(name='op0')]

triton_poi_fused_addmm_t_tanh_0 = async_compile.triton('triton_poi_fused_addmm_t_tanh_0', '''
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
    size_hints={'y0': 4096, 'x1': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_t_tanh_0', 'mutated_arg_names': [], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 4096), ('x1', 16)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_t_tanh_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
    x1_numel = 16
    X1BLOCK_SUB: tl.constexpr = 16
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        x1 = base_x1[None,:]
        tmp0 = tl.load(in_ptr0 + (x1), None)
        tmp1 = tl.load(in_ptr1 + (y0), y0_mask)
        tmp2 = tl.load(in_ptr2 + (x1), None)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl_math.tanh(tmp4)
        tl.store(out_ptr0 + (x1 + 16*y0), tmp5, y0_mask)
''', device_str='npu')


# kernel path: <generated-cache>
# Topologically Sorted Source Nodes: [, input_4], Original ATen: [aten.addmm, aten.tanh]
# Source node to ATen node mapping:
#    => add_tensor_2
#   input_4 => tanh_1
# Graph fragment:
#   %primals_5 : Tensor "f32[16][1]npu:0" = PlaceHolder[target=primals_5]
#   %mm_default_2 : Tensor "f32[4096, 16][16, 1]npu:0" = PlaceHolder[target=mm_default_2]
#   %add_tensor_2 : Tensor "f32[4096, 16][16, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_5, %mm_default_2), kwargs = {})
#   %tanh_1 : Tensor "f32[4096, 16][16, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_tensor_2,), kwargs = {})
#   return %tanh_1
# SchedulerNodes: [SchedulerNode(name='op2')]

triton_poi_fused_addmm_tanh_1 = async_compile.triton('triton_poi_fused_addmm_tanh_1', '''
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
    size_hints={'y0': 4096, 'x1': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_tanh_1', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<build-hash>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [1], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 4096), ('x1', 16)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_tanh_1(in_out_ptr0, in_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr):
    x1_numel = 16
    X1BLOCK_SUB: tl.constexpr = 16
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        x1 = base_x1[None,:]
        tmp0 = tl.load(in_ptr0 + (x1), None)
        tmp1 = tl.load(in_out_ptr0 + (x1 + 16*y0), y0_mask)
        tmp2 = tmp0 + tmp1
        tmp3 = tl_math.tanh(tmp2)
        tl.store(in_out_ptr0 + (x1 + 16*y0), tmp3, y0_mask)
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            primals_3 = copy_if_misaligned(primals_3)
            buf0 = empty_strided_npu((4096, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, , input_2], Original ATen: [aten.t, aten.addmm, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_t_tanh_0.run(primals_2, primals_3, primals_1, buf0, 4096, 16, stream=raw_stream0)
            del primals_1
            del primals_2
            buf1 = empty_strided_npu((4096, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_3, ], Original ATen: [aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:1
            extern_kernels.mm(buf0, reinterpret_tensor(primals_4, (16, 16), (1, 16), 0), out=buf1)
            buf2 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [, input_4], Original ATen: [aten.addmm, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_tanh_1.run(buf2, primals_5, 4096, 16, stream=raw_stream0)
            del primals_5
            buf3 = empty_strided_npu((4096, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_5, ], Original ATen: [aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:2
            extern_kernels.mm(buf2, reinterpret_tensor(primals_6, (16, 16), (1, 16), 0), out=buf3)
            buf4 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [, input_6], Original ATen: [aten.addmm, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_tanh_1.run(buf4, primals_7, 4096, 16, stream=raw_stream0)
            del primals_7
            buf5 = empty_strided_npu((4096, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_7, ], Original ATen: [aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:3
            extern_kernels.mm(buf4, reinterpret_tensor(primals_8, (16, 16), (1, 16), 0), out=buf5)
            buf6 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [, input_8], Original ATen: [aten.addmm, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_tanh_1.run(buf6, primals_9, 4096, 16, stream=raw_stream0)
            del primals_9
            # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.t, aten.addmm]
            # [Provenance debug handles] torch.ops.aten.addmm.default:4
            buf7 = torch.ops.aten.addmm.default(primals_11, buf6, reinterpret_tensor(primals_10, (16, 1), (1, 16), 0), alpha=1, beta=1)
            del primals_11
            buf8 = buf7
            assert_size_stride(buf8, (4096, 1), (1, 1), 'torch.ops.aten.addmm.default')
            assert_alignment(buf8, 16, 'torch.ops.aten.addmm.default')
            del buf7
        return (buf8, primals_3, primals_4, primals_6, primals_8, primals_10, buf0, buf2, buf4, buf6, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((16, 1), (1, 1), device='npu:0', dtype=torch.float32)
    primals_2 = rand_strided((16, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_3 = rand_strided((4096, 1), (1, 1), device='npu:0', dtype=torch.float32)
    primals_4 = rand_strided((16, 16), (16, 1), device='npu:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_6 = rand_strided((16, 16), (16, 1), device='npu:0', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_8 = rand_strided((16, 16), (16, 1), device='npu:0', dtype=torch.float32)
    primals_9 = rand_strided((16, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_10 = rand_strided((1, 16), (16, 1), device='npu:0', dtype=torch.float32)
    primals_11 = rand_strided((1, ), (1, ), device='npu:0', dtype=torch.float32)
    return [primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('lennard_jones', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
