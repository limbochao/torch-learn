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


# kernel path: /tmp/tmp_izavk5m/e6/ce6rkjmj6funp2axxtynwvaeytzvq44b3p47hb5katql43n7bdmw.py
# Topologically Sorted Source Nodes: [_foreach_add], Original ATen: [aten._foreach_add]
# Source node to ATen node mapping:
#   _foreach_add => _foreach_add, getitem
# Graph fragment:
#   %arg0_1 : Tensor "f32[1024, 2048][2048, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %arg3_1 : Tensor "f32[1024, 2048][1, 1024]npu:0" = PlaceHolder[target=arg3_1]
#   %_foreach_add : [num_users=3] = call_function[target=torch.ops.aten._foreach_add.List](args = ([%arg0_1, %arg1_1, %arg2_1], [%arg3_1, %arg4_1, %arg5_1]), kwargs = {})
#   %getitem : Tensor "f32[1024, 2048][2048, 1]npu:0"[num_users=1] = call_function[target=operator.getitem](args = (%_foreach_add, 0), kwargs = {})
#   return %getitem
# SchedulerNodes: [SchedulerNode(name='op0')]

triton_poi_fused__foreach_add_0 = async_compile.triton('triton_poi_fused__foreach_add_0', '''
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
    size_hints={'y0': 1024, 'x1': 2048}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__foreach_add_0', 'mutated_arg_names': [], 'backend_hash': 'AEAB8A5CC9A7D3017E38940399AA7AC23F76B542A0EC23BE6E8F44D2F6FD8A15', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 1024), ('x1', 2048)), 'low_dims': {0, 1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__foreach_add_0(in_ptr0, in_ptr1, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
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
            tmp0 = tl.load(in_ptr0 + (x1 + 2048*y0), x1_mask & y0_mask)
            tmp1 = tl.load(in_ptr1 + (y0 + 1024*x1), x1_mask & y0_mask)
            tmp2 = tmp0 + tmp1
            tl.store(out_ptr0 + (x1 + 2048*y0), tmp2, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: /tmp/tmp_izavk5m/2a/c2anfpu2dczunjh3qk6v2ayob27kpckfkxsmrbly2w6oggu2ov6m.py
# Topologically Sorted Source Nodes: [_foreach_add], Original ATen: [aten._foreach_add]
# Source node to ATen node mapping:
#   _foreach_add => _foreach_add, getitem_1
# Graph fragment:
#   %arg1_1 : Tensor "f32[2048, 2048][2048, 1]npu:0" = PlaceHolder[target=arg1_1]
#   %arg4_1 : Tensor "f32[2048, 2048][1, 2048]npu:0" = PlaceHolder[target=arg4_1]
#   %_foreach_add : [num_users=3] = call_function[target=torch.ops.aten._foreach_add.List](args = ([%arg0_1, %arg1_1, %arg2_1], [%arg3_1, %arg4_1, %arg5_1]), kwargs = {})
#   %getitem_1 : Tensor "f32[2048, 2048][2048, 1]npu:0"[num_users=1] = call_function[target=operator.getitem](args = (%_foreach_add, 1), kwargs = {})
#   return %getitem_1
# SchedulerNodes: [SchedulerNode(name='op1')]

triton_poi_fused__foreach_add_1 = async_compile.triton('triton_poi_fused__foreach_add_1', '''
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
    size_hints={'y0': 2048, 'x1': 2048}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__foreach_add_1', 'mutated_arg_names': [], 'backend_hash': 'AEAB8A5CC9A7D3017E38940399AA7AC23F76B542A0EC23BE6E8F44D2F6FD8A15', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 2048), ('x1', 2048)), 'low_dims': {0, 1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__foreach_add_1(in_ptr0, in_ptr1, out_ptr0, y0_numel, x1_numel, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
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
            tmp0 = tl.load(in_ptr0 + (x1 + 2048*y0), x1_mask & y0_mask)
            tmp1 = tl.load(in_ptr1 + (y0 + 2048*x1), x1_mask & y0_mask)
            tmp2 = tmp0 + tmp1
            tl.store(out_ptr0 + (x1 + 2048*y0), tmp2, x1_mask & y0_mask)
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
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            buf0 = empty_strided((1024, 2048), (2048, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [_foreach_add], Original ATen: [aten._foreach_add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__foreach_add_0.run(arg0_1, arg3_1, buf0, 1024, 2048, stream=stream0)
            del arg0_1
            del arg3_1
            buf1 = empty_strided((2048, 2048), (2048, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [_foreach_add], Original ATen: [aten._foreach_add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__foreach_add_1.run(arg1_1, arg4_1, buf1, 2048, 2048, stream=stream0)
            del arg1_1
            del arg4_1
            buf2 = empty_strided((1024, 2048), (2048, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [_foreach_add], Original ATen: [aten._foreach_add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__foreach_add_0.run(arg2_1, arg5_1, buf2, 1024, 2048, stream=stream0)
            del arg2_1
            del arg5_1
        return (buf0, buf1, buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1024, 2048), (2048, 1), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((2048, 2048), (2048, 1), device='npu:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, 2048), (2048, 1), device='npu:0', dtype=torch.float32)
    arg3_1 = rand_strided((1024, 2048), (1, 1024), device='npu:0', dtype=torch.float32)
    arg4_1 = rand_strided((2048, 2048), (1, 2048), device='npu:0', dtype=torch.float32)
    arg5_1 = rand_strided((1024, 2048), (1, 1024), device='npu:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
