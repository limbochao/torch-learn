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


# kernel path: /home/l30023782/PTA_210/combo_kernel_ut/cache/device3/4s/c4sqcqcprppb64dtdqmkc65jac7x7wizkz7whxboz5bemvfsoda4.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# SchedulerNodes: [ForeachKernelSchedulerNode(nodes=op0_op1_op2_op3_op4_op5)]

triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
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
    size_hints={'x0': 240}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'X0BLOCK': 'i64', 'X0BLOCK_SUB': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'backend_hash': 'AEAB8A5CC9A7D3017E38940399AA7AC23F76B542A0EC23BE6E8F44D2F6FD8A15', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['x0'], 'axis_static_values': (('x0', 240),), 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': (), 'combo_grid_meta': {'num_kernels': 1, 'min_blocks': 0, 'default_config': {'XBLOCK': 'X0BLOCK'}, 'split_numels': [240], 'no_split': [False], 'split_block': 'X0BLOCK', 'no_x_dim_0': False, 'xnumel_0': 240}}
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    pid = tl.program_id(0)
    num_blocks_0 = tl.cdiv(240, X0BLOCK)
    if pid < num_blocks_0:
        pid_offset = pid
        x0_numel = 240
        x0_offset = pid_offset.to(tl.int64) * X0BLOCK
        base_x0= tl.arange(0, X0BLOCK_SUB).to(tl.int64)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
            x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
            tmp0 = tl.load(in_ptr0 + (x0), x0_mask)
            tmp1 = tl_math.tanh(tmp0)
            tl.store(out_ptr0 + (x0), tmp1, x0_mask)
    else:
        pass
''', device_str='npu')


# kernel path: /home/l30023782/PTA_210/combo_kernel_ut/cache/device3/zj/czjnejyj225wfetxd3snupof46bad6lzobc773ts5hrjqcobuhjr.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# SchedulerNodes: [ForeachKernelSchedulerNode(nodes=op0_op1_op2_op3_op4_op5)]

triton_per_fused_1 = async_compile.triton('triton_per_fused_1', '''
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
    size_hints={'x0': 20, 'r1': 20},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i64', 'out_ptr2': '*fp32', 'out_ptr3': '*i64', 'out_ptr4': '*fp32', 'X0BLOCK': 'i64', 'X0BLOCK_SUB': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_1', 'mutated_arg_names': [], 'backend_hash': 'AEAB8A5CC9A7D3017E38940399AA7AC23F76B542A0EC23BE6E8F44D2F6FD8A15', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 20), ('r1', 20)), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': (), 'combo_grid_meta': {'num_kernels': 3, 'min_blocks': 0, 'default_config': {'XBLOCK': 'X0BLOCK'}, 'split_numels': [20, 10, 10], 'no_split': [False, False, False], 'split_block': 'X0BLOCK', 'no_x_dim_0': False, 'xnumel_0': 20, 'no_x_dim_1': False, 'xnumel_1': 10, 'no_x_dim_2': False, 'xnumel_2': 10}}
)
@triton.jit
def triton_per_fused_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    pid = tl.program_id(0)
    num_blocks_0 = tl.cdiv(20, X0BLOCK)
    num_blocks_1 = num_blocks_0 + tl.cdiv(10, X0BLOCK)
    num_blocks_2 = num_blocks_1 + tl.cdiv(10, X0BLOCK)
    if pid < num_blocks_0:
        pid_offset = pid
        x0_numel = 20
        r1_numel = 20
        R1BLOCK_SUB: tl.constexpr = 20
        x0_offset = pid_offset.to(tl.int64) * X0BLOCK
        base_x0= tl.arange(0, X0BLOCK_SUB).to(tl.int64)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        base_r1= tl.arange(0, R1BLOCK_SUB).to(tl.int64)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
            x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
            rindex = base_r1
            r1 = base_r1[:,None]
            r1_mask = r1 < r1_numel
            tmp0 = tl.load(in_ptr0 + (x0 + 20*r1), x0_mask, other=0.0)
            tmp2 = tl.max(tmp0, 0, propagate_nan=True).reshape(1, X0BLOCK_SUB)
            tmp4 = tl.broadcast_to(r1.reshape([R1BLOCK_SUB,1]), tmp0.shape)
            _, tmp3_tmp = triton_helpers.max_with_index(tmp0, tmp4, 0)
            tmp3 = tmp3_tmp.reshape(1, X0BLOCK_SUB)
            tl.store(out_ptr0 + (x0 ), tmp2, x0_mask)
            tl.store(out_ptr1 + (x0 ), tmp3, x0_mask)
    elif pid < num_blocks_1:
        pid_offset = pid - num_blocks_0
        x0_numel = 10
        r1_numel = 10
        R1BLOCK_SUB: tl.constexpr = 10
        x0_offset = pid_offset.to(tl.int64) * X0BLOCK
        base_x0= tl.arange(0, X0BLOCK_SUB).to(tl.int64)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        base_r1= tl.arange(0, R1BLOCK_SUB).to(tl.int64)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
            x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
            rindex = base_r1
            r1 = base_r1[:,None]
            r1_mask = r1 < r1_numel
            tmp5 = tl.load(in_ptr1 + (x0 + 10*r1), x0_mask, other=0.0)
            tmp7 = tl.min(tmp5, 0).reshape(1, X0BLOCK_SUB)
            tmp9 = tl.broadcast_to(r1.reshape([R1BLOCK_SUB,1]), tmp5.shape)
            _, tmp8_tmp = triton_helpers.min_with_index(tmp5, tmp9, 0)
            tmp8 = tmp8_tmp.reshape(1, X0BLOCK_SUB)
            tl.store(out_ptr2 + (x0 ), tmp7, x0_mask)
            tl.store(out_ptr3 + (x0 ), tmp8, x0_mask)
    elif pid < num_blocks_2:
        pid_offset = pid - num_blocks_1
        x0_numel = 10
        r1_numel = 10
        R1BLOCK_SUB: tl.constexpr = 10
        x0_offset = pid_offset.to(tl.int64) * X0BLOCK
        base_x0= tl.arange(0, X0BLOCK_SUB).to(tl.int64)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        base_r1= tl.arange(0, R1BLOCK_SUB).to(tl.int64)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
            x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
            r1 = base_r1[:,None]
            r1_mask = r1 < r1_numel
            tmp10 = tl.load(in_ptr2 + (x0 + 10*r1), x0_mask, other=0.0)
            tmp12 = tl.sum(tmp10, 0).reshape(1, X0BLOCK_SUB)
            tl.store(out_ptr4 + (x0 ), tmp12, x0_mask)
    else:
        pass
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
            buf5 = empty_strided((30, 8), (8, 1), device='npu', dtype=torch.float32)
            buf0 = empty_strided((20, ), (1, ), device='npu', dtype=torch.float32)
            buf1 = empty_strided((20, ), (1, ), device='npu', dtype=torch.int64)
            buf2 = empty_strided((10, ), (1, ), device='npu', dtype=torch.float32)
            buf3 = empty_strided((10, ), (1, ), device='npu', dtype=torch.int64)
            buf4 = empty_strided((10, ), (1, ), device='npu', dtype=torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            stream0 = get_raw_stream(0)
            triton_poi_fused_0.run(arg3_1, buf5, stream=stream0)
            # Unsorted Source Nodes: [], Original ATen: []
            stream0 = get_raw_stream(0)
            triton_per_fused_1.run(arg1_1, arg2_1, arg0_1, buf0, buf1, buf2, buf3, buf4, stream=stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            del arg3_1
        return (buf4, buf0, buf1, buf2, buf3, buf5, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((10, 10), (10, 1), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((20, 20), (20, 1), device='npu:0', dtype=torch.float32)
    arg2_1 = rand_strided((10, 10), (10, 1), device='npu:0', dtype=torch.float32)
    arg3_1 = rand_strided((30, 8), (8, 1), device='npu:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
