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
import torch_npu
empty_strided_npu = torch_npu._C._empty_strided_npu


# kernel path: <workspace>
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# SchedulerNodes: [ForeachKernelSchedulerNode(nodes=op0_op1_op3_op4_op5_op6_op7)]

triton_per_fused_0 = async_compile.triton('triton_per_fused_0', '''
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
    size_hints={'x0': 8192, 'r1': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_0', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<removed>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 8192), ('r1', 1024)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_0(in_out_ptr0, in_ptr0, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 1024
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[:,None]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        r1 = base_r1[None,:]
        r1_mask = True
        tmp0 = tl.load(in_ptr0 + (r1 + 1024*x0), x0_mask, other=0.0)
        tmp2 = tl.where(tl.broadcast_to((x0_mask.reshape([X0BLOCK_SUB, 1])), [X0BLOCK_SUB, R1BLOCK_SUB]), tmp0, 0)
        tmp3 = tl.sum(tmp2, 1).reshape(X0BLOCK_SUB, 1)
        tl.store(in_out_ptr0 + (x0), tmp3, x0_mask)
''', device_str='npu')


# kernel path: <workspace>
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# SchedulerNodes: [ForeachKernelSchedulerNode(nodes=op0_op1_op3_op4_op5_op6_op7)]

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
    size_hints={'r0': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*fp32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_1', 'mutated_arg_names': [], 'backend_hash': '<removed>', 'split_axis': [], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['r0'], 'axis_static_values': (('r0', 2048),), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': None, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': (), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': (), 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': 0, 'default_config': None, 'split_numels': [1, 1], 'no_split': [True, True], 'split_block': None, 'no_x_dim_0': True, 'xnumel_0': 1, 'no_x_dim_1': True, 'xnumel_1': 1}, 'combo_subkernel_meta': [{'axis_names': ['r0'], 'tiling_axis': [0], 'no_loop_axis': [], 'split_axis': [], 'low_dims': {0}, 'split_axis_dtype': None, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'size_hints': {'r0': 2048}, 'persistent_reduction': True}, {'axis_names': ['r0'], 'tiling_axis': [0], 'no_loop_axis': [], 'split_axis': [], 'low_dims': {0}, 'split_axis_dtype': None, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'size_hints': {'r0': 2048}, 'persistent_reduction': True}]}
)
@triton.jit
def triton_per_fused_1(in_ptr0, in_ptr1, out_ptr1, out_ptr3):
    pid = tl.program_id(0)
    num_blocks_0 = 1
    num_blocks_1 = num_blocks_0 + 1
    if pid < num_blocks_0:
        pid_offset = pid
        r0_numel = 2048
        R0BLOCK_SUB: tl.constexpr = 2048
        base_r0= tl.arange(0, R0BLOCK_SUB)
        loops_r0 = (r0_numel + R0BLOCK_SUB - 1) // R0BLOCK_SUB
        r0 = base_r0
        r0_mask = r0 < r0_numel
        tmp0 = tl.load(in_ptr0 + (r0), r0_mask, other=0.0)
        tmp2 = tl.where(tl.broadcast_to((r0_mask.reshape([R0BLOCK_SUB])), [R0BLOCK_SUB]), tmp0, 0)
        tmp3 = triton_helpers.promote_to_tensor(tl.sum(tmp2, 0))
        tl.store(out_ptr1 + (tl.arange(0,1) ), tmp3, None)
    elif pid < num_blocks_1:
        pid_offset = pid - num_blocks_0
        r0_numel = 2048
        R0BLOCK_SUB: tl.constexpr = 2048
        base_r0= tl.arange(0, R0BLOCK_SUB)
        loops_r0 = (r0_numel + R0BLOCK_SUB - 1) // R0BLOCK_SUB
        r0 = base_r0
        r0_mask = r0 < r0_numel
        tmp4 = tl.load(in_ptr1 + (r0), r0_mask, other=0.0)
        tmp6 = tl.where(tl.broadcast_to((r0_mask.reshape([R0BLOCK_SUB])), [R0BLOCK_SUB]), tmp4, 0)
        tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
        tl.store(out_ptr3 + (tl.arange(0,1) ), tmp7, None)
    else:
        pass
''', device_str='npu')


# kernel path: <workspace>
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# SchedulerNodes: [ForeachKernelSchedulerNode(nodes=op0_op1_op3_op4_op5_op6_op7)]

triton_red_fused_2 = async_compile.triton('triton_red_fused_2', '''
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
    size_hints={'x0': 1024, 'r1': 8192},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_2', 'mutated_arg_names': [], 'backend_hash': '<removed>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 1024), ('r1', 8192)), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_red_fused_2(in_ptr0, out_ptr0, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr, R1BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        _tmp2 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        for loop_r1 in range(loops_r1):
            r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[:,None]
            r1_mask = r1 < r1_numel
            tmp0 = tl.load(in_ptr0 + (x0 + 1024*r1), r1_mask & x0_mask, other=0.0)
            tmp1 = tl.reshape(tmp0, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp3 = _tmp2 + tmp1
            _tmp2 = tl.where(tl.reshape((r1_mask & x0_mask), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp3, _tmp2)
        tmp2 = tl.sum(_tmp2, 0).reshape(1, X0BLOCK_SUB)
        tl.store(out_ptr0 + (x0 ), tmp2, x0_mask)
''', device_str='npu')


# kernel path: <workspace>
# Topologically Sorted Source Nodes: [row_sum_reduced], Original ATen: [aten.sum]
# Source node to ATen node mapping:
#   row_sum_reduced => sum_3
# Graph fragment:
#   %buf1 : Tensor "f32[8192][1]npu:0" = PlaceHolder[target=buf1]
#   %sum_3 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%sum_1,), kwargs = {})
#   return %sum_3
# SchedulerNodes: [SchedulerNode(name='op2')]

triton_red_fused_sum_3 = async_compile.triton('triton_red_fused_sum_3', '''
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
    size_hints={'r0': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'r0_numel': 'i32', 'R0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_3', 'mutated_arg_names': [], 'backend_hash': '<removed>', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['r0'], 'axis_static_values': (('r0', 8192),), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('R0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_red_fused_sum_3(in_ptr0, out_ptr0, r0_numel, R0BLOCK, R0BLOCK_SUB : tl.constexpr):
    r0_offset = tl.program_id(0) * R0BLOCK
    base_r0= tl.arange(0, R0BLOCK_SUB)
    loops_r0 = (R0BLOCK + R0BLOCK_SUB - 1) // R0BLOCK_SUB
    _tmp2 = tl.full([R0BLOCK_SUB], 0, tl.float32)
    for loop_r0 in range(loops_r0):
        r0 = (loop_r0 * R0BLOCK_SUB) + base_r0
        r0_mask = r0 < min(R0BLOCK+r0_offset, r0_numel)
        tmp0 = tl.load(in_ptr0 + (r0), r0_mask, other=0.0)
        tmp1 = tl.reshape(tmp0, [R0BLOCK_SUB])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(tl.reshape((r0_mask), [R0BLOCK_SUB]), tmp3, _tmp2)
    tmp2 = triton_helpers.promote_to_tensor(tl.sum(_tmp2, 0))
    tl.store(out_ptr0 + (tl.arange(0,1) ), tmp2, None)
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
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg1_1 = copy_if_misaligned(arg1_1)
            arg2_1 = copy_if_misaligned(arg2_1)
            buf0 = empty_strided_npu((8192, ), (1, ), torch.float32)
            buf1 = buf0; del buf0  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_0.run(buf1, arg0_1, 8192, 1024, stream=raw_stream0)
            buf5 = empty_strided_npu((), (), torch.float32)
            buf7 = empty_strided_npu((), (), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_1.run(arg1_1, arg2_1, buf5, buf7, stream=raw_stream0)
            buf3 = empty_strided_npu((1024, ), (1, ), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_2.run(arg0_1, buf3, 1024, 8192, stream=raw_stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            buf2 = empty_strided_npu((), (), torch.float32)
            # Topologically Sorted Source Nodes: [row_sum_reduced], Original ATen: [aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_sum_3.run(buf1, buf2, 8192, stream=raw_stream0)
            del buf1
        return (buf2, buf3, buf5, buf7, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, 1024), (1024, 1), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    arg2_1 = rand_strided((2048, ), (1, ), device='npu:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))

