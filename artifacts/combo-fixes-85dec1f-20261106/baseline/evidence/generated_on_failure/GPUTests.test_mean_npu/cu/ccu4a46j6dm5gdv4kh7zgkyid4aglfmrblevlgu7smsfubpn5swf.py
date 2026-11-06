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


# kernel path: /tmp/tmpibgr9crs/bo/cbofujqyfnqkwwahrgdszv44txhhyuzha24ka2brzkm5eijx56uz.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# SchedulerNodes: [ForeachKernelSchedulerNode(nodes=op0_op2_op3_op4_op5)]

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
    size_hints={'x0': 32}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'X0BLOCK': 'i32', 'X0BLOCK_SUB': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'backend_hash': '<removed>', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['x0'], 'axis_static_values': (('x0', 32),), 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': (), 'combo_grid_meta': {'num_kernels': 1, 'min_blocks': 0, 'default_config': {'XBLOCK': 'X0BLOCK'}, 'split_numels': [32], 'no_split': [False], 'split_block': 'X0BLOCK', 'no_x_dim_0': False, 'xnumel_0': 32}, 'combo_subkernel_meta': [{'axis_names': ['x0'], 'tiling_axis': [0], 'no_loop_axis': [], 'split_axis': [0], 'low_dims': {0}, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'size_hints': {'x0': 32}, 'persistent_reduction': False}]}
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    pid = tl.program_id(0)
    num_blocks_0 = tl.cdiv(32, X0BLOCK)
    if pid < num_blocks_0:
        pid_offset = pid
        x0_numel = 32
        x0_offset = pid_offset * X0BLOCK
        base_x0= tl.arange(0, X0BLOCK_SUB)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
            x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
            tmp0 = tl.load(in_ptr0 + (x0), x0_mask)
            tl.static_assert(tmp0.dtype == tl.float32)
            tmp1 = tl.load(in_ptr0 + (32 + x0), x0_mask)
            tl.static_assert(tmp1.dtype == tl.float32)
            tmp2 = tmp0 + tmp1
            tl.static_assert(tmp2.dtype == tl.float32)
            tl.static_assert(tmp2.dtype == tl.float32)
            tmp3 = 2.0
            tl.static_assert(tmp3.dtype == tl.float32)
            tl.static_assert(tmp3.dtype == tl.float32)
            tmp4 = (tmp2 / tmp3)
            tl.static_assert(tmp4.dtype == tl.float32)
            tl.static_assert(tmp4.dtype == tl.float32)
            tl.store(out_ptr0 + (x0), tmp4, x0_mask)
    else:
        pass
''', device_str='npu')


# kernel path: /tmp/tmpibgr9crs/tm/ctmngwshe6pcggjxhvyygprrtgeaw2mhofwzikkyqwlomyhkiqpf.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# SchedulerNodes: [ForeachKernelSchedulerNode(nodes=op0_op2_op3_op4_op5)]

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
    size_hints={'x0': 10, 'r1': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'X0BLOCK': 'i32', 'X0BLOCK_SUB': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_1', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<removed>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 2), ('r1', 32)), 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': (), 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': 0, 'default_config': {'XBLOCK': 'X0BLOCK'}, 'split_numels': [2, 8], 'no_split': [False, False], 'split_block': 'X0BLOCK', 'no_x_dim_0': False, 'xnumel_0': 2, 'no_x_dim_1': False, 'xnumel_1': 8}, 'combo_subkernel_meta': [{'axis_names': ['x0', 'r1'], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'split_axis': [0], 'low_dims': {1}, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'size_hints': {'x0': 2, 'r1': 32}, 'persistent_reduction': True}, {'axis_names': ['x0', 'r1'], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'split_axis': [0], 'low_dims': {1}, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'size_hints': {'x0': 8, 'r1': 8}, 'persistent_reduction': True}]}
)
@triton.jit
def triton_per_fused_1(in_out_ptr0, in_ptr0, out_ptr0, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    pid = tl.program_id(0)
    num_blocks_0 = tl.cdiv(2, X0BLOCK)
    num_blocks_1 = num_blocks_0 + tl.cdiv(8, X0BLOCK)
    if pid < num_blocks_0:
        pid_offset = pid
        x0_numel = 2
        r1_numel = 32
        R1BLOCK_SUB: tl.constexpr = 32
        x0_offset = pid_offset * X0BLOCK
        base_x0= tl.arange(0, X0BLOCK_SUB)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        base_r1= tl.arange(0, R1BLOCK_SUB)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[:,None]
            x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
            r1 = base_r1[None,:]
            r1_mask = r1 < r1_numel
            tmp0 = tl.load(in_ptr0 + (r1 + 32*x0), x0_mask, other=0.0)
            tl.static_assert(tmp0.dtype == tl.float32)
            tmp2 = tl.sum(tmp0, 1).reshape(X0BLOCK_SUB, 1)
            tl.static_assert(tmp2.dtype == tl.float32)
            tl.store(out_ptr0 + (x0 ), tmp2, x0_mask)
    elif pid < num_blocks_1:
        pid_offset = pid - num_blocks_0
        x0_numel = 8
        r1_numel = 8
        R1BLOCK_SUB: tl.constexpr = 8
        x0_offset = pid_offset * X0BLOCK
        base_x0= tl.arange(0, X0BLOCK_SUB)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        base_r1= tl.arange(0, R1BLOCK_SUB)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[:,None]
            x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
            r1 = base_r1[None,:]
            r1_mask = r1 < r1_numel
            tmp3 = tl.load(in_ptr0 + (r1 + 8*x0), x0_mask, other=0.0)
            tl.static_assert(tmp3.dtype == tl.float32)
            tmp5 = tl.sum(tmp3, 1).reshape(X0BLOCK_SUB, 1)
            tl.static_assert(tmp5.dtype == tl.float32)
            tmp6 = 8.0
            tl.static_assert(tmp6.dtype == tl.float32)
            tl.static_assert(tmp6.dtype == tl.float32)
            tmp7 = (tmp5 / tmp6)
            tl.static_assert(tmp7.dtype == tl.float32)
            tl.static_assert(tmp7.dtype == tl.float32)
            tl.store(in_out_ptr0 + (x0), tmp7, x0_mask)
            tl.store(in_out_ptr0 + (x0 ), tmp5, x0_mask)
    else:
        pass
''', device_str='npu')


# kernel path: /tmp/tmpibgr9crs/ik/cikwgmryyk5qyao5tkrke7j4cy6av4kcvh2lqcqlef2w54t5u5na.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# SchedulerNodes: [ForeachKernelSchedulerNode(nodes=op0_op2_op3_op4_op5)]

triton_poi_fused_2 = async_compile.triton('triton_poi_fused_2', '''
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
    size_hints={'y0': 2, 'x1': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'Y0BLOCK': 'i32', 'Y0BLOCK_SUB': 'constexpr', 'X1BLOCK_SUB': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'backend_hash': '<removed>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 2), ('x1', 8)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('Y0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': (), 'combo_grid_meta': {'num_kernels': 1, 'min_blocks': 0, 'default_config': {'XBLOCK': 'Y0BLOCK'}, 'split_numels': [2], 'no_split': [False], 'split_block': 'Y0BLOCK', 'no_x_dim_0': False, 'xnumel_0': 2}, 'combo_subkernel_meta': [{'axis_names': ['y0', 'x1'], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'split_axis': [0], 'low_dims': {1}, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'size_hints': {'y0': 2, 'x1': 8}, 'persistent_reduction': False}]}
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, Y0BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    pid = tl.program_id(0)
    num_blocks_0 = tl.cdiv(2, Y0BLOCK)
    if pid < num_blocks_0:
        pid_offset = pid
        y0_numel = 2
        x1_numel = 8
        y0_offset = pid_offset * Y0BLOCK
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
                tmp0 = tl.load(in_ptr0 + (x1 + 32*y0), x1_mask & y0_mask)
                tl.static_assert(tmp0.dtype == tl.float32)
                tmp1 = tl.load(in_ptr0 + (8 + x1 + 32*y0), x1_mask & y0_mask)
                tl.static_assert(tmp1.dtype == tl.float32)
                tmp3 = tl.load(in_ptr0 + (16 + x1 + 32*y0), x1_mask & y0_mask)
                tl.static_assert(tmp3.dtype == tl.float32)
                tmp5 = tl.load(in_ptr0 + (24 + x1 + 32*y0), x1_mask & y0_mask)
                tl.static_assert(tmp5.dtype == tl.float32)
                tmp2 = tmp0 + tmp1
                tl.static_assert(tmp2.dtype == tl.float32)
                tl.static_assert(tmp2.dtype == tl.float32)
                tmp4 = tmp2 + tmp3
                tl.static_assert(tmp4.dtype == tl.float32)
                tl.static_assert(tmp4.dtype == tl.float32)
                tmp6 = tmp4 + tmp5
                tl.static_assert(tmp6.dtype == tl.float32)
                tl.static_assert(tmp6.dtype == tl.float32)
                tmp7 = 4.0
                tl.static_assert(tmp7.dtype == tl.float32)
                tl.static_assert(tmp7.dtype == tl.float32)
                tmp8 = (tmp6 / tmp7)
                tl.static_assert(tmp8.dtype == tl.float32)
                tl.static_assert(tmp8.dtype == tl.float32)
                tl.store(out_ptr0 + (x1 + 8*y0), tmp8, x1_mask & y0_mask)
    else:
        pass
''', device_str='npu')


# kernel path: /tmp/tmpibgr9crs/7d/c7doovmqwinvuhlie6dtpaaegdbhi7gjjoyrr5xuwwazj6z4t42v.py
# Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean => mean
# Graph fragment:
#   %buf0 : Tensor "f32[2][1]npu:0" = PlaceHolder[target=buf0]
#   %buf1 : Tensor "f32[][]npu:0" = PlaceHolder[target=buf1]
#   %mean : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%arg0_1,), kwargs = {})
#   return %buf1,%mean
# SchedulerNodes: [SchedulerNode(name='op1'), SchedulerNode(name='op6')]

triton_per_fused_mean_3 = async_compile.triton('triton_per_fused_mean_3', '''
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
    size_hints={'r0': 2},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'r0_numel': 'i32', 'R0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_3', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<removed>', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['r0'], 'axis_static_values': (('r0', 2),), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('R0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_mean_3(in_out_ptr0, in_ptr0, r0_numel, R0BLOCK):
    R0BLOCK_SUB: tl.constexpr = 2
    r0_offset = tl.program_id(0) * R0BLOCK
    base_r0= tl.arange(0, R0BLOCK_SUB)
    loops_r0 = (R0BLOCK + R0BLOCK_SUB - 1) // R0BLOCK_SUB
    r0 = base_r0
    r0_mask = r0 < min(R0BLOCK+r0_offset, r0_numel)
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tl.static_assert(tmp0.dtype == tl.float32)
    tmp2 = triton_helpers.promote_to_tensor(tl.sum(tmp0, 0))
    tl.static_assert(tmp2.dtype == tl.float32)
    tmp3 = 64.0
    tl.static_assert(tmp3.dtype == tl.float32)
    tl.static_assert(tmp3.dtype == tl.float32)
    tmp4 = (tmp2 / tmp3)
    tl.static_assert(tmp4.dtype == tl.float32)
    tl.static_assert(tmp4.dtype == tl.float32)
    tl.store(in_out_ptr0 + (tl.arange(0,1)), tmp4, None)
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
        arg0_1, = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            buf5 = empty_strided((4, 8), (8, 1), device='npu', dtype=torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            stream0 = get_raw_stream(0)
            triton_poi_fused_0.run(arg0_1, buf5, stream=stream0)
            buf0 = empty_strided((2, ), (1, ), device='npu', dtype=torch.float32)
            buf2 = empty_strided((1, 2, 4), (8, 4, 1), device='npu', dtype=torch.float32)
            buf3 = buf2; del buf2  # reuse
            # Unsorted Source Nodes: [], Original ATen: []
            stream0 = get_raw_stream(0)
            triton_per_fused_1.run(buf3, arg0_1, buf0, stream=stream0)
            buf4 = empty_strided((1, 2, 1, 8), (16, 8, 8, 1), device='npu', dtype=torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            stream0 = get_raw_stream(0)
            triton_poi_fused_2.run(arg0_1, buf4, stream=stream0)
            del arg0_1
            buf1 = empty_strided((), (), device='npu', dtype=torch.float32)
            buf6 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
            stream0 = get_raw_stream(0)
            triton_per_fused_mean_3.run(buf6, buf0, 2, stream=stream0)
            del buf0
        return (buf6, buf3, buf4, buf5, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 2, 4, 8), (64, 32, 8, 1), device='npu:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
