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


# kernel path: <original-profile-cache>
# Topologically Sorted Source Nodes: [le, scalar_tensor, where, sum_2], Original ATen: [aten.threshold_backward, aten.sum]
# Source node to ATen node mapping:
#   le => le
#   scalar_tensor => full_default_3
#   sum_2 => sum_2
#   where => where
# Graph fragment:
#   %relu_4 : Tensor "f32[200000, 64][64, 1]npu:0" = PlaceHolder[target=relu_4]
#   %mm : Tensor "f32[200000, 64][64, 1]npu:0" = PlaceHolder[target=mm]
#   %where : Tensor "f32[200000, 64][64, 1]npu:0" = PlaceHolder[target=where]
#   %le : Tensor "b8[200000, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_4, 0), kwargs = {})
#   %full_default_3 : Tensor "f32[][]npu:0"[num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where : Tensor "f32[200000, 64][64, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default_3, %mm), kwargs = {})
#   %sum_2 : Tensor "f32[1, 64][64, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%where, [0], True), kwargs = {})
#   return %where,%sum_2
# SchedulerNodes: [SchedulerNode(name='op8'), SchedulerNode(name='op11')]

triton_red_fused_sum_threshold_backward_3 = async_compile.triton('triton_red_fused_sum_threshold_backward_3', '''
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
    size_hints={'x0': 64, 'r1': 200000},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_threshold_backward_3', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '24B8565ECE985A50B0202446CE2E5AA39C23B2D84A59413775267D545EE78DFB', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 64), ('r1', 200000)), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_red_fused_sum_threshold_backward_3(in_out_ptr0, in_ptr0, out_ptr0, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr, R1BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        _tmp6 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        for loop_r1 in range(loops_r1):
            r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[:,None]
            r1_mask = r1 < r1_numel
            tmp0 = tl.load(in_out_ptr0 + (x0 + 64*r1), r1_mask & x0_mask, other=0.0)
            tmp3 = tl.load(in_ptr0 + (x0 + 64*r1), r1_mask & x0_mask, other=0.0)
            tmp1 = tl.full([1, 1], 0.0, tl.float32)
            tmp2 = tmp0 <= tmp1
            tmp4 = tl.where(tmp2, tmp1, tmp3)
            tmp5 = tl.reshape(tmp4, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp7 = _tmp6 + tmp5
            _tmp6 = tl.where(tl.reshape((r1_mask & x0_mask), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp7, _tmp6)
            tl.store(in_out_ptr0 + (x0 + 64*r1), tmp4, r1_mask & x0_mask)
        tmp6 = tl.sum(_tmp6, 0).reshape(1, X0BLOCK_SUB)
        tl.store(out_ptr0 + (x0 ), tmp6, x0_mask)
''', device_str='npu')


async_compile.wait(globals())
del async_compile
