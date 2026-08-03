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
from torch_npu._C import _npu_getCurrentRawStream as get_raw_stream
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch_npu._C import _npu_getCurrentRawStream as get_raw_stream
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


# kernel path: /tmp/torchinductor_root/3p/c3pzxyj2ikl27o36ntswhemwekshkkq6agzloffgekjhbbkmhilj.py
# Topologically Sorted Source Nodes: [sign_8, view_160, amax, view_161, gt], Original ATen: [aten.sign, aten.view, aten.amax, aten.gt]
# Source node to ATen node mapping:
#   amax => amax
#   gt => gt
#   sign_8 => sign_8
#   view_160 => view_160
#   view_161 => view_161
# Graph fragment:
#   %arg383_1 : Tensor "i32[s13, 4800][4800, 1]npu:0" = PlaceHolder[target=arg383_1]
#   %sign_8 : Tensor "i32[s13, 4800][4800, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.sign.default](args = (%arg383_1,), kwargs = {})
#   %view_160 : Tensor "i32[s13, 2400, 2][4800, 2, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sign_8, [-1, 2400, 2]), kwargs = {})
#   %amax : Tensor "i32[s13, 2400][2400, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%view_160, [2]), kwargs = {})
#   %view_161 : Tensor "i32[2400*s13][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%amax, [-1]), kwargs = {})
#   %gt : Tensor "b8[2400*s13][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%view_161, 0), kwargs = {})
#   return %gt
# SchedulerNodes: [SchedulerNode(name='op0')]

triton_poi_fused_amax_gt_sign_view_0 = async_compile.triton('triton_poi_fused_amax_gt_sign_view_0', '''
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
    size_hints={'x0': 31200},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i1', 'x0_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_amax_gt_sign_view_0', 'mutated_arg_names': [], 'backend_hash': '74912D1CBCC473172D5335076ED94C082234DC5B012F095796BD676FEFFE0B51', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['x0'], 'axis_static_values': (), 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.bool, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': (), 'enable_auto_blockify': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_amax_gt_sign_view_0(in_ptr0, out_ptr0, x0_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        tmp0 = tl.load(in_ptr0 + (2*x0), x0_mask)
        tmp8 = tl.load(in_ptr0 + (1 + 2*x0), x0_mask)
        tmp1 = tl.full([1], 0, tl.int32)
        tmp2 = tmp1 < tmp0
        tmp3 = tmp2.to(tl.int8)
        tmp4 = tmp0 < tmp1
        tmp5 = tmp4.to(tl.int8)
        tmp6 = tmp3 - tmp5
        tmp7 = tmp6.to(tmp0.dtype)
        tmp9 = tmp1 < tmp8
        tmp10 = tmp9.to(tl.int8)
        tmp11 = tmp8 < tmp1
        tmp12 = tmp11.to(tl.int8)
        tmp13 = tmp10 - tmp12
        tmp14 = tmp13.to(tmp8.dtype)
        tmp15 = triton_helpers.maximum(tmp7, tmp14)
        tmp16 = tmp15 > tmp1
        tl.store(out_ptr0 + (x0), tmp16, x0_mask)
''', device_str='npu')

async_compile.wait(globals())
del async_compile


from torch._dynamo.testing import rand_strided

arg221_1 = 13
arg383_1 = rand_strided((13, 4800), (4800, 1), device='npu:0', dtype=torch.int32)

s13 = arg221_1
torch.npu.set_device(0)
buf0 = empty_strided((2400*s13, ), (1, ), device='npu', dtype=torch.bool)
# Topologically Sorted Source Nodes: [sign_8, view_160, amax, view_161, gt], Original ATen: [aten.sign, aten.view, aten.amax, aten.gt]
triton_poi_fused_amax_gt_sign_view_0_x0_numel = 2400*s13
stream0 = get_raw_stream(0)
triton_poi_fused_amax_gt_sign_view_0.run(arg383_1, buf0, triton_poi_fused_amax_gt_sign_view_0_x0_numel, stream=stream0)
