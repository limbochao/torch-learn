# AOT ID: ['1_inference']
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


# kernel path: /tmp/tmps8vxzxs8/43/c43nhd5pdrqhzphaqzdzj7yqu7o2la2rkiyfyetjx22o47l6dqw5.py
# Topologically Sorted Source Nodes: [_adaptive_avg_pool2d], Original ATen: [aten._adaptive_avg_pool2d]
# Source node to ATen node mapping:
#   _adaptive_avg_pool2d => _adaptive_avg_pool2d
# Graph fragment:
#   %arg0_1 : Tensor "f32[2, 4, 3, 3][36, 9, 3, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %_adaptive_avg_pool2d : Tensor "f32[2, 4, 6, 6][144, 36, 6, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%arg0_1, [6, 6]), kwargs = {})
#   return %_adaptive_avg_pool2d
# SchedulerNodes: [SchedulerNode(name='op0')]

triton_poi_fused__adaptive_avg_pool2d_0 = async_compile.triton('triton_poi_fused__adaptive_avg_pool2d_0', '''
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
    size_hints={'z0': 8, 'y3': 3, 'y4': 2, 'x5': 3, 'x6': 2}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'z0_numel': 'i32', 'y3_numel': 'i32', 'y4_numel': 'i32', 'x5_numel': 'i32', 'x6_numel': 'i32', 'Z0BLOCK': 'i32', 'Y3BLOCK': 'i32', 'Y4BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_0', 'mutated_arg_names': [], 'backend_hash': '<removed>', 'split_axis': [0, 1, 2], 'tiling_axis': [0, 1, 2, 3, 4], 'no_loop_axis': [3, 4], 'axis_names': ['z0', 'y3', 'y4', 'x5', 'x6'], 'axis_static_values': (('z0', 8), ('y3', 3), ('y4', 2), ('x5', 3), ('x6', 2)), 'low_dims': {4}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('Z0BLOCK', 'Y3BLOCK', 'Y4BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_0(in_ptr0, out_ptr0, z0_numel, y3_numel, y4_numel, x5_numel, x6_numel, Z0BLOCK, Y3BLOCK, Y4BLOCK, Z0BLOCK_SUB : tl.constexpr, Y3BLOCK_SUB : tl.constexpr, Y4BLOCK_SUB : tl.constexpr):
    x5_numel = 3
    X5BLOCK_SUB: tl.constexpr = 4
    x6_numel = 2
    X6BLOCK_SUB: tl.constexpr = 2
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y3_offset = tl.program_id(1) * Y3BLOCK
    base_y3= tl.arange(0, Y3BLOCK_SUB)
    loops_y3 = (Y3BLOCK + Y3BLOCK_SUB - 1) // Y3BLOCK_SUB
    y4_offset = tl.program_id(2) * Y4BLOCK
    base_y4= tl.arange(0, Y4BLOCK_SUB)
    loops_y4 = (Y4BLOCK + Y4BLOCK_SUB - 1) // Y4BLOCK_SUB
    base_x5= tl.arange(0, X5BLOCK_SUB)
    base_x6= tl.arange(0, X6BLOCK_SUB)
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y3 in range(loops_y3):
            y3 = y3_offset + (loop_y3 * Y3BLOCK_SUB) + base_y3[None,:,None,None,None]
            y3_mask = y3 < min(Y3BLOCK+y3_offset, y3_numel)
            for loop_y4 in range(loops_y4):
                y4 = y4_offset + (loop_y4 * Y4BLOCK_SUB) + base_y4[None,None,:,None,None]
                y4_mask = y4 < min(Y4BLOCK+y4_offset, y4_numel)
                x5 = base_x5[None,None,None,:,None]
                x5_mask = x5 < x5_numel
                x6 = base_x6[None,None,None,None,:]
                x6_mask = x6 < x6_numel
                tmp11 = tl.load(in_ptr0 + (x5 + 3*y3 + 9*z0), x5_mask & y3_mask & z0_mask, other=0.0)
                tl.static_assert(tmp11.dtype == tl.float32)
                tmp19 = tl.load(in_ptr0 + (1 + x5 + 3*y3 + 9*z0), x5_mask & y3_mask & z0_mask, other=0.0)
                tl.static_assert(tmp19.dtype == tl.float32)
                tmp28 = tl.load(in_ptr0 + (3 + x5 + 3*y3 + 9*z0), x5_mask & y3_mask & z0_mask, other=0.0)
                tl.static_assert(tmp28.dtype == tl.float32)
                tmp35 = tl.load(in_ptr0 + (4 + x5 + 3*y3 + 9*z0), x5_mask & y3_mask & z0_mask, other=0.0)
                tl.static_assert(tmp35.dtype == tl.float32)
                tmp0 = y3
                tl.static_assert(tmp0.dtype == tl.int32)
                tl.static_assert(tmp0.dtype == tl.int32)
                tmp1 = y3 + ((8 + 3*y4) // 6)
                tl.static_assert(tmp1.dtype == tl.int32)
                tl.static_assert(tmp1.dtype == tl.int32)
                tmp2 = tmp0 < tmp1
                tl.static_assert(tmp2.dtype == tl.int1)
                tl.static_assert(tmp2.dtype == tl.int1)
                tmp3 = x5
                tl.static_assert(tmp3.dtype == tl.int32)
                tl.static_assert(tmp3.dtype == tl.int32)
                tmp4 = x5 + ((8 + 3*x6) // 6)
                tl.static_assert(tmp4.dtype == tl.int32)
                tl.static_assert(tmp4.dtype == tl.int32)
                tmp5 = tmp3 < tmp4
                tl.static_assert(tmp5.dtype == tl.int1)
                tl.static_assert(tmp5.dtype == tl.int1)
                tmp6 = tmp2 & tmp5
                tl.static_assert(tmp6.dtype == tl.int1)
                tl.static_assert(tmp6.dtype == tl.int1)
                tmp7 = tmp6 & x5_mask
                tl.static_assert(tmp7.dtype == tl.int1)
                tl.static_assert(tmp7.dtype == tl.int1)
                tmp8 = tmp7 & y3_mask
                tl.static_assert(tmp8.dtype == tl.int1)
                tl.static_assert(tmp8.dtype == tl.int1)
                tmp9 = tmp8 & y4_mask
                tl.static_assert(tmp9.dtype == tl.int1)
                tl.static_assert(tmp9.dtype == tl.int1)
                tmp10 = tmp9 & x6_mask
                tl.static_assert(tmp10.dtype == tl.int1)
                tl.static_assert(tmp10.dtype == tl.int1)
                tl.static_assert(tmp11.dtype == tl.float32)
                tl.static_assert(tmp0.dtype == tl.int32)
                tl.static_assert(tmp1.dtype == tl.int32)
                tl.static_assert(tmp2.dtype == tl.int1)
                tmp12 = 1 + x5
                tl.static_assert(tmp12.dtype == tl.int32)
                tl.static_assert(tmp12.dtype == tl.int32)
                tl.static_assert(tmp4.dtype == tl.int32)
                tmp13 = tmp12 < tmp4
                tl.static_assert(tmp13.dtype == tl.int1)
                tl.static_assert(tmp13.dtype == tl.int1)
                tmp14 = tmp2 & tmp13
                tl.static_assert(tmp14.dtype == tl.int1)
                tl.static_assert(tmp14.dtype == tl.int1)
                tmp15 = tmp14 & x5_mask
                tl.static_assert(tmp15.dtype == tl.int1)
                tl.static_assert(tmp15.dtype == tl.int1)
                tmp16 = tmp15 & y3_mask
                tl.static_assert(tmp16.dtype == tl.int1)
                tl.static_assert(tmp16.dtype == tl.int1)
                tmp17 = tmp16 & y4_mask
                tl.static_assert(tmp17.dtype == tl.int1)
                tl.static_assert(tmp17.dtype == tl.int1)
                tmp18 = tmp17 & x6_mask
                tl.static_assert(tmp18.dtype == tl.int1)
                tl.static_assert(tmp18.dtype == tl.int1)
                tl.static_assert(tmp19.dtype == tl.float32)
                tmp20 = tmp19 + tmp11
                tl.static_assert(tmp20.dtype == tl.float32)
                tl.static_assert(tmp20.dtype == tl.float32)
                tmp21 = 1 + y3
                tl.static_assert(tmp21.dtype == tl.int32)
                tl.static_assert(tmp21.dtype == tl.int32)
                tl.static_assert(tmp1.dtype == tl.int32)
                tmp22 = tmp21 < tmp1
                tl.static_assert(tmp22.dtype == tl.int1)
                tl.static_assert(tmp22.dtype == tl.int1)
                tl.static_assert(tmp3.dtype == tl.int32)
                tl.static_assert(tmp4.dtype == tl.int32)
                tl.static_assert(tmp5.dtype == tl.int1)
                tmp23 = tmp22 & tmp5
                tl.static_assert(tmp23.dtype == tl.int1)
                tl.static_assert(tmp23.dtype == tl.int1)
                tmp24 = tmp23 & x5_mask
                tl.static_assert(tmp24.dtype == tl.int1)
                tl.static_assert(tmp24.dtype == tl.int1)
                tmp25 = tmp24 & y3_mask
                tl.static_assert(tmp25.dtype == tl.int1)
                tl.static_assert(tmp25.dtype == tl.int1)
                tmp26 = tmp25 & y4_mask
                tl.static_assert(tmp26.dtype == tl.int1)
                tl.static_assert(tmp26.dtype == tl.int1)
                tmp27 = tmp26 & x6_mask
                tl.static_assert(tmp27.dtype == tl.int1)
                tl.static_assert(tmp27.dtype == tl.int1)
                tl.static_assert(tmp28.dtype == tl.float32)
                tmp29 = tmp28 + tmp20
                tl.static_assert(tmp29.dtype == tl.float32)
                tl.static_assert(tmp29.dtype == tl.float32)
                tl.static_assert(tmp21.dtype == tl.int32)
                tl.static_assert(tmp1.dtype == tl.int32)
                tl.static_assert(tmp22.dtype == tl.int1)
                tl.static_assert(tmp12.dtype == tl.int32)
                tl.static_assert(tmp4.dtype == tl.int32)
                tl.static_assert(tmp13.dtype == tl.int1)
                tmp30 = tmp22 & tmp13
                tl.static_assert(tmp30.dtype == tl.int1)
                tl.static_assert(tmp30.dtype == tl.int1)
                tmp31 = tmp30 & x5_mask
                tl.static_assert(tmp31.dtype == tl.int1)
                tl.static_assert(tmp31.dtype == tl.int1)
                tmp32 = tmp31 & y3_mask
                tl.static_assert(tmp32.dtype == tl.int1)
                tl.static_assert(tmp32.dtype == tl.int1)
                tmp33 = tmp32 & y4_mask
                tl.static_assert(tmp33.dtype == tl.int1)
                tl.static_assert(tmp33.dtype == tl.int1)
                tmp34 = tmp33 & x6_mask
                tl.static_assert(tmp34.dtype == tl.int1)
                tl.static_assert(tmp34.dtype == tl.int1)
                tl.static_assert(tmp35.dtype == tl.float32)
                tmp36 = tmp35 + tmp29
                tl.static_assert(tmp36.dtype == tl.float32)
                tl.static_assert(tmp36.dtype == tl.float32)
                tl.static_assert(tmp0.dtype == tl.int32)
                tl.static_assert(tmp1.dtype == tl.int32)
                tl.static_assert(tmp2.dtype == tl.int1)
                tl.static_assert(tmp3.dtype == tl.int32)
                tl.static_assert(tmp4.dtype == tl.int32)
                tl.static_assert(tmp5.dtype == tl.int1)
                tl.static_assert(tmp6.dtype == tl.int1)
                tl.static_assert(tmp7.dtype == tl.int1)
                tl.static_assert(tmp8.dtype == tl.int1)
                tl.static_assert(tmp9.dtype == tl.int1)
                tl.static_assert(tmp10.dtype == tl.int1)
                tmp37 = 1.0
                tl.static_assert(tmp37.dtype == tl.float32)
                tl.static_assert(tmp37.dtype == tl.float32)
                tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
                tl.static_assert(tmp38.dtype == tl.float32)
                tmp39 = tl.where(tmp10, tmp37, tmp38)
                tl.static_assert(tmp39.dtype == tl.float32)
                tl.static_assert(tmp39.dtype == tl.float32)
                tl.static_assert(tmp39.dtype == tl.float32)
                tl.static_assert(tmp0.dtype == tl.int32)
                tl.static_assert(tmp1.dtype == tl.int32)
                tl.static_assert(tmp2.dtype == tl.int1)
                tl.static_assert(tmp12.dtype == tl.int32)
                tl.static_assert(tmp4.dtype == tl.int32)
                tl.static_assert(tmp13.dtype == tl.int1)
                tl.static_assert(tmp14.dtype == tl.int1)
                tl.static_assert(tmp15.dtype == tl.int1)
                tl.static_assert(tmp16.dtype == tl.int1)
                tl.static_assert(tmp17.dtype == tl.int1)
                tl.static_assert(tmp18.dtype == tl.int1)
                tmp40 = 1.0
                tl.static_assert(tmp40.dtype == tl.float32)
                tl.static_assert(tmp40.dtype == tl.float32)
                tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
                tl.static_assert(tmp41.dtype == tl.float32)
                tmp42 = tl.where(tmp18, tmp40, tmp41)
                tl.static_assert(tmp42.dtype == tl.float32)
                tl.static_assert(tmp42.dtype == tl.float32)
                tl.static_assert(tmp42.dtype == tl.float32)
                tmp43 = tmp42 + tmp39
                tl.static_assert(tmp43.dtype == tl.float32)
                tl.static_assert(tmp43.dtype == tl.float32)
                tl.static_assert(tmp21.dtype == tl.int32)
                tl.static_assert(tmp1.dtype == tl.int32)
                tl.static_assert(tmp22.dtype == tl.int1)
                tl.static_assert(tmp3.dtype == tl.int32)
                tl.static_assert(tmp4.dtype == tl.int32)
                tl.static_assert(tmp5.dtype == tl.int1)
                tl.static_assert(tmp23.dtype == tl.int1)
                tl.static_assert(tmp24.dtype == tl.int1)
                tl.static_assert(tmp25.dtype == tl.int1)
                tl.static_assert(tmp26.dtype == tl.int1)
                tl.static_assert(tmp27.dtype == tl.int1)
                tmp44 = 1.0
                tl.static_assert(tmp44.dtype == tl.float32)
                tl.static_assert(tmp44.dtype == tl.float32)
                tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
                tl.static_assert(tmp45.dtype == tl.float32)
                tmp46 = tl.where(tmp27, tmp44, tmp45)
                tl.static_assert(tmp46.dtype == tl.float32)
                tl.static_assert(tmp46.dtype == tl.float32)
                tl.static_assert(tmp46.dtype == tl.float32)
                tmp47 = tmp46 + tmp43
                tl.static_assert(tmp47.dtype == tl.float32)
                tl.static_assert(tmp47.dtype == tl.float32)
                tl.static_assert(tmp21.dtype == tl.int32)
                tl.static_assert(tmp1.dtype == tl.int32)
                tl.static_assert(tmp22.dtype == tl.int1)
                tl.static_assert(tmp12.dtype == tl.int32)
                tl.static_assert(tmp4.dtype == tl.int32)
                tl.static_assert(tmp13.dtype == tl.int1)
                tl.static_assert(tmp30.dtype == tl.int1)
                tl.static_assert(tmp31.dtype == tl.int1)
                tl.static_assert(tmp32.dtype == tl.int1)
                tl.static_assert(tmp33.dtype == tl.int1)
                tl.static_assert(tmp34.dtype == tl.int1)
                tmp48 = 1.0
                tl.static_assert(tmp48.dtype == tl.float32)
                tl.static_assert(tmp48.dtype == tl.float32)
                tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
                tl.static_assert(tmp49.dtype == tl.float32)
                tmp50 = tl.where(tmp34, tmp48, tmp49)
                tl.static_assert(tmp50.dtype == tl.float32)
                tl.static_assert(tmp50.dtype == tl.float32)
                tl.static_assert(tmp50.dtype == tl.float32)
                tmp51 = tmp50 + tmp47
                tl.static_assert(tmp51.dtype == tl.float32)
                tl.static_assert(tmp51.dtype == tl.float32)
                tmp52 = (tmp36 / tmp51)
                tl.static_assert(tmp52.dtype == tl.float32)
                tl.static_assert(tmp52.dtype == tl.float32)
                tl.store(out_ptr0 + (x6 + 2*x5 + 6*y4 + 12*y3 + 36*z0), tmp52, x5_mask & x6_mask & y3_mask & y4_mask & z0_mask)
''', device_str='npu')


# kernel path: /tmp/tmps8vxzxs8/dv/cdvztojpwpp62us6vkt4vfl5axgajd7p2lj5vehi7vu4d2dzlran.py
# Topologically Sorted Source Nodes: [add, _adaptive_avg_pool2d_1], Original ATen: [aten.add, aten._adaptive_avg_pool2d]
# Source node to ATen node mapping:
#   _adaptive_avg_pool2d_1 => _adaptive_avg_pool2d_1
#   add => add
# Graph fragment:
#   %arg0_1 : Tensor "f32[2, 4, 3, 3][36, 9, 3, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %add : Tensor "f32[2, 4, 3, 3][36, 9, 3, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, 1), kwargs = {})
#   %_adaptive_avg_pool2d_1 : Tensor "f32[2, 4, 2, 5][40, 10, 5, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%add, [2, 5]), kwargs = {})
#   return %_adaptive_avg_pool2d_1
# SchedulerNodes: [SchedulerNode(name='op1')]

triton_poi_fused__adaptive_avg_pool2d_add_1 = async_compile.triton('triton_poi_fused__adaptive_avg_pool2d_add_1', '''
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
    size_hints={'z0': 8, 'y1': 2, 'x2': 5}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32', 'X2BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_add_1', 'mutated_arg_names': [], 'backend_hash': '<removed>', 'split_axis': [0, 1, 2], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['z0', 'y1', 'x2'], 'axis_static_values': (('z0', 8), ('y1', 2), ('x2', 5)), 'low_dims': {2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK', 'X2BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_add_1(in_ptr0, out_ptr0, z0_numel, y1_numel, x2_numel, Z0BLOCK, Y1BLOCK, X2BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr):
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(1) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    x2_offset = tl.program_id(2) * X2BLOCK
    base_x2= tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (X2BLOCK + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[None,:,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            for loop_x2 in range(loops_x2):
                x2 = x2_offset + (loop_x2 * X2BLOCK_SUB) + base_x2[None,None,:]
                x2_mask = x2 < min(X2BLOCK+x2_offset, x2_numel)
                tmp0 = (3*y1) // 2
                tl.static_assert(tmp0.dtype == tl.int32)
                tl.static_assert(tmp0.dtype == tl.int32)
                tmp1 = 2 + ((3*y1) // 2)
                tl.static_assert(tmp1.dtype == tl.int32)
                tl.static_assert(tmp1.dtype == tl.int32)
                tmp2 = tmp0 < tmp1
                tl.static_assert(tmp2.dtype == tl.int1)
                tl.static_assert(tmp2.dtype == tl.int1)
                tmp3 = (3*x2) // 5
                tl.static_assert(tmp3.dtype == tl.int32)
                tl.static_assert(tmp3.dtype == tl.int32)
                tmp4 = (7 + 3*x2) // 5
                tl.static_assert(tmp4.dtype == tl.int32)
                tl.static_assert(tmp4.dtype == tl.int32)
                tmp5 = tmp3 < tmp4
                tl.static_assert(tmp5.dtype == tl.int1)
                tl.static_assert(tmp5.dtype == tl.int1)
                tmp6 = tmp2 & tmp5
                tl.static_assert(tmp6.dtype == tl.int1)
                tl.static_assert(tmp6.dtype == tl.int1)
                tmp7 = tmp6 & y1_mask
                tl.static_assert(tmp7.dtype == tl.int1)
                tl.static_assert(tmp7.dtype == tl.int1)
                tmp8 = tmp7 & x2_mask
                tl.static_assert(tmp8.dtype == tl.int1)
                tl.static_assert(tmp8.dtype == tl.int1)
                tmp9 = tl.load(in_ptr0 + (3*((3*y1) // 2) + 9*z0 + ((3*x2) // 5)), tmp8 & z0_mask, other=0.0)
                tl.static_assert(tmp9.dtype == tl.float32)
                tmp10 = 1.0
                tl.static_assert(tmp10.dtype == tl.float32)
                tl.static_assert(tmp10.dtype == tl.float32)
                tmp11 = tmp9 + tmp10
                tl.static_assert(tmp11.dtype == tl.float32)
                tl.static_assert(tmp11.dtype == tl.float32)
                tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
                tl.static_assert(tmp12.dtype == tl.float32)
                tmp13 = tl.where(tmp8, tmp11, tmp12)
                tl.static_assert(tmp13.dtype == tl.float32)
                tl.static_assert(tmp13.dtype == tl.float32)
                tl.static_assert(tmp13.dtype == tl.float32)
                tl.static_assert(tmp0.dtype == tl.int32)
                tl.static_assert(tmp1.dtype == tl.int32)
                tl.static_assert(tmp2.dtype == tl.int1)
                tmp14 = 1 + ((3*x2) // 5)
                tl.static_assert(tmp14.dtype == tl.int32)
                tl.static_assert(tmp14.dtype == tl.int32)
                tl.static_assert(tmp4.dtype == tl.int32)
                tmp15 = tmp14 < tmp4
                tl.static_assert(tmp15.dtype == tl.int1)
                tl.static_assert(tmp15.dtype == tl.int1)
                tmp16 = tmp2 & tmp15
                tl.static_assert(tmp16.dtype == tl.int1)
                tl.static_assert(tmp16.dtype == tl.int1)
                tmp17 = tmp16 & y1_mask
                tl.static_assert(tmp17.dtype == tl.int1)
                tl.static_assert(tmp17.dtype == tl.int1)
                tmp18 = tmp17 & x2_mask
                tl.static_assert(tmp18.dtype == tl.int1)
                tl.static_assert(tmp18.dtype == tl.int1)
                tmp19 = tl.load(in_ptr0 + (1 + 3*((3*y1) // 2) + 9*z0 + ((3*x2) // 5)), tmp18 & z0_mask, other=0.0)
                tl.static_assert(tmp19.dtype == tl.float32)
                tmp20 = 1.0
                tl.static_assert(tmp20.dtype == tl.float32)
                tl.static_assert(tmp20.dtype == tl.float32)
                tmp21 = tmp19 + tmp20
                tl.static_assert(tmp21.dtype == tl.float32)
                tl.static_assert(tmp21.dtype == tl.float32)
                tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
                tl.static_assert(tmp22.dtype == tl.float32)
                tmp23 = tl.where(tmp18, tmp21, tmp22)
                tl.static_assert(tmp23.dtype == tl.float32)
                tl.static_assert(tmp23.dtype == tl.float32)
                tl.static_assert(tmp23.dtype == tl.float32)
                tmp24 = tmp23 + tmp13
                tl.static_assert(tmp24.dtype == tl.float32)
                tl.static_assert(tmp24.dtype == tl.float32)
                tmp25 = 1 + ((3*y1) // 2)
                tl.static_assert(tmp25.dtype == tl.int32)
                tl.static_assert(tmp25.dtype == tl.int32)
                tl.static_assert(tmp1.dtype == tl.int32)
                tmp26 = tmp25 < tmp1
                tl.static_assert(tmp26.dtype == tl.int1)
                tl.static_assert(tmp26.dtype == tl.int1)
                tl.static_assert(tmp3.dtype == tl.int32)
                tl.static_assert(tmp4.dtype == tl.int32)
                tl.static_assert(tmp5.dtype == tl.int1)
                tmp27 = tmp26 & tmp5
                tl.static_assert(tmp27.dtype == tl.int1)
                tl.static_assert(tmp27.dtype == tl.int1)
                tmp28 = tmp27 & y1_mask
                tl.static_assert(tmp28.dtype == tl.int1)
                tl.static_assert(tmp28.dtype == tl.int1)
                tmp29 = tmp28 & x2_mask
                tl.static_assert(tmp29.dtype == tl.int1)
                tl.static_assert(tmp29.dtype == tl.int1)
                tmp30 = tl.load(in_ptr0 + (3 + 3*((3*y1) // 2) + 9*z0 + ((3*x2) // 5)), tmp29 & z0_mask, other=0.0)
                tl.static_assert(tmp30.dtype == tl.float32)
                tmp31 = 1.0
                tl.static_assert(tmp31.dtype == tl.float32)
                tl.static_assert(tmp31.dtype == tl.float32)
                tmp32 = tmp30 + tmp31
                tl.static_assert(tmp32.dtype == tl.float32)
                tl.static_assert(tmp32.dtype == tl.float32)
                tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
                tl.static_assert(tmp33.dtype == tl.float32)
                tmp34 = tl.where(tmp29, tmp32, tmp33)
                tl.static_assert(tmp34.dtype == tl.float32)
                tl.static_assert(tmp34.dtype == tl.float32)
                tl.static_assert(tmp34.dtype == tl.float32)
                tmp35 = tmp34 + tmp24
                tl.static_assert(tmp35.dtype == tl.float32)
                tl.static_assert(tmp35.dtype == tl.float32)
                tl.static_assert(tmp25.dtype == tl.int32)
                tl.static_assert(tmp1.dtype == tl.int32)
                tl.static_assert(tmp26.dtype == tl.int1)
                tl.static_assert(tmp14.dtype == tl.int32)
                tl.static_assert(tmp4.dtype == tl.int32)
                tl.static_assert(tmp15.dtype == tl.int1)
                tmp36 = tmp26 & tmp15
                tl.static_assert(tmp36.dtype == tl.int1)
                tl.static_assert(tmp36.dtype == tl.int1)
                tmp37 = tmp36 & y1_mask
                tl.static_assert(tmp37.dtype == tl.int1)
                tl.static_assert(tmp37.dtype == tl.int1)
                tmp38 = tmp37 & x2_mask
                tl.static_assert(tmp38.dtype == tl.int1)
                tl.static_assert(tmp38.dtype == tl.int1)
                tmp39 = tl.load(in_ptr0 + (4 + 3*((3*y1) // 2) + 9*z0 + ((3*x2) // 5)), tmp38 & z0_mask, other=0.0)
                tl.static_assert(tmp39.dtype == tl.float32)
                tmp40 = 1.0
                tl.static_assert(tmp40.dtype == tl.float32)
                tl.static_assert(tmp40.dtype == tl.float32)
                tmp41 = tmp39 + tmp40
                tl.static_assert(tmp41.dtype == tl.float32)
                tl.static_assert(tmp41.dtype == tl.float32)
                tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
                tl.static_assert(tmp42.dtype == tl.float32)
                tmp43 = tl.where(tmp38, tmp41, tmp42)
                tl.static_assert(tmp43.dtype == tl.float32)
                tl.static_assert(tmp43.dtype == tl.float32)
                tl.static_assert(tmp43.dtype == tl.float32)
                tmp44 = tmp43 + tmp35
                tl.static_assert(tmp44.dtype == tl.float32)
                tl.static_assert(tmp44.dtype == tl.float32)
                tl.static_assert(tmp0.dtype == tl.int32)
                tl.static_assert(tmp1.dtype == tl.int32)
                tl.static_assert(tmp2.dtype == tl.int1)
                tl.static_assert(tmp3.dtype == tl.int32)
                tl.static_assert(tmp4.dtype == tl.int32)
                tl.static_assert(tmp5.dtype == tl.int1)
                tl.static_assert(tmp6.dtype == tl.int1)
                tl.static_assert(tmp7.dtype == tl.int1)
                tl.static_assert(tmp8.dtype == tl.int1)
                tl.static_assert(tmp10.dtype == tl.float32)
                tmp45 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
                tl.static_assert(tmp45.dtype == tl.float32)
                tmp46 = tl.where(tmp8, tmp10, tmp45)
                tl.static_assert(tmp46.dtype == tl.float32)
                tl.static_assert(tmp46.dtype == tl.float32)
                tl.static_assert(tmp46.dtype == tl.float32)
                tl.static_assert(tmp0.dtype == tl.int32)
                tl.static_assert(tmp1.dtype == tl.int32)
                tl.static_assert(tmp2.dtype == tl.int1)
                tl.static_assert(tmp14.dtype == tl.int32)
                tl.static_assert(tmp4.dtype == tl.int32)
                tl.static_assert(tmp15.dtype == tl.int1)
                tl.static_assert(tmp16.dtype == tl.int1)
                tl.static_assert(tmp17.dtype == tl.int1)
                tl.static_assert(tmp18.dtype == tl.int1)
                tl.static_assert(tmp20.dtype == tl.float32)
                tmp47 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
                tl.static_assert(tmp47.dtype == tl.float32)
                tmp48 = tl.where(tmp18, tmp20, tmp47)
                tl.static_assert(tmp48.dtype == tl.float32)
                tl.static_assert(tmp48.dtype == tl.float32)
                tl.static_assert(tmp48.dtype == tl.float32)
                tmp49 = tmp48 + tmp46
                tl.static_assert(tmp49.dtype == tl.float32)
                tl.static_assert(tmp49.dtype == tl.float32)
                tl.static_assert(tmp25.dtype == tl.int32)
                tl.static_assert(tmp1.dtype == tl.int32)
                tl.static_assert(tmp26.dtype == tl.int1)
                tl.static_assert(tmp3.dtype == tl.int32)
                tl.static_assert(tmp4.dtype == tl.int32)
                tl.static_assert(tmp5.dtype == tl.int1)
                tl.static_assert(tmp27.dtype == tl.int1)
                tl.static_assert(tmp28.dtype == tl.int1)
                tl.static_assert(tmp29.dtype == tl.int1)
                tl.static_assert(tmp31.dtype == tl.float32)
                tmp50 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
                tl.static_assert(tmp50.dtype == tl.float32)
                tmp51 = tl.where(tmp29, tmp31, tmp50)
                tl.static_assert(tmp51.dtype == tl.float32)
                tl.static_assert(tmp51.dtype == tl.float32)
                tl.static_assert(tmp51.dtype == tl.float32)
                tmp52 = tmp51 + tmp49
                tl.static_assert(tmp52.dtype == tl.float32)
                tl.static_assert(tmp52.dtype == tl.float32)
                tl.static_assert(tmp25.dtype == tl.int32)
                tl.static_assert(tmp1.dtype == tl.int32)
                tl.static_assert(tmp26.dtype == tl.int1)
                tl.static_assert(tmp14.dtype == tl.int32)
                tl.static_assert(tmp4.dtype == tl.int32)
                tl.static_assert(tmp15.dtype == tl.int1)
                tl.static_assert(tmp36.dtype == tl.int1)
                tl.static_assert(tmp37.dtype == tl.int1)
                tl.static_assert(tmp38.dtype == tl.int1)
                tl.static_assert(tmp40.dtype == tl.float32)
                tmp53 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
                tl.static_assert(tmp53.dtype == tl.float32)
                tmp54 = tl.where(tmp38, tmp40, tmp53)
                tl.static_assert(tmp54.dtype == tl.float32)
                tl.static_assert(tmp54.dtype == tl.float32)
                tl.static_assert(tmp54.dtype == tl.float32)
                tmp55 = tmp54 + tmp52
                tl.static_assert(tmp55.dtype == tl.float32)
                tl.static_assert(tmp55.dtype == tl.float32)
                tmp56 = (tmp44 / tmp55)
                tl.static_assert(tmp56.dtype == tl.float32)
                tl.static_assert(tmp56.dtype == tl.float32)
                tl.store(out_ptr0 + (x2 + 5*y1 + 10*z0), tmp56, x2_mask & y1_mask & z0_mask)
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
            buf0 = empty_strided((2, 4, 6, 6), (144, 36, 6, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [_adaptive_avg_pool2d], Original ATen: [aten._adaptive_avg_pool2d]
            stream0 = get_raw_stream(0)
            triton_poi_fused__adaptive_avg_pool2d_0.run(arg0_1, buf0, 8, 3, 2, 3, 2, stream=stream0)
            buf1 = empty_strided((2, 4, 2, 5), (40, 10, 5, 1), device='npu', dtype=torch.float32)
            # Topologically Sorted Source Nodes: [add, _adaptive_avg_pool2d_1], Original ATen: [aten.add, aten._adaptive_avg_pool2d]
            stream0 = get_raw_stream(0)
            triton_poi_fused__adaptive_avg_pool2d_add_1.run(arg0_1, buf1, 8, 2, 5, stream=stream0)
            del arg0_1
        return (buf0, buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 4, 3, 3), (36, 9, 3, 1), device='npu:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
