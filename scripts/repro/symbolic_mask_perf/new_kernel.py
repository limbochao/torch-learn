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

# kernel path: /opt/tiger/dump_backup_clear_hourly/aml.inference.test/qianchuan_torch_v4143_mlu_r19150212_0/Float/worker_0/inductor/e3/ce35zbvcz3vxudnoonlyagxjxzx3n4mt2vpfinpvvzbg24npq5of.py
# Topologically Sorted Source Nodes: [merged_17, split_15, adapt1_hidden_5], Original ATen: [aten.add, aten.split_with_sizes, aten.gelu]
# Source node to ATen node mapping:
#   adapt1_hidden_5 => add_14764, mul_9678, mul_9679, mul_9680, mul_9681, mul_9682, sigmoid_18
#   merged_17 => add_14751
#   split_15 => split_with_sizes_15
# Graph fragment:
#   %bmm_26 : Tensor "f16[32, s0, 1160][1160*s0, 1160, 1]npu:0" = PlaceHolder[target=bmm_26]
#   %arg1288_1 : Tensor "f16[32, 1, 1160][1160, 1160, 1]npu:0" = PlaceHolder[target=arg1288_1]
#   %add_14751 : Tensor "f16[32, s0, 1160][1160*s0, 1160, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%bmm_26, %arg1288_1), kwargs = {})
#   %split_with_sizes_15 : [num_users=2] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%add_14751, [200, 960], -1), kwargs = {})
#   %mul_9678 : Tensor "f16[32, s0, 200][200*s0, 200, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_470, %getitem_470), kwargs = {})
#   %mul_9679 : Tensor "f16[32, s0, 200][200*s0, 200, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9678, %getitem_470), kwargs = {})
#   %mul_9680 : Tensor "f16[32, s0, 200][200*s0, 200, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9679, 0.044715), kwargs = {})
#   %add_14764 : Tensor "f16[32, s0, 200][200*s0, 200, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_470, %mul_9680), kwargs = {})
#   %mul_9681 : Tensor "f16[32, s0, 200][200*s0, 200, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_14764, 1.5957691216057308), kwargs = {})
#   %sigmoid_18 : Tensor "f16[32, s0, 200][200*s0, 200, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mul_9681,), kwargs = {})
#   %mul_9682 : Tensor "f16[32, s0, 200][200*s0, 200, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_470, %sigmoid_18), kwargs = {})
#   return %mul_9682
# SchedulerNodes: [SchedulerNode(name='op2427')]

triton_poi_fused_add_gelu_split_with_sizes_229 = async_compile.triton('triton_poi_fused_add_gelu_split_with_sizes_229', '''
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
    size_hints={'z0': 32, 'y1': 256, 'x2': 200}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'ks0': 'i64', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32', 'Z0BLOCK': 'i32', 'Y1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_split_with_sizes_229', 'mutated_arg_names': [], 'backend_hash': '34EA3B6D85563EC0F53332A8219440D39CEAE6547484345EEAC9502540493460', 'split_axis': [0, 1], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [2], 'axis_names': ['z0', 'y1', 'x2'], 'axis_static_values': (('z0', 32), ('x2', 200)), 'low_dims': {2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float16, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('Z0BLOCK', 'Y1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_gelu_split_with_sizes_229(in_ptr0, in_ptr1, out_ptr0, ks0, z0_numel, y1_numel, x2_numel, Z0BLOCK, Y1BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr):
    x2_numel = 200
    X2BLOCK_SUB: tl.constexpr = 200
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    y1_offset = tl.program_id(1) * Y1BLOCK
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[None,:,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            x2 = base_x2[None,None,:]
            tmp0 = tl.load(in_ptr0 + (x2 + 1160*y1 + 1160*ks0*z0), y1_mask & z0_mask).to(tl.float32)
            tmp1 = tl.load(in_ptr1 + (x2 + 1160*z0), z0_mask).to(tl.float32)
            tmp2 = tmp0 + tmp1
            tmp3 = tmp2 * tmp2
            tmp4 = tmp3 * tmp2
            tmp5 = 0.044715
            tmp6 = tmp4 * tmp5
            tmp7 = tmp2 + tmp6
            tmp8 = 1.5957691216057308
            tmp9 = tmp7 * tmp8
            tmp10 = tl.sigmoid(tmp9)
            tmp11 = tmp2 * tmp10
            tl.store(out_ptr0 + (x2 + 200*y1 + 200*ks0*z0), tmp11, y1_mask & z0_mask)
''', device_str='npu')

arg125_1 = 256
s0 = arg125_1
arg1288_1 = empty_strided((32, 1, 1160), (1160, 1160, 1), device='npu:0', dtype=torch.float16)
buf2440 = empty_strided((32, s0, 1160), (1160*s0, 1160, 1), device='npu', dtype=torch.float16)
buf2441 = empty_strided((32, s0, 200), (200*s0, 200, 1), device='npu', dtype=torch.float16)
stream0 = get_raw_stream(0)
triton_poi_fused_add_gelu_split_with_sizes_229.run(buf2440, arg1288_1, buf2441, s0, 32, s0, 200, stream=stream0)