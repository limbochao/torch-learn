# AOT ID: ['1_backward']
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


# kernel path: <cache>/generated_kernel.py
# Topologically Sorted Source Nodes: [full, _generalized_scatter, sub, mul_24, mul_30, sum_3, sum_4], Original ATen: [aten.select_backward, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   _generalized_scatter => select_scatter_default
#   full => full_default
#   mul_24 => mul_24
#   mul_30 => mul_30
#   sub => sub
#   sum_3 => sum_3
#   sum_4 => sum_4
# Graph fragment:
#   %tangents_1 : Tensor "f32[16, 768][768, 1]npu:0" = PlaceHolder[target=tangents_1]
#   %add_24 : Tensor "f32[16, 1370, 768][1052160, 768, 1]npu:0" = PlaceHolder[target=add_24]
#   %getitem_193 : Tensor "f32[16, 1370, 1][1370, 1, 1]npu:0" = PlaceHolder[target=getitem_193]
#   %getitem_194 : Tensor "f32[16, 1370, 1][1370, 1, 1]npu:0" = PlaceHolder[target=getitem_194]
#   %full_default : Tensor "f32[16, 1370, 768][1052160, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([16, 1370, 768], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %select_scatter_default : Tensor "f32[16, 1370, 768][1052160, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %tangents_1, 1, 0), kwargs = {})
#   %sub : Tensor "f32[16, 1370, 768][1052160, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_24, %getitem_193), kwargs = {})
#   %mul_24 : Tensor "f32[16, 1370, 768][1052160, 768, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %getitem_194), kwargs = {})
#   %mul_30 : Tensor "f32[16, 1370, 768][1052160, 768, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_scatter_default, %mul_24), kwargs = {})
#   %sum_3 : Tensor "f32[768][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_30, [0, 1]), kwargs = {})
#   %sum_4 : Tensor "f32[768][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%select_scatter_default, [0, 1]), kwargs = {})
#   return %sum_3,%sum_4
triton_red_fused_native_layer_norm_backward_select_backward_0 = async_compile.triton('triton_red_fused_native_layer_norm_backward_select_backward_0', '''
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
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 32768},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_native_layer_norm_backward_select_backward_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'backend_hash': '<omitted>', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True, 'requires_no_linear_block_remap': True, 'npu_kernel_type': 'simd_simt_mix', 'split_axis': [0], 'tiling_axis': [0, 1], 'low_dims': [1], 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'axis_names': ['x', 'r0_']},
)
@triton.jit
def triton_red_fused_native_layer_norm_backward_select_backward_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 768
    r0_numel = 21920
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp16 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = (r0_index % 1370)
        r0_2 = r0_index // 1370
        r0_3 = r0_index
        tmp3 = tl.load(in_ptr0 + (x0 + 768*r0_2), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr1 + (x0 + 768*r0_3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r0_3), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr3 + (r0_3), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp0 = (r0_1).to(tl.int32)
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = tl.full([1, 1], 0.0, tl.float32)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp8 = tmp6 - tmp7
        tmp10 = tmp8 * tmp9
        tmp11 = tmp5 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask & xmask, tmp14, _tmp13)
        tmp15 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(r0_mask & xmask, tmp17, _tmp16)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''', device_str='npu')


async_compile.wait(globals())
del async_compile


torch.npu.set_device(0)
tangents_1 = copy_if_misaligned(tangents_1)
buf3 = empty_strided_npu((768, ), (1, ), torch.float32)
buf4 = empty_strided_npu((768, ), (1, ), torch.float32)
raw_stream0 = get_raw_stream(0)
triton_red_fused_native_layer_norm_backward_select_backward_0.run(tangents_1, add_24, getitem_193, getitem_194, buf3, buf4, 768, 21920, stream=raw_stream0)
