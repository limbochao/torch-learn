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
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

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


# kernel path: /mnt/raid1/gaojg/data/dev/cooperative-strategy-20261123/cache/lb/clbqudtx6chcpbcd2vn5zgxipqnsv3fe2azdhb2zpmazuyfz5jex.py
# Topologically Sorted Source Nodes: [z], Original ATen: [aten.add, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   z => add
# Graph fragment:
#   %arg0_1 : Tensor "f32[1, 1048576][1048576, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[1, 1048576][1048576, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %add : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %prepare_softmax_online_default_3 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add, -1), kwargs = {})
#   return %buf1,%buf2
triton_red_fused_add_prepare_softmax_online_0 = async_compile.triton('triton_red_fused_add_prepare_softmax_online_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_add_prepare_softmax_online_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 2048, 'r0_': 8388608}, 'backend_hash': 'B2C00AD225F1945839503C666727F7E33E52219DAB5FB3E9C1893C177F82CD58', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_add_prepare_softmax_online_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp6_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1, 1], 1.0, tl.float32)
        tmp4 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])

        _tmp6_max_next, _tmp6_sum_next = triton_helpers.online_softmax_combine_with_sum(
            _tmp6_max, _tmp6_sum, tmp4, tmp5, False
        )

        _tmp6_max = tl.where(r0_mask & xmask, _tmp6_max_next, _tmp6_max)
        _tmp6_sum = tl.where(r0_mask & xmask, _tmp6_sum_next, _tmp6_sum)

    tmp6, tmp7 = triton_helpers.online_softmax_reduce(
        _tmp6_max, _tmp6_sum, 1, False)
    tmp6 = tmp6[:, None]
    tmp7 = tmp7[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: /mnt/raid1/gaojg/data/dev/cooperative-strategy-20261123/cache/ot/cotbhkrkknkguo4doyto6jjdupv363bxf2t4xh2nap4zqbqcfu3y.py
# Topologically Sorted Source Nodes: [z], Original ATen: [aten.add, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   z => add
# Graph fragment:
#   %buf1 : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0" = PlaceHolder[target=buf1]
#   %buf2 : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0" = PlaceHolder[target=buf2]
#   %add : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %prepare_softmax_online_default_3 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add, -1), kwargs = {})
#   return %getitem_6,%getitem_7
triton_per_fused_add_prepare_softmax_online_1 = async_compile.triton('triton_per_fused_add_prepare_softmax_online_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_add_prepare_softmax_online_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'r0_': 1024}, 'backend_hash': 'B2C00AD225F1945839503C666727F7E33E52219DAB5FB3E9C1893C177F82CD58', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused_add_prepare_softmax_online_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr1 + (r0_0), None, eviction_policy='evict_first')
    tmp2 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])

    tmp4, tmp5 = triton_helpers.online_softmax_reduce(
        tmp2, tmp3, 1, False)
    tmp4 = tmp4[:, None]
    tmp5 = tmp5[:, None]
    tl.store(out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp4, None)
    tl.store(out_ptr1 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp5, None)
''', device_str='cuda')


# kernel path: /mnt/raid1/gaojg/data/dev/cooperative-strategy-20261123/cache/lx/clx5pehqavcv62mtg46aj3bqr6r5fd6eupuiuvy5zzxazncz5dje.py
# Topologically Sorted Source Nodes: [z, softmax, z_1], Original ATen: [aten.add, aten._softmax, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   softmax => div, exp_default_3, sub_tensor_3
#   z => add
#   z_1 => add_1
# Graph fragment:
#   %arg0_1 : Tensor "f32[1, 1048576][1048576, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[1, 1048576][1048576, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %add : Tensor "f32[1, 1048576][1048576, 1]cuda:0" = PlaceHolder[target=add]
#   %getitem_6 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem_6]
#   %getitem_7 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem_7]
#   %add : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %sub_tensor_3 : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_6), kwargs = {})
#   %exp_default_3 : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_3,), kwargs = {})
#   %div : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_3, %getitem_7), kwargs = {})
#   %add_1 : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %div), kwargs = {})
#   %prepare_softmax_online_default_2 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_1, -1), kwargs = {})
#   return %add,%add_1,%buf6,%buf7
triton_red_fused__softmax_add_prepare_softmax_online_2 = async_compile.triton('triton_red_fused__softmax_add_prepare_softmax_online_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__softmax_add_prepare_softmax_online_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 3, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 2048, 'r0_': 16777216}, 'backend_hash': 'B2C00AD225F1945839503C666727F7E33E52219DAB5FB3E9C1893C177F82CD58', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__softmax_add_prepare_softmax_online_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp3 = tl.load(in_ptr2 + (0))
    tmp4 = tl.broadcast_to(tmp3, [1, 1])
    tmp7 = tl.load(in_ptr3 + (0))
    tmp8 = tl.broadcast_to(tmp7, [1, 1])
    _tmp14_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp14_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp2 - tmp4
        tmp6 = libdevice.exp(tmp5)
        tmp9 = (tmp6 / tmp8)
        tmp10 = tmp2 + tmp9
        tmp11 = tl.full([1, 1], 1.0, tl.float32)
        tmp12 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])

        _tmp14_max_next, _tmp14_sum_next = triton_helpers.online_softmax_combine_with_sum(
            _tmp14_max, _tmp14_sum, tmp12, tmp13, False
        )

        _tmp14_max = tl.where(r0_mask & xmask, _tmp14_max_next, _tmp14_max)
        _tmp14_sum = tl.where(r0_mask & xmask, _tmp14_sum_next, _tmp14_sum)
        tl.store(out_ptr1 + (r0_1 + 8192*x0), tmp10, r0_mask & xmask)

    tmp14, tmp15 = triton_helpers.online_softmax_reduce(
        _tmp14_max, _tmp14_sum, 1, False)
    tmp14 = tmp14[:, None]
    tmp15 = tmp15[:, None]
    tl.store(out_ptr2 + (x0), tmp14, xmask)
    tl.store(out_ptr3 + (x0), tmp15, xmask)
''', device_str='cuda')


# kernel path: /mnt/raid1/gaojg/data/dev/cooperative-strategy-20261123/cache/yv/cyvtbmispdb6bb3icx74irw2deo4xuhkyjhlq5ybn6rwjsp7fs6d.py
# Topologically Sorted Source Nodes: [softmax_1, z_2], Original ATen: [aten._softmax, aten.add, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   softmax_1 => div_1, exp_default_2, sub_tensor_2
#   z_2 => add_2
# Graph fragment:
#   %add_1 : Tensor "f32[1, 1048576][1048576, 1]cuda:0" = PlaceHolder[target=add_1]
#   %getitem_4 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem_4]
#   %getitem_5 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem_5]
#   %sub_tensor_2 : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_4), kwargs = {})
#   %exp_default_2 : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_2,), kwargs = {})
#   %div_1 : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_2, %getitem_5), kwargs = {})
#   %add_2 : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %div_1), kwargs = {})
#   %prepare_softmax_online_default_1 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_2, -1), kwargs = {})
#   return %buf11,%buf12
triton_red_fused__softmax_add_prepare_softmax_online_3 = async_compile.triton('triton_red_fused__softmax_add_prepare_softmax_online_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__softmax_add_prepare_softmax_online_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 2048, 'r0_': 4194304}, 'backend_hash': 'B2C00AD225F1945839503C666727F7E33E52219DAB5FB3E9C1893C177F82CD58', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__softmax_add_prepare_softmax_online_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [1, 1])
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [1, 1])
    _tmp12_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp12_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp0 - tmp2
        tmp4 = libdevice.exp(tmp3)
        tmp7 = (tmp4 / tmp6)
        tmp8 = tmp0 + tmp7
        tmp9 = tl.full([1, 1], 1.0, tl.float32)
        tmp10 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])

        _tmp12_max_next, _tmp12_sum_next = triton_helpers.online_softmax_combine_with_sum(
            _tmp12_max, _tmp12_sum, tmp10, tmp11, False
        )

        _tmp12_max = tl.where(r0_mask & xmask, _tmp12_max_next, _tmp12_max)
        _tmp12_sum = tl.where(r0_mask & xmask, _tmp12_sum_next, _tmp12_sum)

    tmp12, tmp13 = triton_helpers.online_softmax_reduce(
        _tmp12_max, _tmp12_sum, 1, False)
    tmp12 = tmp12[:, None]
    tmp13 = tmp13[:, None]
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tl.store(out_ptr1 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: /mnt/raid1/gaojg/data/dev/cooperative-strategy-20261123/cache/ls/clsnjvtg4uwc7o6zywfbrmwpppagwvcbj6g2tryid2sx62zq3qpo.py
# Topologically Sorted Source Nodes: [softmax_1, z_2, softmax_2, z_3], Original ATen: [aten._softmax, aten.add, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   softmax_1 => div_1, exp_default_2, sub_tensor_2
#   softmax_2 => div_2, exp_default_1, sub_tensor_1
#   z_2 => add_2
#   z_3 => add_3
# Graph fragment:
#   %add_1 : Tensor "f32[1, 1048576][1048576, 1]cuda:0" = PlaceHolder[target=add_1]
#   %getitem_4 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem_4]
#   %getitem_5 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem_5]
#   %add_2 : Tensor "f32[1, 1048576][1048576, 1]cuda:0" = PlaceHolder[target=add_2]
#   %getitem_2 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem_2]
#   %getitem_3 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem_3]
#   %sub_tensor_2 : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_4), kwargs = {})
#   %exp_default_2 : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_2,), kwargs = {})
#   %div_1 : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_2, %getitem_5), kwargs = {})
#   %add_2 : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %div_1), kwargs = {})
#   %sub_tensor_1 : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %getitem_2), kwargs = {})
#   %exp_default_1 : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_1,), kwargs = {})
#   %div_2 : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_1, %getitem_3), kwargs = {})
#   %add_3 : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %div_2), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_3, -1), kwargs = {})
#   return %add_2,%add_3,%buf16,%buf17
triton_red_fused__softmax_add_prepare_softmax_online_4 = async_compile.triton('triton_red_fused__softmax_add_prepare_softmax_online_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__softmax_add_prepare_softmax_online_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 2048, 'r0_': 12582912}, 'backend_hash': 'B2C00AD225F1945839503C666727F7E33E52219DAB5FB3E9C1893C177F82CD58', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__softmax_add_prepare_softmax_online_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [1, 1])
    tmp5 = tl.load(in_ptr1 + (0))
    tmp6 = tl.broadcast_to(tmp5, [1, 1])
    tmp9 = tl.load(in_ptr2 + (0))
    tmp10 = tl.broadcast_to(tmp9, [1, 1])
    tmp13 = tl.load(in_ptr3 + (0))
    tmp14 = tl.broadcast_to(tmp13, [1, 1])
    _tmp20_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp20_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp0 - tmp2
        tmp4 = libdevice.exp(tmp3)
        tmp7 = (tmp4 / tmp6)
        tmp8 = tmp0 + tmp7
        tmp11 = tmp8 - tmp10
        tmp12 = libdevice.exp(tmp11)
        tmp15 = (tmp12 / tmp14)
        tmp16 = tmp8 + tmp15
        tmp17 = tl.full([1, 1], 1.0, tl.float32)
        tmp18 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])

        _tmp20_max_next, _tmp20_sum_next = triton_helpers.online_softmax_combine_with_sum(
            _tmp20_max, _tmp20_sum, tmp18, tmp19, False
        )

        _tmp20_max = tl.where(r0_mask & xmask, _tmp20_max_next, _tmp20_max)
        _tmp20_sum = tl.where(r0_mask & xmask, _tmp20_sum_next, _tmp20_sum)
        tl.store(out_ptr0 + (r0_1 + 8192*x0), tmp16, r0_mask & xmask)

    tmp20, tmp21 = triton_helpers.online_softmax_reduce(
        _tmp20_max, _tmp20_sum, 1, False)
    tmp20 = tmp20[:, None]
    tmp21 = tmp21[:, None]
    tl.store(out_ptr1 + (x0), tmp20, xmask)
    tl.store(out_ptr2 + (x0), tmp21, xmask)
''', device_str='cuda')


# kernel path: /mnt/raid1/gaojg/data/dev/cooperative-strategy-20261123/cache/nh/cnhuwowjq3bpyamkrfom6s6haydgojn6hgcqfz764xccbswhek6p.py
# Topologically Sorted Source Nodes: [softmax_3, z_4], Original ATen: [aten._softmax, aten.add]
# Source node to ATen node mapping:
#   softmax_3 => div_3, exp_default, sub_tensor
#   z_4 => add_4
# Graph fragment:
#   %add_3 : Tensor "f32[1, 1048576][1048576, 1]cuda:0" = PlaceHolder[target=add_3]
#   %getitem : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem]
#   %getitem_1 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem_1]
#   %sub_tensor : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem), kwargs = {})
#   %exp_default : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div_3 : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_1), kwargs = {})
#   %add_4 : Tensor "f32[1, 1048576][1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %div_3), kwargs = {})
#   return %add_4
triton_poi_fused__softmax_add_5 = async_compile.triton('triton_poi_fused__softmax_add_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__softmax_add_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 12582912}, 'backend_hash': 'B2C00AD225F1945839503C666727F7E33E52219DAB5FB3E9C1893C177F82CD58', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_add_5(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp5 = tl.load(in_ptr1 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp3 = tmp0 - tmp2
    tmp4 = libdevice.exp(tmp3)
    tmp7 = (tmp4 / tmp6)
    tmp8 = tmp0 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


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
        arg0_1, arg1_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 1048576), (1048576, 1), 'input')
        assert_size_stride(arg1_1, (1, 1048576), (1048576, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg1_1 = copy_if_misaligned(arg1_1)
            buf1 = empty_strided_cuda((1, 1, 128), (128, 128, 1), torch.float32)
            buf2 = empty_strided_cuda((1, 1, 128), (128, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [z], Original ATen: [aten.add, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_add_prepare_softmax_online_0.run(arg0_1, arg1_1, buf1, buf2, 128, 8192, stream=raw_stream0)
            buf3 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
            buf4 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [z], Original ATen: [aten.add, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_add_prepare_softmax_online_1.run(buf1, buf2, buf3, buf4, 1, 128, stream=raw_stream0)
            buf5 = empty_strided_cuda((1, 1048576), (1048576, 1), torch.float32)
            buf6 = buf2; del buf2  # reuse
            buf7 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [z, softmax, z_1], Original ATen: [aten.add, aten._softmax, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__softmax_add_prepare_softmax_online_2.run(arg0_1, arg1_1, buf3, buf4, buf5, buf6, buf7, 128, 8192, stream=raw_stream0)
            del arg0_1
            del arg1_1
            buf8 = buf4; del buf4  # reuse
            buf9 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [softmax, z_1], Original ATen: [aten._softmax, aten.add, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_add_prepare_softmax_online_1.run(buf6, buf7, buf8, buf9, 1, 128, stream=raw_stream0)
            buf11 = buf7; del buf7  # reuse
            buf12 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [softmax_1, z_2], Original ATen: [aten._softmax, aten.add, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__softmax_add_prepare_softmax_online_3.run(buf5, buf8, buf9, buf11, buf12, 128, 8192, stream=raw_stream0)
            buf13 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
            buf14 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [softmax_1, z_2], Original ATen: [aten._softmax, aten.add, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_add_prepare_softmax_online_1.run(buf11, buf12, buf13, buf14, 1, 128, stream=raw_stream0)
            buf10 = buf5; del buf5  # reuse
            buf15 = empty_strided_cuda((1, 1048576), (1048576, 1), torch.float32)
            buf16 = buf12; del buf12  # reuse
            buf17 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [softmax_1, z_2, softmax_2, z_3], Original ATen: [aten._softmax, aten.add, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__softmax_add_prepare_softmax_online_4.run(buf10, buf8, buf9, buf13, buf14, buf15, buf16, buf17, 128, 8192, stream=raw_stream0)
            del buf10
            del buf13
            del buf14
            buf18 = buf9; del buf9  # reuse
            buf19 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [softmax_2, z_3], Original ATen: [aten._softmax, aten.add, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_add_prepare_softmax_online_1.run(buf16, buf17, buf18, buf19, 1, 128, stream=raw_stream0)
            del buf16
            del buf17
            buf20 = buf15; del buf15  # reuse
            # Topologically Sorted Source Nodes: [softmax_3, z_4], Original ATen: [aten._softmax, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_add_5.run(buf20, buf18, buf19, 1048576, stream=raw_stream0)
            del buf18
            del buf19
        return (buf20, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 1048576), (1048576, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 1048576), (1048576, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
