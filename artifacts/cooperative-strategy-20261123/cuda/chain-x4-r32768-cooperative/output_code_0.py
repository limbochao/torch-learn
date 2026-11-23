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


# kernel path: /mnt/raid1/gaojg/data/dev/cooperative-strategy-20261123/cache/qq/cqqjkasndwwg6kzhmbs2ytmrf4xfygikpq6vhzem3b5jiilykdxn.py
# Topologically Sorted Source Nodes: [z], Original ATen: [aten.add, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   z => add
# Graph fragment:
#   %arg0_1 : Tensor "f32[4, 32768][32768, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[4, 32768][32768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %add : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %prepare_softmax_online_default_3 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add, -1), kwargs = {})
#   return %buf0,%buf1
triton_red_fused_add_prepare_softmax_online_0 = async_compile.triton('triton_red_fused_add_prepare_softmax_online_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_add_prepare_softmax_online_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 256, 'r0_': 1048576}, 'backend_hash': 'B2C00AD225F1945839503C666727F7E33E52219DAB5FB3E9C1893C177F82CD58', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_add_prepare_softmax_online_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16
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


# kernel path: /mnt/raid1/gaojg/data/dev/cooperative-strategy-20261123/cache/tn/ctndygj3lr7jogqi6pujggkvg26nfj2meufr2bsxmc3d2kp36o22.py
# Topologically Sorted Source Nodes: [z], Original ATen: [aten.add, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   z => add
# Graph fragment:
#   %buf0 : Tensor "f32[4, 1, 4][4, 16, 1]cuda:0" = PlaceHolder[target=buf0]
#   %buf1 : Tensor "f32[4, 1, 4][4, 16, 1]cuda:0" = PlaceHolder[target=buf1]
#   %add : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
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
    size_hints={'x': 4, 'r0_': 4},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_add_prepare_softmax_online_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 32, 'r0_': 64}, 'backend_hash': 'B2C00AD225F1945839503C666727F7E33E52219DAB5FB3E9C1893C177F82CD58', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused_add_prepare_softmax_online_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4
    r0_numel = 4
    R0_BLOCK: tl.constexpr = 4
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 4*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 4*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
    tmp2 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(r0_mask & xmask, tmp2, float("-inf"))
    tmp6 = tl.where(r0_mask & xmask, tmp3, 0)

    tmp4, tmp7 = triton_helpers.online_softmax_reduce(
        tmp5, tmp6, 1, False)
    tmp4 = tmp4[:, None]
    tmp7 = tmp7[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: /mnt/raid1/gaojg/data/dev/cooperative-strategy-20261123/cache/h6/ch6zxfi5uvph2sbe45duabsnerax4hftpc3kvkchif7em3kea4he.py
# Topologically Sorted Source Nodes: [z, softmax, z_1], Original ATen: [aten.add, aten._softmax, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   softmax => div, exp_default_3, sub_tensor_3
#   z => add
#   z_1 => add_1
# Graph fragment:
#   %arg0_1 : Tensor "f32[4, 32768][32768, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[4, 32768][32768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %getitem_6 : Tensor "f32[4, 1][1, 4]cuda:0" = PlaceHolder[target=getitem_6]
#   %getitem_7 : Tensor "f32[4, 1][1, 4]cuda:0" = PlaceHolder[target=getitem_7]
#   %add : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %sub_tensor_3 : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_6), kwargs = {})
#   %exp_default_3 : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_3,), kwargs = {})
#   %div : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_3, %getitem_7), kwargs = {})
#   %add_1 : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %div), kwargs = {})
#   %prepare_softmax_online_default_2 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_1, -1), kwargs = {})
#   return %buf4,%buf5
triton_red_fused__softmax_add_prepare_softmax_online_2 = async_compile.triton('triton_red_fused__softmax_add_prepare_softmax_online_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__softmax_add_prepare_softmax_online_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 256, 'r0_': 1048576}, 'backend_hash': 'B2C00AD225F1945839503C666727F7E33E52219DAB5FB3E9C1893C177F82CD58', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__softmax_add_prepare_softmax_online_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x1 = xindex // 4
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    _tmp12_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp12_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 8192*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 8192*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp5 = libdevice.exp(tmp4)
        tmp7 = (tmp5 / tmp6)
        tmp8 = tmp2 + tmp7
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
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''', device_str='cuda')


# kernel path: /mnt/raid1/gaojg/data/dev/cooperative-strategy-20261123/cache/o2/co2hlapxntrm6fwzm54mj3mqliisoamzd7r3zspn6dncna2bhifa.py
# Topologically Sorted Source Nodes: [z, softmax, z_1, softmax_1, z_2], Original ATen: [aten.add, aten._softmax, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   softmax => div, exp_default_3, sub_tensor_3
#   softmax_1 => div_1, exp_default_2, sub_tensor_2
#   z => add
#   z_1 => add_1
#   z_2 => add_2
# Graph fragment:
#   %arg0_1 : Tensor "f32[4, 32768][32768, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[4, 32768][32768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %getitem_6 : Tensor "f32[4, 1][1, 4]cuda:0" = PlaceHolder[target=getitem_6]
#   %getitem_7 : Tensor "f32[4, 1][1, 4]cuda:0" = PlaceHolder[target=getitem_7]
#   %getitem_4 : Tensor "f32[4, 1][1, 4]cuda:0" = PlaceHolder[target=getitem_4]
#   %getitem_5 : Tensor "f32[4, 1][1, 4]cuda:0" = PlaceHolder[target=getitem_5]
#   %add_2 : Tensor "f32[4, 32768][32768, 1]cuda:0" = PlaceHolder[target=add_2]
#   %add : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %sub_tensor_3 : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_6), kwargs = {})
#   %exp_default_3 : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_3,), kwargs = {})
#   %div : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_3, %getitem_7), kwargs = {})
#   %add_1 : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %div), kwargs = {})
#   %sub_tensor_2 : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_4), kwargs = {})
#   %exp_default_2 : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_2,), kwargs = {})
#   %div_1 : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_2, %getitem_5), kwargs = {})
#   %add_2 : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %div_1), kwargs = {})
#   %prepare_softmax_online_default_1 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_2, -1), kwargs = {})
#   return %add_2,%buf9,%buf10
triton_red_fused__softmax_add_prepare_softmax_online_3 = async_compile.triton('triton_red_fused__softmax_add_prepare_softmax_online_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__softmax_add_prepare_softmax_online_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 3, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 256, 'r0_': 2621440}, 'backend_hash': 'B2C00AD225F1945839503C666727F7E33E52219DAB5FB3E9C1893C177F82CD58', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__softmax_add_prepare_softmax_online_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x1 = xindex // 4
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    _tmp18_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp18_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 8192*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 8192*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp5 = libdevice.exp(tmp4)
        tmp7 = (tmp5 / tmp6)
        tmp8 = tmp2 + tmp7
        tmp10 = tmp8 - tmp9
        tmp11 = libdevice.exp(tmp10)
        tmp13 = (tmp11 / tmp12)
        tmp14 = tmp8 + tmp13
        tmp15 = tl.full([1, 1], 1.0, tl.float32)
        tmp16 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp17 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])

        _tmp18_max_next, _tmp18_sum_next = triton_helpers.online_softmax_combine_with_sum(
            _tmp18_max, _tmp18_sum, tmp16, tmp17, False
        )

        _tmp18_max = tl.where(r0_mask & xmask, _tmp18_max_next, _tmp18_max)
        _tmp18_sum = tl.where(r0_mask & xmask, _tmp18_sum_next, _tmp18_sum)
        tl.store(out_ptr0 + (r0_2 + 8192*x3), tmp14, r0_mask & xmask)

    tmp18, tmp19 = triton_helpers.online_softmax_reduce(
        _tmp18_max, _tmp18_sum, 1, False)
    tmp18 = tmp18[:, None]
    tmp19 = tmp19[:, None]
    tl.store(out_ptr1 + (x3), tmp18, xmask)
    tl.store(out_ptr2 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: /mnt/raid1/gaojg/data/dev/cooperative-strategy-20261123/cache/2i/c2it75hgvgldaekrmxzfnzdncl5kj5m2qw5q4lqrsgl6muur4o6s.py
# Topologically Sorted Source Nodes: [softmax_2, z_3], Original ATen: [aten._softmax, aten.add, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   softmax_2 => div_2, exp_default_1, sub_tensor_1
#   z_3 => add_3
# Graph fragment:
#   %add_2 : Tensor "f32[4, 32768][32768, 1]cuda:0" = PlaceHolder[target=add_2]
#   %getitem_2 : Tensor "f32[4, 1][1, 4]cuda:0" = PlaceHolder[target=getitem_2]
#   %getitem_3 : Tensor "f32[4, 1][1, 4]cuda:0" = PlaceHolder[target=getitem_3]
#   %sub_tensor_1 : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %getitem_2), kwargs = {})
#   %exp_default_1 : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_1,), kwargs = {})
#   %div_2 : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_1, %getitem_3), kwargs = {})
#   %add_3 : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %div_2), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_3, -1), kwargs = {})
#   return %buf13,%buf14
triton_red_fused__softmax_add_prepare_softmax_online_4 = async_compile.triton('triton_red_fused__softmax_add_prepare_softmax_online_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__softmax_add_prepare_softmax_online_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 256, 'r0_': 524288}, 'backend_hash': 'B2C00AD225F1945839503C666727F7E33E52219DAB5FB3E9C1893C177F82CD58', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__softmax_add_prepare_softmax_online_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x1 = xindex // 4
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp10_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp10_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 8192*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp3 = libdevice.exp(tmp2)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tmp0 + tmp5
        tmp7 = tl.full([1, 1], 1.0, tl.float32)
        tmp8 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])

        _tmp10_max_next, _tmp10_sum_next = triton_helpers.online_softmax_combine_with_sum(
            _tmp10_max, _tmp10_sum, tmp8, tmp9, False
        )

        _tmp10_max = tl.where(r0_mask & xmask, _tmp10_max_next, _tmp10_max)
        _tmp10_sum = tl.where(r0_mask & xmask, _tmp10_sum_next, _tmp10_sum)

    tmp10, tmp11 = triton_helpers.online_softmax_reduce(
        _tmp10_max, _tmp10_sum, 1, False)
    tmp10 = tmp10[:, None]
    tmp11 = tmp11[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: /mnt/raid1/gaojg/data/dev/cooperative-strategy-20261123/cache/2d/c2doqxudo7x3z4k3mwllxfy3mipnqvurmyvxvryitrshn7jtfniu.py
# Topologically Sorted Source Nodes: [softmax_2, z_3, softmax_3, z_4], Original ATen: [aten._softmax, aten.add]
# Source node to ATen node mapping:
#   softmax_2 => div_2, exp_default_1, sub_tensor_1
#   softmax_3 => div_3, exp_default, sub_tensor
#   z_3 => add_3
#   z_4 => add_4
# Graph fragment:
#   %add_2 : Tensor "f32[4, 32768][32768, 1]cuda:0" = PlaceHolder[target=add_2]
#   %getitem_2 : Tensor "f32[4, 1][1, 4]cuda:0" = PlaceHolder[target=getitem_2]
#   %getitem_3 : Tensor "f32[4, 1][1, 4]cuda:0" = PlaceHolder[target=getitem_3]
#   %getitem : Tensor "f32[4, 1][1, 4]cuda:0" = PlaceHolder[target=getitem]
#   %getitem_1 : Tensor "f32[4, 1][1, 4]cuda:0" = PlaceHolder[target=getitem_1]
#   %sub_tensor_1 : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %getitem_2), kwargs = {})
#   %exp_default_1 : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_1,), kwargs = {})
#   %div_2 : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_1, %getitem_3), kwargs = {})
#   %add_3 : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %div_2), kwargs = {})
#   %sub_tensor : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem), kwargs = {})
#   %exp_default : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div_3 : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_1), kwargs = {})
#   %add_4 : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %div_3), kwargs = {})
#   return %add_4
triton_poi_fused__softmax_add_5 = async_compile.triton('triton_poi_fused__softmax_add_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__softmax_add_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 1572928}, 'backend_hash': 'B2C00AD225F1945839503C666727F7E33E52219DAB5FB3E9C1893C177F82CD58', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_add_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x1 = xindex // 32768
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp5 = (tmp3 / tmp4)
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 - tmp7
    tmp9 = libdevice.exp(tmp8)
    tmp11 = (tmp9 / tmp10)
    tmp12 = tmp6 + tmp11
    tl.store(in_out_ptr0 + (x2), tmp12, None)
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
        assert_size_stride(arg0_1, (4, 32768), (32768, 1), 'input')
        assert_size_stride(arg1_1, (4, 32768), (32768, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg1_1 = copy_if_misaligned(arg1_1)
            buf0 = empty_strided_cuda((4, 1, 4), (4, 16, 1), torch.float32)
            buf1 = empty_strided_cuda((4, 1, 4), (4, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [z], Original ATen: [aten.add, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_add_prepare_softmax_online_0.run(arg0_1, arg1_1, buf0, buf1, 16, 8192, stream=raw_stream0)
            buf2 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
            buf3 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
            # Topologically Sorted Source Nodes: [z], Original ATen: [aten.add, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_add_prepare_softmax_online_1.run(buf0, buf1, buf2, buf3, 4, 4, stream=raw_stream0)
            buf4 = buf1; del buf1  # reuse
            buf5 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [z, softmax, z_1], Original ATen: [aten.add, aten._softmax, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__softmax_add_prepare_softmax_online_2.run(arg0_1, arg1_1, buf2, buf3, buf4, buf5, 16, 8192, stream=raw_stream0)
            buf6 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
            buf7 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
            # Topologically Sorted Source Nodes: [z, softmax, z_1], Original ATen: [aten.add, aten._softmax, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_add_prepare_softmax_online_1.run(buf4, buf5, buf6, buf7, 4, 4, stream=raw_stream0)
            buf8 = empty_strided_cuda((4, 32768), (32768, 1), torch.float32)
            buf9 = buf5; del buf5  # reuse
            buf10 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [z, softmax, z_1, softmax_1, z_2], Original ATen: [aten.add, aten._softmax, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__softmax_add_prepare_softmax_online_3.run(arg0_1, arg1_1, buf2, buf3, buf6, buf7, buf8, buf9, buf10, 16, 8192, stream=raw_stream0)
            del arg0_1
            del arg1_1
            buf11 = buf7; del buf7  # reuse
            buf12 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_add_prepare_softmax_online_1.run(buf9, buf10, buf11, buf12, 4, 4, stream=raw_stream0)
            buf13 = buf9; del buf9  # reuse
            buf14 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [softmax_2, z_3], Original ATen: [aten._softmax, aten.add, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__softmax_add_prepare_softmax_online_4.run(buf8, buf11, buf12, buf13, buf14, 16, 8192, stream=raw_stream0)
            buf15 = buf3; del buf3  # reuse
            buf16 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [softmax_2, z_3], Original ATen: [aten._softmax, aten.add, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_add_prepare_softmax_online_1.run(buf13, buf14, buf15, buf16, 4, 4, stream=raw_stream0)
            del buf13
            del buf14
            buf17 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [softmax_2, z_3, softmax_3, z_4], Original ATen: [aten._softmax, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_add_5.run(buf17, buf11, buf12, buf15, buf16, 131072, stream=raw_stream0)
            del buf11
            del buf12
            del buf15
            del buf16
        return (buf17, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((4, 32768), (32768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 32768), (32768, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
