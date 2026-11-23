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


# kernel path: /mnt/raid1/gaojg/data/dev/cooperative-strategy-20261123/cache/c6/cc6vcb77wqpb7o7sdo76yqtdei7ociqzqtn67igog3segajawasp.py
# Topologically Sorted Source Nodes: [z, softmax, z_1, softmax_1, z_2, softmax_2, z_3, softmax_3, z_4], Original ATen: [aten.add, prims.prepare_softmax_online, aten._softmax]
# Source node to ATen node mapping:
#   softmax => div, exp_default_3, sub_tensor_3
#   softmax_1 => div_1, exp_default_2, sub_tensor_2
#   softmax_2 => div_2, exp_default_1, sub_tensor_1
#   softmax_3 => div_3, exp_default, sub_tensor
#   z => add
#   z_1 => add_1
#   z_2 => add_2
#   z_3 => add_3
#   z_4 => add_4
# Graph fragment:
#   %arg0_1 : Tensor "f32[1, 32768][32768, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[1, 32768][32768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %getitem_6 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem_6]
#   %getitem_7 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem_7]
#   %getitem_4 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem_4]
#   %getitem_5 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem_5]
#   %add_2 : Tensor "f32[1, 32768][32768, 1]cuda:0" = PlaceHolder[target=add_2]
#   %getitem_2 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem_2]
#   %getitem_3 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem_3]
#   %getitem : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem]
#   %getitem_1 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem_1]
#   %add : Tensor "f32[1, 32768][32768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %prepare_softmax_online_default_3 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add, -1), kwargs = {})
#   %sub_tensor_3 : Tensor "f32[1, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_6), kwargs = {})
#   %exp_default_3 : Tensor "f32[1, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_3,), kwargs = {})
#   %div : Tensor "f32[1, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_3, %getitem_7), kwargs = {})
#   %add_1 : Tensor "f32[1, 32768][32768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %div), kwargs = {})
#   %prepare_softmax_online_default_2 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_1, -1), kwargs = {})
#   %sub_tensor_2 : Tensor "f32[1, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_4), kwargs = {})
#   %exp_default_2 : Tensor "f32[1, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_2,), kwargs = {})
#   %div_1 : Tensor "f32[1, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_2, %getitem_5), kwargs = {})
#   %add_2 : Tensor "f32[1, 32768][32768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %div_1), kwargs = {})
#   %prepare_softmax_online_default_1 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_2, -1), kwargs = {})
#   %sub_tensor_1 : Tensor "f32[1, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %getitem_2), kwargs = {})
#   %exp_default_1 : Tensor "f32[1, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_1,), kwargs = {})
#   %div_2 : Tensor "f32[1, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_1, %getitem_3), kwargs = {})
#   %add_3 : Tensor "f32[1, 32768][32768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %div_2), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_3, -1), kwargs = {})
#   %sub_tensor : Tensor "f32[1, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem), kwargs = {})
#   %exp_default : Tensor "f32[1, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div_3 : Tensor "f32[1, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_1), kwargs = {})
#   %add_4 : Tensor "f32[1, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %div_3), kwargs = {})
#   return %getitem_6,%getitem_7,%getitem_4,%getitem_5,%add_2,%getitem_2,%getitem_3,%getitem,%getitem_1,%add_4
triton_unk_fused__softmax_add_prepare_softmax_online_0 = async_compile.triton('triton_unk_fused__softmax_add_prepare_softmax_online_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.cooperative_reduction(
    size_hints={'x': 1, 'r0_': 32768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'sem_ptr': '*u32', 'ws_ptr': '*u8', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'launch_cooperative_grid': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'CooperativeReductionGrid', 'kernel_name': 'triton_unk_fused__softmax_add_prepare_softmax_online_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 16, 'autotune_hints': set(), 'tiling_scores': {'r0_': 524288}, 'persistent_reduction': True, 'backend_hash': 'B2C00AD225F1945839503C666727F7E33E52219DAB5FB3E9C1893C177F82CD58', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_unk_fused__softmax_add_prepare_softmax_online_0(in_out_ptr0, in_ptr0, in_ptr1, sem_ptr, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT : tl.constexpr):
    xnumel = 1
    r0_numel = 32768
    R0_BLOCK: tl.constexpr = triton_helpers.constexpr_next_power_of_2((32768 + RSPLIT - 1) // RSPLIT)
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    RSPLIT_NEXT_POWER_OF_2: tl.constexpr = triton_helpers.constexpr_next_power_of_2(RSPLIT)
    RSPLIT_IS_POWER_OF_2: tl.constexpr = RSPLIT == RSPLIT_NEXT_POWER_OF_2
    HAS_RSPLIT: tl.constexpr = RSPLIT > 1
    rsplit_id = tl.program_id(0)
    num_rblocks = (rnumel + RBLOCK - 1) // RBLOCK
    rsplit_chunk = (num_rblocks + RSPLIT - 1) // RSPLIT * RBLOCK
    rsplit_start = rsplit_chunk * rsplit_id
    rsplit_end = rsplit_chunk * (rsplit_id + 1)
    rsplit_end = tl.where(rsplit_end < rnumel, rsplit_end, rnumel)
    xoffset = tl.program_id(1) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :] + rsplit_start
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    rsplit_arange = tl.arange(0, RSPLIT_NEXT_POWER_OF_2)[None, :]
    if RSPLIT_IS_POWER_OF_2:
        rsplit_mask: tl.constexpr = None
    else:
        rsplit_mask = rsplit_arange < RSPLIT
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None].to(tl.float32)
    tmp9 = tmp3 - tmp8
    tmp10 = libdevice.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(r0_mask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    if HAS_RSPLIT:
        tmp8_ws = (ws_ptr + 0).to(tl.pointer_type(tl.float32))
        tl.store(tmp8_ws + (xindex * RSPLIT + rsplit_id), tmp8, None)
    if HAS_RSPLIT:
        tmp14_ws = (ws_ptr + 256).to(tl.pointer_type(tl.float32))
        tl.store(tmp14_ws + (xindex * RSPLIT + rsplit_id), tmp14, None)
    if HAS_RSPLIT:
        tmp8_ws = (ws_ptr + 512).to(tl.pointer_type(tl.float32))
        tl.store(tmp8_ws + (xindex * RSPLIT + rsplit_id), tmp8, None)
        tmp14_ws = (ws_ptr + 768).to(tl.pointer_type(tl.float32))
        tl.store(tmp14_ws + (xindex * RSPLIT + rsplit_id), tmp14, None)
    if HAS_RSPLIT:
        triton_helpers.x_grid_barrier(sem_ptr + tl.program_id(1))
    if HAS_RSPLIT:
        tmp8_peers = tl.load(tmp8_ws + (xindex * RSPLIT + rsplit_arange), rsplit_mask, eviction_policy='evict_first', other=triton_helpers.if_mask(rsplit_mask, float("-inf")))
        tmp8 = triton_helpers.max2(tmp8_peers, 1)[:, None]
    if HAS_RSPLIT:
        tmp14_peers = tl.load(tmp14_ws + (xindex * RSPLIT + rsplit_arange), rsplit_mask, eviction_policy='evict_first', other=triton_helpers.if_mask(rsplit_mask, 0))
        tmp14 = tl.sum(tmp14_peers, 1)[:, None]
    if HAS_RSPLIT:
        tmp8_peers = tl.load(tmp8_ws + (xindex * RSPLIT + rsplit_arange), rsplit_mask, eviction_policy='evict_first', other=triton_helpers.if_mask(rsplit_mask, float("-inf")))
        tmp14_peers = tl.load(tmp14_ws + (xindex * RSPLIT + rsplit_arange), rsplit_mask, eviction_policy='evict_first', other=triton_helpers.if_mask(rsplit_mask, 0))

        tmp8, tmp14 = triton_helpers.online_softmax_reduce(
            tmp8_peers, tmp14_peers, 1, False)
        tmp8 = tmp8[:, None]
        tmp14 = tmp14[:, None]
    tmp15 = tmp2 - tmp8
    tmp16 = libdevice.exp(tmp15)
    tmp17 = (tmp16 / tmp14)
    tmp18 = tmp2 + tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(r0_mask, tmp21, float("-inf"))
    tmp24 = triton_helpers.max2(tmp23, 1)[:, None].to(tl.float32)
    tmp25 = tmp19 - tmp24
    tmp26 = libdevice.exp(tmp25)
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
    tmp29 = tl.where(r0_mask, tmp27, 0)
    tmp30 = tl.sum(tmp29, 1)[:, None].to(tl.float32)
    if HAS_RSPLIT:
        tmp24_ws = (ws_ptr + 1024).to(tl.pointer_type(tl.float32))
        tl.store(tmp24_ws + (xindex * RSPLIT + rsplit_id), tmp24, None)
    if HAS_RSPLIT:
        tmp30_ws = (ws_ptr + 1280).to(tl.pointer_type(tl.float32))
        tl.store(tmp30_ws + (xindex * RSPLIT + rsplit_id), tmp30, None)
    if HAS_RSPLIT:
        tmp24_ws = (ws_ptr + 1536).to(tl.pointer_type(tl.float32))
        tl.store(tmp24_ws + (xindex * RSPLIT + rsplit_id), tmp24, None)
        tmp30_ws = (ws_ptr + 1792).to(tl.pointer_type(tl.float32))
        tl.store(tmp30_ws + (xindex * RSPLIT + rsplit_id), tmp30, None)
    if HAS_RSPLIT:
        triton_helpers.x_grid_barrier(sem_ptr + tl.program_id(1))
    if HAS_RSPLIT:
        tmp24_peers = tl.load(tmp24_ws + (xindex * RSPLIT + rsplit_arange), rsplit_mask, eviction_policy='evict_first', other=triton_helpers.if_mask(rsplit_mask, float("-inf")))
        tmp24 = triton_helpers.max2(tmp24_peers, 1)[:, None]
    if HAS_RSPLIT:
        tmp30_peers = tl.load(tmp30_ws + (xindex * RSPLIT + rsplit_arange), rsplit_mask, eviction_policy='evict_first', other=triton_helpers.if_mask(rsplit_mask, 0))
        tmp30 = tl.sum(tmp30_peers, 1)[:, None]
    if HAS_RSPLIT:
        tmp24_peers = tl.load(tmp24_ws + (xindex * RSPLIT + rsplit_arange), rsplit_mask, eviction_policy='evict_first', other=triton_helpers.if_mask(rsplit_mask, float("-inf")))
        tmp30_peers = tl.load(tmp30_ws + (xindex * RSPLIT + rsplit_arange), rsplit_mask, eviction_policy='evict_first', other=triton_helpers.if_mask(rsplit_mask, 0))

        tmp24, tmp30 = triton_helpers.online_softmax_reduce(
            tmp24_peers, tmp30_peers, 1, False)
        tmp24 = tmp24[:, None]
        tmp30 = tmp30[:, None]
    tmp31 = tmp18 - tmp24
    tmp32 = libdevice.exp(tmp31)
    tmp33 = (tmp32 / tmp30)
    tmp34 = tmp18 + tmp33
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, R0_BLOCK])
    tmp37 = tl.broadcast_to(tmp35, [XBLOCK, R0_BLOCK])
    tmp39 = tl.where(r0_mask, tmp37, float("-inf"))
    tmp40 = triton_helpers.max2(tmp39, 1)[:, None].to(tl.float32)
    tmp41 = tmp35 - tmp40
    tmp42 = libdevice.exp(tmp41)
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK, R0_BLOCK])
    tmp45 = tl.where(r0_mask, tmp43, 0)
    tmp46 = tl.sum(tmp45, 1)[:, None].to(tl.float32)
    if HAS_RSPLIT:
        tmp40_ws = (ws_ptr + 0).to(tl.pointer_type(tl.float32))
        tl.store(tmp40_ws + (xindex * RSPLIT + rsplit_id), tmp40, None)
    if HAS_RSPLIT:
        tmp46_ws = (ws_ptr + 256).to(tl.pointer_type(tl.float32))
        tl.store(tmp46_ws + (xindex * RSPLIT + rsplit_id), tmp46, None)
    if HAS_RSPLIT:
        tmp40_ws = (ws_ptr + 512).to(tl.pointer_type(tl.float32))
        tl.store(tmp40_ws + (xindex * RSPLIT + rsplit_id), tmp40, None)
        tmp46_ws = (ws_ptr + 768).to(tl.pointer_type(tl.float32))
        tl.store(tmp46_ws + (xindex * RSPLIT + rsplit_id), tmp46, None)
    if HAS_RSPLIT:
        triton_helpers.x_grid_barrier(sem_ptr + tl.program_id(1))
    if HAS_RSPLIT:
        tmp40_peers = tl.load(tmp40_ws + (xindex * RSPLIT + rsplit_arange), rsplit_mask, eviction_policy='evict_first', other=triton_helpers.if_mask(rsplit_mask, float("-inf")))
        tmp40 = triton_helpers.max2(tmp40_peers, 1)[:, None]
    if HAS_RSPLIT:
        tmp46_peers = tl.load(tmp46_ws + (xindex * RSPLIT + rsplit_arange), rsplit_mask, eviction_policy='evict_first', other=triton_helpers.if_mask(rsplit_mask, 0))
        tmp46 = tl.sum(tmp46_peers, 1)[:, None]
    if HAS_RSPLIT:
        tmp40_peers = tl.load(tmp40_ws + (xindex * RSPLIT + rsplit_arange), rsplit_mask, eviction_policy='evict_first', other=triton_helpers.if_mask(rsplit_mask, float("-inf")))
        tmp46_peers = tl.load(tmp46_ws + (xindex * RSPLIT + rsplit_arange), rsplit_mask, eviction_policy='evict_first', other=triton_helpers.if_mask(rsplit_mask, 0))

        tmp40, tmp46 = triton_helpers.online_softmax_reduce(
            tmp40_peers, tmp46_peers, 1, False)
        tmp40 = tmp40[:, None]
        tmp46 = tmp46[:, None]
    tmp47 = tmp34 - tmp40
    tmp48 = libdevice.exp(tmp47)
    tmp49 = (tmp48 / tmp46)
    tmp50 = tmp34 + tmp49
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, R0_BLOCK])
    tmp53 = tl.broadcast_to(tmp51, [XBLOCK, R0_BLOCK])
    tmp55 = tl.where(r0_mask, tmp53, float("-inf"))
    tmp56 = triton_helpers.max2(tmp55, 1)[:, None].to(tl.float32)
    tmp57 = tmp51 - tmp56
    tmp58 = libdevice.exp(tmp57)
    tmp59 = tl.broadcast_to(tmp58, [XBLOCK, R0_BLOCK])
    tmp61 = tl.where(r0_mask, tmp59, 0)
    tmp62 = tl.sum(tmp61, 1)[:, None].to(tl.float32)
    if HAS_RSPLIT:
        tmp56_ws = (ws_ptr + 1024).to(tl.pointer_type(tl.float32))
        tl.store(tmp56_ws + (xindex * RSPLIT + rsplit_id), tmp56, None)
    if HAS_RSPLIT:
        tmp62_ws = (ws_ptr + 1280).to(tl.pointer_type(tl.float32))
        tl.store(tmp62_ws + (xindex * RSPLIT + rsplit_id), tmp62, None)
    if HAS_RSPLIT:
        tmp56_ws = (ws_ptr + 1536).to(tl.pointer_type(tl.float32))
        tl.store(tmp56_ws + (xindex * RSPLIT + rsplit_id), tmp56, None)
        tmp62_ws = (ws_ptr + 1792).to(tl.pointer_type(tl.float32))
        tl.store(tmp62_ws + (xindex * RSPLIT + rsplit_id), tmp62, None)
    if HAS_RSPLIT:
        triton_helpers.x_grid_barrier(sem_ptr + tl.program_id(1))
    if HAS_RSPLIT:
        tmp56_peers = tl.load(tmp56_ws + (xindex * RSPLIT + rsplit_arange), rsplit_mask, eviction_policy='evict_first', other=triton_helpers.if_mask(rsplit_mask, float("-inf")))
        tmp56 = triton_helpers.max2(tmp56_peers, 1)[:, None]
    if HAS_RSPLIT:
        tmp62_peers = tl.load(tmp62_ws + (xindex * RSPLIT + rsplit_arange), rsplit_mask, eviction_policy='evict_first', other=triton_helpers.if_mask(rsplit_mask, 0))
        tmp62 = tl.sum(tmp62_peers, 1)[:, None]
    if HAS_RSPLIT:
        tmp56_peers = tl.load(tmp56_ws + (xindex * RSPLIT + rsplit_arange), rsplit_mask, eviction_policy='evict_first', other=triton_helpers.if_mask(rsplit_mask, float("-inf")))
        tmp62_peers = tl.load(tmp62_ws + (xindex * RSPLIT + rsplit_arange), rsplit_mask, eviction_policy='evict_first', other=triton_helpers.if_mask(rsplit_mask, 0))

        tmp56, tmp62 = triton_helpers.online_softmax_reduce(
            tmp56_peers, tmp62_peers, 1, False)
        tmp56 = tmp56[:, None]
        tmp62 = tmp62[:, None]
    tmp63 = tmp50 - tmp56
    tmp64 = libdevice.exp(tmp63)
    tmp65 = (tmp64 / tmp62)
    tmp66 = tmp50 + tmp65
    tl.store(in_out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp66, r0_mask)
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
        assert_size_stride(arg0_1, (1, 32768), (32768, 1), 'input')
        assert_size_stride(arg1_1, (1, 32768), (32768, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg1_1 = copy_if_misaligned(arg1_1)
            buf4 = empty_strided_cuda((1, 32768), (32768, 1), torch.float32)
            buf9 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [z, softmax, z_1, softmax_1, z_2, softmax_2, z_3, softmax_3, z_4], Original ATen: [aten.add, prims.prepare_softmax_online, aten._softmax]
            semaphores_cuda_0 = empty_strided_cuda((1, ), (1, ), torch.uint32)
            semaphores_cuda_0.zero_()
            workspace_0 = empty_strided_cuda((2048, ), (1, ), torch.uint8)
            raw_stream0 = get_raw_stream(0)
            triton_unk_fused__softmax_add_prepare_softmax_online_0.run(buf9, arg0_1, arg1_1, semaphores_cuda_0, workspace_0, 1, 32768, stream=raw_stream0)
            del workspace_0
            del arg0_1
            del arg1_1
        return (buf9, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 32768), (32768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 32768), (32768, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
