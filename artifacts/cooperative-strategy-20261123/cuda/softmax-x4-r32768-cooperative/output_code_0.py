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
# Topologically Sorted Source Nodes: [add], Original ATen: [aten.add, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   add => add
# Graph fragment:
#   %arg0_1 : Tensor "f32[4, 32768][32768, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[4, 32768][32768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %add : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add, -1), kwargs = {})
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
# Topologically Sorted Source Nodes: [add], Original ATen: [aten.add, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   add => add
# Graph fragment:
#   %buf0 : Tensor "f32[4, 1, 4][4, 16, 1]cuda:0" = PlaceHolder[target=buf0]
#   %buf1 : Tensor "f32[4, 1, 4][4, 16, 1]cuda:0" = PlaceHolder[target=buf1]
#   %add : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add, -1), kwargs = {})
#   return %getitem,%getitem_1
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


# kernel path: /mnt/raid1/gaojg/data/dev/cooperative-strategy-20261123/cache/yc/cycrut2lhjyana4yvm7e2n5ilqmfbxjg7wzftlmtjjdikpoe5aaj.py
# Topologically Sorted Source Nodes: [add, softmax], Original ATen: [aten.add, aten._softmax]
# Source node to ATen node mapping:
#   add => add
#   softmax => div, exp_default, sub_tensor
# Graph fragment:
#   %arg0_1 : Tensor "f32[4, 32768][32768, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[4, 32768][32768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %getitem : Tensor "f32[4, 1][1, 4]cuda:0" = PlaceHolder[target=getitem]
#   %getitem_1 : Tensor "f32[4, 1][1, 4]cuda:0" = PlaceHolder[target=getitem_1]
#   %add : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %sub_tensor : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem), kwargs = {})
#   %exp_default : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : Tensor "f32[4, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_1), kwargs = {})
#   return %div
triton_poi_fused__softmax_add_2 = async_compile.triton('triton_poi_fused__softmax_add_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__softmax_add_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 2097184}, 'backend_hash': 'B2C00AD225F1945839503C666727F7E33E52219DAB5FB3E9C1893C177F82CD58', 'assert_indirect_indexing': True, 'autotune_local_cache': False, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_add_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x1 = xindex // 32768
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp5 = libdevice.exp(tmp4)
    tmp7 = (tmp5 / tmp6)
    tl.store(out_ptr0 + (x2), tmp7, None)
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
            # Topologically Sorted Source Nodes: [add], Original ATen: [aten.add, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_add_prepare_softmax_online_0.run(arg0_1, arg1_1, buf0, buf1, 16, 8192, stream=raw_stream0)
            buf2 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
            buf3 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
            # Topologically Sorted Source Nodes: [add], Original ATen: [aten.add, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_add_prepare_softmax_online_1.run(buf0, buf1, buf2, buf3, 4, 4, stream=raw_stream0)
            del buf0
            del buf1
            buf4 = empty_strided_cuda((4, 32768), (32768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [add, softmax], Original ATen: [aten.add, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__softmax_add_2.run(arg0_1, arg1_1, buf2, buf3, buf4, 131072, stream=raw_stream0)
            del arg0_1
            del arg1_1
            del buf2
            del buf3
        return (buf4, )

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
