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


# kernel path: /tmp/tmp4p8wdo5v/wp/cwpn7uwpnwu24jw5wjua6rz3ookm6swps7ova2ulvzcqmbj5cdll.py
# Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean => mean
# Graph fragment:
#   %arg3_1 : Tensor "f32[1, s27, s53, s0][s0*s27*s53, s0*s53, s0, 1]npu:0" = PlaceHolder[target=arg3_1]
#   %mean : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%arg3_1,), kwargs = {})
#   return %buf0
triton_red_fused_mean_0 = async_compile.triton('triton_red_fused_mean_0', '''
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
    size_hints={'x': 2, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_mean_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'requires_no_linear_block_remap': True, 'npu_kernel_type': 'simd_simt_mix', 'split_axis': [0], 'tiling_axis': [0, 1], 'low_dims': [1], 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'axis_names': ['x', 'r0_']},
)
@triton.jit
def triton_red_fused_mean_0(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = (r0_1 + x0*((1 + ks0*ks1*ks2) // 2)).to(tl.int32)
        tl.static_assert(tmp0.dtype == tl.int32)
        tl.static_assert(tmp0.dtype == tl.int32)
        tl.static_assert(tmp0.shape == (XBLOCK, R0_BLOCK))
        tmp1 = (ks0*ks1*ks2).to(tl.int32)
        tl.static_assert(tmp1.dtype == tl.int32)
        tl.static_assert(tmp1.dtype == tl.int32)
        tl.static_assert(tmp1.shape == ())
        tmp2 = tmp0 < tmp1
        tl.static_assert(tmp2.dtype == tl.int1)
        tl.static_assert(tmp2.dtype == tl.int1)
        tl.static_assert(tmp2.shape == (XBLOCK, R0_BLOCK))
        tmp3 = tl.load(in_ptr0 + (((r0_1 + x0*((1 + ks0*ks1*ks2) // 2)) % (ks0*ks1*ks2))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tl.static_assert(tmp3.dtype == tl.float32)
        tl.static_assert(tmp3.dtype == tl.float32)
        tl.static_assert(tmp3.shape == (XBLOCK, R0_BLOCK))
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tl.static_assert(tmp4.dtype == tl.float32)
        tmp6 = _tmp5 + tmp4
        tl.static_assert(tmp6.dtype == tl.float32)
        tl.static_assert(tmp6.dtype == tl.float32)
        tl.static_assert(tmp6.shape == (XBLOCK, R0_BLOCK))
        _tmp5 = tl.where(r0_mask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''', device_str='npu')


# kernel path: /tmp/tmp4p8wdo5v/ov/covsrjecobcaztknnftk67tpjak3ueb2m6ienkhn7rzoe5nhrfqz.py
# Topologically Sorted Source Nodes: [mean_1], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean_1 => mean_1
# Graph fragment:
#   %arg3_1 : Tensor "f32[1, s27, s53, s0][s0*s27*s53, s0*s53, s0, 1]npu:0" = PlaceHolder[target=arg3_1]
#   %buf2 : Tensor "f32[1, s27, 1, s0][s0*s27, s0, s0*s27, 1]npu:0" = PlaceHolder[target=buf2]
#   %mean_1 : Tensor "f32[1, s27, 1, s0][s0*s27, s0, s0, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%arg3_1, [-2], True), kwargs = {})
#   return %buf2,%mean_1
triton_red_fused_mean_1 = async_compile.triton('triton_red_fused_mean_1', '''
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
    size_hints={'x': 16, 'r0_': 4},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_mean_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'requires_no_linear_block_remap': True, 'npu_kernel_type': 'simd_simt_mix', 'split_axis': [0], 'tiling_axis': [0, 1], 'low_dims': [1], 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'axis_names': ['x', 'r0_']},
)
@triton.jit
def triton_red_fused_mean_1(in_out_ptr0, in_ptr0, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + ks0*r0_2 + ks0*ks1*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tl.static_assert(tmp0.dtype == tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tl.static_assert(tmp1.dtype == tl.float32)
        tmp3 = _tmp2 + tmp1
        tl.static_assert(tmp3.dtype == tl.float32)
        tl.static_assert(tmp3.dtype == tl.float32)
        tl.static_assert(tmp3.shape == (XBLOCK, R0_BLOCK))
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = (ks1).to(tl.float32)
    tl.static_assert(tmp4.dtype == tl.float32)
    tmp5 = (tmp4).to(tl.float32)
    tl.static_assert(tmp5.dtype == tl.float32)
    tl.static_assert(tmp5.dtype == tl.float32)
    tl.static_assert(tmp5.shape == ())
    tmp6 = (tmp2 / tmp5)
    tl.static_assert(tmp6.dtype == tl.float32)
    tl.static_assert(tmp6.dtype == tl.float32)
    tl.static_assert(tmp6.shape == (XBLOCK, 1))
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''', device_str='npu')


# kernel path: /tmp/tmp4p8wdo5v/kn/cknjq46rxre5niio6eyk6oyom7d3uf5o4mtpcvubdqzqaicjxttm.py
# Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean => mean
# Graph fragment:
#   %buf0 : Tensor "f32[2][1]npu:0" = PlaceHolder[target=buf0]
#   %buf1 : Tensor "f32[][]npu:0" = PlaceHolder[target=buf1]
#   %mean : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%arg3_1,), kwargs = {})
#   return %buf1,%mean
triton_per_fused_mean_2 = async_compile.triton('triton_per_fused_mean_2', '''
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

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 2},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_mean_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'requires_no_linear_block_remap': True, 'npu_kernel_type': 'simd_simt_mix', 'split_axis': [0], 'tiling_axis': [0, 1], 'low_dims': [1], 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'axis_names': ['x', 'r0_']},
)
@triton.jit
def triton_per_fused_mean_2(in_out_ptr0, in_ptr0, ks0, ks1, ks2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
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
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tl.static_assert(tmp1.dtype == tl.float32)
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tl.static_assert(tmp3.dtype == tl.float32)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.static_assert(tmp4.dtype == tl.float32)
    tmp5 = (ks0*ks1*ks2).to(tl.float32)
    tl.static_assert(tmp5.dtype == tl.float32)
    tmp6 = (tmp5).to(tl.float32)
    tl.static_assert(tmp6.dtype == tl.float32)
    tl.static_assert(tmp6.dtype == tl.float32)
    tl.static_assert(tmp6.shape == ())
    tmp7 = (tmp4 / tmp6)
    tl.static_assert(tmp7.dtype == tl.float32)
    tl.static_assert(tmp7.dtype == tl.float32)
    tl.static_assert(tmp7.shape == (XBLOCK, 1))
    tl.store(in_out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp7, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        s27 = arg0_1
        s53 = arg1_1
        s0 = arg2_1
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg3_1 = copy_if_misaligned(arg3_1)
            buf0 = empty_strided_npu((2, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
            triton_red_fused_mean_0_r0_numel = (1 + s0*s27*s53) // 2
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_mean_0.run(arg3_1, buf0, s0, s27, s53, 2, triton_red_fused_mean_0_r0_numel, stream=raw_stream0)
            buf2 = empty_strided_npu((1, s27, 1, s0), (s0*s27, s0, s0*s27, 1), torch.float32)
            buf3 = reinterpret_tensor(buf2, (1, s27, 1, s0), (s0*s27, s0, s0, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [mean_1], Original ATen: [aten.mean]
            triton_red_fused_mean_1_xnumel = s0*s27
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_mean_1.run(buf3, arg3_1, s0, s53, triton_red_fused_mean_1_xnumel, s53, stream=raw_stream0)
            del arg3_1
            buf1 = empty_strided_npu((), (), torch.float32)
            buf4 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_mean_2.run(buf4, buf0, s0, s27, s53, 1, 2, stream=raw_stream0)
            del buf0
        return (buf4, buf3, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = 2
    arg1_1 = 4
    arg2_1 = 8
    arg3_1 = rand_strided((1, 2, 4, 8), (64, 32, 8, 1), device='npu:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))

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


# kernel path: /tmp/tmp2bt0f70d/m6/cm6hhuq3dmw2moksrsieowudqsh7oo5n7paopnbe4g3kaleiu5ue.py
# Topologically Sorted Source Nodes: [min_1], Original ATen: [aten.min]
# Source node to ATen node mapping:
#   min_1 => min_1
# Graph fragment:
#   %arg2_1 : Tensor "f32[256, 256][256, 1]npu:0" = PlaceHolder[target=arg2_1]
#   %min_1 : [num_users=2] = call_function[target=torch.ops.aten.min.dim](args = (%arg2_1, -1), kwargs = {})
#   return %getitem,%getitem_1
triton_per_fused_min_0 = async_compile.triton('triton_per_fused_min_0', '''
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

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_min_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'requires_no_linear_block_remap': True, 'npu_kernel_type': 'simd_simt_mix', 'split_axis': [0], 'tiling_axis': [0, 1], 'low_dims': [1], 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'axis_names': ['x', 'r0_']},
)
@triton.jit
def triton_per_fused_min_0(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tl.static_assert(tmp1.dtype == tl.float32)
    tmp3 = tl.where(r0_mask & xmask, tmp1, float("inf"))
    tl.static_assert(tmp3.dtype == tl.float32)
    tmp4 = triton_helpers.min2(tmp3, 1)[:, None].to(tl.float32)
    tl.static_assert(tmp4.dtype == tl.float32)
    tmp6 = tl.broadcast_to(rindex, tmp3.shape)
    tl.static_assert(tmp6.dtype == tl.int32)
    tmp5_val, tmp5_idx = triton_helpers.min_with_index(tmp3, tmp6, 1)
    tmp5 = tmp5_idx[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='npu')


# kernel path: /tmp/tmp2bt0f70d/c2/cc2ojan5a44g7e2hwfcc3vt5bk542joz4sfnhtivov6w6iymu6gv.py
# Topologically Sorted Source Nodes: [max_1], Original ATen: [aten.max]
# Source node to ATen node mapping:
#   max_1 => max_1
# Graph fragment:
#   %arg3_1 : Tensor "f32[256, 256][256, 1]npu:0" = PlaceHolder[target=arg3_1]
#   %max_1 : [num_users=2] = call_function[target=torch.ops.aten.max.dim](args = (%arg3_1, -1), kwargs = {})
#   return %getitem_2,%getitem_3
triton_per_fused_max_1 = async_compile.triton('triton_per_fused_max_1', '''
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

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_max_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'requires_no_linear_block_remap': True, 'npu_kernel_type': 'simd_simt_mix', 'split_axis': [0], 'tiling_axis': [0, 1], 'low_dims': [1], 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'axis_names': ['x', 'r0_']},
)
@triton.jit
def triton_per_fused_max_1(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tl.static_assert(tmp1.dtype == tl.float32)
    tmp3 = tl.where(r0_mask & xmask, tmp1, float("-inf"))
    tl.static_assert(tmp3.dtype == tl.float32)
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None].to(tl.float32)
    tl.static_assert(tmp4.dtype == tl.float32)
    tmp6 = tl.broadcast_to(rindex, tmp3.shape)
    tl.static_assert(tmp6.dtype == tl.int32)
    tmp5_val, tmp5_idx = triton_helpers.max_with_index(tmp3, tmp6, 1)
    tmp5 = tmp5_idx[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='npu')


# kernel path: /tmp/tmp2bt0f70d/zw/czwxygrvynjalopyuxzjrwyb7jxqdeklvza3m5ati3t7e6zxwtvw.py
# Topologically Sorted Source Nodes: [a], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   a => mul
# Graph fragment:
#   %arg0_1 : Tensor "f32[20, 20][20, 1]npu:0" = PlaceHolder[target=arg0_1]
#   %mul : Tensor "f32[20, 20][20, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 4), kwargs = {})
#   return %mul
# SchedulerNodes: [SchedulerNode(name='op4')]

triton_poi_fused_mul_2 = async_compile.triton('triton_poi_fused_mul_2', '''
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
    size_hints={'y0': 20, 'x1': 20}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_2', 'mutated_arg_names': [], 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 20), ('x1', 20)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_2(in_ptr0, out_ptr0, y0_numel, x1_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    x1_offset = tl.program_id(1) * X1BLOCK
    base_x1= tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (X1BLOCK + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        for loop_x1 in range(loops_x1):
            x1 = x1_offset + (loop_x1 * X1BLOCK_SUB) + base_x1[None,:]
            x1_mask = x1 < min(X1BLOCK+x1_offset, x1_numel)
            tmp0 = tl.load(in_ptr0 + (x1 + 20*y0), x1_mask & y0_mask)
            tl.static_assert(tmp0.dtype == tl.float32)
            tmp1 = tl.full([1, 1], 4.0, tl.float32)
            tl.static_assert(tmp1.dtype == tl.float32)
            tl.static_assert(tmp1.dtype == tl.float32)
            tmp2 = tmp0 * tmp1
            tl.static_assert(tmp2.dtype == tl.float32)
            tl.static_assert(tmp2.dtype == tl.float32)
            tl.store(out_ptr0 + (x1 + 20*y0), tmp2, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: /tmp/tmp2bt0f70d/s6/cs6adzcsrttmzcboxm7wxfvcm7tg6ximnnndkykalzud4wvxtkhq.py
# Topologically Sorted Source Nodes: [b], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   b => add
# Graph fragment:
#   %arg1_1 : Tensor "f32[30, 30][30, 1]npu:0" = PlaceHolder[target=arg1_1]
#   %add : Tensor "f32[30, 30][30, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, 8), kwargs = {})
#   return %add
# SchedulerNodes: [SchedulerNode(name='op5')]

triton_poi_fused_add_3 = async_compile.triton('triton_poi_fused_add_3', '''
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
    size_hints={'y0': 30, 'x1': 30}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_3', 'mutated_arg_names': [], 'backend_hash': '904EABC54D771636701C89D131527E16DC1636875AB1277CC69C0713ED99AC8E', 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'axis_static_values': (('y0', 30), ('x1', 30)), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_3(in_ptr0, out_ptr0, y0_numel, x1_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    x1_offset = tl.program_id(1) * X1BLOCK
    base_x1= tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (X1BLOCK + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        for loop_x1 in range(loops_x1):
            x1 = x1_offset + (loop_x1 * X1BLOCK_SUB) + base_x1[None,:]
            x1_mask = x1 < min(X1BLOCK+x1_offset, x1_numel)
            tmp0 = tl.load(in_ptr0 + (x1 + 30*y0), x1_mask & y0_mask)
            tl.static_assert(tmp0.dtype == tl.float32)
            tmp1 = tl.full([1, 1], 8.0, tl.float32)
            tl.static_assert(tmp1.dtype == tl.float32)
            tl.static_assert(tmp1.dtype == tl.float32)
            tmp2 = tmp0 + tmp1
            tl.static_assert(tmp2.dtype == tl.float32)
            tl.static_assert(tmp2.dtype == tl.float32)
            tl.store(out_ptr0 + (x1 + 30*y0), tmp2, x1_mask & y0_mask)
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
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg2_1 = copy_if_misaligned(arg2_1)
            arg3_1 = copy_if_misaligned(arg3_1)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg1_1 = copy_if_misaligned(arg1_1)
            buf0 = empty_strided_npu((256, ), (1, ), torch.float32)
            buf1 = empty_strided_npu((256, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [min_1], Original ATen: [aten.min]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_min_0.run(arg2_1, buf0, buf1, 256, 256, stream=raw_stream0)
            del arg2_1
            buf2 = empty_strided_npu((256, ), (1, ), torch.float32)
            buf3 = empty_strided_npu((256, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [max_1], Original ATen: [aten.max]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_max_1.run(arg3_1, buf2, buf3, 256, 256, stream=raw_stream0)
            del arg3_1
            buf4 = empty_strided_npu((20, 20), (20, 1), torch.float32)
            # Topologically Sorted Source Nodes: [a], Original ATen: [aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_mul_2.run(arg0_1, buf4, 20, 20, stream=raw_stream0)
            del arg0_1
            buf5 = empty_strided_npu((30, 30), (30, 1), torch.float32)
            # Topologically Sorted Source Nodes: [b], Original ATen: [aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_3.run(arg1_1, buf5, 30, 30, stream=raw_stream0)
            del arg1_1
        return (buf4, buf5, buf0, buf1, buf2, buf3, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((20, 20), (20, 1), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((30, 30), (30, 1), device='npu:0', dtype=torch.float32)
    arg2_1 = rand_strided((256, 256), (256, 1), device='npu:0', dtype=torch.float32)
    arg3_1 = rand_strided((256, 256), (256, 1), device='npu:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))

