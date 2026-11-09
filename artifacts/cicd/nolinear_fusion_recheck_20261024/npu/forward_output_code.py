# AOT ID: ['0_forward']
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


# kernel path: <validation>/branch/cache/inductor/mu/cmufz67whtx2ih3qqrh674pvl6ub3hzetqf73dyuas5pghxy2cpg.py
# Topologically Sorted Source Nodes: [mean, sigmoid], Original ATen: [aten.mean, aten.sigmoid]
# Source node to ATen node mapping:
#   mean => mean
#   sigmoid => sigmoid
# Graph fragment:
#   %convolution : Tensor "f32[s4, s14, (((s43 - 1)//32)) + 1, (((s43 - 1)//32)) + 1][s14*((((s43 - 1)//32)) + 1)**2, ((((s43 - 1)//32)) + 1)**2, (((s43 - 1)//32)) + 1, 1]npu:0" = PlaceHolder[target=convolution]
#   %buf2 : Tensor "f32[s4, s14, 1, 1][s14, 1, s14*s4, s14*s4]npu:0" = PlaceHolder[target=buf2]
#   %mean : Tensor "f32[s4, s14, 1, 1][s14, 1, 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution, [-2, -1], True), kwargs = {})
#   %sigmoid : Tensor "f32[s4, s14, 1, 1][s14, 1, 1, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mean,), kwargs = {})
#   return %buf2,%sigmoid
triton_red_fused_mean_sigmoid_0 = async_compile.triton('triton_red_fused_mean_sigmoid_0', '''
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
    size_hints={'x': 32, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_mean_sigmoid_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'backend_hash': '<backend-hash>', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'requires_no_linear_block_remap': True, 'npu_kernel_type': 'simd_simt_mix', 'split_axis': [0], 'tiling_axis': [0, 1], 'low_dims': [1], 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'axis_names': ['x', 'r0_']},
)
@triton.jit
def triton_red_fused_mean_sigmoid_0(in_out_ptr0, in_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + x0 + x0*(triton_helpers.div_floor_integer((-1) + ks0,  32))*(triton_helpers.div_floor_integer((-1) + ks0,  32)) + 2*x0*(triton_helpers.div_floor_integer((-1) + ks0,  32))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = (1 + (triton_helpers.div_floor_integer((-1) + ks0,  32))*(triton_helpers.div_floor_integer((-1) + ks0,  32)) + 2*(triton_helpers.div_floor_integer((-1) + ks0,  32))).to(tl.float32)
    tmp5 = (tmp4).to(tl.float32)
    tmp6 = (tmp2 / tmp5)
    tmp7 = tl.sigmoid(tmp6)
    tl.store(in_out_ptr0 + (x0), tmp7, xmask)
''', device_str='npu')


# kernel path: <validation>/branch/cache/inductor/7o/c7oxg4fux5uvp426knottoa6oync3szj4qt27l7q6o4gk46juatj.py
# Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul => mul_13
# Graph fragment:
#   %convolution : Tensor "f32[s4, s14, (((s43 - 1)//32)) + 1, (((s43 - 1)//32)) + 1][s14*((((s43 - 1)//32)) + 1)**2, ((((s43 - 1)//32)) + 1)**2, (((s43 - 1)//32)) + 1, 1]npu:0" = PlaceHolder[target=convolution]
#   %sigmoid : Tensor "f32[s4, s14, 1, 1][s14, 1, 1, 1]npu:0" = PlaceHolder[target=sigmoid]
#   %mul_13 : Tensor "f32[s4, s14, (((s43 - 1)//32)) + 1, (((s43 - 1)//32)) + 1][s14*((((s43 - 1)//32)) + 1)**2, ((((s43 - 1)//32)) + 1)**2, (((s43 - 1)//32)) + 1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, %sigmoid), kwargs = {})
#   return %mul_13
triton_poi_fused_mul_1 = async_compile.triton('triton_poi_fused_mul_1', '''
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

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_mul_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'backend_hash': '<backend-hash>', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'requires_no_linear_block_remap': True, 'npu_kernel_type': 'simd_simt_mix', 'split_axis': [0], 'tiling_axis': [0], 'low_dims': [0], 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'axis_names': ['x']},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_1(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // ks0
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, xmask)
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
        primals_1, primals_2, primals_3, primals_4, primals_5 = args
        args.clear()
        s4 = primals_1
        s14 = primals_2
        s43 = primals_3
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            primals_4 = copy_if_misaligned(primals_4)
            primals_5 = copy_if_misaligned(primals_5)
            # Topologically Sorted Source Nodes: [value], Original ATen: [aten.convolution]
            # [Provenance debug handles] torch.ops.aten.convolution.default:1
            buf0 = torch.ops.aten.convolution.default(primals_4, primals_5, None, [32, 32], [0, 0], [1, 1], False, [0, 0], 1)
            buf1 = buf0
            assert_size_stride(buf1, (s4, s14, 1 + (((-1) + s43) // 32), 1 + (((-1) + s43) // 32)), (s14 + s14*(((-1) + s43) // 32)*(((-1) + s43) // 32) + 2*s14*(((-1) + s43) // 32), 1 + (((-1) + s43) // 32)*(((-1) + s43) // 32) + 2*(((-1) + s43) // 32), 1 + (((-1) + s43) // 32), 1), 'torch.ops.aten.convolution.default')
            assert_alignment(buf1, 16, 'torch.ops.aten.convolution.default')
            del buf0
            buf2 = empty_strided_npu((s4, s14, 1, 1), (s14, 1, s14*s4, s14*s4), torch.float32)
            buf3 = reinterpret_tensor(buf2, (s4, s14, 1, 1), (s14, 1, 1, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [mean, sigmoid], Original ATen: [aten.mean, aten.sigmoid]
            # [Provenance debug handles] triton_red_fused_mean_sigmoid_0:2
            triton_red_fused_mean_sigmoid_0_xnumel = s14*s4
            triton_red_fused_mean_sigmoid_0_r0_numel = 1 + (((-1) + s43) // 32)*(((-1) + s43) // 32) + 2*(((-1) + s43) // 32)
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_mean_sigmoid_0.run(buf3, buf1, s43, triton_red_fused_mean_sigmoid_0_xnumel, triton_red_fused_mean_sigmoid_0_r0_numel, stream=raw_stream0)
            ps0 = 1 + (((-1) + s43) // 32)*(((-1) + s43) // 32) + 2*(((-1) + s43) // 32)
            buf4 = empty_strided_npu((s4, s14, 1 + (((-1) + s43) // 32), 1 + (((-1) + s43) // 32)), (s14 + s14*(((-1) + s43) // 32)*(((-1) + s43) // 32) + 2*s14*(((-1) + s43) // 32), 1 + (((-1) + s43) // 32)*(((-1) + s43) // 32) + 2*(((-1) + s43) // 32), 1 + (((-1) + s43) // 32), 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
            # [Provenance debug handles] triton_poi_fused_mul_1:3
            triton_poi_fused_mul_1_xnumel = s14*s4 + s14*s4*(((-1) + s43) // 32)*(((-1) + s43) // 32) + 2*s14*s4*(((-1) + s43) // 32)
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_mul_1.run(buf1, buf3, buf4, ps0, triton_poi_fused_mul_1_xnumel, stream=raw_stream0)
        return (buf4, primals_4, primals_5, buf1, buf3, s4, s14, s43, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = 2
    primals_2 = 16
    primals_3 = 224
    primals_4 = rand_strided((2, 16, 224, 224), (802816, 50176, 224, 1), device='npu:0', dtype=torch.float32)
    primals_5 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='npu:0', dtype=torch.float32)
    return [primals_1, primals_2, primals_3, primals_4, primals_5]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
