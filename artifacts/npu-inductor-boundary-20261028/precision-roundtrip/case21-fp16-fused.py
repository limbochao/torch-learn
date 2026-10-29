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
from torch._C._dynamo.guards import copy_if_misaligned
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
import torch_npu
empty_strided_npu = torch_npu._C._empty_strided_npu


# kernel path: /tmp/tmpi17w3qbp/22/c22hptprpocdlv6s3zm2c2mbo77jbxemc2pxe2anrooecsacjpiw.py
# Topologically Sorted Source Nodes: [addmm, relu, sub], Original ATen: [aten.addmm, aten.relu, aten.sub]
# Source node to ATen node mapping:
#   addmm => addmm
#   relu => relu
#   sub => sub
# Graph fragment:
#   %arg0_1 : Tensor "f16[128][1]npu:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f16[128, 128][128, 1]npu:0" = PlaceHolder[target=arg1_1]
#   %addmm : Tensor "f16[128, 128][128, 1]npu:0" = PlaceHolder[target=addmm]
#   %addmm : Tensor "f16[128, 128][128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%arg0_1, %arg1_1, %arg1_1), kwargs = {})
#   %relu : Tensor "f16[128, 128][128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%addmm,), kwargs = {})
#   %sub : Tensor "f16[128, 128][128, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%relu, 1.0), kwargs = {})
#   return %addmm,%sub
# SchedulerNodes: [SchedulerNode(name='op0'), SchedulerNode(name='op1')]

triton_tem_fused_addmm_relu_sub_0 = async_compile.triton('triton_tem_fused_addmm_relu_sub_0', '''
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

@triton_heuristics.template(
    num_stages=2,
    num_warps=4,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'arg_B': '*fp16', 'out_ptr1': '*fp16'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'kernel_name': 'triton_tem_fused_addmm_relu_sub_0', 'backend_hash': '<redacted>', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'BLOCK_M': 128, 'BLOCK_N': 128},
)
@triton.jit
def triton_tem_fused_addmm_relu_sub_0(in_ptr0, arg_B, out_ptr1):
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 1
    ALLOW_HF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    MASKED_M : tl.constexpr = False
    MASKED_N : tl.constexpr = False
    MASKED_K : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_B
    B = arg_B

    M = 128
    N = 128
    K = 128
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 128
    stride_ak = 1
    stride_bk = 128
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    # Use super-grouping (GROUP_M) for improved L2 cache hit rate
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)




    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)


        a = tl.load(A + (rm[:, None] * stride_am + offs_k[None, :] * stride_ak))


        b = tl.load(B + (offs_k[:, None] * stride_bk + rn[None, :] * stride_bn))

        acc += tl.dot(a, b, out_dtype=ACC_TYPE)

    # rm, rn, and m_mask are reused from above (no rematerialization).
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 128*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tmp2 = tmp1.to(tl.float16)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = triton_helpers.maximum(tmp4, tmp3)
    tmp6 = tl.full([1], 1.0, tl.float32)
    tmp7 = tmp5 - tmp6
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp7, mask)
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp7, mask)
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
        arg0_1, arg1_1 = args
        args.clear()
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg1_1 = copy_if_misaligned(arg1_1)
            buf1 = empty_strided_npu((128, 128), (128, 1), torch.float16)
            # Topologically Sorted Source Nodes: [addmm, relu, sub], Original ATen: [aten.addmm, aten.relu, aten.sub]
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused_addmm_relu_sub_0.run(arg0_1, arg1_1, buf1, 1, 1, 1, stream=raw_stream0)
            del arg0_1
            del arg1_1
        return (buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float16)
    arg1_1 = rand_strided((128, 128), (128, 1), device='npu:0', dtype=torch.float16)
    return [arg0_1, arg1_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
