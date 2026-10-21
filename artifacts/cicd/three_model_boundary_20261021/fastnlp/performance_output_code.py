# AOT ID: ['2_inference']
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


# kernel path: <workspace>/three_model_boundary_20261021/matrix/inductor_cache/fastNLP_Bert/p7/cp75cs4jtrr3t2v2krfonjzsoez5vps6rms63s3h7qkqpypg2fvq.py
# Topologically Sorted Source Nodes: [word_mask, eq, batch_word_pieces_length, getitem_2, sum_3, max_1, seq_len, word_pieces_lengths], Original ATen: [aten.ne, aten.eq, aten.masked_fill, aten.index, aten.sum, aten.max]
# Source node to ATen node mapping:
#   batch_word_pieces_length => full_default, where
#   eq => eq_2
#   getitem_2 => index
#   max_1 => max_1
#   seq_len => sum_1
#   sum_3 => sum_3
#   word_mask => ne
#   word_pieces_lengths => sum_2
# Graph fragment:
#   %arg1_1 : Tensor "i64[1, s4][s4, 1]npu:0" = PlaceHolder[target=arg1_1]
#   %ne : Tensor "b8[1, s4][s4, 1]npu:0" = PlaceHolder[target=ne]
#   %arg2_1 : Tensor "i64[2869][1]npu:0" = PlaceHolder[target=arg2_1]
#   %where : Tensor "i64[1, s4][s4, 1]npu:0" = PlaceHolder[target=where]
#   %sum_3 : Tensor "i64[1][1]npu:0" = PlaceHolder[target=sum_3]
#   %ne : Tensor "b8[1, s4][s4, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.ne.Scalar](args = (%arg1_1, 0), kwargs = {})
#   %eq_2 : Tensor "b8[1, s4][s4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%ne, False), kwargs = {})
#   %full_default : Tensor "i64[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: npu:0, pin_memory: False})
#   %index : Tensor "i64[1, s4][s4, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg2_1, [%arg1_1]), kwargs = {})
#   %where : Tensor "i64[1, s4][s4, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%eq_2, %full_default, %index), kwargs = {})
#   %sum_3 : Tensor "i64[1][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%where, [-1]), kwargs = {})
#   %max_1 : Tensor "i64[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.max.default](args = (%sum_3,), kwargs = {})
#   %sum_1 : Tensor "i64[1][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%ne, [-1]), kwargs = {})
#   %sum_2 : Tensor "i64[1][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%where, [-1]), kwargs = {})
#   return %ne,%where,%sum_3,%sum_2,%sum_1,%max_1
# SchedulerNodes: [SchedulerNode(name='op0'), SchedulerNode(name='op1'), SchedulerNode(name='op2'), SchedulerNode(name='op4'), SchedulerNode(name='op3'), SchedulerNode(name='op5')]

triton_red_fused_eq_index_masked_fill_max_ne_s_0 = async_compile.triton('triton_red_fused_eq_index_masked_fill_max_ne_s_0', '''
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

@triton_heuristics.reduction(
    size_hints={'r0': 473},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*i64', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'out_ptr0': '*i1', 'out_ptr1': '*i64', 'out_ptr2': '*i64', 'out_ptr3': '*i64', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_eq_index_masked_fill_max_ne_s_0', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '<redacted>', 'split_axis': [], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['r0'], 'axis_static_values': (), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': None, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': (), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_red_fused_eq_index_masked_fill_max_ne_s_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, r0_numel, R0BLOCK_SUB : tl.constexpr):
    base_r0= tl.arange(0, R0BLOCK_SUB)
    loops_r0 = (r0_numel + R0BLOCK_SUB - 1) // R0BLOCK_SUB
    _tmp13 = tl.full([R0BLOCK_SUB], 0, tl.int64)
    _tmp17 = tl.full([R0BLOCK_SUB], 0, tl.int64)
    for loop_r0 in range(loops_r0):
        r0 = (loop_r0 * R0BLOCK_SUB) + base_r0
        r0_mask = r0 < r0_numel
        tmp0 = tl.load(in_ptr0 + (r0), r0_mask, other=0.0)
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tl.full([1], False, tl.int1)
        tmp4 = tmp2 == tmp3
        tmp5 = tl.full([R0BLOCK_SUB], 2869, tl.int32)
        tmp6 = tmp0 + tmp5
        tmp7 = tmp0 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp0)
        tl.device_assert(((0 <= tmp8) & (tmp8 < 2869)) | ~(r0_mask), "index out of bounds: 0 <= tmp8 < 2869")
        tmp10 = tl.load(in_ptr1 + (tmp8), r0_mask)
        tmp11 = tl.where(tmp4, tmp1, tmp10)
        tmp12 = tl.reshape(tmp11, [R0BLOCK_SUB])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(tl.reshape((r0_mask), [R0BLOCK_SUB]), tmp14, _tmp13)
        tmp15 = tmp2.to(tl.int64)
        tmp16 = tl.reshape(tmp15, [R0BLOCK_SUB])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(tl.reshape((r0_mask), [R0BLOCK_SUB]), tmp18, _tmp17)
        tl.store(out_ptr0 + (r0), tmp2, r0_mask)
        tl.store(out_ptr1 + (r0), tmp11, r0_mask)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(_tmp13, 0))
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(_tmp17, 0))
    tl.store(out_ptr2 + (tl.arange(0,1) ), tmp13, None)
    tl.store(out_ptr3 + (tl.arange(0,1) ), tmp17, None)
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
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        s4 = arg0_1
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            arg1_1 = copy_if_misaligned(arg1_1)
            buf0 = empty_strided((1, s4), (s4, 1), device='npu', dtype=torch.bool)
            buf1 = empty_strided((1, s4), (s4, 1), device='npu', dtype=torch.int64)
            buf2 = empty_strided((1, ), (1, ), device='npu', dtype=torch.int64)
            buf4 = empty_strided((1, ), (1, ), device='npu', dtype=torch.int64)
            buf3 = empty_strided((1, ), (1, ), device='npu', dtype=torch.int64)
            buf5 = reinterpret_tensor(buf2, (), (), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [word_mask, eq, batch_word_pieces_length, getitem_2, sum_3, max_1, seq_len, word_pieces_lengths], Original ATen: [aten.ne, aten.eq, aten.masked_fill, aten.index, aten.sum, aten.max]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_eq_index_masked_fill_max_ne_s_0.run(buf5, arg1_1, arg2_1, buf0, buf1, buf4, buf3, s4, stream=raw_stream0)
            del arg1_1
            del arg2_1
        return (buf5, s4, buf0, buf3, buf1, buf4, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = 473
    arg1_1 = rand_strided((1, 473), (473, 1), device='npu:0', dtype=torch.int64)
    arg2_1 = rand_strided((2869, ), (1, ), device='npu:0', dtype=torch.int64)
    return [arg0_1, arg1_1, arg2_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('fastNLP_Bert', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
