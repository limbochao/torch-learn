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


# kernel path: /home/l30023782/PTA_213/combo-case-observation-198521f-20261115/cache-inductor/ss/cssvnjsqgqtlfkhil7xq54m5wgnlsdrvcy5znljexatswgdaba3v.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# SchedulerNodes: [ForeachKernelSchedulerNode(nodes=op0_op1)]

triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
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
    size_hints={'y1': 27430, 'y0': 256, 'x2': 44}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'Y1BLOCK': 'i32', 'Y1BLOCK_SUB': 'constexpr', 'Y0BLOCK_SUB': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'backend_hash': '139C9AD8FB0E3F910BF6CA8D4166949EB020778AFDF1D1A0ABA3AF3BC313E710', 'split_axis': [0], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [2], 'axis_names': ['y1', 'y0', 'x2'], 'axis_static_values': (('y1', 4800), ('y0', 256), ('x2', 34)), 'low_dims': {1, 2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('Y1BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': (), 'optimize_mem': True, 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': 0, 'default_config': {'XBLOCK': 'Y1BLOCK'}, 'split_numels': [4800, 22630], 'no_split': [False, False], 'split_block': 'Y1BLOCK', 'no_x_dim_0': False, 'xnumel_0': 4800, 'no_x_dim_1': False, 'xnumel_1': 22630}, 'combo_subkernel_meta': [{'axis_names': ['y1', 'y0', 'x2'], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [2], 'split_axis': [0], 'low_dims': {1, 2}, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'size_hints': {'y1': 4800, 'y0': 256, 'x2': 34}, 'persistent_reduction': False}, {'axis_names': ['y1', 'y0', 'x2'], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [2], 'split_axis': [0], 'low_dims': {1, 2}, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'size_hints': {'y1': 22630, 'y0': 256, 'x2': 44}, 'persistent_reduction': False}]}
)
@triton.jit
def triton_poi_fused_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, Y1BLOCK, Y1BLOCK_SUB : tl.constexpr, Y0BLOCK_SUB : tl.constexpr):
    pid = tl.program_id(0)
    num_blocks_0 = tl.cdiv(4800, Y1BLOCK)
    num_blocks_1 = num_blocks_0 + tl.cdiv(22630, Y1BLOCK)
    if pid < num_blocks_0:
        pid_offset = pid
        y1_numel = 4800
        y0_numel = 256
        x2_numel = 34
        X2BLOCK_SUB: tl.constexpr = 34
        y1_offset = pid_offset * Y1BLOCK
        base_y1= tl.arange(0, Y1BLOCK_SUB)
        loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
        base_y0= tl.arange(0, Y0BLOCK_SUB)
        loops_y0 = (y0_numel + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
        base_x2= tl.arange(0, X2BLOCK_SUB)
        for loop_y1 in range(loops_y1):
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[:,None,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            for loop_y0 in range(loops_y0):
                y0 = (loop_y0 * Y0BLOCK_SUB) + base_y0[None,None,:]
                y0_mask = y0 < y0_numel
                x2 = base_x2[None,:,None]
                tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 8704*y1), y0_mask & y1_mask)
                tl.store(out_ptr0 + (x2 + 34*y0 + 8704*y1), tmp0, y0_mask & y1_mask)
    elif pid < num_blocks_1:
        pid_offset = pid - num_blocks_0
        y1_numel = 22630
        y0_numel = 256
        x2_numel = 44
        X2BLOCK_SUB: tl.constexpr = 44
        y1_offset = pid_offset * Y1BLOCK
        base_y1= tl.arange(0, Y1BLOCK_SUB)
        loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
        base_y0= tl.arange(0, Y0BLOCK_SUB)
        loops_y0 = (y0_numel + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
        base_x2= tl.arange(0, X2BLOCK_SUB)
        for loop_y1 in range(loops_y1):
            y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[:,None,None]
            y1_mask = y1 < min(Y1BLOCK+y1_offset, y1_numel)
            for loop_y0 in range(loops_y0):
                y0 = (loop_y0 * Y0BLOCK_SUB) + base_y0[None,None,:]
                y0_mask = y0 < y0_numel
                x2 = base_x2[None,:,None]
                tmp1 = tl.load(in_ptr1 + (y0 + 256*x2 + 11264*y1), y0_mask & y1_mask)
                tl.store(out_ptr1 + (x2 + 44*y0 + 11264*y1), tmp1, y0_mask & y1_mask)
    else:
        pass
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
            buf0 = empty_strided_npu((4800, 256, 34), (8704, 34, 1), torch.float32)
            buf1 = empty_strided_npu((22630, 256, 44), (11264, 44, 1), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_0.run(arg0_1, arg1_1, buf0, buf1, stream=raw_stream0)
            del arg0_1
            del arg1_1
        return (reinterpret_tensor(buf0, (1228800, 34), (34, 1), 0), reinterpret_tensor(buf1, (5793280, 44), (44, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((4800, 34, 256), (8704, 256, 1), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((22630, 44, 256), (11264, 256, 1), device='npu:0', dtype=torch.float32)
    return [arg0_1, arg1_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='npu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))

