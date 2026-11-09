# PR !48448; exact failing pre-PR DSL captured on 2026-10-29.
# Source: artifacts/npu-inductor-boundary-20261028/pr48448-split-with-sizes/before/output_code.py
# Original wrapper SHA256: c21c909eeac668eb16ec257968ef22982fdf38a3521dbd0cef935ba46ac80742
# Exact DSL string SHA256: 46c48ca300360ef02be5b5c167e125d187febf0590c0751d6cf9d8bc845e5678
# r0_numel=7 for sizes=(7,16,9); r0_mask is calculated but omitted from load and sum.
# Evidence only: original definition/decorators/metadata, without a launch harness.

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

@triton_heuristics.persistent_reduction(
    size_hints={'r0': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_split_with_sizes_sum_0', 'mutated_arg_names': [], 'backend_hash': '<redacted>', 'split_axis': [], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['r0'], 'axis_static_values': (), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': None, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': (), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_split_with_sizes_sum_0(in_ptr0, out_ptr0, r0_numel, R0BLOCK_SUB : tl.constexpr):
    base_r0= tl.arange(0, R0BLOCK_SUB)
    loops_r0 = (r0_numel + R0BLOCK_SUB - 1) // R0BLOCK_SUB
    r0 = base_r0
    r0_mask = r0 < r0_numel
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp2 = triton_helpers.promote_to_tensor(tl.sum(tmp0, 0))
    tl.store(out_ptr0 + (tl.arange(0,1) ), tmp2, None)
