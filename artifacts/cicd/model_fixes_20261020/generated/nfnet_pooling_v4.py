
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
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_avg_pool2d_convolution_view_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'backend_hash': '<redacted>', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'requires_no_linear_block_remap': True, 'npu_kernel_type': 'simd_simt_mix', 'split_axis': [0], 'tiling_axis': [0], 'low_dims': [0], 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'axis_names': ['x']},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_avg_pool2d_convolution_view_21(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks0)
    x2 = xindex // ks1
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x2 + 2*x0 + 2*x1 + x2*(triton_helpers.div_floor_integer((-1) + ks2,  4))*(triton_helpers.div_floor_integer((-1) + ks2,  4)) + 2*x1*(triton_helpers.div_floor_integer((-1) + ks2,  4)) + 2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  4))), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + x2 + 2*x0 + 2*x1 + x2*(triton_helpers.div_floor_integer((-1) + ks2,  4))*(triton_helpers.div_floor_integer((-1) + ks2,  4)) + 2*x1*(triton_helpers.div_floor_integer((-1) + ks2,  4)) + 2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  4))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + x2 + 2*x0 + 2*x1 + x2*(triton_helpers.div_floor_integer((-1) + ks2,  4))*(triton_helpers.div_floor_integer((-1) + ks2,  4)) + 2*x1*(triton_helpers.div_floor_integer((-1) + ks2,  4)) + 2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  4)) + (triton_helpers.div_floor_integer((-1) + ks2,  4))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + x2 + 2*x0 + 2*x1 + x2*(triton_helpers.div_floor_integer((-1) + ks2,  4))*(triton_helpers.div_floor_integer((-1) + ks2,  4)) + 2*x1*(triton_helpers.div_floor_integer((-1) + ks2,  4)) + 2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  4)) + (triton_helpers.div_floor_integer((-1) + ks2,  4))), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1], 4.0, tl.float32)
    tmp8 = (tmp6 / tmp7)
    tl.store(out_ptr0 + (x3), tmp8, xmask)
