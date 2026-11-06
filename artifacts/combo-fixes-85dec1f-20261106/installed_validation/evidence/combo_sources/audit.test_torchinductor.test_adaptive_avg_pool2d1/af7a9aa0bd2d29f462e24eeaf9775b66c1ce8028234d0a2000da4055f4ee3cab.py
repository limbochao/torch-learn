
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
    size_hints={'z0': 8, 'y3': 3, 'y4': 2, 'x5': 3, 'x6': 2}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'Z0BLOCK': 'i32', 'Z0BLOCK_SUB': 'constexpr', 'Y3BLOCK_SUB': 'constexpr', 'Y4BLOCK_SUB': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': [], 'backend_hash': '<removed>', 'split_axis': [0], 'tiling_axis': [0, 1, 2, 3, 4], 'no_loop_axis': [3, 4], 'axis_names': ['z0', 'y3', 'y4', 'x5', 'x6'], 'axis_static_values': (('z0', 8), ('y3', 3), ('y4', 2), ('x5', 3), ('x6', 2)), 'low_dims': {4}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('Z0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': (), 'combo_grid_meta': {'num_kernels': 1, 'min_blocks': 0, 'default_config': {'XBLOCK': 'Z0BLOCK'}, 'split_numels': [8], 'no_split': [False], 'split_block': 'Z0BLOCK', 'no_x_dim_0': False, 'xnumel_0': 8}, 'combo_subkernel_meta': [{'axis_names': ['z0', 'y3', 'y4', 'x5', 'x6'], 'tiling_axis': [0, 1, 2, 3, 4], 'no_loop_axis': [3, 4], 'split_axis': [0], 'low_dims': {4}, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'size_hints': {'z0': 8, 'y3': 3, 'y4': 2, 'x5': 3, 'x6': 2}, 'persistent_reduction': False}]}
)
@triton.jit
def Placeholder.KERNEL_NAME(in_ptr0, out_ptr0, Z0BLOCK, Z0BLOCK_SUB : tl.constexpr, Y3BLOCK_SUB : tl.constexpr, Y4BLOCK_SUB : tl.constexpr):
    pid = tl.program_id(0)
    num_blocks_0 = tl.cdiv(8, Z0BLOCK)
    if pid < num_blocks_0:
        pid_offset = pid
        z0_numel = 8
        y3_numel = 3
        y4_numel = 2
        x5_numel = 3
        X5BLOCK_SUB: tl.constexpr = 4
        x6_numel = 2
        X6BLOCK_SUB: tl.constexpr = 2
        z0_offset = pid_offset * Z0BLOCK
        base_z0= tl.arange(0, Z0BLOCK_SUB)
        loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
        base_y3= tl.arange(0, Y3BLOCK_SUB)
        loops_y3 = (y3_numel + Y3BLOCK_SUB - 1) // Y3BLOCK_SUB
        base_y4= tl.arange(0, Y4BLOCK_SUB)
        loops_y4 = (y4_numel + Y4BLOCK_SUB - 1) // Y4BLOCK_SUB
        base_x5= tl.arange(0, X5BLOCK_SUB)
        base_x6= tl.arange(0, X6BLOCK_SUB)
        for loop_z0 in range(loops_z0):
            z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None,None,None]
            z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
            for loop_y3 in range(loops_y3):
                y3 = (loop_y3 * Y3BLOCK_SUB) + base_y3[None,:,None,None,None]
                y3_mask = y3 < y3_numel
                for loop_y4 in range(loops_y4):
                    y4 = (loop_y4 * Y4BLOCK_SUB) + base_y4[None,None,:,None,None]
                    y4_mask = y4 < y4_numel
                    x5 = base_x5[None,None,None,:,None]
                    x5_mask = x5 < x5_numel
                    x6 = base_x6[None,None,None,None,:]
                    x6_mask = x6 < x6_numel
                    tmp0 = y3
                    tl.static_assert(tmp0.dtype == tl.int32)
                    tl.static_assert(tmp0.dtype == tl.int32)
                    tmp1 = y3 + ((8 + 3*y4) // 6)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tmp2 = tmp0 < tmp1
                    tl.static_assert(tmp2.dtype == tl.int1)
                    tl.static_assert(tmp2.dtype == tl.int1)
                    tmp3 = x5
                    tl.static_assert(tmp3.dtype == tl.int32)
                    tl.static_assert(tmp3.dtype == tl.int32)
                    tmp4 = x5 + ((8 + 3*x6) // 6)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tmp5 = tmp3 < tmp4
                    tl.static_assert(tmp5.dtype == tl.int1)
                    tl.static_assert(tmp5.dtype == tl.int1)
                    tmp6 = tmp2 & tmp5
                    tl.static_assert(tmp6.dtype == tl.int1)
                    tl.static_assert(tmp6.dtype == tl.int1)
                    tmp7 = tmp6 & x5_mask
                    tl.static_assert(tmp7.dtype == tl.int1)
                    tl.static_assert(tmp7.dtype == tl.int1)
                    tmp8 = tmp7 & x6_mask
                    tl.static_assert(tmp8.dtype == tl.int1)
                    tl.static_assert(tmp8.dtype == tl.int1)
                    tmp9 = tmp8 & y4_mask
                    tl.static_assert(tmp9.dtype == tl.int1)
                    tl.static_assert(tmp9.dtype == tl.int1)
                    tmp10 = tmp9 & y3_mask
                    tl.static_assert(tmp10.dtype == tl.int1)
                    tl.static_assert(tmp10.dtype == tl.int1)
                    tmp11 = tl.load(in_ptr0 + (x5 + 3*y3 + 9*z0), tmp10 & x5_mask & y3_mask & z0_mask, other=0.0)
                    tl.static_assert(tmp11.dtype == tl.float32)
                    tl.static_assert(tmp11.dtype == tl.float32)
                    tl.static_assert(tmp0.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp2.dtype == tl.int1)
                    tmp12 = 1 + x5
                    tl.static_assert(tmp12.dtype == tl.int32)
                    tl.static_assert(tmp12.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tmp13 = tmp12 < tmp4
                    tl.static_assert(tmp13.dtype == tl.int1)
                    tl.static_assert(tmp13.dtype == tl.int1)
                    tmp14 = tmp2 & tmp13
                    tl.static_assert(tmp14.dtype == tl.int1)
                    tl.static_assert(tmp14.dtype == tl.int1)
                    tmp15 = tmp14 & x5_mask
                    tl.static_assert(tmp15.dtype == tl.int1)
                    tl.static_assert(tmp15.dtype == tl.int1)
                    tmp16 = tmp15 & x6_mask
                    tl.static_assert(tmp16.dtype == tl.int1)
                    tl.static_assert(tmp16.dtype == tl.int1)
                    tmp17 = tmp16 & y4_mask
                    tl.static_assert(tmp17.dtype == tl.int1)
                    tl.static_assert(tmp17.dtype == tl.int1)
                    tmp18 = tmp17 & y3_mask
                    tl.static_assert(tmp18.dtype == tl.int1)
                    tl.static_assert(tmp18.dtype == tl.int1)
                    tmp19 = tl.load(in_ptr0 + (1 + x5 + 3*y3 + 9*z0), tmp18 & x5_mask & y3_mask & z0_mask, other=0.0)
                    tl.static_assert(tmp19.dtype == tl.float32)
                    tl.static_assert(tmp19.dtype == tl.float32)
                    tmp20 = tmp19 + tmp11
                    tl.static_assert(tmp20.dtype == tl.float32)
                    tl.static_assert(tmp20.dtype == tl.float32)
                    tmp21 = 1 + y3
                    tl.static_assert(tmp21.dtype == tl.int32)
                    tl.static_assert(tmp21.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tmp22 = tmp21 < tmp1
                    tl.static_assert(tmp22.dtype == tl.int1)
                    tl.static_assert(tmp22.dtype == tl.int1)
                    tl.static_assert(tmp3.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp5.dtype == tl.int1)
                    tmp23 = tmp22 & tmp5
                    tl.static_assert(tmp23.dtype == tl.int1)
                    tl.static_assert(tmp23.dtype == tl.int1)
                    tmp24 = tmp23 & x5_mask
                    tl.static_assert(tmp24.dtype == tl.int1)
                    tl.static_assert(tmp24.dtype == tl.int1)
                    tmp25 = tmp24 & x6_mask
                    tl.static_assert(tmp25.dtype == tl.int1)
                    tl.static_assert(tmp25.dtype == tl.int1)
                    tmp26 = tmp25 & y4_mask
                    tl.static_assert(tmp26.dtype == tl.int1)
                    tl.static_assert(tmp26.dtype == tl.int1)
                    tmp27 = tmp26 & y3_mask
                    tl.static_assert(tmp27.dtype == tl.int1)
                    tl.static_assert(tmp27.dtype == tl.int1)
                    tmp28 = tl.load(in_ptr0 + (3 + x5 + 3*y3 + 9*z0), tmp27 & x5_mask & y3_mask & z0_mask, other=0.0)
                    tl.static_assert(tmp28.dtype == tl.float32)
                    tl.static_assert(tmp28.dtype == tl.float32)
                    tmp29 = tmp28 + tmp20
                    tl.static_assert(tmp29.dtype == tl.float32)
                    tl.static_assert(tmp29.dtype == tl.float32)
                    tl.static_assert(tmp21.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp22.dtype == tl.int1)
                    tl.static_assert(tmp12.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp13.dtype == tl.int1)
                    tmp30 = tmp22 & tmp13
                    tl.static_assert(tmp30.dtype == tl.int1)
                    tl.static_assert(tmp30.dtype == tl.int1)
                    tmp31 = tmp30 & x5_mask
                    tl.static_assert(tmp31.dtype == tl.int1)
                    tl.static_assert(tmp31.dtype == tl.int1)
                    tmp32 = tmp31 & x6_mask
                    tl.static_assert(tmp32.dtype == tl.int1)
                    tl.static_assert(tmp32.dtype == tl.int1)
                    tmp33 = tmp32 & y4_mask
                    tl.static_assert(tmp33.dtype == tl.int1)
                    tl.static_assert(tmp33.dtype == tl.int1)
                    tmp34 = tmp33 & y3_mask
                    tl.static_assert(tmp34.dtype == tl.int1)
                    tl.static_assert(tmp34.dtype == tl.int1)
                    tmp35 = tl.load(in_ptr0 + (4 + x5 + 3*y3 + 9*z0), tmp34 & x5_mask & y3_mask & z0_mask, other=0.0)
                    tl.static_assert(tmp35.dtype == tl.float32)
                    tl.static_assert(tmp35.dtype == tl.float32)
                    tmp36 = tmp35 + tmp29
                    tl.static_assert(tmp36.dtype == tl.float32)
                    tl.static_assert(tmp36.dtype == tl.float32)
                    tl.static_assert(tmp0.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp2.dtype == tl.int1)
                    tl.static_assert(tmp3.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp5.dtype == tl.int1)
                    tl.static_assert(tmp6.dtype == tl.int1)
                    tl.static_assert(tmp7.dtype == tl.int1)
                    tl.static_assert(tmp8.dtype == tl.int1)
                    tl.static_assert(tmp9.dtype == tl.int1)
                    tl.static_assert(tmp10.dtype == tl.int1)
                    tmp37 = 1.0
                    tl.static_assert(tmp37.dtype == tl.float32)
                    tl.static_assert(tmp37.dtype == tl.float32)
                    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
                    tl.static_assert(tmp38.dtype == tl.float32)
                    tmp39 = tl.where(tmp10, tmp37, tmp38)
                    tl.static_assert(tmp39.dtype == tl.float32)
                    tl.static_assert(tmp39.dtype == tl.float32)
                    tl.static_assert(tmp39.dtype == tl.float32)
                    tl.static_assert(tmp0.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp2.dtype == tl.int1)
                    tl.static_assert(tmp12.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp13.dtype == tl.int1)
                    tl.static_assert(tmp14.dtype == tl.int1)
                    tl.static_assert(tmp15.dtype == tl.int1)
                    tl.static_assert(tmp16.dtype == tl.int1)
                    tl.static_assert(tmp17.dtype == tl.int1)
                    tl.static_assert(tmp18.dtype == tl.int1)
                    tmp40 = 1.0
                    tl.static_assert(tmp40.dtype == tl.float32)
                    tl.static_assert(tmp40.dtype == tl.float32)
                    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
                    tl.static_assert(tmp41.dtype == tl.float32)
                    tmp42 = tl.where(tmp18, tmp40, tmp41)
                    tl.static_assert(tmp42.dtype == tl.float32)
                    tl.static_assert(tmp42.dtype == tl.float32)
                    tl.static_assert(tmp42.dtype == tl.float32)
                    tmp43 = tmp42 + tmp39
                    tl.static_assert(tmp43.dtype == tl.float32)
                    tl.static_assert(tmp43.dtype == tl.float32)
                    tl.static_assert(tmp21.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp22.dtype == tl.int1)
                    tl.static_assert(tmp3.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp5.dtype == tl.int1)
                    tl.static_assert(tmp23.dtype == tl.int1)
                    tl.static_assert(tmp24.dtype == tl.int1)
                    tl.static_assert(tmp25.dtype == tl.int1)
                    tl.static_assert(tmp26.dtype == tl.int1)
                    tl.static_assert(tmp27.dtype == tl.int1)
                    tmp44 = 1.0
                    tl.static_assert(tmp44.dtype == tl.float32)
                    tl.static_assert(tmp44.dtype == tl.float32)
                    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
                    tl.static_assert(tmp45.dtype == tl.float32)
                    tmp46 = tl.where(tmp27, tmp44, tmp45)
                    tl.static_assert(tmp46.dtype == tl.float32)
                    tl.static_assert(tmp46.dtype == tl.float32)
                    tl.static_assert(tmp46.dtype == tl.float32)
                    tmp47 = tmp46 + tmp43
                    tl.static_assert(tmp47.dtype == tl.float32)
                    tl.static_assert(tmp47.dtype == tl.float32)
                    tl.static_assert(tmp21.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp22.dtype == tl.int1)
                    tl.static_assert(tmp12.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp13.dtype == tl.int1)
                    tl.static_assert(tmp30.dtype == tl.int1)
                    tl.static_assert(tmp31.dtype == tl.int1)
                    tl.static_assert(tmp32.dtype == tl.int1)
                    tl.static_assert(tmp33.dtype == tl.int1)
                    tl.static_assert(tmp34.dtype == tl.int1)
                    tmp48 = 1.0
                    tl.static_assert(tmp48.dtype == tl.float32)
                    tl.static_assert(tmp48.dtype == tl.float32)
                    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
                    tl.static_assert(tmp49.dtype == tl.float32)
                    tmp50 = tl.where(tmp34, tmp48, tmp49)
                    tl.static_assert(tmp50.dtype == tl.float32)
                    tl.static_assert(tmp50.dtype == tl.float32)
                    tl.static_assert(tmp50.dtype == tl.float32)
                    tmp51 = tmp50 + tmp47
                    tl.static_assert(tmp51.dtype == tl.float32)
                    tl.static_assert(tmp51.dtype == tl.float32)
                    tmp52 = (tmp36 / tmp51)
                    tl.static_assert(tmp52.dtype == tl.float32)
                    tl.static_assert(tmp52.dtype == tl.float32)
                    tl.store(out_ptr0 + (x6 + 2*x5 + 6*y4 + 12*y3 + 36*z0), tmp52, x5_mask & x6_mask & y3_mask & y4_mask & z0_mask)
    else:
        pass
