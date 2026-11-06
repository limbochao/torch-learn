
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
    size_hints={'z0': 8, 'y1': 2, 'x2': 5}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'Z0BLOCK': 'i32', 'Z0BLOCK_SUB': 'constexpr', 'Y1BLOCK_SUB': 'constexpr', 'X2BLOCK_SUB': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': [], 'backend_hash': '<removed>', 'split_axis': [0], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'axis_names': ['z0', 'y1', 'x2'], 'axis_static_values': (('z0', 8), ('y1', 2), ('x2', 5)), 'low_dims': {2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float16, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('Z0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': False, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': (), 'combo_grid_meta': {'num_kernels': 1, 'min_blocks': 0, 'default_config': {'XBLOCK': 'Z0BLOCK'}, 'split_numels': [8], 'no_split': [False], 'split_block': 'Z0BLOCK', 'no_x_dim_0': False, 'xnumel_0': 8}, 'combo_subkernel_meta': [{'axis_names': ['z0', 'y1', 'x2'], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [], 'split_axis': [0], 'low_dims': {2}, 'split_axis_dtype': torch.float16, 'dual_reduction': False, 'npu_kernel_type': 'simd_simt_mix', 'size_hints': {'z0': 8, 'y1': 2, 'x2': 5}, 'persistent_reduction': False}]}
)
@triton.jit
def Placeholder.KERNEL_NAME(in_ptr0, out_ptr0, Z0BLOCK, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr):
    pid = tl.program_id(0)
    num_blocks_0 = tl.cdiv(8, Z0BLOCK)
    if pid < num_blocks_0:
        pid_offset = pid
        z0_numel = 8
        y1_numel = 2
        x2_numel = 5
        z0_offset = pid_offset * Z0BLOCK
        base_z0= tl.arange(0, Z0BLOCK_SUB)
        loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
        base_y1= tl.arange(0, Y1BLOCK_SUB)
        loops_y1 = (y1_numel + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
        base_x2= tl.arange(0, X2BLOCK_SUB)
        loops_x2 = (x2_numel + X2BLOCK_SUB - 1) // X2BLOCK_SUB
        for loop_z0 in range(loops_z0):
            z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None]
            z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
            for loop_y1 in range(loops_y1):
                y1 = (loop_y1 * Y1BLOCK_SUB) + base_y1[None,:,None]
                y1_mask = y1 < y1_numel
                for loop_x2 in range(loops_x2):
                    x2 = (loop_x2 * X2BLOCK_SUB) + base_x2[None,None,:]
                    x2_mask = x2 < x2_numel
                    tmp0 = 3*y1
                    tl.static_assert(tmp0.dtype == tl.int32)
                    tl.static_assert(tmp0.dtype == tl.int32)
                    tmp1 = 3 + 3*y1
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tmp2 = tmp0 < tmp1
                    tl.static_assert(tmp2.dtype == tl.int1)
                    tl.static_assert(tmp2.dtype == tl.int1)
                    tmp3 = (6*x2) // 5
                    tl.static_assert(tmp3.dtype == tl.int32)
                    tl.static_assert(tmp3.dtype == tl.int32)
                    tmp4 = 2 + ((6*x2) // 5)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tmp5 = tmp3 < tmp4
                    tl.static_assert(tmp5.dtype == tl.int1)
                    tl.static_assert(tmp5.dtype == tl.int1)
                    tmp6 = tmp2 & tmp5
                    tl.static_assert(tmp6.dtype == tl.int1)
                    tl.static_assert(tmp6.dtype == tl.int1)
                    tmp7 = tmp6 & x2_mask
                    tl.static_assert(tmp7.dtype == tl.int1)
                    tl.static_assert(tmp7.dtype == tl.int1)
                    tmp8 = tmp7 & y1_mask
                    tl.static_assert(tmp8.dtype == tl.int1)
                    tl.static_assert(tmp8.dtype == tl.int1)
                    tmp9 = tl.load(in_ptr0 + (18*y1 + 36*z0 + ((6*x2) // 5)), tmp8 & z0_mask, other=0.0).to(tl.float32)
                    tl.static_assert(tmp9.dtype == tl.float32)
                    tmp10 = 1.0
                    tl.static_assert(tmp10.dtype == tl.float32)
                    tl.static_assert(tmp10.dtype == tl.float32)
                    tmp11 = tmp9 + tmp10
                    tl.static_assert(tmp11.dtype == tl.float32)
                    tl.static_assert(tmp11.dtype == tl.float32)
                    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
                    tl.static_assert(tmp12.dtype == tl.float32)
                    tmp13 = tl.where(tmp8, tmp11, tmp12)
                    tl.static_assert(tmp13.dtype == tl.float32)
                    tl.static_assert(tmp13.dtype == tl.float32)
                    tl.static_assert(tmp13.dtype == tl.float32)
                    tl.static_assert(tmp0.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp2.dtype == tl.int1)
                    tmp14 = 1 + ((6*x2) // 5)
                    tl.static_assert(tmp14.dtype == tl.int32)
                    tl.static_assert(tmp14.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tmp15 = tmp14 < tmp4
                    tl.static_assert(tmp15.dtype == tl.int1)
                    tl.static_assert(tmp15.dtype == tl.int1)
                    tmp16 = tmp2 & tmp15
                    tl.static_assert(tmp16.dtype == tl.int1)
                    tl.static_assert(tmp16.dtype == tl.int1)
                    tmp17 = tmp16 & x2_mask
                    tl.static_assert(tmp17.dtype == tl.int1)
                    tl.static_assert(tmp17.dtype == tl.int1)
                    tmp18 = tmp17 & y1_mask
                    tl.static_assert(tmp18.dtype == tl.int1)
                    tl.static_assert(tmp18.dtype == tl.int1)
                    tmp19 = tl.load(in_ptr0 + (1 + 18*y1 + 36*z0 + ((6*x2) // 5)), tmp18 & z0_mask, other=0.0).to(tl.float32)
                    tl.static_assert(tmp19.dtype == tl.float32)
                    tmp20 = 1.0
                    tl.static_assert(tmp20.dtype == tl.float32)
                    tl.static_assert(tmp20.dtype == tl.float32)
                    tmp21 = tmp19 + tmp20
                    tl.static_assert(tmp21.dtype == tl.float32)
                    tl.static_assert(tmp21.dtype == tl.float32)
                    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
                    tl.static_assert(tmp22.dtype == tl.float32)
                    tmp23 = tl.where(tmp18, tmp21, tmp22)
                    tl.static_assert(tmp23.dtype == tl.float32)
                    tl.static_assert(tmp23.dtype == tl.float32)
                    tl.static_assert(tmp23.dtype == tl.float32)
                    tmp24 = tmp23 + tmp13
                    tl.static_assert(tmp24.dtype == tl.float32)
                    tl.static_assert(tmp24.dtype == tl.float32)
                    tmp25 = 1 + 3*y1
                    tl.static_assert(tmp25.dtype == tl.int32)
                    tl.static_assert(tmp25.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tmp26 = tmp25 < tmp1
                    tl.static_assert(tmp26.dtype == tl.int1)
                    tl.static_assert(tmp26.dtype == tl.int1)
                    tl.static_assert(tmp3.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp5.dtype == tl.int1)
                    tmp27 = tmp26 & tmp5
                    tl.static_assert(tmp27.dtype == tl.int1)
                    tl.static_assert(tmp27.dtype == tl.int1)
                    tmp28 = tmp27 & x2_mask
                    tl.static_assert(tmp28.dtype == tl.int1)
                    tl.static_assert(tmp28.dtype == tl.int1)
                    tmp29 = tmp28 & y1_mask
                    tl.static_assert(tmp29.dtype == tl.int1)
                    tl.static_assert(tmp29.dtype == tl.int1)
                    tmp30 = tl.load(in_ptr0 + (6 + 18*y1 + 36*z0 + ((6*x2) // 5)), tmp29 & z0_mask, other=0.0).to(tl.float32)
                    tl.static_assert(tmp30.dtype == tl.float32)
                    tmp31 = 1.0
                    tl.static_assert(tmp31.dtype == tl.float32)
                    tl.static_assert(tmp31.dtype == tl.float32)
                    tmp32 = tmp30 + tmp31
                    tl.static_assert(tmp32.dtype == tl.float32)
                    tl.static_assert(tmp32.dtype == tl.float32)
                    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
                    tl.static_assert(tmp33.dtype == tl.float32)
                    tmp34 = tl.where(tmp29, tmp32, tmp33)
                    tl.static_assert(tmp34.dtype == tl.float32)
                    tl.static_assert(tmp34.dtype == tl.float32)
                    tl.static_assert(tmp34.dtype == tl.float32)
                    tmp35 = tmp34 + tmp24
                    tl.static_assert(tmp35.dtype == tl.float32)
                    tl.static_assert(tmp35.dtype == tl.float32)
                    tl.static_assert(tmp25.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp26.dtype == tl.int1)
                    tl.static_assert(tmp14.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp15.dtype == tl.int1)
                    tmp36 = tmp26 & tmp15
                    tl.static_assert(tmp36.dtype == tl.int1)
                    tl.static_assert(tmp36.dtype == tl.int1)
                    tmp37 = tmp36 & x2_mask
                    tl.static_assert(tmp37.dtype == tl.int1)
                    tl.static_assert(tmp37.dtype == tl.int1)
                    tmp38 = tmp37 & y1_mask
                    tl.static_assert(tmp38.dtype == tl.int1)
                    tl.static_assert(tmp38.dtype == tl.int1)
                    tmp39 = tl.load(in_ptr0 + (7 + 18*y1 + 36*z0 + ((6*x2) // 5)), tmp38 & z0_mask, other=0.0).to(tl.float32)
                    tl.static_assert(tmp39.dtype == tl.float32)
                    tmp40 = 1.0
                    tl.static_assert(tmp40.dtype == tl.float32)
                    tl.static_assert(tmp40.dtype == tl.float32)
                    tmp41 = tmp39 + tmp40
                    tl.static_assert(tmp41.dtype == tl.float32)
                    tl.static_assert(tmp41.dtype == tl.float32)
                    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
                    tl.static_assert(tmp42.dtype == tl.float32)
                    tmp43 = tl.where(tmp38, tmp41, tmp42)
                    tl.static_assert(tmp43.dtype == tl.float32)
                    tl.static_assert(tmp43.dtype == tl.float32)
                    tl.static_assert(tmp43.dtype == tl.float32)
                    tmp44 = tmp43 + tmp35
                    tl.static_assert(tmp44.dtype == tl.float32)
                    tl.static_assert(tmp44.dtype == tl.float32)
                    tmp45 = 2 + 3*y1
                    tl.static_assert(tmp45.dtype == tl.int32)
                    tl.static_assert(tmp45.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tmp46 = tmp45 < tmp1
                    tl.static_assert(tmp46.dtype == tl.int1)
                    tl.static_assert(tmp46.dtype == tl.int1)
                    tl.static_assert(tmp3.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp5.dtype == tl.int1)
                    tmp47 = tmp46 & tmp5
                    tl.static_assert(tmp47.dtype == tl.int1)
                    tl.static_assert(tmp47.dtype == tl.int1)
                    tmp48 = tmp47 & x2_mask
                    tl.static_assert(tmp48.dtype == tl.int1)
                    tl.static_assert(tmp48.dtype == tl.int1)
                    tmp49 = tmp48 & y1_mask
                    tl.static_assert(tmp49.dtype == tl.int1)
                    tl.static_assert(tmp49.dtype == tl.int1)
                    tmp50 = tl.load(in_ptr0 + (12 + 18*y1 + 36*z0 + ((6*x2) // 5)), tmp49 & z0_mask, other=0.0).to(tl.float32)
                    tl.static_assert(tmp50.dtype == tl.float32)
                    tmp51 = 1.0
                    tl.static_assert(tmp51.dtype == tl.float32)
                    tl.static_assert(tmp51.dtype == tl.float32)
                    tmp52 = tmp50 + tmp51
                    tl.static_assert(tmp52.dtype == tl.float32)
                    tl.static_assert(tmp52.dtype == tl.float32)
                    tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
                    tl.static_assert(tmp53.dtype == tl.float32)
                    tmp54 = tl.where(tmp49, tmp52, tmp53)
                    tl.static_assert(tmp54.dtype == tl.float32)
                    tl.static_assert(tmp54.dtype == tl.float32)
                    tl.static_assert(tmp54.dtype == tl.float32)
                    tmp55 = tmp54 + tmp44
                    tl.static_assert(tmp55.dtype == tl.float32)
                    tl.static_assert(tmp55.dtype == tl.float32)
                    tl.static_assert(tmp45.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp46.dtype == tl.int1)
                    tl.static_assert(tmp14.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp15.dtype == tl.int1)
                    tmp56 = tmp46 & tmp15
                    tl.static_assert(tmp56.dtype == tl.int1)
                    tl.static_assert(tmp56.dtype == tl.int1)
                    tmp57 = tmp56 & x2_mask
                    tl.static_assert(tmp57.dtype == tl.int1)
                    tl.static_assert(tmp57.dtype == tl.int1)
                    tmp58 = tmp57 & y1_mask
                    tl.static_assert(tmp58.dtype == tl.int1)
                    tl.static_assert(tmp58.dtype == tl.int1)
                    tmp59 = tl.load(in_ptr0 + (13 + 18*y1 + 36*z0 + ((6*x2) // 5)), tmp58 & z0_mask, other=0.0).to(tl.float32)
                    tl.static_assert(tmp59.dtype == tl.float32)
                    tmp60 = 1.0
                    tl.static_assert(tmp60.dtype == tl.float32)
                    tl.static_assert(tmp60.dtype == tl.float32)
                    tmp61 = tmp59 + tmp60
                    tl.static_assert(tmp61.dtype == tl.float32)
                    tl.static_assert(tmp61.dtype == tl.float32)
                    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
                    tl.static_assert(tmp62.dtype == tl.float32)
                    tmp63 = tl.where(tmp58, tmp61, tmp62)
                    tl.static_assert(tmp63.dtype == tl.float32)
                    tl.static_assert(tmp63.dtype == tl.float32)
                    tl.static_assert(tmp63.dtype == tl.float32)
                    tmp64 = tmp63 + tmp55
                    tl.static_assert(tmp64.dtype == tl.float32)
                    tl.static_assert(tmp64.dtype == tl.float32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tmp65 = tmp1 < tmp1
                    tl.static_assert(tmp65.dtype == tl.int1)
                    tl.static_assert(tmp65.dtype == tl.int1)
                    tl.static_assert(tmp3.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp5.dtype == tl.int1)
                    tmp66 = tmp65 & tmp5
                    tl.static_assert(tmp66.dtype == tl.int1)
                    tl.static_assert(tmp66.dtype == tl.int1)
                    tmp67 = tmp66 & x2_mask
                    tl.static_assert(tmp67.dtype == tl.int1)
                    tl.static_assert(tmp67.dtype == tl.int1)
                    tmp68 = tmp67 & y1_mask
                    tl.static_assert(tmp68.dtype == tl.int1)
                    tl.static_assert(tmp68.dtype == tl.int1)
                    tmp69 = tl.load(in_ptr0 + (18 + 18*y1 + 36*z0 + ((6*x2) // 5)), tmp68 & z0_mask, other=0.0).to(tl.float32)
                    tl.static_assert(tmp69.dtype == tl.float32)
                    tmp70 = 1.0
                    tl.static_assert(tmp70.dtype == tl.float32)
                    tl.static_assert(tmp70.dtype == tl.float32)
                    tmp71 = tmp69 + tmp70
                    tl.static_assert(tmp71.dtype == tl.float32)
                    tl.static_assert(tmp71.dtype == tl.float32)
                    tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
                    tl.static_assert(tmp72.dtype == tl.float32)
                    tmp73 = tl.where(tmp68, tmp71, tmp72)
                    tl.static_assert(tmp73.dtype == tl.float32)
                    tl.static_assert(tmp73.dtype == tl.float32)
                    tl.static_assert(tmp73.dtype == tl.float32)
                    tmp74 = tmp73 + tmp64
                    tl.static_assert(tmp74.dtype == tl.float32)
                    tl.static_assert(tmp74.dtype == tl.float32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp65.dtype == tl.int1)
                    tl.static_assert(tmp14.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp15.dtype == tl.int1)
                    tmp75 = tmp65 & tmp15
                    tl.static_assert(tmp75.dtype == tl.int1)
                    tl.static_assert(tmp75.dtype == tl.int1)
                    tmp76 = tmp75 & x2_mask
                    tl.static_assert(tmp76.dtype == tl.int1)
                    tl.static_assert(tmp76.dtype == tl.int1)
                    tmp77 = tmp76 & y1_mask
                    tl.static_assert(tmp77.dtype == tl.int1)
                    tl.static_assert(tmp77.dtype == tl.int1)
                    tmp78 = tl.load(in_ptr0 + (19 + 18*y1 + 36*z0 + ((6*x2) // 5)), tmp77 & z0_mask, other=0.0).to(tl.float32)
                    tl.static_assert(tmp78.dtype == tl.float32)
                    tmp79 = 1.0
                    tl.static_assert(tmp79.dtype == tl.float32)
                    tl.static_assert(tmp79.dtype == tl.float32)
                    tmp80 = tmp78 + tmp79
                    tl.static_assert(tmp80.dtype == tl.float32)
                    tl.static_assert(tmp80.dtype == tl.float32)
                    tmp81 = tl.full(tmp80.shape, 0.0, tmp80.dtype)
                    tl.static_assert(tmp81.dtype == tl.float32)
                    tmp82 = tl.where(tmp77, tmp80, tmp81)
                    tl.static_assert(tmp82.dtype == tl.float32)
                    tl.static_assert(tmp82.dtype == tl.float32)
                    tl.static_assert(tmp82.dtype == tl.float32)
                    tmp83 = tmp82 + tmp74
                    tl.static_assert(tmp83.dtype == tl.float32)
                    tl.static_assert(tmp83.dtype == tl.float32)
                    tl.static_assert(tmp0.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp2.dtype == tl.int1)
                    tl.static_assert(tmp3.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp5.dtype == tl.int1)
                    tl.static_assert(tmp6.dtype == tl.int1)
                    tl.static_assert(tmp7.dtype == tl.int1)
                    tl.static_assert(tmp8.dtype == tl.int1)
                    tl.static_assert(tmp10.dtype == tl.float32)
                    tmp84 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
                    tl.static_assert(tmp84.dtype == tl.float32)
                    tmp85 = tl.where(tmp8, tmp10, tmp84)
                    tl.static_assert(tmp85.dtype == tl.float32)
                    tl.static_assert(tmp85.dtype == tl.float32)
                    tl.static_assert(tmp85.dtype == tl.float32)
                    tl.static_assert(tmp0.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp2.dtype == tl.int1)
                    tl.static_assert(tmp14.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp15.dtype == tl.int1)
                    tl.static_assert(tmp16.dtype == tl.int1)
                    tl.static_assert(tmp17.dtype == tl.int1)
                    tl.static_assert(tmp18.dtype == tl.int1)
                    tl.static_assert(tmp20.dtype == tl.float32)
                    tmp86 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
                    tl.static_assert(tmp86.dtype == tl.float32)
                    tmp87 = tl.where(tmp18, tmp20, tmp86)
                    tl.static_assert(tmp87.dtype == tl.float32)
                    tl.static_assert(tmp87.dtype == tl.float32)
                    tl.static_assert(tmp87.dtype == tl.float32)
                    tmp88 = tmp87 + tmp85
                    tl.static_assert(tmp88.dtype == tl.float32)
                    tl.static_assert(tmp88.dtype == tl.float32)
                    tl.static_assert(tmp25.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp26.dtype == tl.int1)
                    tl.static_assert(tmp3.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp5.dtype == tl.int1)
                    tl.static_assert(tmp27.dtype == tl.int1)
                    tl.static_assert(tmp28.dtype == tl.int1)
                    tl.static_assert(tmp29.dtype == tl.int1)
                    tl.static_assert(tmp31.dtype == tl.float32)
                    tmp89 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
                    tl.static_assert(tmp89.dtype == tl.float32)
                    tmp90 = tl.where(tmp29, tmp31, tmp89)
                    tl.static_assert(tmp90.dtype == tl.float32)
                    tl.static_assert(tmp90.dtype == tl.float32)
                    tl.static_assert(tmp90.dtype == tl.float32)
                    tmp91 = tmp90 + tmp88
                    tl.static_assert(tmp91.dtype == tl.float32)
                    tl.static_assert(tmp91.dtype == tl.float32)
                    tl.static_assert(tmp25.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp26.dtype == tl.int1)
                    tl.static_assert(tmp14.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp15.dtype == tl.int1)
                    tl.static_assert(tmp36.dtype == tl.int1)
                    tl.static_assert(tmp37.dtype == tl.int1)
                    tl.static_assert(tmp38.dtype == tl.int1)
                    tl.static_assert(tmp40.dtype == tl.float32)
                    tmp92 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
                    tl.static_assert(tmp92.dtype == tl.float32)
                    tmp93 = tl.where(tmp38, tmp40, tmp92)
                    tl.static_assert(tmp93.dtype == tl.float32)
                    tl.static_assert(tmp93.dtype == tl.float32)
                    tl.static_assert(tmp93.dtype == tl.float32)
                    tmp94 = tmp93 + tmp91
                    tl.static_assert(tmp94.dtype == tl.float32)
                    tl.static_assert(tmp94.dtype == tl.float32)
                    tl.static_assert(tmp45.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp46.dtype == tl.int1)
                    tl.static_assert(tmp3.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp5.dtype == tl.int1)
                    tl.static_assert(tmp47.dtype == tl.int1)
                    tl.static_assert(tmp48.dtype == tl.int1)
                    tl.static_assert(tmp49.dtype == tl.int1)
                    tl.static_assert(tmp51.dtype == tl.float32)
                    tmp95 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
                    tl.static_assert(tmp95.dtype == tl.float32)
                    tmp96 = tl.where(tmp49, tmp51, tmp95)
                    tl.static_assert(tmp96.dtype == tl.float32)
                    tl.static_assert(tmp96.dtype == tl.float32)
                    tl.static_assert(tmp96.dtype == tl.float32)
                    tmp97 = tmp96 + tmp94
                    tl.static_assert(tmp97.dtype == tl.float32)
                    tl.static_assert(tmp97.dtype == tl.float32)
                    tl.static_assert(tmp45.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp46.dtype == tl.int1)
                    tl.static_assert(tmp14.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp15.dtype == tl.int1)
                    tl.static_assert(tmp56.dtype == tl.int1)
                    tl.static_assert(tmp57.dtype == tl.int1)
                    tl.static_assert(tmp58.dtype == tl.int1)
                    tl.static_assert(tmp60.dtype == tl.float32)
                    tmp98 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
                    tl.static_assert(tmp98.dtype == tl.float32)
                    tmp99 = tl.where(tmp58, tmp60, tmp98)
                    tl.static_assert(tmp99.dtype == tl.float32)
                    tl.static_assert(tmp99.dtype == tl.float32)
                    tl.static_assert(tmp99.dtype == tl.float32)
                    tmp100 = tmp99 + tmp97
                    tl.static_assert(tmp100.dtype == tl.float32)
                    tl.static_assert(tmp100.dtype == tl.float32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp65.dtype == tl.int1)
                    tl.static_assert(tmp3.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp5.dtype == tl.int1)
                    tl.static_assert(tmp66.dtype == tl.int1)
                    tl.static_assert(tmp67.dtype == tl.int1)
                    tl.static_assert(tmp68.dtype == tl.int1)
                    tl.static_assert(tmp70.dtype == tl.float32)
                    tmp101 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
                    tl.static_assert(tmp101.dtype == tl.float32)
                    tmp102 = tl.where(tmp68, tmp70, tmp101)
                    tl.static_assert(tmp102.dtype == tl.float32)
                    tl.static_assert(tmp102.dtype == tl.float32)
                    tl.static_assert(tmp102.dtype == tl.float32)
                    tmp103 = tmp102 + tmp100
                    tl.static_assert(tmp103.dtype == tl.float32)
                    tl.static_assert(tmp103.dtype == tl.float32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp1.dtype == tl.int32)
                    tl.static_assert(tmp65.dtype == tl.int1)
                    tl.static_assert(tmp14.dtype == tl.int32)
                    tl.static_assert(tmp4.dtype == tl.int32)
                    tl.static_assert(tmp15.dtype == tl.int1)
                    tl.static_assert(tmp75.dtype == tl.int1)
                    tl.static_assert(tmp76.dtype == tl.int1)
                    tl.static_assert(tmp77.dtype == tl.int1)
                    tl.static_assert(tmp79.dtype == tl.float32)
                    tmp104 = tl.full(tmp79.shape, 0.0, tmp79.dtype)
                    tl.static_assert(tmp104.dtype == tl.float32)
                    tmp105 = tl.where(tmp77, tmp79, tmp104)
                    tl.static_assert(tmp105.dtype == tl.float32)
                    tl.static_assert(tmp105.dtype == tl.float32)
                    tl.static_assert(tmp105.dtype == tl.float32)
                    tmp106 = tmp105 + tmp103
                    tl.static_assert(tmp106.dtype == tl.float32)
                    tl.static_assert(tmp106.dtype == tl.float32)
                    tmp107 = (tmp83 / tmp106)
                    tl.static_assert(tmp107.dtype == tl.float32)
                    tl.static_assert(tmp107.dtype == tl.float32)
                    tl.store(out_ptr0 + (x2 + 5*y1 + 10*z0), tmp107, x2_mask & y1_mask & z0_mask)
    else:
        pass
