
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
    size_hints={'x0': 4096, 'r1': 2048},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'in_ptr25': '*fp32', 'in_ptr26': '*fp32', 'in_ptr27': '*fp32', 'in_ptr28': '*fp32', 'in_ptr29': '*fp32', 'in_ptr30': '*fp32', 'in_ptr31': '*fp32', 'in_ptr32': '*fp32', 'in_ptr33': '*fp32', 'in_ptr34': '*fp32', 'in_ptr35': '*fp32', 'in_ptr36': '*fp32', 'in_ptr37': '*fp32', 'in_ptr38': '*fp32', 'in_ptr39': '*fp32', 'in_ptr40': '*fp32', 'in_ptr41': '*fp32', 'in_ptr42': '*fp32', 'in_ptr43': '*fp32', 'in_ptr44': '*fp32', 'in_ptr45': '*fp32', 'in_ptr46': '*fp32', 'in_ptr47': '*fp32', 'in_ptr48': '*fp32', 'in_ptr49': '*fp32', 'in_ptr50': '*fp32', 'in_ptr51': '*fp32', 'in_ptr52': '*fp32', 'in_ptr53': '*fp32', 'in_ptr54': '*fp32', 'in_ptr55': '*fp32', 'in_ptr56': '*fp32', 'in_ptr57': '*fp32', 'in_ptr58': '*fp32', 'in_ptr59': '*fp32', 'in_ptr60': '*fp32', 'in_ptr61': '*fp32', 'in_ptr62': '*fp32', 'in_ptr63': '*fp32', 'in_ptr64': '*fp32', 'in_ptr65': '*fp32', 'in_ptr66': '*fp32', 'in_ptr67': '*fp32', 'in_ptr68': '*fp32', 'in_ptr69': '*fp32', 'in_ptr70': '*fp32', 'in_ptr71': '*fp32', 'in_ptr72': '*fp32', 'in_ptr73': '*fp32', 'in_ptr74': '*fp32', 'in_ptr75': '*fp32', 'in_ptr76': '*fp32', 'in_ptr77': '*fp32', 'in_ptr78': '*fp32', 'in_ptr79': '*fp32', 'in_ptr80': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backwar_14', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'backend_hash': '<redacted>', 'split_axis': [0], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['x0', 'r1'], 'axis_static_values': (('x0', 4096), ('r1', 2048)), 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_backwar_14(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 2048
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        r1 = base_r1[:,None]
        r1_mask = True
        tmp0 = tl.load(in_ptr0 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1), None)
        tmp3 = tmp1 - tmp2
        tmp4 = tl.load(in_ptr3 + (r1), None)
        tmp5 = tmp3 * tmp4
        tmp6 = tmp0 * tmp5
        tmp8 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp6, 0)
        tmp9 = tl.sum(tmp8, 0).reshape(1, X0BLOCK_SUB)
        tmp11 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp0, 0)
        tmp12 = tl.sum(tmp11, 0).reshape(1, X0BLOCK_SUB)
        tmp13 = tl.load(in_ptr4 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp14 = tl.load(in_ptr5 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp15 = tmp13 + tmp14
        tmp16 = tl.load(in_ptr6 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp17 = tmp15 + tmp16
        tmp18 = tl.load(in_ptr7 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp19 = tmp17 + tmp18
        tmp20 = tl.load(in_ptr8 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp21 = tl.load(in_ptr9 + (r1), None)
        tmp22 = tmp20 - tmp21
        tmp23 = tl.load(in_ptr10 + (r1), None)
        tmp24 = tmp22 * tmp23
        tmp25 = tmp19 * tmp24
        tmp27 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp25, 0)
        tmp28 = tl.sum(tmp27, 0).reshape(1, X0BLOCK_SUB)
        tmp30 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp19, 0)
        tmp31 = tl.sum(tmp30, 0).reshape(1, X0BLOCK_SUB)
        tmp32 = tl.load(in_ptr11 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp33 = tl.load(in_ptr12 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp34 = tmp32 + tmp33
        tmp35 = tl.load(in_ptr13 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp36 = tmp34 + tmp35
        tmp37 = tl.load(in_ptr14 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp38 = tmp36 + tmp37
        tmp39 = tl.load(in_ptr15 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp40 = tl.load(in_ptr16 + (r1), None)
        tmp41 = tmp39 - tmp40
        tmp42 = tl.load(in_ptr17 + (r1), None)
        tmp43 = tmp41 * tmp42
        tmp44 = tmp38 * tmp43
        tmp46 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp44, 0)
        tmp47 = tl.sum(tmp46, 0).reshape(1, X0BLOCK_SUB)
        tmp49 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp38, 0)
        tmp50 = tl.sum(tmp49, 0).reshape(1, X0BLOCK_SUB)
        tmp51 = tl.load(in_ptr18 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp52 = tl.load(in_ptr19 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp53 = tmp51 + tmp52
        tmp54 = tl.load(in_ptr20 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp55 = tmp53 + tmp54
        tmp56 = tl.load(in_ptr21 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp57 = tmp55 + tmp56
        tmp58 = tl.load(in_ptr22 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp59 = tl.load(in_ptr23 + (r1), None)
        tmp60 = tmp58 - tmp59
        tmp61 = tl.load(in_ptr24 + (r1), None)
        tmp62 = tmp60 * tmp61
        tmp63 = tmp57 * tmp62
        tmp65 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp63, 0)
        tmp66 = tl.sum(tmp65, 0).reshape(1, X0BLOCK_SUB)
        tmp68 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp57, 0)
        tmp69 = tl.sum(tmp68, 0).reshape(1, X0BLOCK_SUB)
        tmp70 = tl.load(in_ptr25 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp71 = tl.load(in_ptr26 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp72 = tmp70 + tmp71
        tmp73 = tl.load(in_ptr27 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp74 = tmp72 + tmp73
        tmp75 = tl.load(in_ptr28 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp76 = tmp74 + tmp75
        tmp77 = tl.load(in_ptr29 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp78 = tl.load(in_ptr30 + (r1), None)
        tmp79 = tmp77 - tmp78
        tmp80 = tl.load(in_ptr31 + (r1), None)
        tmp81 = tmp79 * tmp80
        tmp82 = tmp76 * tmp81
        tmp84 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp82, 0)
        tmp85 = tl.sum(tmp84, 0).reshape(1, X0BLOCK_SUB)
        tmp87 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp76, 0)
        tmp88 = tl.sum(tmp87, 0).reshape(1, X0BLOCK_SUB)
        tmp89 = tl.load(in_ptr32 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp90 = tl.load(in_ptr33 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp91 = tmp89 + tmp90
        tmp92 = tl.load(in_ptr34 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp93 = tmp91 + tmp92
        tmp94 = tl.load(in_ptr35 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp95 = tmp93 + tmp94
        tmp96 = tl.load(in_ptr36 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp97 = tl.load(in_ptr37 + (r1), None)
        tmp98 = tmp96 - tmp97
        tmp99 = tl.load(in_ptr38 + (r1), None)
        tmp100 = tmp98 * tmp99
        tmp101 = tmp95 * tmp100
        tmp103 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp101, 0)
        tmp104 = tl.sum(tmp103, 0).reshape(1, X0BLOCK_SUB)
        tmp106 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp95, 0)
        tmp107 = tl.sum(tmp106, 0).reshape(1, X0BLOCK_SUB)
        tmp108 = tl.load(in_ptr39 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp109 = tl.load(in_ptr40 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp110 = tmp108 + tmp109
        tmp111 = tl.load(in_ptr41 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp112 = tmp110 + tmp111
        tmp113 = tl.load(in_ptr42 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp114 = tmp112 + tmp113
        tmp115 = tl.load(in_ptr43 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp116 = tl.load(in_ptr44 + (r1), None)
        tmp117 = tmp115 - tmp116
        tmp118 = tl.load(in_ptr45 + (r1), None)
        tmp119 = tmp117 * tmp118
        tmp120 = tmp114 * tmp119
        tmp122 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp120, 0)
        tmp123 = tl.sum(tmp122, 0).reshape(1, X0BLOCK_SUB)
        tmp125 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp114, 0)
        tmp126 = tl.sum(tmp125, 0).reshape(1, X0BLOCK_SUB)
        tmp127 = tl.load(in_ptr46 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp128 = tl.load(in_ptr47 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp129 = tmp127 + tmp128
        tmp130 = tl.load(in_ptr48 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp131 = tmp129 + tmp130
        tmp132 = tl.load(in_ptr49 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp133 = tmp131 + tmp132
        tmp134 = tl.load(in_ptr50 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp135 = tl.load(in_ptr51 + (r1), None)
        tmp136 = tmp134 - tmp135
        tmp137 = tl.load(in_ptr52 + (r1), None)
        tmp138 = tmp136 * tmp137
        tmp139 = tmp133 * tmp138
        tmp141 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp139, 0)
        tmp142 = tl.sum(tmp141, 0).reshape(1, X0BLOCK_SUB)
        tmp144 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp133, 0)
        tmp145 = tl.sum(tmp144, 0).reshape(1, X0BLOCK_SUB)
        tmp146 = tl.load(in_ptr53 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp147 = tl.load(in_ptr54 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp148 = tmp146 + tmp147
        tmp149 = tl.load(in_ptr55 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp150 = tmp148 + tmp149
        tmp151 = tl.load(in_ptr56 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp152 = tmp150 + tmp151
        tmp153 = tl.load(in_ptr57 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp154 = tl.load(in_ptr58 + (r1), None)
        tmp155 = tmp153 - tmp154
        tmp156 = tl.load(in_ptr59 + (r1), None)
        tmp157 = tmp155 * tmp156
        tmp158 = tmp152 * tmp157
        tmp160 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp158, 0)
        tmp161 = tl.sum(tmp160, 0).reshape(1, X0BLOCK_SUB)
        tmp163 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp152, 0)
        tmp164 = tl.sum(tmp163, 0).reshape(1, X0BLOCK_SUB)
        tmp165 = tl.load(in_ptr60 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp166 = tl.load(in_ptr61 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp167 = tmp165 + tmp166
        tmp168 = tl.load(in_ptr62 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp169 = tmp167 + tmp168
        tmp170 = tl.load(in_ptr63 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp171 = tmp169 + tmp170
        tmp172 = tl.load(in_ptr64 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp173 = tl.load(in_ptr65 + (r1), None)
        tmp174 = tmp172 - tmp173
        tmp175 = tl.load(in_ptr66 + (r1), None)
        tmp176 = tmp174 * tmp175
        tmp177 = tmp171 * tmp176
        tmp179 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp177, 0)
        tmp180 = tl.sum(tmp179, 0).reshape(1, X0BLOCK_SUB)
        tmp182 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp171, 0)
        tmp183 = tl.sum(tmp182, 0).reshape(1, X0BLOCK_SUB)
        tmp184 = tl.load(in_ptr67 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp185 = tl.load(in_ptr68 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp186 = tmp184 + tmp185
        tmp187 = tl.load(in_ptr69 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp188 = tmp186 + tmp187
        tmp189 = tl.load(in_ptr70 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp190 = tmp188 + tmp189
        tmp191 = tl.load(in_ptr71 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp192 = tl.load(in_ptr72 + (r1), None)
        tmp193 = tmp191 - tmp192
        tmp194 = tl.load(in_ptr73 + (r1), None)
        tmp195 = tmp193 * tmp194
        tmp196 = tmp190 * tmp195
        tmp198 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp196, 0)
        tmp199 = tl.sum(tmp198, 0).reshape(1, X0BLOCK_SUB)
        tmp201 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp190, 0)
        tmp202 = tl.sum(tmp201, 0).reshape(1, X0BLOCK_SUB)
        tmp203 = tl.load(in_ptr74 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp204 = tl.load(in_ptr75 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp205 = tmp203 + tmp204
        tmp206 = tl.load(in_ptr76 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp207 = tmp205 + tmp206
        tmp208 = tl.load(in_ptr77 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp209 = tmp207 + tmp208
        tmp210 = tl.load(in_ptr78 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp211 = tl.load(in_ptr79 + (r1), None)
        tmp212 = tmp210 - tmp211
        tmp213 = tl.load(in_ptr80 + (r1), None)
        tmp214 = tmp212 * tmp213
        tmp215 = tmp209 * tmp214
        tmp217 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp215, 0)
        tmp218 = tl.sum(tmp217, 0).reshape(1, X0BLOCK_SUB)
        tmp220 = tl.where(tl.broadcast_to((x0_mask.reshape([1, X0BLOCK_SUB])), [R1BLOCK_SUB, X0BLOCK_SUB]), tmp209, 0)
        tmp221 = tl.sum(tmp220, 0).reshape(1, X0BLOCK_SUB)
        tmp222 = tmp9 + tmp28
        tmp223 = tmp222 + tmp47
        tmp224 = tmp223 + tmp66
        tmp225 = tmp224 + tmp85
        tmp226 = tmp225 + tmp104
        tmp227 = tmp226 + tmp142
        tmp228 = tmp227 + tmp180
        tmp229 = tmp228 + tmp218
        tmp230 = tmp229 + tmp123
        tmp231 = tmp230 + tmp161
        tmp232 = tmp231 + tmp199
        tmp233 = tmp12 + tmp31
        tmp234 = tmp233 + tmp50
        tmp235 = tmp234 + tmp69
        tmp236 = tmp235 + tmp88
        tmp237 = tmp236 + tmp107
        tmp238 = tmp237 + tmp145
        tmp239 = tmp238 + tmp183
        tmp240 = tmp239 + tmp221
        tmp241 = tmp240 + tmp126
        tmp242 = tmp241 + tmp164
        tmp243 = tmp242 + tmp202
        tl.store(in_out_ptr0 + (x0), tmp232, x0_mask)
        tl.store(in_out_ptr1 + (x0), tmp243, x0_mask)
