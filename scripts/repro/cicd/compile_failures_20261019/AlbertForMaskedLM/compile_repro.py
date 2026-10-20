# Standalone single-kernel compile reproducer.
# Extracted by repro_compile.py for triton_per_fused_add_native_layer_norm_backwar_14.
# It calls Triton's compiler/runtime directly with one fixed candidate config.
# It does not instantiate Inductor's autotuner or rerun Dynamo/AOTAutograd.

import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
tl_math = tl.math
import torch
import torch_npu
if not torch_npu.npu.is_initialized() and torch_npu.npu._is_in_bad_fork():
    torch_npu.npu._initialized = True
import triton.language.extra.cann.libdevice as libdevice
import triton.language.extra.cann.extension as extension
tl_math = tl.math

@triton.jit
def triton_per_fused_add_native_layer_norm_backwar_14(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB: tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 2048
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0 = tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1 = tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + loop_x0 * X0BLOCK_SUB + base_x0[None, :]
        x0_mask = x0 < min(X0BLOCK + x0_offset, x0_numel)
        r1 = base_r1[:, None]
        r1_mask = r1 < r1_numel
        tmp0 = tl.load(in_ptr0 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp2 = tl.load(in_ptr2 + r1, None)
        tmp4 = tl.load(in_ptr3 + r1, None)
        tmp11 = tl.load(in_ptr4 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp12 = tl.load(in_ptr5 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp14 = tl.load(in_ptr6 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp16 = tl.load(in_ptr7 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp18 = tl.load(in_ptr8 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp19 = tl.load(in_ptr9 + r1, None)
        tmp21 = tl.load(in_ptr10 + r1, None)
        tmp28 = tl.load(in_ptr11 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp29 = tl.load(in_ptr12 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp31 = tl.load(in_ptr13 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp33 = tl.load(in_ptr14 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp35 = tl.load(in_ptr15 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp36 = tl.load(in_ptr16 + r1, None)
        tmp38 = tl.load(in_ptr17 + r1, None)
        tmp45 = tl.load(in_ptr18 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp46 = tl.load(in_ptr19 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp48 = tl.load(in_ptr20 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp50 = tl.load(in_ptr21 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp52 = tl.load(in_ptr22 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp53 = tl.load(in_ptr23 + r1, None)
        tmp55 = tl.load(in_ptr24 + r1, None)
        tmp62 = tl.load(in_ptr25 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp63 = tl.load(in_ptr26 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp65 = tl.load(in_ptr27 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp67 = tl.load(in_ptr28 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp69 = tl.load(in_ptr29 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp70 = tl.load(in_ptr30 + r1, None)
        tmp72 = tl.load(in_ptr31 + r1, None)
        tmp79 = tl.load(in_ptr32 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp80 = tl.load(in_ptr33 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp82 = tl.load(in_ptr34 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp84 = tl.load(in_ptr35 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp86 = tl.load(in_ptr36 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp87 = tl.load(in_ptr37 + r1, None)
        tmp89 = tl.load(in_ptr38 + r1, None)
        tmp96 = tl.load(in_ptr39 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp97 = tl.load(in_ptr40 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp99 = tl.load(in_ptr41 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp101 = tl.load(in_ptr42 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp103 = tl.load(in_ptr43 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp104 = tl.load(in_ptr44 + r1, None)
        tmp106 = tl.load(in_ptr45 + r1, None)
        tmp113 = tl.load(in_ptr46 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp114 = tl.load(in_ptr47 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp116 = tl.load(in_ptr48 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp118 = tl.load(in_ptr49 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp120 = tl.load(in_ptr50 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp121 = tl.load(in_ptr51 + r1, None)
        tmp123 = tl.load(in_ptr52 + r1, None)
        tmp130 = tl.load(in_ptr53 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp131 = tl.load(in_ptr54 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp133 = tl.load(in_ptr55 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp135 = tl.load(in_ptr56 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp137 = tl.load(in_ptr57 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp138 = tl.load(in_ptr58 + r1, None)
        tmp140 = tl.load(in_ptr59 + r1, None)
        tmp147 = tl.load(in_ptr60 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp148 = tl.load(in_ptr61 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp150 = tl.load(in_ptr62 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp152 = tl.load(in_ptr63 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp154 = tl.load(in_ptr64 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp155 = tl.load(in_ptr65 + r1, None)
        tmp157 = tl.load(in_ptr66 + r1, None)
        tmp164 = tl.load(in_ptr67 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp165 = tl.load(in_ptr68 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp167 = tl.load(in_ptr69 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp169 = tl.load(in_ptr70 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp171 = tl.load(in_ptr71 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp172 = tl.load(in_ptr72 + r1, None)
        tmp174 = tl.load(in_ptr73 + r1, None)
        tmp181 = tl.load(in_ptr74 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp182 = tl.load(in_ptr75 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp184 = tl.load(in_ptr76 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp186 = tl.load(in_ptr77 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp188 = tl.load(in_ptr78 + (x0 + 4096 * r1), x0_mask, other=0.0)
        tmp189 = tl.load(in_ptr79 + r1, None)
        tmp191 = tl.load(in_ptr80 + r1, None)
        tmp3 = tmp1 - tmp2
        tmp5 = tmp3 * tmp4
        tmp6 = tmp0 * tmp5
        tmp8 = tl.sum(tmp6, 0).reshape(1, X0BLOCK_SUB)
        tmp10 = tl.sum(tmp0, 0).reshape(1, X0BLOCK_SUB)
        tmp13 = tmp11 + tmp12
        tmp15 = tmp13 + tmp14
        tmp17 = tmp15 + tmp16
        tmp20 = tmp18 - tmp19
        tmp22 = tmp20 * tmp21
        tmp23 = tmp17 * tmp22
        tmp25 = tl.sum(tmp23, 0).reshape(1, X0BLOCK_SUB)
        tmp27 = tl.sum(tmp17, 0).reshape(1, X0BLOCK_SUB)
        tmp30 = tmp28 + tmp29
        tmp32 = tmp30 + tmp31
        tmp34 = tmp32 + tmp33
        tmp37 = tmp35 - tmp36
        tmp39 = tmp37 * tmp38
        tmp40 = tmp34 * tmp39
        tmp42 = tl.sum(tmp40, 0).reshape(1, X0BLOCK_SUB)
        tmp44 = tl.sum(tmp34, 0).reshape(1, X0BLOCK_SUB)
        tmp47 = tmp45 + tmp46
        tmp49 = tmp47 + tmp48
        tmp51 = tmp49 + tmp50
        tmp54 = tmp52 - tmp53
        tmp56 = tmp54 * tmp55
        tmp57 = tmp51 * tmp56
        tmp59 = tl.sum(tmp57, 0).reshape(1, X0BLOCK_SUB)
        tmp61 = tl.sum(tmp51, 0).reshape(1, X0BLOCK_SUB)
        tmp64 = tmp62 + tmp63
        tmp66 = tmp64 + tmp65
        tmp68 = tmp66 + tmp67
        tmp71 = tmp69 - tmp70
        tmp73 = tmp71 * tmp72
        tmp74 = tmp68 * tmp73
        tmp76 = tl.sum(tmp74, 0).reshape(1, X0BLOCK_SUB)
        tmp78 = tl.sum(tmp68, 0).reshape(1, X0BLOCK_SUB)
        tmp81 = tmp79 + tmp80
        tmp83 = tmp81 + tmp82
        tmp85 = tmp83 + tmp84
        tmp88 = tmp86 - tmp87
        tmp90 = tmp88 * tmp89
        tmp91 = tmp85 * tmp90
        tmp93 = tl.sum(tmp91, 0).reshape(1, X0BLOCK_SUB)
        tmp95 = tl.sum(tmp85, 0).reshape(1, X0BLOCK_SUB)
        tmp98 = tmp96 + tmp97
        tmp100 = tmp98 + tmp99
        tmp102 = tmp100 + tmp101
        tmp105 = tmp103 - tmp104
        tmp107 = tmp105 * tmp106
        tmp108 = tmp102 * tmp107
        tmp110 = tl.sum(tmp108, 0).reshape(1, X0BLOCK_SUB)
        tmp112 = tl.sum(tmp102, 0).reshape(1, X0BLOCK_SUB)
        tmp115 = tmp113 + tmp114
        tmp117 = tmp115 + tmp116
        tmp119 = tmp117 + tmp118
        tmp122 = tmp120 - tmp121
        tmp124 = tmp122 * tmp123
        tmp125 = tmp119 * tmp124
        tmp127 = tl.sum(tmp125, 0).reshape(1, X0BLOCK_SUB)
        tmp129 = tl.sum(tmp119, 0).reshape(1, X0BLOCK_SUB)
        tmp132 = tmp130 + tmp131
        tmp134 = tmp132 + tmp133
        tmp136 = tmp134 + tmp135
        tmp139 = tmp137 - tmp138
        tmp141 = tmp139 * tmp140
        tmp142 = tmp136 * tmp141
        tmp144 = tl.sum(tmp142, 0).reshape(1, X0BLOCK_SUB)
        tmp146 = tl.sum(tmp136, 0).reshape(1, X0BLOCK_SUB)
        tmp149 = tmp147 + tmp148
        tmp151 = tmp149 + tmp150
        tmp153 = tmp151 + tmp152
        tmp156 = tmp154 - tmp155
        tmp158 = tmp156 * tmp157
        tmp159 = tmp153 * tmp158
        tmp161 = tl.sum(tmp159, 0).reshape(1, X0BLOCK_SUB)
        tmp163 = tl.sum(tmp153, 0).reshape(1, X0BLOCK_SUB)
        tmp166 = tmp164 + tmp165
        tmp168 = tmp166 + tmp167
        tmp170 = tmp168 + tmp169
        tmp173 = tmp171 - tmp172
        tmp175 = tmp173 * tmp174
        tmp176 = tmp170 * tmp175
        tmp178 = tl.sum(tmp176, 0).reshape(1, X0BLOCK_SUB)
        tmp180 = tl.sum(tmp170, 0).reshape(1, X0BLOCK_SUB)
        tmp183 = tmp181 + tmp182
        tmp185 = tmp183 + tmp184
        tmp187 = tmp185 + tmp186
        tmp190 = tmp188 - tmp189
        tmp192 = tmp190 * tmp191
        tmp193 = tmp187 * tmp192
        tmp195 = tl.sum(tmp193, 0).reshape(1, X0BLOCK_SUB)
        tmp197 = tl.sum(tmp187, 0).reshape(1, X0BLOCK_SUB)
        tmp198 = tmp8 + tmp25
        tmp199 = tmp198 + tmp42
        tmp200 = tmp199 + tmp59
        tmp201 = tmp200 + tmp76
        tmp202 = tmp201 + tmp93
        tmp203 = tmp202 + tmp127
        tmp204 = tmp203 + tmp161
        tmp205 = tmp204 + tmp195
        tmp206 = tmp205 + tmp110
        tmp207 = tmp206 + tmp144
        tmp208 = tmp207 + tmp178
        tmp209 = tmp10 + tmp27
        tmp210 = tmp209 + tmp44
        tmp211 = tmp210 + tmp61
        tmp212 = tmp211 + tmp78
        tmp213 = tmp212 + tmp95
        tmp214 = tmp213 + tmp129
        tmp215 = tmp214 + tmp163
        tmp216 = tmp215 + tmp197
        tmp217 = tmp216 + tmp112
        tmp218 = tmp217 + tmp146
        tmp219 = tmp218 + tmp180
        tl.store(in_out_ptr0 + x0, tmp208, x0_mask)
        tl.store(in_out_ptr1 + x0, tmp219, x0_mask)

# ---- direct standalone reproducer helpers ----
_STANDALONE_FN_NAME = 'triton_per_fused_add_native_layer_norm_backwar_14'
_STANDALONE_WANTED_CONFIG = {'X0BLOCK_SUB': 1, 'R1BLOCK_SUB': 2048, 'compile_mode': 'simd_simt_template', 'multibuffer': False, 'split_axis': [0], 'num_warps': 1, 'num_stages': 2}
_STANDALONE_SIGNATURE = {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'in_ptr25': '*fp32', 'in_ptr26': '*fp32', 'in_ptr27': '*fp32', 'in_ptr28': '*fp32', 'in_ptr29': '*fp32', 'in_ptr30': '*fp32', 'in_ptr31': '*fp32', 'in_ptr32': '*fp32', 'in_ptr33': '*fp32', 'in_ptr34': '*fp32', 'in_ptr35': '*fp32', 'in_ptr36': '*fp32', 'in_ptr37': '*fp32', 'in_ptr38': '*fp32', 'in_ptr39': '*fp32', 'in_ptr40': '*fp32', 'in_ptr41': '*fp32', 'in_ptr42': '*fp32', 'in_ptr43': '*fp32', 'in_ptr44': '*fp32', 'in_ptr45': '*fp32', 'in_ptr46': '*fp32', 'in_ptr47': '*fp32', 'in_ptr48': '*fp32', 'in_ptr49': '*fp32', 'in_ptr50': '*fp32', 'in_ptr51': '*fp32', 'in_ptr52': '*fp32', 'in_ptr53': '*fp32', 'in_ptr54': '*fp32', 'in_ptr55': '*fp32', 'in_ptr56': '*fp32', 'in_ptr57': '*fp32', 'in_ptr58': '*fp32', 'in_ptr59': '*fp32', 'in_ptr60': '*fp32', 'in_ptr61': '*fp32', 'in_ptr62': '*fp32', 'in_ptr63': '*fp32', 'in_ptr64': '*fp32', 'in_ptr65': '*fp32', 'in_ptr66': '*fp32', 'in_ptr67': '*fp32', 'in_ptr68': '*fp32', 'in_ptr69': '*fp32', 'in_ptr70': '*fp32', 'in_ptr71': '*fp32', 'in_ptr72': '*fp32', 'in_ptr73': '*fp32', 'in_ptr74': '*fp32', 'in_ptr75': '*fp32', 'in_ptr76': '*fp32', 'in_ptr77': '*fp32', 'in_ptr78': '*fp32', 'in_ptr79': '*fp32', 'in_ptr80': '*fp32', 'x0_numel': 'i32', 'r1_numel': 'i32', 'X0BLOCK': 'i32'}
_STANDALONE_CONSTANTS = {'X0BLOCK_SUB': 1}
_STANDALONE_ATTRS = {}
_STANDALONE_TARGET = {'backend': 'npu', 'arch': 'Ascend950PR_9579', 'warp_size': 32}
_STANDALONE_DEVICE_INDEX = 0
_STANDALONE_OPTIONS = {'num_warps': 1, 'num_stages': 2, 'debug': False, 'compile_mode': 'simd_simt_template', 'multibuffer': False}
_STANDALONE_INDUCTOR_META = {'grid_type': 'GridNpu', 'split_axis': [0], 'axis_names': ['x0', 'r1'], 'runtime_block_arg_names': ['X0BLOCK'], 'group_enabled': False}
_STANDALONE_LAUNCH_ARG_NAMES = ['in_out_ptr0', 'in_out_ptr1', 'in_ptr0', 'in_ptr1', 'in_ptr2', 'in_ptr3', 'in_ptr4', 'in_ptr5', 'in_ptr6', 'in_ptr7', 'in_ptr8', 'in_ptr9', 'in_ptr10', 'in_ptr11', 'in_ptr12', 'in_ptr13', 'in_ptr14', 'in_ptr15', 'in_ptr16', 'in_ptr17', 'in_ptr18', 'in_ptr19', 'in_ptr20', 'in_ptr21', 'in_ptr22', 'in_ptr23', 'in_ptr24', 'in_ptr25', 'in_ptr26', 'in_ptr27', 'in_ptr28', 'in_ptr29', 'in_ptr30', 'in_ptr31', 'in_ptr32', 'in_ptr33', 'in_ptr34', 'in_ptr35', 'in_ptr36', 'in_ptr37', 'in_ptr38', 'in_ptr39', 'in_ptr40', 'in_ptr41', 'in_ptr42', 'in_ptr43', 'in_ptr44', 'in_ptr45', 'in_ptr46', 'in_ptr47', 'in_ptr48', 'in_ptr49', 'in_ptr50', 'in_ptr51', 'in_ptr52', 'in_ptr53', 'in_ptr54', 'in_ptr55', 'in_ptr56', 'in_ptr57', 'in_ptr58', 'in_ptr59', 'in_ptr60', 'in_ptr61', 'in_ptr62', 'in_ptr63', 'in_ptr64', 'in_ptr65', 'in_ptr66', 'in_ptr67', 'in_ptr68', 'in_ptr69', 'in_ptr70', 'in_ptr71', 'in_ptr72', 'in_ptr73', 'in_ptr74', 'in_ptr75', 'in_ptr76', 'in_ptr77', 'in_ptr78', 'in_ptr79', 'in_ptr80', 'x0_numel', 'r1_numel', 'X0BLOCK']
_STANDALONE_RUNTIME_BLOCK_NAMES = ['X0BLOCK']
_STANDALONE_BASE_LAUNCH_ARG_NAMES = ['in_out_ptr0', 'in_out_ptr1', 'in_ptr0', 'in_ptr1', 'in_ptr2', 'in_ptr3', 'in_ptr4', 'in_ptr5', 'in_ptr6', 'in_ptr7', 'in_ptr8', 'in_ptr9', 'in_ptr10', 'in_ptr11', 'in_ptr12', 'in_ptr13', 'in_ptr14', 'in_ptr15', 'in_ptr16', 'in_ptr17', 'in_ptr18', 'in_ptr19', 'in_ptr20', 'in_ptr21', 'in_ptr22', 'in_ptr23', 'in_ptr24', 'in_ptr25', 'in_ptr26', 'in_ptr27', 'in_ptr28', 'in_ptr29', 'in_ptr30', 'in_ptr31', 'in_ptr32', 'in_ptr33', 'in_ptr34', 'in_ptr35', 'in_ptr36', 'in_ptr37', 'in_ptr38', 'in_ptr39', 'in_ptr40', 'in_ptr41', 'in_ptr42', 'in_ptr43', 'in_ptr44', 'in_ptr45', 'in_ptr46', 'in_ptr47', 'in_ptr48', 'in_ptr49', 'in_ptr50', 'in_ptr51', 'in_ptr52', 'in_ptr53', 'in_ptr54', 'in_ptr55', 'in_ptr56', 'in_ptr57', 'in_ptr58', 'in_ptr59', 'in_ptr60', 'in_ptr61', 'in_ptr62', 'in_ptr63', 'in_ptr64', 'in_ptr65', 'in_ptr66', 'in_ptr67', 'in_ptr68', 'in_ptr69', 'in_ptr70', 'in_ptr71', 'in_ptr72', 'in_ptr73', 'in_ptr74', 'in_ptr75', 'in_ptr76', 'in_ptr77', 'in_ptr78', 'in_ptr79', 'in_ptr80', 'x0_numel', 'r1_numel']
_STANDALONE_EXTRA_LAUNCH_ARG_NAMES = ()


def _standalone_repro_compile():
    import triton
    try:
        import torch
        if hasattr(torch, "npu") and hasattr(torch.npu, "set_device"):
            torch.npu.set_device(_STANDALONE_DEVICE_INDEX)
    except (ImportError, RuntimeError):
        # Triton can still select its active device in environments where the
        # torch_npu convenience namespace is unavailable (for example during
        # compile-only diagnostics with a mocked driver).
        pass
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource

    kernel_fn = globals().get(_STANDALONE_FN_NAME)
    if kernel_fn is None:
        raise RuntimeError(
            f"Kernel function {_STANDALONE_FN_NAME!r} is missing from export"
        )
    target_info = _STANDALONE_TARGET
    if not target_info.get("arch"):
        raise RuntimeError(
            "Captured device architecture is unavailable; recapture with "
            "trace_compile.py so direct compile metadata is journaled"
        )
    target = GPUTarget(
        target_info.get("backend", "npu"),
        target_info["arch"],
        target_info.get("warp_size") or 32,
    )
    attrs = _STANDALONE_ATTRS or None
    if isinstance(attrs, dict):
        # Triton-Ascend 3.2 serializes an AttrsDescriptor as
        # ``{'arg_properties': ..., 'cls': ...}``; older releases use keyword
        # fields such as ``divisible_by_16`` and newer releases consume the
        # raw tuple-key mapping directly.  Rebuild only the descriptor shapes
        # that identify themselves, leaving the newest raw mapping untouched.
        descriptor_wrapper = attrs.get("__attrs_descriptor__")
        if isinstance(descriptor_wrapper, dict):
            attrs = descriptor_wrapper
        descriptor_data = attrs
        try:
            from triton.compiler.compiler import AttrsDescriptor
        except ImportError:
            try:
                from triton.backends.compiler import AttrsDescriptor
            except ImportError:
                AttrsDescriptor = None
        if AttrsDescriptor is not None:
            if {
                "arg_properties",
                "cls",
            }.issubset(descriptor_data) and hasattr(
                AttrsDescriptor,
                "from_dict",
            ):
                attrs = AttrsDescriptor.from_dict(descriptor_data)
            elif any(
                key in descriptor_data
                for key in (
                    "divisible_by_16",
                    "equal_to_1",
                    "pointer_range_32",
                )
            ):
                try:
                    attrs = AttrsDescriptor(**descriptor_data)
                except (TypeError, ValueError):
                    # A backend-specific descriptor may require a different
                    # constructor; passing its mapping through is preferable
                    # to turning a useful capture into an export-time crash.
                    attrs = descriptor_data
    if attrs is None:
        source = ASTSource(
            kernel_fn,
            _STANDALONE_SIGNATURE,
            _STANDALONE_CONSTANTS,
        )
    else:
        source = ASTSource(
            kernel_fn,
            _STANDALONE_SIGNATURE,
            _STANDALONE_CONSTANTS,
            attrs,
        )
    return triton.compile(
        source,
        target=target,
        options=dict(_STANDALONE_OPTIONS),
    )


def _standalone_repro_current_stream():
    """Return the backend's raw current stream without an Inductor import."""
    import torch

    npu = getattr(torch, "npu", None)
    current_stream = getattr(npu, "current_stream", None)
    if not callable(current_stream):
        return None
    stream = current_stream()
    # torch_npu streams expose the integer ABI handle as ``npu_stream``;
    # keeping the object as a fallback also works with lightweight test
    # doubles and Triton versions that accept a stream wrapper directly.
    return getattr(stream, "npu_stream", stream)


def _standalone_repro_runtime_names():
    meta = _STANDALONE_INDUCTOR_META or {}
    return tuple(
        _STANDALONE_RUNTIME_BLOCK_NAMES
        or meta.get("runtime_block_append_order")
        or meta.get("runtime_block_arg_names")
        or ()
    )


def _standalone_repro_extra_names():
    meta = _STANDALONE_INDUCTOR_META or {}
    return tuple(
        _STANDALONE_EXTRA_LAUNCH_ARG_NAMES
        or meta.get("extra_launcher_args")
        or ()
    )


def _standalone_repro_legacy_prefix(args, legacy_launch_args):
    """Strip candidate-specific runtime blocks from an old payload tuple."""
    args = tuple(args)
    if legacy_launch_args is None:
        return args
    legacy = tuple(legacy_launch_args)
    runtime_count = len(_STANDALONE_RUNTIME_BLOCKS)
    if runtime_count and len(legacy) >= len(args) + runtime_count:
        prefix = legacy[:-runtime_count]
        if len(prefix) >= len(args):
            return prefix
    return legacy if len(legacy) >= len(args) else args


def _standalone_repro_split_launch_args(raw_launch_args):
    """Split Inductor's ``kernel args + extra grid args + runtime blocks``."""
    raw_launch_args = tuple(raw_launch_args)
    launch_names = tuple(_STANDALONE_LAUNCH_ARG_NAMES)
    runtime_names = _standalone_repro_runtime_names()
    base_names = tuple(
        _STANDALONE_BASE_LAUNCH_ARG_NAMES
        or (name for name in launch_names if name not in runtime_names)
    )
    extra_names = _standalone_repro_extra_names()
    raw_names = (*base_names, *extra_names, *runtime_names)

    if len(raw_launch_args) == len(raw_names):
        raw_values = dict(zip(raw_names, raw_launch_args))
        kernel_args = tuple(raw_values[name] for name in launch_names)
        return kernel_args, raw_values
    if len(raw_launch_args) == len(launch_names):
        # Newer capture records may already have removed extra launcher args.
        return raw_launch_args, dict(zip(launch_names, raw_launch_args))

    raise RuntimeError(
        "Captured launch argument count does not match the direct signature: "
        f"got {len(raw_launch_args)}, expected {len(raw_names)} (raw) or "
        f"{len(launch_names)} (kernel)"
    )


def _standalone_repro_value(values, name):
    if name in values:
        return values[name]
    lowered = str(name).lower()
    for key, value in values.items():
        if str(key).lower() == lowered:
            return value
    return None


def _standalone_repro_numel(values, axis):
    candidates = (
        f"{axis}_numel",
        f"{axis}numel",
        f"{str(axis).upper()}_NUMEL",
        f"{str(axis).upper()}NUMEL",
    )
    for name in candidates:
        value = _standalone_repro_value(values, name)
        if value is not None:
            return int(value)
    raise RuntimeError(f"No numel argument found for axis {axis!r}")


def _standalone_repro_block(values, axis):
    candidates = (
        f"{str(axis).upper()}BLOCK",
        f"{axis}block",
        f"{axis}_block",
    )
    for name in candidates:
        value = _standalone_repro_value(values, name)
        if value is not None:
            return int(value)
    # A non-runtime block is normally a constexpr/config value.
    for name in candidates:
        value = _standalone_repro_value(_STANDALONE_CONSTANTS, name)
        if value is None:
            value = _standalone_repro_value(_STANDALONE_WANTED_CONFIG, name)
        if value is not None:
            return int(value)
    raise RuntimeError(f"No block value found for axis {axis!r}")


def _standalone_repro_grid_value(expression, values):
    if isinstance(expression, bool):
        return int(expression)
    if isinstance(expression, (int, float)):
        return int(expression)
    if not isinstance(expression, str):
        raise RuntimeError(f"Unsupported captured grid expression: {expression!r}")
    direct = _standalone_repro_value(values, expression)
    if direct is not None:
        return int(direct)
    try:
        return int(expression)
    except ValueError:
        pass
    # Fixed/precomputed grids emitted by Inductor generally contain only
    # symbols and arithmetic.  Evaluate that narrow expression against the
    # captured values, with Python builtins disabled.
    try:
        return int(eval(expression, {"__builtins__": {}}, dict(values)))
    except Exception as error:
        raise RuntimeError(
            f"Cannot evaluate captured grid expression {expression!r}: {error}"
        ) from error


def _standalone_repro_grid(raw_launch_args):
    _, values = _standalone_repro_split_launch_args(raw_launch_args)
    meta = _STANDALONE_INDUCTOR_META or {}

    fixed_grid = meta.get("fixed_grid")
    if fixed_grid is not None:
        if not isinstance(fixed_grid, (tuple, list)):
            raise RuntimeError(f"Unsupported captured fixed_grid: {fixed_grid!r}")
        result = tuple(
            _standalone_repro_grid_value(expression, values)
            for expression in fixed_grid
        )
        return (result + (1, 1, 1))[:3]

    precomputed = meta.get("precomputed_grids")
    if precomputed:
        selected = None
        for candidate in precomputed:
            candidate_config = candidate.get("config", {})
            if all(
                _STANDALONE_WANTED_CONFIG.get(name) == value
                for name, value in candidate_config.items()
            ):
                selected = candidate
                break
        if selected is None:
            raise RuntimeError(
                "No precomputed grid matches the captured fixed config: "
                f"{_STANDALONE_WANTED_CONFIG!r}"
            )
        expressions = selected.get("python") or selected.get("python_slow")
        if expressions is None:
            raise RuntimeError(f"Precomputed grid has no Python expression: {selected!r}")
        result = tuple(
            _standalone_repro_grid_value(expression, values)
            for expression in expressions
        )
        return (result + (1, 1, 1))[:3]

    axis_names = tuple(meta.get("axis_names") or ())
    explicit_runtime_names = _standalone_repro_runtime_names()
    runtime_names = explicit_runtime_names
    grid_type = str(meta.get("grid_type") or "")
    # If an older record has only ``split_blocks`` metadata, defer the
    # runtime-name inference below.  Inferring ``Z0BLOCK``/etc. too early
    # would make the split-block fallback unreachable and incorrectly demand
    # values that were never appended to the launch ABI.
    if not runtime_names and not (
        grid_type == "GridNpu" and meta.get("split_blocks") is not None
    ):
        # Older records did not preserve runtime-block names.  GridNpu uses
        # <AXIS>BLOCK names in split-axis order; infer those conservatively.
        split_axis = tuple(meta.get("split_axis") or range(min(3, len(axis_names))))
        runtime_names = tuple(
            f"{str(axis_names[index]).upper()}BLOCK"
            for index in split_axis[:3]
            if index < len(axis_names)
        )

    grids = []
    if runtime_names and axis_names:
        for block_name in runtime_names[:3]:
            axis = str(block_name).removesuffix("BLOCK").lower()
            if axis not in axis_names:
                raise RuntimeError(
                    f"Cannot map runtime block {block_name!r} to captured axes "
                    f"{axis_names!r}"
                )
            block = _standalone_repro_value(values, block_name)
            if block is None:
                raise RuntimeError(f"No runtime value found for {block_name!r}")
            block = int(block)
            if block <= 0:
                raise RuntimeError(f"Invalid runtime block {block_name}={block}")
            numel = _standalone_repro_numel(values, axis)
            grids.append((numel + block - 1) // block)
        return tuple((grids + [1, 1, 1])[:3])

    # A few older NPU captures did not expose runtime block argument names;
    # their GridNpu metadata instead records the split blocks directly.  Keep
    # this path independent of Inductor's GridExpr implementation.
    if grid_type == "GridNpu" and axis_names:
        split_axis = tuple(meta.get("split_axis") or range(len(axis_names)))
        split_blocks = meta.get("split_blocks")
        for position, axis_index in enumerate(split_axis[:3]):
            if not isinstance(axis_index, int) or axis_index >= len(axis_names):
                raise RuntimeError(
                    f"Invalid captured split axis {axis_index!r} for {axis_names!r}"
                )
            axis = str(axis_names[axis_index])
            numel = _standalone_repro_numel(values, axis)
            if isinstance(split_blocks, (tuple, list)) and position < len(split_blocks):
                block_expr = split_blocks[position]
                block = (
                    1
                    if block_expr is None
                    else _standalone_repro_grid_value(block_expr, values)
                )
            else:
                block = _standalone_repro_block(values, axis)
            if block <= 0:
                raise RuntimeError(f"Invalid split block for {axis}: {block}")
            grids.append(numel if block == 1 else (numel + block - 1) // block)
        return tuple((grids + [1, 1, 1])[:3])

    # Common upstream grid classes use X/Y/ZBLOCK constexprs and xnumel /
    # ynumel / znumel arguments.  This also makes old non-NPU Grid1D/2D/3D
    # captures useful without importing Inductor's grid classes.
    dimension_axes = {
        "Grid1D": ("x",),
        "Grid2D": ("x", "y"),
        "Grid3D": ("x", "y", "z"),
        "Grid2DWithYZOverflow": ("x", "y"),
    }.get(grid_type)
    if dimension_axes:
        for axis in dimension_axes:
            block = _standalone_repro_block(values, axis)
            numel = _standalone_repro_numel(values, axis)
            grids.append((numel + block - 1) // block)
        if grid_type == "Grid2DWithYZOverflow":
            grids.append(1)
        return tuple((grids + [1, 1, 1])[:3])

    return (1, 1, 1)



def _standalone_repro_main():
    print(f"Compiling fixed config only: {_STANDALONE_FN_NAME}")
    print(f"Config: {_STANDALONE_WANTED_CONFIG}")
    _standalone_repro_compile()
    print("Kernel compiled successfully; the original compiler failure did not reproduce.")


if __name__ == "__main__":
    _standalone_repro_main()
