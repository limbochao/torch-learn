def triton_per_fused_add_native_layer_norm_backwar_11(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
    R1BLOCK_SUB: tl.constexpr = 512
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        r1 = base_r1[:,None]
        r1_mask = r1 < r1_numel
        tmp0 = tl.load(in_ptr0 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1), None)
        tmp4 = tl.load(in_ptr3 + (r1), None)
        tmp11 = tl.load(in_ptr4 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp12 = tl.load(in_ptr5 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp14 = tl.load(in_ptr6 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp16 = tl.load(in_ptr7 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp18 = tl.load(in_ptr8 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp19 = tl.load(in_ptr9 + (r1), None)
        tmp21 = tl.load(in_ptr10 + (r1), None)
        tmp28 = tl.load(in_ptr11 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp29 = tl.load(in_ptr12 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp31 = tl.load(in_ptr13 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp33 = tl.load(in_ptr14 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp35 = tl.load(in_ptr15 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp36 = tl.load(in_ptr16 + (r1), None)
        tmp38 = tl.load(in_ptr17 + (r1), None)
        tmp45 = tl.load(in_ptr18 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp46 = tl.load(in_ptr19 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp48 = tl.load(in_ptr20 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp50 = tl.load(in_ptr21 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp52 = tl.load(in_ptr22 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp53 = tl.load(in_ptr23 + (r1), None)
        tmp55 = tl.load(in_ptr24 + (r1), None)
        tmp62 = tl.load(in_ptr25 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp63 = tl.load(in_ptr26 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp65 = tl.load(in_ptr27 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp67 = tl.load(in_ptr28 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp69 = tl.load(in_ptr29 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp70 = tl.load(in_ptr30 + (r1), None)
        tmp72 = tl.load(in_ptr31 + (r1), None)
        tmp79 = tl.load(in_ptr32 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp80 = tl.load(in_ptr33 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp82 = tl.load(in_ptr34 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp84 = tl.load(in_ptr35 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp86 = tl.load(in_ptr36 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp87 = tl.load(in_ptr37 + (r1), None)
        tmp89 = tl.load(in_ptr38 + (r1), None)
        tmp96 = tl.load(in_ptr39 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp97 = tl.load(in_ptr40 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp99 = tl.load(in_ptr41 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp101 = tl.load(in_ptr42 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp103 = tl.load(in_ptr43 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp104 = tl.load(in_ptr44 + (r1), None)
        tmp106 = tl.load(in_ptr45 + (r1), None)
        tmp113 = tl.load(in_ptr46 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp114 = tl.load(in_ptr47 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp116 = tl.load(in_ptr48 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp118 = tl.load(in_ptr49 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp120 = tl.load(in_ptr50 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp121 = tl.load(in_ptr51 + (r1), None)
        tmp123 = tl.load(in_ptr52 + (r1), None)
        tmp130 = tl.load(in_ptr53 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp131 = tl.load(in_ptr54 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp133 = tl.load(in_ptr55 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp135 = tl.load(in_ptr56 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp137 = tl.load(in_ptr57 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp138 = tl.load(in_ptr58 + (r1), None)
        tmp140 = tl.load(in_ptr59 + (r1), None)
        tmp147 = tl.load(in_ptr60 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp148 = tl.load(in_ptr61 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp150 = tl.load(in_ptr62 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp152 = tl.load(in_ptr63 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp154 = tl.load(in_ptr64 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp155 = tl.load(in_ptr65 + (r1), None)
        tmp157 = tl.load(in_ptr66 + (r1), None)
        tmp164 = tl.load(in_ptr67 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp165 = tl.load(in_ptr68 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp167 = tl.load(in_ptr69 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp169 = tl.load(in_ptr70 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp171 = tl.load(in_ptr71 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp172 = tl.load(in_ptr72 + (r1), None)
        tmp174 = tl.load(in_ptr73 + (r1), None)
        tmp181 = tl.load(in_ptr74 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp182 = tl.load(in_ptr75 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp184 = tl.load(in_ptr76 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp186 = tl.load(in_ptr77 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp188 = tl.load(in_ptr78 + (x0 + 4096*r1), x0_mask, other=0.0)
        tmp189 = tl.load(in_ptr79 + (r1), None)
        tmp191 = tl.load(in_ptr80 + (r1), None)
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
        tl.store(in_out_ptr0 + (x0), tmp208, x0_mask)
        tl.store(in_out_ptr1 + (x0), tmp219, x0_mask)
