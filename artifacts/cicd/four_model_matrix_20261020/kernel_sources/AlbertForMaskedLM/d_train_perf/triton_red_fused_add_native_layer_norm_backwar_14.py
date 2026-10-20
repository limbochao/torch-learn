def triton_red_fused_add_native_layer_norm_backwar_14(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, x0_numel, r1_numel, X0BLOCK, X0BLOCK_SUB : tl.constexpr, R1BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[None,:]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        _tmp8 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp11 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp27 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp30 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp46 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp49 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp65 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp68 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp84 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp87 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp103 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp106 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp122 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp125 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp141 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp144 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp160 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp163 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp179 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp182 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp198 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp201 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp217 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        _tmp220 = tl.full([R1BLOCK_SUB, X0BLOCK_SUB], 0, tl.float32)
        for loop_r1 in range(loops_r1):
            r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[:,None]
            r1_mask = r1 < r1_numel
            tmp0 = tl.load(in_ptr0 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp1 = tl.load(in_ptr1 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp2 = tl.load(in_ptr2 + (r1), r1_mask, other=0.0)
            tmp4 = tl.load(in_ptr3 + (r1), r1_mask, other=0.0)
            tmp13 = tl.load(in_ptr4 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp14 = tl.load(in_ptr5 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp16 = tl.load(in_ptr6 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp18 = tl.load(in_ptr7 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp20 = tl.load(in_ptr8 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp21 = tl.load(in_ptr9 + (r1), r1_mask, other=0.0)
            tmp23 = tl.load(in_ptr10 + (r1), r1_mask, other=0.0)
            tmp32 = tl.load(in_ptr11 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp33 = tl.load(in_ptr12 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp35 = tl.load(in_ptr13 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp37 = tl.load(in_ptr14 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp39 = tl.load(in_ptr15 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp40 = tl.load(in_ptr16 + (r1), r1_mask, other=0.0)
            tmp42 = tl.load(in_ptr17 + (r1), r1_mask, other=0.0)
            tmp51 = tl.load(in_ptr18 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp52 = tl.load(in_ptr19 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp54 = tl.load(in_ptr20 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp56 = tl.load(in_ptr21 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp58 = tl.load(in_ptr22 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp59 = tl.load(in_ptr23 + (r1), r1_mask, other=0.0)
            tmp61 = tl.load(in_ptr24 + (r1), r1_mask, other=0.0)
            tmp70 = tl.load(in_ptr25 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp71 = tl.load(in_ptr26 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp73 = tl.load(in_ptr27 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp75 = tl.load(in_ptr28 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp77 = tl.load(in_ptr29 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp78 = tl.load(in_ptr30 + (r1), r1_mask, other=0.0)
            tmp80 = tl.load(in_ptr31 + (r1), r1_mask, other=0.0)
            tmp89 = tl.load(in_ptr32 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp90 = tl.load(in_ptr33 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp92 = tl.load(in_ptr34 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp94 = tl.load(in_ptr35 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp96 = tl.load(in_ptr36 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp97 = tl.load(in_ptr37 + (r1), r1_mask, other=0.0)
            tmp99 = tl.load(in_ptr38 + (r1), r1_mask, other=0.0)
            tmp108 = tl.load(in_ptr39 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp109 = tl.load(in_ptr40 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp111 = tl.load(in_ptr41 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp113 = tl.load(in_ptr42 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp115 = tl.load(in_ptr43 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp116 = tl.load(in_ptr44 + (r1), r1_mask, other=0.0)
            tmp118 = tl.load(in_ptr45 + (r1), r1_mask, other=0.0)
            tmp127 = tl.load(in_ptr46 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp128 = tl.load(in_ptr47 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp130 = tl.load(in_ptr48 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp132 = tl.load(in_ptr49 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp134 = tl.load(in_ptr50 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp135 = tl.load(in_ptr51 + (r1), r1_mask, other=0.0)
            tmp137 = tl.load(in_ptr52 + (r1), r1_mask, other=0.0)
            tmp146 = tl.load(in_ptr53 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp147 = tl.load(in_ptr54 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp149 = tl.load(in_ptr55 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp151 = tl.load(in_ptr56 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp153 = tl.load(in_ptr57 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp154 = tl.load(in_ptr58 + (r1), r1_mask, other=0.0)
            tmp156 = tl.load(in_ptr59 + (r1), r1_mask, other=0.0)
            tmp165 = tl.load(in_ptr60 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp166 = tl.load(in_ptr61 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp168 = tl.load(in_ptr62 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp170 = tl.load(in_ptr63 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp172 = tl.load(in_ptr64 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp173 = tl.load(in_ptr65 + (r1), r1_mask, other=0.0)
            tmp175 = tl.load(in_ptr66 + (r1), r1_mask, other=0.0)
            tmp184 = tl.load(in_ptr67 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp185 = tl.load(in_ptr68 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp187 = tl.load(in_ptr69 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp189 = tl.load(in_ptr70 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp191 = tl.load(in_ptr71 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp192 = tl.load(in_ptr72 + (r1), r1_mask, other=0.0)
            tmp194 = tl.load(in_ptr73 + (r1), r1_mask, other=0.0)
            tmp203 = tl.load(in_ptr74 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp204 = tl.load(in_ptr75 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp206 = tl.load(in_ptr76 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp208 = tl.load(in_ptr77 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp210 = tl.load(in_ptr78 + (x0 + 4096*r1), r1_mask & x0_mask, other=0.0)
            tmp211 = tl.load(in_ptr79 + (r1), r1_mask, other=0.0)
            tmp213 = tl.load(in_ptr80 + (r1), r1_mask, other=0.0)
            tmp3 = tmp1 - tmp2
            tmp5 = tmp3 * tmp4
            tmp6 = tmp0 * tmp5
            tmp7 = tl.reshape(tmp6, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp9 = _tmp8 + tmp7
            _tmp8 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp9, _tmp8)
            tmp10 = tl.reshape(tmp0, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp12 = _tmp11 + tmp10
            _tmp11 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp12, _tmp11)
            tmp15 = tmp13 + tmp14
            tmp17 = tmp15 + tmp16
            tmp19 = tmp17 + tmp18
            tmp22 = tmp20 - tmp21
            tmp24 = tmp22 * tmp23
            tmp25 = tmp19 * tmp24
            tmp26 = tl.reshape(tmp25, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp28 = _tmp27 + tmp26
            _tmp27 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp28, _tmp27)
            tmp29 = tl.reshape(tmp19, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp31 = _tmp30 + tmp29
            _tmp30 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp31, _tmp30)
            tmp34 = tmp32 + tmp33
            tmp36 = tmp34 + tmp35
            tmp38 = tmp36 + tmp37
            tmp41 = tmp39 - tmp40
            tmp43 = tmp41 * tmp42
            tmp44 = tmp38 * tmp43
            tmp45 = tl.reshape(tmp44, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp47 = _tmp46 + tmp45
            _tmp46 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp47, _tmp46)
            tmp48 = tl.reshape(tmp38, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp50 = _tmp49 + tmp48
            _tmp49 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp50, _tmp49)
            tmp53 = tmp51 + tmp52
            tmp55 = tmp53 + tmp54
            tmp57 = tmp55 + tmp56
            tmp60 = tmp58 - tmp59
            tmp62 = tmp60 * tmp61
            tmp63 = tmp57 * tmp62
            tmp64 = tl.reshape(tmp63, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp66 = _tmp65 + tmp64
            _tmp65 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp66, _tmp65)
            tmp67 = tl.reshape(tmp57, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp69 = _tmp68 + tmp67
            _tmp68 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp69, _tmp68)
            tmp72 = tmp70 + tmp71
            tmp74 = tmp72 + tmp73
            tmp76 = tmp74 + tmp75
            tmp79 = tmp77 - tmp78
            tmp81 = tmp79 * tmp80
            tmp82 = tmp76 * tmp81
            tmp83 = tl.reshape(tmp82, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp85 = _tmp84 + tmp83
            _tmp84 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp85, _tmp84)
            tmp86 = tl.reshape(tmp76, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp88 = _tmp87 + tmp86
            _tmp87 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp88, _tmp87)
            tmp91 = tmp89 + tmp90
            tmp93 = tmp91 + tmp92
            tmp95 = tmp93 + tmp94
            tmp98 = tmp96 - tmp97
            tmp100 = tmp98 * tmp99
            tmp101 = tmp95 * tmp100
            tmp102 = tl.reshape(tmp101, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp104 = _tmp103 + tmp102
            _tmp103 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp104, _tmp103)
            tmp105 = tl.reshape(tmp95, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp107 = _tmp106 + tmp105
            _tmp106 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp107, _tmp106)
            tmp110 = tmp108 + tmp109
            tmp112 = tmp110 + tmp111
            tmp114 = tmp112 + tmp113
            tmp117 = tmp115 - tmp116
            tmp119 = tmp117 * tmp118
            tmp120 = tmp114 * tmp119
            tmp121 = tl.reshape(tmp120, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp123 = _tmp122 + tmp121
            _tmp122 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp123, _tmp122)
            tmp124 = tl.reshape(tmp114, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp126 = _tmp125 + tmp124
            _tmp125 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp126, _tmp125)
            tmp129 = tmp127 + tmp128
            tmp131 = tmp129 + tmp130
            tmp133 = tmp131 + tmp132
            tmp136 = tmp134 - tmp135
            tmp138 = tmp136 * tmp137
            tmp139 = tmp133 * tmp138
            tmp140 = tl.reshape(tmp139, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp142 = _tmp141 + tmp140
            _tmp141 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp142, _tmp141)
            tmp143 = tl.reshape(tmp133, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp145 = _tmp144 + tmp143
            _tmp144 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp145, _tmp144)
            tmp148 = tmp146 + tmp147
            tmp150 = tmp148 + tmp149
            tmp152 = tmp150 + tmp151
            tmp155 = tmp153 - tmp154
            tmp157 = tmp155 * tmp156
            tmp158 = tmp152 * tmp157
            tmp159 = tl.reshape(tmp158, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp161 = _tmp160 + tmp159
            _tmp160 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp161, _tmp160)
            tmp162 = tl.reshape(tmp152, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp164 = _tmp163 + tmp162
            _tmp163 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp164, _tmp163)
            tmp167 = tmp165 + tmp166
            tmp169 = tmp167 + tmp168
            tmp171 = tmp169 + tmp170
            tmp174 = tmp172 - tmp173
            tmp176 = tmp174 * tmp175
            tmp177 = tmp171 * tmp176
            tmp178 = tl.reshape(tmp177, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp180 = _tmp179 + tmp178
            _tmp179 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp180, _tmp179)
            tmp181 = tl.reshape(tmp171, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp183 = _tmp182 + tmp181
            _tmp182 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp183, _tmp182)
            tmp186 = tmp184 + tmp185
            tmp188 = tmp186 + tmp187
            tmp190 = tmp188 + tmp189
            tmp193 = tmp191 - tmp192
            tmp195 = tmp193 * tmp194
            tmp196 = tmp190 * tmp195
            tmp197 = tl.reshape(tmp196, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp199 = _tmp198 + tmp197
            _tmp198 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp199, _tmp198)
            tmp200 = tl.reshape(tmp190, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp202 = _tmp201 + tmp200
            _tmp201 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp202, _tmp201)
            tmp205 = tmp203 + tmp204
            tmp207 = tmp205 + tmp206
            tmp209 = tmp207 + tmp208
            tmp212 = tmp210 - tmp211
            tmp214 = tmp212 * tmp213
            tmp215 = tmp209 * tmp214
            tmp216 = tl.reshape(tmp215, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp218 = _tmp217 + tmp216
            _tmp217 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp218, _tmp217)
            tmp219 = tl.reshape(tmp209, [R1BLOCK_SUB, X0BLOCK_SUB])
            tmp221 = _tmp220 + tmp219
            _tmp220 = tl.where((r1_mask & x0_mask).reshape([R1BLOCK_SUB, X0BLOCK_SUB]), tmp221, _tmp220)
        tmp8 = tl.sum(_tmp8, 0).reshape(1, X0BLOCK_SUB)
        tmp11 = tl.sum(_tmp11, 0).reshape(1, X0BLOCK_SUB)
        tmp27 = tl.sum(_tmp27, 0).reshape(1, X0BLOCK_SUB)
        tmp30 = tl.sum(_tmp30, 0).reshape(1, X0BLOCK_SUB)
        tmp46 = tl.sum(_tmp46, 0).reshape(1, X0BLOCK_SUB)
        tmp49 = tl.sum(_tmp49, 0).reshape(1, X0BLOCK_SUB)
        tmp65 = tl.sum(_tmp65, 0).reshape(1, X0BLOCK_SUB)
        tmp68 = tl.sum(_tmp68, 0).reshape(1, X0BLOCK_SUB)
        tmp84 = tl.sum(_tmp84, 0).reshape(1, X0BLOCK_SUB)
        tmp87 = tl.sum(_tmp87, 0).reshape(1, X0BLOCK_SUB)
        tmp103 = tl.sum(_tmp103, 0).reshape(1, X0BLOCK_SUB)
        tmp106 = tl.sum(_tmp106, 0).reshape(1, X0BLOCK_SUB)
        tmp122 = tl.sum(_tmp122, 0).reshape(1, X0BLOCK_SUB)
        tmp125 = tl.sum(_tmp125, 0).reshape(1, X0BLOCK_SUB)
        tmp141 = tl.sum(_tmp141, 0).reshape(1, X0BLOCK_SUB)
        tmp144 = tl.sum(_tmp144, 0).reshape(1, X0BLOCK_SUB)
        tmp160 = tl.sum(_tmp160, 0).reshape(1, X0BLOCK_SUB)
        tmp163 = tl.sum(_tmp163, 0).reshape(1, X0BLOCK_SUB)
        tmp179 = tl.sum(_tmp179, 0).reshape(1, X0BLOCK_SUB)
        tmp182 = tl.sum(_tmp182, 0).reshape(1, X0BLOCK_SUB)
        tmp198 = tl.sum(_tmp198, 0).reshape(1, X0BLOCK_SUB)
        tmp201 = tl.sum(_tmp201, 0).reshape(1, X0BLOCK_SUB)
        tmp217 = tl.sum(_tmp217, 0).reshape(1, X0BLOCK_SUB)
        tmp220 = tl.sum(_tmp220, 0).reshape(1, X0BLOCK_SUB)
        tmp222 = tmp8 + tmp27
        tmp223 = tmp222 + tmp46
        tmp224 = tmp223 + tmp65
        tmp225 = tmp224 + tmp84
        tmp226 = tmp225 + tmp103
        tmp227 = tmp226 + tmp141
        tmp228 = tmp227 + tmp179
        tmp229 = tmp228 + tmp217
        tmp230 = tmp229 + tmp122
        tmp231 = tmp230 + tmp160
        tmp232 = tmp231 + tmp198
        tmp233 = tmp11 + tmp30
        tmp234 = tmp233 + tmp49
        tmp235 = tmp234 + tmp68
        tmp236 = tmp235 + tmp87
        tmp237 = tmp236 + tmp106
        tmp238 = tmp237 + tmp144
        tmp239 = tmp238 + tmp182
        tmp240 = tmp239 + tmp220
        tmp241 = tmp240 + tmp125
        tmp242 = tmp241 + tmp163
        tmp243 = tmp242 + tmp201
        tl.store(in_out_ptr0 + (x0), tmp232, x0_mask)
        tl.store(in_out_ptr1 + (x0), tmp243, x0_mask)
