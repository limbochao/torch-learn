import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(isnan, convert_element_type_175, arg6_1, arg7_1, arg5_1, arg1656_1, arg1654_1, cat_215, addmm_495, where_314, arg1708_1, addmm_498, arg1715_1, addmm_489, arg1694_1, addmm_492, arg1701_1, addmm_486, arg1687_1, arg1675_1, add_16274, add_16298, arg1676_1, add_16226, add_16250, add_16202, add_15184, where_316, convert_element_type_207, isnan_2, convert_element_type_183, add_15316, convert_element_type_209, isnan_3, convert_element_type_187, add_15382, convert_element_type_211, isnan_4, convert_element_type_191, add_15448, convert_element_type_213, isnan_1, convert_element_type_179, add_15250, clone_default_8, clone_default_7, clone_default_6, clone_default_5, clone_default_9, where_304, bitwise_or_14, neg_4, where_277, where_295, bitwise_or_10, neg, where_265, where_287, bitwise_or_11, neg_1, where_268, where_289, bitwise_or_12, neg_2, where_271, where_291, bitwise_or_13, neg_3, where_274, where_293, clone_default_3, clone_default_4, clone_default_2, clone_default_1, clone_default, arg13_1, arg1555_1, arg1564_1, arg1573_1, arg1582_1, arg1591_1, mm_default_62, mm_default_53, mm_default_44, mm_default_35, mm_default_26):
    arg4_1 = isnan.shape[0]
    arg124_1 = isnan.shape[0]
    eq_16 = torch.ops.aten.eq.Scalar(arg6_1, 96)
    eq_11 = torch.ops.aten.eq.Scalar(arg6_1, 412)
    eq_13 = torch.ops.aten.eq.Scalar(arg7_1, 102)
    logical_and = torch.ops.aten.logical_and.default(eq_11, eq_13)
    logical_or_2 = torch.ops.aten.logical_or.default(eq_16, logical_and)
    full_default_7 = torch.ops.aten.full.default([arg4_1, 1], 9998, dtype=torch.int64, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    where = torch.ops.aten.where.self(logical_or_2, full_default_7, arg5_1)
    full_default_13 = torch.ops.aten.full.default([arg4_1, 1], 0, dtype=torch.int64, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    eq_55 = torch.ops.aten.eq.Tensor(where, full_default_13)
    full_default_14 = torch.ops.aten.full.default([arg4_1, 1], 1, dtype=torch.int64, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    full_default_201 = torch.ops.aten.full.default([arg4_1], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    le = torch.ops.aten.le.Scalar(arg13_1, 0)
    where_8 = torch.ops.aten.where.self(le, full_default_13, full_default_14)
    full_default_221 = torch.ops.aten.full.default([arg4_1, 1], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    eq_8102 = torch.ops.aten.eq.Scalar(where_8, 1)
    add_tensor_62 = torch.ops.aten.add.Tensor(arg1555_1, mm_default_62)
    sub_4770 = torch.ops.aten.sub.Tensor(0.0, add_tensor_62)
    exp = torch.ops.aten.exp.default(sub_4770)
    add_14555 = torch.ops.aten.add.Tensor(exp, 1)
    log = torch.ops.aten.log.default(add_14555)
    neg = torch.ops.aten.neg.default(log)
    where_265 = torch.ops.aten.where.self(eq_8102, neg, full_default_221)
    squeeze_271 = torch.ops.aten.squeeze.dim(where_265, 1)
    full_default_309 = torch.ops.aten.full.default([], -2.197265625, dtype=torch.float16, layout=torch.strided, device=torch.device('cpu'), pin_memory=False)
    eq_8141 = torch.ops.aten.eq.Scalar(where_8, 1)
    add_tensor_53 = torch.ops.aten.add.Tensor(arg1564_1, mm_default_53)
    sub_4797 = torch.ops.aten.sub.Tensor(0.0, add_tensor_53)
    exp_1 = torch.ops.aten.exp.default(sub_4797)
    add_14629 = torch.ops.aten.add.Tensor(exp_1, 1)
    log_1 = torch.ops.aten.log.default(add_14629)
    neg_1 = torch.ops.aten.neg.default(log_1)
    where_268 = torch.ops.aten.where.self(eq_8141, neg_1, full_default_221)
    squeeze_276 = torch.ops.aten.squeeze.dim(where_268, 1)
    eq_8180 = torch.ops.aten.eq.Scalar(where_8, 1)
    add_tensor_44 = torch.ops.aten.add.Tensor(arg1573_1, mm_default_44)
    sub_4824 = torch.ops.aten.sub.Tensor(0.0, add_tensor_44)
    exp_2 = torch.ops.aten.exp.default(sub_4824)
    add_14703 = torch.ops.aten.add.Tensor(exp_2, 1)
    log_2 = torch.ops.aten.log.default(add_14703)
    neg_2 = torch.ops.aten.neg.default(log_2)
    where_271 = torch.ops.aten.where.self(eq_8180, neg_2, full_default_221)
    squeeze_281 = torch.ops.aten.squeeze.dim(where_271, 1)
    eq_8219 = torch.ops.aten.eq.Scalar(where_8, 1)
    add_tensor_35 = torch.ops.aten.add.Tensor(arg1582_1, mm_default_35)
    sub_4851 = torch.ops.aten.sub.Tensor(0.0, add_tensor_35)
    exp_3 = torch.ops.aten.exp.default(sub_4851)
    add_14777 = torch.ops.aten.add.Tensor(exp_3, 1)
    log_3 = torch.ops.aten.log.default(add_14777)
    neg_3 = torch.ops.aten.neg.default(log_3)
    where_274 = torch.ops.aten.where.self(eq_8219, neg_3, full_default_221)
    squeeze_286 = torch.ops.aten.squeeze.dim(where_274, 1)
    eq_8259 = torch.ops.aten.eq.Scalar(where_8, 1)
    add_tensor_26 = torch.ops.aten.add.Tensor(arg1591_1, mm_default_26)
    sub_4879 = torch.ops.aten.sub.Tensor(0.0, add_tensor_26)
    exp_4 = torch.ops.aten.exp.default(sub_4879)
    add_14854 = torch.ops.aten.add.Tensor(exp_4, 1)
    log_4 = torch.ops.aten.log.default(add_14854)
    neg_4 = torch.ops.aten.neg.default(log_4)
    where_277 = torch.ops.aten.where.self(eq_8259, neg_4, full_default_221)
    squeeze_291 = torch.ops.aten.squeeze.dim(where_277, 1)
    where_286 = torch.ops.aten.where.self(isnan, full_default_201, convert_element_type_175)
    add_15184 = torch.ops.aten.add.Tensor(where_286, full_default_309)
    view_728 = torch.ops.aten.reshape.default(arg1675_1, [-1])
    eq_9373 = torch.ops.aten.eq.Scalar(view_728, 30)
    squeeze_dims_10 = torch.ops.aten.squeeze.dims(addmm_495, [1])
    view_722 = torch.ops.aten.reshape.default(arg1656_1, [1, arg124_1, 1])
    squeeze_dims_9 = torch.ops.aten.squeeze.dims(view_722, [0])
    view_719 = torch.ops.aten.reshape.default(arg1654_1, [1, arg124_1, 1])
    squeeze_dims_8 = torch.ops.aten.squeeze.dims(view_719, [0])
    add_16081 = torch.ops.aten.add.Tensor(squeeze_dims_9, squeeze_dims_8)
    view_725 = torch.ops.aten.reshape.default(cat_215, [3, arg124_1, 1])
    sum_265 = torch.ops.aten.sum.dim_IntList(view_725, [0])
    add_16085 = torch.ops.aten.add.Tensor(sum_265, squeeze_dims_8)
    where_314 = torch.ops.aten.where.self(eq_55, add_16081, add_16085)
    squeeze_330 = torch.ops.aten.squeeze.dim(where_314, 1)
    mul_10562 = torch.ops.aten.mul.Tensor(squeeze_330, arg1708_1)
    add_16274 = torch.ops.aten.add.Tensor(squeeze_dims_10, mul_10562)
    squeeze_dims_7 = torch.ops.aten.squeeze.dims(addmm_498, [1])
    squeeze_331 = torch.ops.aten.squeeze.dim(where_314, 1)
    mul_10576 = torch.ops.aten.mul.Tensor(squeeze_331, arg1715_1)
    add_16298 = torch.ops.aten.add.Tensor(squeeze_dims_7, mul_10576)
    full_default_326 = torch.ops.aten.full.default([], 1000, dtype=torch.int64, layout=torch.strided, device=torch.device('cpu'), pin_memory=False)
    div_4_replacement = torch.ops.aten.div.Tensor(arg1676_1, full_default_326)
    view_729_replacement = torch.ops.aten.reshape.default(div_4_replacement, [-1])
    convert_element_type_204 = torch.ops.prims.convert_element_type.default(view_729_replacement, torch.float32)
    full_default_327 = torch.ops.aten.full.default([], 1024.0, dtype=torch.float32, layout=torch.strided, device=torch.device('cpu'), pin_memory=False)
    div_5 = torch.ops.aten.div.Tensor(convert_element_type_204, full_default_327)
    convert_element_type_205 = torch.ops.prims.convert_element_type.default(div_5, torch.float16)
    mul_10582 = torch.ops.aten.mul.Tensor(add_16298, convert_element_type_205)
    add_16308 = torch.ops.aten.add.Tensor(add_16274, mul_10582)
    eq_9371 = torch.ops.aten.eq.Scalar(view_728, 26)
    squeeze_dims_6 = torch.ops.aten.squeeze.dims(addmm_489, [1])
    squeeze_328 = torch.ops.aten.squeeze.dim(where_314, 1)
    mul_10534 = torch.ops.aten.mul.Tensor(squeeze_328, arg1694_1)
    add_16226 = torch.ops.aten.add.Tensor(squeeze_dims_6, mul_10534)
    squeeze_dims_5 = torch.ops.aten.squeeze.dims(addmm_492, [1])
    squeeze_329 = torch.ops.aten.squeeze.dim(where_314, 1)
    mul_10548 = torch.ops.aten.mul.Tensor(squeeze_329, arg1701_1)
    add_16250 = torch.ops.aten.add.Tensor(squeeze_dims_5, mul_10548)
    mul_10579 = torch.ops.aten.mul.Tensor(add_16250, convert_element_type_205)
    add_16303 = torch.ops.aten.add.Tensor(add_16226, mul_10579)
    squeeze_dims_4 = torch.ops.aten.squeeze.dims(addmm_486, [1])
    squeeze_327 = torch.ops.aten.squeeze.dim(where_314, 1)
    mul_10520 = torch.ops.aten.mul.Tensor(squeeze_327, arg1687_1)
    add_16202 = torch.ops.aten.add.Tensor(squeeze_dims_4, mul_10520)
    where_315 = torch.ops.aten.where.self(eq_9371, add_16303, add_16202)
    where_316 = torch.ops.aten.where.self(eq_9373, add_16308, where_315)
    add_16317 = torch.ops.aten.add.Tensor(add_15184, where_316)
    convert_element_type_206 = torch.ops.prims.convert_element_type.default(add_16317, torch.float32)
    clamp_min_26 = torch.ops.aten.clamp_min.default(convert_element_type_206, -15)
    clamp_max_24 = torch.ops.aten.clamp_max.default(clamp_min_26, 15)
    convert_element_type_207 = torch.ops.prims.convert_element_type.default(clamp_max_24, torch.float16)
    neg_25 = torch.ops.aten.neg.default(convert_element_type_207)
    exp_15 = torch.ops.aten.exp.default(neg_25)
    add_16326 = torch.ops.aten.add.Tensor(exp_15, 1)
    reciprocal_20 = torch.ops.aten.reciprocal.default(add_16326)
    clone_default_13 = torch.ops.aten.clone.default(reciprocal_20)
    where_290 = torch.ops.aten.where.self(isnan_2, full_default_201, convert_element_type_183)
    add_15316 = torch.ops.aten.add.Tensor(where_290, full_default_309)
    add_16337 = torch.ops.aten.add.Tensor(add_15316, where_316)
    convert_element_type_208 = torch.ops.prims.convert_element_type.default(add_16337, torch.float32)
    clamp_min_27 = torch.ops.aten.clamp_min.default(convert_element_type_208, -15)
    clamp_max_25 = torch.ops.aten.clamp_max.default(clamp_min_27, 15)
    convert_element_type_209 = torch.ops.prims.convert_element_type.default(clamp_max_25, torch.float16)
    neg_26 = torch.ops.aten.neg.default(convert_element_type_209)
    exp_16 = torch.ops.aten.exp.default(neg_26)
    add_16346 = torch.ops.aten.add.Tensor(exp_16, 1)
    reciprocal_21 = torch.ops.aten.reciprocal.default(add_16346)
    clone_default_12 = torch.ops.aten.clone.default(reciprocal_21)
    where_292 = torch.ops.aten.where.self(isnan_3, full_default_201, convert_element_type_187)
    add_15382 = torch.ops.aten.add.Tensor(where_292, full_default_309)
    add_16357 = torch.ops.aten.add.Tensor(add_15382, where_316)
    convert_element_type_210 = torch.ops.prims.convert_element_type.default(add_16357, torch.float32)
    clamp_min_28 = torch.ops.aten.clamp_min.default(convert_element_type_210, -15)
    clamp_max_26 = torch.ops.aten.clamp_max.default(clamp_min_28, 15)
    convert_element_type_211 = torch.ops.prims.convert_element_type.default(clamp_max_26, torch.float16)
    neg_27 = torch.ops.aten.neg.default(convert_element_type_211)
    exp_17 = torch.ops.aten.exp.default(neg_27)
    add_16366 = torch.ops.aten.add.Tensor(exp_17, 1)
    reciprocal_22 = torch.ops.aten.reciprocal.default(add_16366)
    clone_default_11 = torch.ops.aten.clone.default(reciprocal_22)
    where_294 = torch.ops.aten.where.self(isnan_4, full_default_201, convert_element_type_191)
    add_15448 = torch.ops.aten.add.Tensor(where_294, full_default_309)
    add_16377 = torch.ops.aten.add.Tensor(add_15448, where_316)
    convert_element_type_212 = torch.ops.prims.convert_element_type.default(add_16377, torch.float32)
    clamp_min_29 = torch.ops.aten.clamp_min.default(convert_element_type_212, -15)
    clamp_max_27 = torch.ops.aten.clamp_max.default(clamp_min_29, 15)
    convert_element_type_213 = torch.ops.prims.convert_element_type.default(clamp_max_27, torch.float16)
    neg_28 = torch.ops.aten.neg.default(convert_element_type_213)
    exp_18 = torch.ops.aten.exp.default(neg_28)
    add_16386 = torch.ops.aten.add.Tensor(exp_18, 1)
    reciprocal_23 = torch.ops.aten.reciprocal.default(add_16386)
    clone_default_10 = torch.ops.aten.clone.default(reciprocal_23)
    squeeze_304 = torch.ops.aten.squeeze.dims(arg6_1, [1])
    eq_9181 = torch.ops.aten.eq.Scalar(squeeze_304, 412)
    neg_23 = torch.ops.aten.neg.default(add_15448)
    exp_13 = torch.ops.aten.exp.default(neg_23)
    add_15791 = torch.ops.aten.add.Tensor(exp_13, 1)
    reciprocal_18 = torch.ops.aten.reciprocal.default(add_15791)
    clone_default_9 = torch.ops.aten.clone.default(reciprocal_18)
    squeeze_302 = torch.ops.aten.squeeze.dim(arg6_1, 1)
    eq_9173 = torch.ops.aten.eq.Scalar(squeeze_302, 412)
    squeeze_303 = torch.ops.aten.squeeze.dim(arg7_1, 1)
    eq_9176 = torch.ops.aten.eq.Scalar(squeeze_303, 102)
    logical_and_6 = torch.ops.aten.logical_and.default(eq_9173, eq_9176)
    eq_9170 = torch.ops.aten.eq.Scalar(squeeze_302, 96)
    neg_15 = torch.ops.aten.neg.default(add_15184)
    exp_5 = torch.ops.aten.exp.default(neg_15)
    add_15687 = torch.ops.aten.add.Tensor(exp_5, 1)
    reciprocal_10 = torch.ops.aten.reciprocal.default(add_15687)
    clone_default_8 = torch.ops.aten.clone.default(reciprocal_10)
    eq_9167 = torch.ops.aten.eq.Scalar(squeeze_302, 167)
    where_288 = torch.ops.aten.where.self(isnan_1, full_default_201, convert_element_type_179)
    add_15250 = torch.ops.aten.add.Tensor(where_288, full_default_309)
    neg_17 = torch.ops.aten.neg.default(add_15250)
    exp_7 = torch.ops.aten.exp.default(neg_17)
    add_15713 = torch.ops.aten.add.Tensor(exp_7, 1)
    reciprocal_12 = torch.ops.aten.reciprocal.default(add_15713)
    clone_default_7 = torch.ops.aten.clone.default(reciprocal_12)
    eq_9164 = torch.ops.aten.eq.Scalar(squeeze_302, 172)
    neg_19 = torch.ops.aten.neg.default(add_15316)
    exp_9 = torch.ops.aten.exp.default(neg_19)
    add_15739 = torch.ops.aten.add.Tensor(exp_9, 1)
    reciprocal_14 = torch.ops.aten.reciprocal.default(add_15739)
    clone_default_6 = torch.ops.aten.clone.default(reciprocal_14)
    neg_21 = torch.ops.aten.neg.default(add_15382)
    exp_11 = torch.ops.aten.exp.default(neg_21)
    add_15765 = torch.ops.aten.add.Tensor(exp_11, 1)
    reciprocal_16 = torch.ops.aten.reciprocal.default(add_15765)
    clone_default_5 = torch.ops.aten.clone.default(reciprocal_16)
    where_301 = torch.ops.aten.where.self(eq_9164, clone_default_6, clone_default_5)
    where_302 = torch.ops.aten.where.self(eq_9167, clone_default_7, where_301)
    where_303 = torch.ops.aten.where.self(eq_9170, clone_default_8, where_302)
    where_304 = torch.ops.aten.where.self(logical_and_6, clone_default_8, where_303)
    where_305 = torch.ops.aten.where.self(eq_9181, clone_default_9, where_304)
    eq_9482 = torch.ops.aten.eq.Scalar(squeeze_302, 412)
    bitwise_and_10 = torch.ops.aten.bitwise_and.Tensor(eq_9482, logical_and_6)
    eq_9479 = torch.ops.aten.eq.Scalar(squeeze_302, 412)
    squeeze_292 = torch.ops.aten.squeeze.dim(neg_4, 1)
    full_default_307 = torch.ops.aten.full.default([arg4_1], float('nan'), dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    sub_5119 = torch.ops.aten.sub.Tensor(squeeze_292, squeeze_291)
    add_15453 = torch.ops.aten.add.Tensor(add_15448, sub_5119)
    convert_element_type_192 = torch.ops.prims.convert_element_type.default(add_15453, torch.float32)
    clamp_min_20 = torch.ops.aten.clamp_min.default(convert_element_type_192, -15)
    clamp_max_18 = torch.ops.aten.clamp_max.default(clamp_min_20, 15)
    convert_element_type_193 = torch.ops.prims.convert_element_type.default(clamp_max_18, torch.float16)
    where_295 = torch.ops.aten.where.self(bitwise_or_14, full_default_307, convert_element_type_193)
    neg_24 = torch.ops.aten.neg.default(where_295)
    exp_14 = torch.ops.aten.exp.default(neg_24)
    add_15804 = torch.ops.aten.add.Tensor(exp_14, 1)
    reciprocal_19 = torch.ops.aten.reciprocal.default(add_15804)
    clone_default_4 = torch.ops.aten.clone.default(reciprocal_19)
    eq_9476 = torch.ops.aten.eq.Scalar(squeeze_302, 96)
    squeeze_272 = torch.ops.aten.squeeze.dim(neg, 1)
    full_default_291 = torch.ops.aten.full.default([arg4_1], float('nan'), dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    sub_4999 = torch.ops.aten.sub.Tensor(squeeze_272, squeeze_271)
    add_15189 = torch.ops.aten.add.Tensor(add_15184, sub_4999)
    convert_element_type_176 = torch.ops.prims.convert_element_type.default(add_15189, torch.float32)
    clamp_min_12 = torch.ops.aten.clamp_min.default(convert_element_type_176, -15)
    clamp_max_10 = torch.ops.aten.clamp_max.default(clamp_min_12, 15)
    convert_element_type_177 = torch.ops.prims.convert_element_type.default(clamp_max_10, torch.float16)
    where_287 = torch.ops.aten.where.self(bitwise_or_10, full_default_291, convert_element_type_177)
    neg_16 = torch.ops.aten.neg.default(where_287)
    exp_6 = torch.ops.aten.exp.default(neg_16)
    add_15700 = torch.ops.aten.add.Tensor(exp_6, 1)
    reciprocal_11 = torch.ops.aten.reciprocal.default(add_15700)
    clone_default_3 = torch.ops.aten.clone.default(reciprocal_11)
    eq_9473 = torch.ops.aten.eq.Scalar(squeeze_302, 167)
    squeeze_277 = torch.ops.aten.squeeze.dim(neg_1, 1)
    full_default_295 = torch.ops.aten.full.default([arg4_1], float('nan'), dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    sub_5029 = torch.ops.aten.sub.Tensor(squeeze_277, squeeze_276)
    add_15255 = torch.ops.aten.add.Tensor(add_15250, sub_5029)
    convert_element_type_180 = torch.ops.prims.convert_element_type.default(add_15255, torch.float32)
    clamp_min_14 = torch.ops.aten.clamp_min.default(convert_element_type_180, -15)
    clamp_max_12 = torch.ops.aten.clamp_max.default(clamp_min_14, 15)
    convert_element_type_181 = torch.ops.prims.convert_element_type.default(clamp_max_12, torch.float16)
    where_289 = torch.ops.aten.where.self(bitwise_or_11, full_default_295, convert_element_type_181)
    neg_18 = torch.ops.aten.neg.default(where_289)
    exp_8 = torch.ops.aten.exp.default(neg_18)
    add_15726 = torch.ops.aten.add.Tensor(exp_8, 1)
    reciprocal_13 = torch.ops.aten.reciprocal.default(add_15726)
    clone_default_2 = torch.ops.aten.clone.default(reciprocal_13)
    eq_9470 = torch.ops.aten.eq.Scalar(squeeze_302, 172)
    squeeze_282 = torch.ops.aten.squeeze.dim(neg_2, 1)
    full_default_299 = torch.ops.aten.full.default([arg4_1], float('nan'), dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    sub_5059 = torch.ops.aten.sub.Tensor(squeeze_282, squeeze_281)
    add_15321 = torch.ops.aten.add.Tensor(add_15316, sub_5059)
    convert_element_type_184 = torch.ops.prims.convert_element_type.default(add_15321, torch.float32)
    clamp_min_16 = torch.ops.aten.clamp_min.default(convert_element_type_184, -15)
    clamp_max_14 = torch.ops.aten.clamp_max.default(clamp_min_16, 15)
    convert_element_type_185 = torch.ops.prims.convert_element_type.default(clamp_max_14, torch.float16)
    where_291 = torch.ops.aten.where.self(bitwise_or_12, full_default_299, convert_element_type_185)
    neg_20 = torch.ops.aten.neg.default(where_291)
    exp_10 = torch.ops.aten.exp.default(neg_20)
    add_15752 = torch.ops.aten.add.Tensor(exp_10, 1)
    reciprocal_15 = torch.ops.aten.reciprocal.default(add_15752)
    clone_default_1 = torch.ops.aten.clone.default(reciprocal_15)
    squeeze_287 = torch.ops.aten.squeeze.dim(neg_3, 1)
    full_default_303 = torch.ops.aten.full.default([arg4_1], float('nan'), dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    sub_5089 = torch.ops.aten.sub.Tensor(squeeze_287, squeeze_286)
    add_15387 = torch.ops.aten.add.Tensor(add_15382, sub_5089)
    convert_element_type_188 = torch.ops.prims.convert_element_type.default(add_15387, torch.float32)
    clamp_min_18 = torch.ops.aten.clamp_min.default(convert_element_type_188, -15)
    clamp_max_16 = torch.ops.aten.clamp_max.default(clamp_min_18, 15)
    convert_element_type_189 = torch.ops.prims.convert_element_type.default(clamp_max_16, torch.float16)
    where_293 = torch.ops.aten.where.self(bitwise_or_13, full_default_303, convert_element_type_189)
    neg_22 = torch.ops.aten.neg.default(where_293)
    exp_12 = torch.ops.aten.exp.default(neg_22)
    add_15778 = torch.ops.aten.add.Tensor(exp_12, 1)
    reciprocal_17 = torch.ops.aten.reciprocal.default(add_15778)
    clone_default = torch.ops.aten.clone.default(reciprocal_17)
    where_317 = torch.ops.aten.where.self(eq_9470, clone_default_1, clone_default)
    where_318 = torch.ops.aten.where.self(eq_9473, clone_default_2, where_317)
    where_319 = torch.ops.aten.where.self(eq_9476, clone_default_3, where_318)
    where_320 = torch.ops.aten.where.self(eq_9479, clone_default_4, where_319)
    where_321 = torch.ops.aten.where.self(bitwise_and_10, clone_default_3, where_320)
    eq_9492 = torch.ops.aten.eq.Scalar(squeeze_302, 96)
    eq_9489 = torch.ops.aten.eq.Scalar(squeeze_302, 412)
    eq_9486 = torch.ops.aten.eq.Scalar(squeeze_302, 172)
    where_322 = torch.ops.aten.where.self(eq_9486, add_15316, add_15382)
    where_323 = torch.ops.aten.where.self(eq_9489, add_15448, where_322)
    where_324 = torch.ops.aten.where.self(eq_9492, add_15184, where_323)
    return add_15184,where_314,add_16274,add_16298,add_16226,add_16250,add_16202,where_316,convert_element_type_207,clone_default_13,add_15316,convert_element_type_209,clone_default_12,add_15382,convert_element_type_211,clone_default_11,add_15448,convert_element_type_213,clone_default_10,clone_default_9,clone_default_8,add_15250,clone_default_7,clone_default_6,clone_default_5,where_304,where_305,where_295,clone_default_4,where_287,clone_default_3,where_289,clone_default_2,where_291,clone_default_1,where_293,clone_default,where_321,where_324


SAMPLE_BINDINGS = [
    {'s0': 64},
    {'s0': 112},
    {'s0': 160},
    {'s0': 208},
    {'s0': 256},
]


COMPILE_BINDINGS = [
    {'s0': 64},
    {'s0': 256},
]


def make_inputs(binding, device):
    s0 = binding['s0']
    isnan = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.bool,
    )
    convert_element_type_175 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    arg6_1 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.int64,
    )
    arg7_1 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.int64,
    )
    arg5_1 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.int64,
    )
    arg1656_1 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1654_1 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    cat_215 = rand_strided(
        (3*s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    addmm_495 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    where_314 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1708_1 = rand_strided(
        (1,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    addmm_498 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1715_1 = rand_strided(
        (1,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    addmm_489 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1694_1 = rand_strided(
        (1,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    addmm_492 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1701_1 = rand_strided(
        (1,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    addmm_486 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1687_1 = rand_strided(
        (1,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    arg1675_1 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.int64,
    )
    add_16274 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    add_16298 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    arg1676_1 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.int64,
    )
    add_16226 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    add_16250 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    add_16202 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    add_15184 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    where_316 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    convert_element_type_207 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    isnan_2 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.bool,
    )
    convert_element_type_183 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    add_15316 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    convert_element_type_209 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    isnan_3 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.bool,
    )
    convert_element_type_187 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    add_15382 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    convert_element_type_211 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    isnan_4 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.bool,
    )
    convert_element_type_191 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    add_15448 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    convert_element_type_213 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    isnan_1 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.bool,
    )
    convert_element_type_179 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    add_15250 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    clone_default_8 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    clone_default_7 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    clone_default_6 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    clone_default_5 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    clone_default_9 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    where_304 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    bitwise_or_14 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.bool,
    )
    neg_4 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    where_277 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    where_295 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    bitwise_or_10 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.bool,
    )
    neg = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    where_265 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    where_287 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    bitwise_or_11 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.bool,
    )
    neg_1 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    where_268 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    where_289 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    bitwise_or_12 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.bool,
    )
    neg_2 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    where_271 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    where_291 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    bitwise_or_13 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.bool,
    )
    neg_3 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    where_274 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    where_293 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    clone_default_3 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    clone_default_4 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    clone_default_2 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    clone_default_1 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    clone_default = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    # These values are upstream buffers referenced by the grouped Graph fragment but omitted as placeholders.
    arg13_1 = rand_strided((s0, 1), (1, 1), device=device, dtype=torch.int64)
    arg1555_1 = rand_strided((1,), (1,), device=device, dtype=torch.float16)
    arg1564_1 = rand_strided((1,), (1,), device=device, dtype=torch.float16)
    arg1573_1 = rand_strided((1,), (1,), device=device, dtype=torch.float16)
    arg1582_1 = rand_strided((1,), (1,), device=device, dtype=torch.float16)
    arg1591_1 = rand_strided((1,), (1,), device=device, dtype=torch.float16)
    mm_default_62 = rand_strided((s0, 1), (1, 1), device=device, dtype=torch.float16)
    mm_default_53 = rand_strided((s0, 1), (1, 1), device=device, dtype=torch.float16)
    mm_default_44 = rand_strided((s0, 1), (1, 1), device=device, dtype=torch.float16)
    mm_default_35 = rand_strided((s0, 1), (1, 1), device=device, dtype=torch.float16)
    mm_default_26 = rand_strided((s0, 1), (1, 1), device=device, dtype=torch.float16)
    return (isnan, convert_element_type_175, arg6_1, arg7_1, arg5_1, arg1656_1, arg1654_1, cat_215, addmm_495, where_314, arg1708_1, addmm_498, arg1715_1, addmm_489, arg1694_1, addmm_492, arg1701_1, addmm_486, arg1687_1, arg1675_1, add_16274, add_16298, arg1676_1, add_16226, add_16250, add_16202, add_15184, where_316, convert_element_type_207, isnan_2, convert_element_type_183, add_15316, convert_element_type_209, isnan_3, convert_element_type_187, add_15382, convert_element_type_211, isnan_4, convert_element_type_191, add_15448, convert_element_type_213, isnan_1, convert_element_type_179, add_15250, clone_default_8, clone_default_7, clone_default_6, clone_default_5, clone_default_9, where_304, bitwise_or_14, neg_4, where_277, where_295, bitwise_or_10, neg, where_265, where_287, bitwise_or_11, neg_1, where_268, where_289, bitwise_or_12, neg_2, where_271, where_291, bitwise_or_13, neg_3, where_274, where_293, clone_default_3, clone_default_4, clone_default_2, clone_default_1, clone_default, arg13_1, arg1555_1, arg1564_1, arg1573_1, arg1582_1, arg1591_1, mm_default_62, mm_default_53, mm_default_44, mm_default_35, mm_default_26), {}


DYNAMIC_DIMS = {'args[0]': (0,),
 'args[11]': (0,),
 'args[13]': (0,),
 'args[15]': (0,),
 'args[17]': (0,),
 'args[19]': (0,),
 'args[1]': (0,),
 'args[20]': (0,),
 'args[21]': (0,),
 'args[22]': (0,),
 'args[23]': (0,),
 'args[24]': (0,),
 'args[25]': (0,),
 'args[26]': (0,),
 'args[27]': (0,),
 'args[28]': (0,),
 'args[29]': (0,),
 'args[2]': (0,),
 'args[30]': (0,),
 'args[31]': (0,),
 'args[32]': (0,),
 'args[33]': (0,),
 'args[34]': (0,),
 'args[35]': (0,),
 'args[36]': (0,),
 'args[37]': (0,),
 'args[38]': (0,),
 'args[39]': (0,),
 'args[3]': (0,),
 'args[40]': (0,),
 'args[41]': (0,),
 'args[42]': (0,),
 'args[43]': (0,),
 'args[44]': (0,),
 'args[45]': (0,),
 'args[46]': (0,),
 'args[47]': (0,),
 'args[48]': (0,),
 'args[49]': (0,),
 'args[4]': (0,),
 'args[50]': (0,),
 'args[51]': (0,),
 'args[52]': (0,),
 'args[53]': (0,),
 'args[54]': (0,),
 'args[55]': (0,),
 'args[56]': (0,),
 'args[57]': (0,),
 'args[58]': (0,),
 'args[59]': (0,),
 'args[5]': (0,),
 'args[60]': (0,),
 'args[61]': (0,),
 'args[62]': (0,),
 'args[63]': (0,),
 'args[64]': (0,),
 'args[65]': (0,),
 'args[66]': (0,),
 'args[67]': (0,),
 'args[68]': (0,),
 'args[69]': (0,),
 'args[6]': (0,),
 'args[70]': (0,),
 'args[71]': (0,),
 'args[72]': (0,),
 'args[73]': (0,),
 'args[74]': (0,),
 'args[75]': (0,),
 'args[81]': (0,),
 'args[82]': (0,),
 'args[83]': (0,),
 'args[84]': (0,),
 'args[85]': (0,),
 'args[7]': (0,),
 'args[8]': (0,),
 'args[9]': (0,)}


CASE = {
    'name': 'pointwise_triton_poi_fused__to_copy_add_addmm_bitwise_an_272',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
