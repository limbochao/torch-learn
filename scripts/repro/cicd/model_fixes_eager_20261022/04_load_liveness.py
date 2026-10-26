"""Albert LayerNorm parameter-gradient subgraph for eager and fresh Inductor compilation.

Source: static training performance kernel
triton_per_fused_add_native_layer_norm_backwar_14 (R=2048, X=4096).
The extracted backward ATen operations run directly as eager_forward; no backward() is needed.
Preserves 81 input layouts, 12 contributions, reduction/add order, and two final outputs.
Inputs are seeded synthetic tensors with positive rstd, rather than captured model values.
See README.md and artifacts/cicd/model_fixes_eager_20261022/extraction_manifest.json for provenance.
The file can be copied and run alone; those provenance files are not runtime dependencies.
"""
import torch


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(
    mm_2,
    add_61,
    getitem_73,
    getitem_74,
    mul_90,
    mm_10,
    mm_12,
    mm_14,
    add_56,
    getitem_67,
    getitem_68,
    mul_114,
    mm_22,
    mm_24,
    mm_26,
    add_51,
    getitem_61,
    getitem_62,
    mul_138,
    mm_34,
    mm_36,
    mm_38,
    add_46,
    getitem_55,
    getitem_56,
    mul_162,
    mm_46,
    mm_48,
    mm_50,
    add_41,
    getitem_49,
    getitem_50,
    mul_186,
    mm_58,
    mm_60,
    mm_62,
    add_36,
    getitem_43,
    getitem_44,
    mul_282,
    mm_106,
    mm_108,
    mm_110,
    add_16,
    getitem_19,
    getitem_20,
    mul_210,
    mm_70,
    mm_72,
    mm_74,
    add_31,
    getitem_37,
    getitem_38,
    mul_306,
    mm_118,
    mm_120,
    mm_122,
    add_11,
    getitem_13,
    getitem_14,
    mul_234,
    mm_82,
    mm_84,
    mm_86,
    add_26,
    getitem_31,
    getitem_32,
    mul_330,
    mm_130,
    mm_132,
    mm_134,
    add_6,
    getitem_7,
    getitem_8,
    mul_258,
    mm_94,
    mm_96,
    mm_98,
    add_21,
    getitem_25,
    getitem_26,
):
    view_278 = torch.ops.aten.reshape.default(mm_2, [4, 512, 4096])
    sub_19 = torch.ops.aten.sub.Tensor(add_61, getitem_73)
    mul_69 = torch.ops.aten.mul.Tensor(sub_19, getitem_74)
    mul_75 = torch.ops.aten.mul.Tensor(view_278, mul_69)
    sum_22 = torch.ops.aten.sum.dim_IntList(mul_75, [0, 1])
    sum_23 = torch.ops.aten.sum.dim_IntList(view_278, [0, 1])
    view_306 = torch.ops.aten.reshape.default(mm_10, [4, 512, 4096])
    add_70 = torch.ops.aten.add.Tensor(mul_90, view_306)
    view_309 = torch.ops.aten.reshape.default(mm_12, [4, 512, 4096])
    add_71 = torch.ops.aten.add.Tensor(add_70, view_309)
    view_312 = torch.ops.aten.reshape.default(mm_14, [4, 512, 4096])
    add_72 = torch.ops.aten.add.Tensor(add_71, view_312)
    sub_26 = torch.ops.aten.sub.Tensor(add_56, getitem_67)
    mul_93 = torch.ops.aten.mul.Tensor(sub_26, getitem_68)
    mul_99 = torch.ops.aten.mul.Tensor(add_72, mul_93)
    sum_37 = torch.ops.aten.sum.dim_IntList(mul_99, [0, 1])
    sum_38 = torch.ops.aten.sum.dim_IntList(add_72, [0, 1])
    add_73 = torch.ops.aten.add.Tensor(sum_22, sum_37)
    add_74 = torch.ops.aten.add.Tensor(sum_23, sum_38)
    view_340 = torch.ops.aten.reshape.default(mm_22, [4, 512, 4096])
    add_88 = torch.ops.aten.add.Tensor(mul_114, view_340)
    view_343 = torch.ops.aten.reshape.default(mm_24, [4, 512, 4096])
    add_91 = torch.ops.aten.add.Tensor(add_88, view_343)
    view_346 = torch.ops.aten.reshape.default(mm_26, [4, 512, 4096])
    add_94 = torch.ops.aten.add.Tensor(add_91, view_346)
    sub_33 = torch.ops.aten.sub.Tensor(add_51, getitem_61)
    mul_117 = torch.ops.aten.mul.Tensor(sub_33, getitem_62)
    mul_123 = torch.ops.aten.mul.Tensor(add_94, mul_117)
    sum_52 = torch.ops.aten.sum.dim_IntList(mul_123, [0, 1])
    sum_53 = torch.ops.aten.sum.dim_IntList(add_94, [0, 1])
    add_95 = torch.ops.aten.add.Tensor(add_73, sum_52)
    add_96 = torch.ops.aten.add.Tensor(add_74, sum_53)
    view_374 = torch.ops.aten.reshape.default(mm_34, [4, 512, 4096])
    add_110 = torch.ops.aten.add.Tensor(mul_138, view_374)
    view_377 = torch.ops.aten.reshape.default(mm_36, [4, 512, 4096])
    add_113 = torch.ops.aten.add.Tensor(add_110, view_377)
    view_380 = torch.ops.aten.reshape.default(mm_38, [4, 512, 4096])
    add_116 = torch.ops.aten.add.Tensor(add_113, view_380)
    sub_40 = torch.ops.aten.sub.Tensor(add_46, getitem_55)
    mul_141 = torch.ops.aten.mul.Tensor(sub_40, getitem_56)
    mul_147 = torch.ops.aten.mul.Tensor(add_116, mul_141)
    sum_67 = torch.ops.aten.sum.dim_IntList(mul_147, [0, 1])
    sum_68 = torch.ops.aten.sum.dim_IntList(add_116, [0, 1])
    add_117 = torch.ops.aten.add.Tensor(add_95, sum_67)
    add_118 = torch.ops.aten.add.Tensor(add_96, sum_68)
    view_408 = torch.ops.aten.reshape.default(mm_46, [4, 512, 4096])
    add_132 = torch.ops.aten.add.Tensor(mul_162, view_408)
    view_411 = torch.ops.aten.reshape.default(mm_48, [4, 512, 4096])
    add_135 = torch.ops.aten.add.Tensor(add_132, view_411)
    view_414 = torch.ops.aten.reshape.default(mm_50, [4, 512, 4096])
    add_138 = torch.ops.aten.add.Tensor(add_135, view_414)
    sub_47 = torch.ops.aten.sub.Tensor(add_41, getitem_49)
    mul_165 = torch.ops.aten.mul.Tensor(sub_47, getitem_50)
    mul_171 = torch.ops.aten.mul.Tensor(add_138, mul_165)
    sum_82 = torch.ops.aten.sum.dim_IntList(mul_171, [0, 1])
    sum_83 = torch.ops.aten.sum.dim_IntList(add_138, [0, 1])
    add_139 = torch.ops.aten.add.Tensor(add_117, sum_82)
    add_140 = torch.ops.aten.add.Tensor(add_118, sum_83)
    view_442 = torch.ops.aten.reshape.default(mm_58, [4, 512, 4096])
    add_154 = torch.ops.aten.add.Tensor(mul_186, view_442)
    view_445 = torch.ops.aten.reshape.default(mm_60, [4, 512, 4096])
    add_157 = torch.ops.aten.add.Tensor(add_154, view_445)
    view_448 = torch.ops.aten.reshape.default(mm_62, [4, 512, 4096])
    add_160 = torch.ops.aten.add.Tensor(add_157, view_448)
    sub_54 = torch.ops.aten.sub.Tensor(add_36, getitem_43)
    mul_189 = torch.ops.aten.mul.Tensor(sub_54, getitem_44)
    mul_195 = torch.ops.aten.mul.Tensor(add_160, mul_189)
    sum_97 = torch.ops.aten.sum.dim_IntList(mul_195, [0, 1])
    sum_98 = torch.ops.aten.sum.dim_IntList(add_160, [0, 1])
    add_161 = torch.ops.aten.add.Tensor(add_139, sum_97)
    add_162 = torch.ops.aten.add.Tensor(add_140, sum_98)
    view_476 = torch.ops.aten.reshape.default(mm_70, [4, 512, 4096])
    add_176 = torch.ops.aten.add.Tensor(mul_210, view_476)
    view_479 = torch.ops.aten.reshape.default(mm_72, [4, 512, 4096])
    add_179 = torch.ops.aten.add.Tensor(add_176, view_479)
    view_482 = torch.ops.aten.reshape.default(mm_74, [4, 512, 4096])
    add_182 = torch.ops.aten.add.Tensor(add_179, view_482)
    sub_61 = torch.ops.aten.sub.Tensor(add_31, getitem_37)
    mul_213 = torch.ops.aten.mul.Tensor(sub_61, getitem_38)
    mul_219 = torch.ops.aten.mul.Tensor(add_182, mul_213)
    sum_112 = torch.ops.aten.sum.dim_IntList(mul_219, [0, 1])
    sum_113 = torch.ops.aten.sum.dim_IntList(add_182, [0, 1])
    add_183 = torch.ops.aten.add.Tensor(add_161, sum_112)
    add_184 = torch.ops.aten.add.Tensor(add_162, sum_113)
    view_510 = torch.ops.aten.reshape.default(mm_82, [4, 512, 4096])
    add_198 = torch.ops.aten.add.Tensor(mul_234, view_510)
    view_513 = torch.ops.aten.reshape.default(mm_84, [4, 512, 4096])
    add_201 = torch.ops.aten.add.Tensor(add_198, view_513)
    view_516 = torch.ops.aten.reshape.default(mm_86, [4, 512, 4096])
    add_204 = torch.ops.aten.add.Tensor(add_201, view_516)
    sub_68 = torch.ops.aten.sub.Tensor(add_26, getitem_31)
    mul_237 = torch.ops.aten.mul.Tensor(sub_68, getitem_32)
    mul_243 = torch.ops.aten.mul.Tensor(add_204, mul_237)
    sum_127 = torch.ops.aten.sum.dim_IntList(mul_243, [0, 1])
    sum_128 = torch.ops.aten.sum.dim_IntList(add_204, [0, 1])
    add_205 = torch.ops.aten.add.Tensor(add_183, sum_127)
    add_206 = torch.ops.aten.add.Tensor(add_184, sum_128)
    view_544 = torch.ops.aten.reshape.default(mm_94, [4, 512, 4096])
    add_220 = torch.ops.aten.add.Tensor(mul_258, view_544)
    view_547 = torch.ops.aten.reshape.default(mm_96, [4, 512, 4096])
    add_223 = torch.ops.aten.add.Tensor(add_220, view_547)
    view_550 = torch.ops.aten.reshape.default(mm_98, [4, 512, 4096])
    add_226 = torch.ops.aten.add.Tensor(add_223, view_550)
    sub_75 = torch.ops.aten.sub.Tensor(add_21, getitem_25)
    mul_261 = torch.ops.aten.mul.Tensor(sub_75, getitem_26)
    mul_267 = torch.ops.aten.mul.Tensor(add_226, mul_261)
    sum_142 = torch.ops.aten.sum.dim_IntList(mul_267, [0, 1])
    sum_143 = torch.ops.aten.sum.dim_IntList(add_226, [0, 1])
    add_227 = torch.ops.aten.add.Tensor(add_205, sum_142)
    add_228 = torch.ops.aten.add.Tensor(add_206, sum_143)
    view_578 = torch.ops.aten.reshape.default(mm_106, [4, 512, 4096])
    add_242 = torch.ops.aten.add.Tensor(mul_282, view_578)
    view_581 = torch.ops.aten.reshape.default(mm_108, [4, 512, 4096])
    add_245 = torch.ops.aten.add.Tensor(add_242, view_581)
    view_584 = torch.ops.aten.reshape.default(mm_110, [4, 512, 4096])
    add_248 = torch.ops.aten.add.Tensor(add_245, view_584)
    sub_82 = torch.ops.aten.sub.Tensor(add_16, getitem_19)
    mul_285 = torch.ops.aten.mul.Tensor(sub_82, getitem_20)
    mul_291 = torch.ops.aten.mul.Tensor(add_248, mul_285)
    sum_157 = torch.ops.aten.sum.dim_IntList(mul_291, [0, 1])
    sum_158 = torch.ops.aten.sum.dim_IntList(add_248, [0, 1])
    add_249 = torch.ops.aten.add.Tensor(add_227, sum_157)
    add_250 = torch.ops.aten.add.Tensor(add_228, sum_158)
    view_612 = torch.ops.aten.reshape.default(mm_118, [4, 512, 4096])
    add_264 = torch.ops.aten.add.Tensor(mul_306, view_612)
    view_615 = torch.ops.aten.reshape.default(mm_120, [4, 512, 4096])
    add_267 = torch.ops.aten.add.Tensor(add_264, view_615)
    view_618 = torch.ops.aten.reshape.default(mm_122, [4, 512, 4096])
    add_270 = torch.ops.aten.add.Tensor(add_267, view_618)
    sub_89 = torch.ops.aten.sub.Tensor(add_11, getitem_13)
    mul_309 = torch.ops.aten.mul.Tensor(sub_89, getitem_14)
    mul_315 = torch.ops.aten.mul.Tensor(add_270, mul_309)
    sum_172 = torch.ops.aten.sum.dim_IntList(mul_315, [0, 1])
    sum_173 = torch.ops.aten.sum.dim_IntList(add_270, [0, 1])
    add_271 = torch.ops.aten.add.Tensor(add_249, sum_172)
    add_272 = torch.ops.aten.add.Tensor(add_250, sum_173)
    view_646 = torch.ops.aten.reshape.default(mm_130, [4, 512, 4096])
    add_286 = torch.ops.aten.add.Tensor(mul_330, view_646)
    view_649 = torch.ops.aten.reshape.default(mm_132, [4, 512, 4096])
    add_289 = torch.ops.aten.add.Tensor(add_286, view_649)
    view_652 = torch.ops.aten.reshape.default(mm_134, [4, 512, 4096])
    add_292 = torch.ops.aten.add.Tensor(add_289, view_652)
    sub_96 = torch.ops.aten.sub.Tensor(add_6, getitem_7)
    mul_333 = torch.ops.aten.mul.Tensor(sub_96, getitem_8)
    mul_339 = torch.ops.aten.mul.Tensor(add_292, mul_333)
    sum_187 = torch.ops.aten.sum.dim_IntList(mul_339, [0, 1])
    sum_188 = torch.ops.aten.sum.dim_IntList(add_292, [0, 1])
    add_293 = torch.ops.aten.add.Tensor(add_271, sum_187)
    add_294 = torch.ops.aten.add.Tensor(add_272, sum_188)
    return add_293,add_294


DEFAULT_DYNAMIC = False
INPUT_NAMES = (
    'mm_2',
    'add_61',
    'getitem_73',
    'getitem_74',
    'mul_90',
    'mm_10',
    'mm_12',
    'mm_14',
    'add_56',
    'getitem_67',
    'getitem_68',
    'mul_114',
    'mm_22',
    'mm_24',
    'mm_26',
    'add_51',
    'getitem_61',
    'getitem_62',
    'mul_138',
    'mm_34',
    'mm_36',
    'mm_38',
    'add_46',
    'getitem_55',
    'getitem_56',
    'mul_162',
    'mm_46',
    'mm_48',
    'mm_50',
    'add_41',
    'getitem_49',
    'getitem_50',
    'mul_186',
    'mm_58',
    'mm_60',
    'mm_62',
    'add_36',
    'getitem_43',
    'getitem_44',
    'mul_282',
    'mm_106',
    'mm_108',
    'mm_110',
    'add_16',
    'getitem_19',
    'getitem_20',
    'mul_210',
    'mm_70',
    'mm_72',
    'mm_74',
    'add_31',
    'getitem_37',
    'getitem_38',
    'mul_306',
    'mm_118',
    'mm_120',
    'mm_122',
    'add_11',
    'getitem_13',
    'getitem_14',
    'mul_234',
    'mm_82',
    'mm_84',
    'mm_86',
    'add_26',
    'getitem_31',
    'getitem_32',
    'mul_330',
    'mm_130',
    'mm_132',
    'mm_134',
    'add_6',
    'getitem_7',
    'getitem_8',
    'mul_258',
    'mm_94',
    'mm_96',
    'mm_98',
    'add_21',
    'getitem_25',
    'getitem_26',
)


def make_inputs(device):
    specs = (
        ((2048, 4096), (4096, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((2048, 4096), (4096, 1), torch.float32),
        ((4, 512, 4096), (2097152, 4096, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
        ((4, 512, 1), (512, 1, 1), torch.float32),
    )
    inputs = []
    for shape, stride, dtype in specs:
        value = torch.empty_strided(shape, stride, dtype=dtype, device=device)
        if dtype.is_floating_point:
            value.normal_()
        else:
            value.zero_()
        inputs.append(value)
    inputs[3].abs_().add_(0.5)
    inputs[10].abs_().add_(0.5)
    inputs[17].abs_().add_(0.5)
    inputs[24].abs_().add_(0.5)
    inputs[31].abs_().add_(0.5)
    inputs[38].abs_().add_(0.5)
    inputs[45].abs_().add_(0.5)
    inputs[52].abs_().add_(0.5)
    inputs[59].abs_().add_(0.5)
    inputs[66].abs_().add_(0.5)
    inputs[73].abs_().add_(0.5)
    inputs[80].abs_().add_(0.5)
    return tuple(inputs)


def main():
    import argparse
    from torch.utils import _pytree

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action="store_true", help="compile the extracted ATen graph with Inductor")
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=DEFAULT_DYNAMIC)
    parser.add_argument(
        "--persistent",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable/disable the persistent reduction heuristic; default uses the installed compiler setting",
    )
    args = parser.parse_args()
    # Register the NPU device before asking torch.device to parse its name.
    if args.device.split(":", 1)[0] == "npu":
        import torch_npu
        print(f"torch_npu={torch_npu.__version__}", flush=True)
    device = torch.device(args.device)
    torch.manual_seed(0)
    print(f"torch={torch.__version__} cuda={torch.version.cuda} device={args.device} dynamic={args.dynamic}")
    if device.type in ("cuda", "npu"):
        import triton
        device_api = getattr(torch, device.type)
        print(f"hardware={device_api.get_device_name(device)} triton={triton.__version__}", flush=True)
    inputs = make_inputs(args.device)
    for name, value in zip(INPUT_NAMES, inputs):
        print(f"input={name} shape={tuple(value.shape)} stride={value.stride()} dtype={value.dtype}")
    expected = eager_forward(*inputs)
    leaves = _pytree.tree_leaves(expected)
    for value in leaves:
        assert torch.isfinite(value).all().item(), "non-finite eager result"
    print(f"PASS eager outputs={[tuple(value.shape) for value in leaves]}", flush=True)
    if args.compile:
        from torch._inductor import config

        overrides = {}
        if args.persistent is not None:
            overrides["triton.persistent_reductions"] = args.persistent
        with config.patch(overrides):
            print(f"persistent_reductions={config.triton.persistent_reductions}", flush=True)
            compiled = torch.compile(eager_forward, backend="inductor", fullgraph=True, dynamic=args.dynamic)
            actual = compiled(*inputs)
            if device.type in ("cuda", "npu"):
                device_api.synchronize(device)
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
        errors = [(a - b).abs().max().item() for a, b in zip(actual, expected)]
        print(f"PASS compiled_vs_eager max_abs={errors} rtol=0.002 atol=0.002", flush=True)


if __name__ == "__main__":
    main()
