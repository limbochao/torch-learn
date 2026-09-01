import torch
from math import inf, nan
from cmath import nanj
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_87, arg18_1):
    arg4_1 = view_87.shape[0]
    view_41 = view_87
    full_default_46 = torch.ops.aten.full.default([arg4_1, 16], 0, dtype=torch.float16, layout=torch.strided, device=torch.device(device), pin_memory=False)
    slice_119 = torch.ops.aten.slice.Tensor(arg18_1, 1, 42865, 42881)
    where_46 = torch.ops.aten.where.self(view_41, full_default_46, slice_119)
    slice_120 = torch.ops.aten.slice.Tensor(arg18_1, 1, 42886, 42902)
    where_47 = torch.ops.aten.where.self(view_41, full_default_46, slice_120)
    slice_121 = torch.ops.aten.slice.Tensor(arg18_1, 1, 42908, 42924)
    where_48 = torch.ops.aten.where.self(view_41, full_default_46, slice_121)
    slice_122 = torch.ops.aten.slice.Tensor(arg18_1, 1, 42574, 42590)
    where_49 = torch.ops.aten.where.self(view_41, full_default_46, slice_122)
    slice_123 = torch.ops.aten.slice.Tensor(arg18_1, 1, 42531, 42547)
    where_50 = torch.ops.aten.where.self(view_41, full_default_46, slice_123)
    slice_124 = torch.ops.aten.slice.Tensor(arg18_1, 1, 42552, 42568)
    where_51 = torch.ops.aten.where.self(view_41, full_default_46, slice_124)
    slice_125 = torch.ops.aten.slice.Tensor(arg18_1, 1, 42595, 42611)
    where_52 = torch.ops.aten.where.self(view_41, full_default_46, slice_125)
    slice_126 = torch.ops.aten.slice.Tensor(arg18_1, 1, 42700, 42716)
    where_53 = torch.ops.aten.where.self(view_41, full_default_46, slice_126)
    slice_127 = torch.ops.aten.slice.Tensor(arg18_1, 1, 42679, 42695)
    where_54 = torch.ops.aten.where.self(view_41, full_default_46, slice_127)
    slice_128 = torch.ops.aten.slice.Tensor(arg18_1, 1, 42637, 42653)
    where_55 = torch.ops.aten.where.self(view_41, full_default_46, slice_128)
    slice_129 = torch.ops.aten.slice.Tensor(arg18_1, 1, 42827, 42843)
    where_56 = torch.ops.aten.where.self(view_41, full_default_46, slice_129)
    slice_130 = torch.ops.aten.slice.Tensor(arg18_1, 1, 42742, 42758)
    where_57 = torch.ops.aten.where.self(view_41, full_default_46, slice_130)
    slice_131 = torch.ops.aten.slice.Tensor(arg18_1, 1, 42763, 42779)
    where_58 = torch.ops.aten.where.self(view_41, full_default_46, slice_131)
    slice_132 = torch.ops.aten.slice.Tensor(arg18_1, 1, 42785, 42801)
    where_59 = torch.ops.aten.where.self(view_41, full_default_46, slice_132)
    slice_133 = torch.ops.aten.slice.Tensor(arg18_1, 1, 42489, 42505)
    where_60 = torch.ops.aten.where.self(view_41, full_default_46, slice_133)
    slice_134 = torch.ops.aten.slice.Tensor(arg18_1, 1, 42721, 42737)
    where_61 = torch.ops.aten.where.self(view_41, full_default_46, slice_134)
    slice_135 = torch.ops.aten.slice.Tensor(arg18_1, 1, 42447, 42463)
    where_62 = torch.ops.aten.where.self(view_41, full_default_46, slice_135)
    slice_136 = torch.ops.aten.slice.Tensor(arg18_1, 1, 42468, 42484)
    where_63 = torch.ops.aten.where.self(view_41, full_default_46, slice_136)
    slice_137 = torch.ops.aten.slice.Tensor(arg18_1, 1, 42510, 42526)
    where_64 = torch.ops.aten.where.self(view_41, full_default_46, slice_137)
    slice_174 = torch.ops.aten.slice.Tensor(arg18_1, 1, 42658, 42674)
    where_65 = torch.ops.aten.where.self(view_41, full_default_46, slice_174)
    slice_175 = torch.ops.aten.slice.Tensor(arg18_1, 1, 42616, 42632)
    where_66 = torch.ops.aten.where.self(view_41, full_default_46, slice_175)
    return where_46,where_47,where_48,where_49,where_50,where_51,where_52,where_53,where_54,where_55,where_56,where_57,where_58,where_59,where_60,where_61,where_62,where_63,where_64,where_65,where_66


SAMPLE_BINDINGS = [
    {'s0': 64},
    {'s0': 112},
    {'s0': 160},
    {'s0': 208},
    {'s0': 256},
]


COMPILE_BINDINGS = [
    {'s0': 64},
]


def make_inputs(binding, device):
    s0 = binding['s0']
    view_87 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.bool,
    )
    arg18_1 = rand_strided(
        (s0, 69876),
        (69876, 1),
        device=device,
        dtype=torch.float16,
    )
    return (view_87, arg18_1), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (0,)}


CASE = {
    'name': 'triton_poi_fused_slice_where_zeros_like_27',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
