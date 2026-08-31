import torch
from math import inf, nan
from cmath import nanj
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg18_1):
    arg4_1 = view_723.shape[0]
    view_125 = view_723
    view_127 = view_723
    full_default_125 = torch.ops.aten.full.default([arg4_1, 32], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    slice_369 = torch.ops.aten.slice.Tensor(arg18_1, 1, 22942, 22974)
    where_143 = torch.ops.aten.where.self(view_125, full_default_125, slice_369)
    slice_371 = torch.ops.aten.slice.Tensor(arg18_1, 1, 27179, 27211)
    where_144 = torch.ops.aten.where.self(view_127, full_default_125, slice_371)
    slice_372 = torch.ops.aten.slice.Tensor(arg18_1, 1, 23964, 23996)
    where_145 = torch.ops.aten.where.self(view_127, full_default_125, slice_372)
    slice_373 = torch.ops.aten.slice.Tensor(arg18_1, 1, 24810, 24842)
    where_146 = torch.ops.aten.where.self(view_127, full_default_125, slice_373)
    slice_374 = torch.ops.aten.slice.Tensor(arg18_1, 1, 25640, 25672)
    where_147 = torch.ops.aten.where.self(view_127, full_default_125, slice_374)
    slice_375 = torch.ops.aten.slice.Tensor(arg18_1, 1, 26390, 26422)
    where_148 = torch.ops.aten.where.self(view_127, full_default_125, slice_375)
    return where_143,where_144,where_145,where_146,where_147,where_148


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
    view_723 = rand_strided(
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
    return (view_723, arg18_1), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (0,)}


CASE = {
    'name': 'triton_poi_fused_slice_where_zeros_like_143',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
