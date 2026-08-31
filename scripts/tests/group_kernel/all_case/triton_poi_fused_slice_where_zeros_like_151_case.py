import torch
from math import inf, nan
from cmath import nanj
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg18_1):
    arg4_1 = view_723.shape[0]
    view_161 = view_723
    view_163 = view_723
    full_default_125 = torch.ops.aten.full.default([arg4_1, 32], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    slice_507 = torch.ops.aten.slice.Tensor(arg18_1, 1, 23166, 23198)
    where_185 = torch.ops.aten.where.self(view_161, full_default_125, slice_507)
    slice_509 = torch.ops.aten.slice.Tensor(arg18_1, 1, 27403, 27435)
    where_186 = torch.ops.aten.where.self(view_163, full_default_125, slice_509)
    slice_510 = torch.ops.aten.slice.Tensor(arg18_1, 1, 24188, 24220)
    where_187 = torch.ops.aten.where.self(view_163, full_default_125, slice_510)
    slice_511 = torch.ops.aten.slice.Tensor(arg18_1, 1, 25034, 25066)
    where_188 = torch.ops.aten.where.self(view_163, full_default_125, slice_511)
    slice_512 = torch.ops.aten.slice.Tensor(arg18_1, 1, 25864, 25896)
    where_189 = torch.ops.aten.where.self(view_163, full_default_125, slice_512)
    slice_513 = torch.ops.aten.slice.Tensor(arg18_1, 1, 26614, 26646)
    where_190 = torch.ops.aten.where.self(view_163, full_default_125, slice_513)
    return where_185,where_186,where_187,where_188,where_189,where_190


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
    'name': 'triton_poi_fused_slice_where_zeros_like_151',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
