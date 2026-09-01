import torch
from math import inf, nan
from cmath import nanj
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg18_1):
    arg4_1 = view_723.shape[0]
    view_137 = view_723
    view_139 = view_723
    full_default_161 = torch.ops.aten.full.default([arg4_1, 64], 0, dtype=torch.float16, layout=torch.strided, device=torch.device(device), pin_memory=False)
    slice_415 = torch.ops.aten.slice.Tensor(arg18_1, 1, 23006, 23070)
    where_157 = torch.ops.aten.where.self(view_137, full_default_161, slice_415)
    slice_417 = torch.ops.aten.slice.Tensor(arg18_1, 1, 27243, 27307)
    where_158 = torch.ops.aten.where.self(view_139, full_default_161, slice_417)
    slice_418 = torch.ops.aten.slice.Tensor(arg18_1, 1, 24028, 24092)
    where_159 = torch.ops.aten.where.self(view_139, full_default_161, slice_418)
    slice_419 = torch.ops.aten.slice.Tensor(arg18_1, 1, 24874, 24938)
    where_160 = torch.ops.aten.where.self(view_139, full_default_161, slice_419)
    slice_420 = torch.ops.aten.slice.Tensor(arg18_1, 1, 25704, 25768)
    where_161 = torch.ops.aten.where.self(view_139, full_default_161, slice_420)
    slice_421 = torch.ops.aten.slice.Tensor(arg18_1, 1, 26454, 26518)
    where_162 = torch.ops.aten.where.self(view_139, full_default_161, slice_421)
    return where_157,where_158,where_159,where_160,where_161,where_162


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
    'name': 'triton_poi_fused_slice_where_zeros_like_31',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
