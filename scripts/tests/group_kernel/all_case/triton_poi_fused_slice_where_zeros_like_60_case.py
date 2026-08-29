import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg18_1):
    full_default_125 = torch.ops.aten.full.default([arg4_1, 32], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    slice_300 = torch.ops.aten.slice.Tensor(arg18_1, 1, 22846, 22878)
    where_122 = torch.ops.aten.where.self(view_107, full_default_125, slice_300)
    slice_302 = torch.ops.aten.slice.Tensor(arg18_1, 1, 27083, 27115)
    where_123 = torch.ops.aten.where.self(view_109, full_default_125, slice_302)
    slice_303 = torch.ops.aten.slice.Tensor(arg18_1, 1, 23868, 23900)
    where_124 = torch.ops.aten.where.self(view_109, full_default_125, slice_303)
    slice_304 = torch.ops.aten.slice.Tensor(arg18_1, 1, 24714, 24746)
    where_125 = torch.ops.aten.where.self(view_109, full_default_125, slice_304)
    slice_305 = torch.ops.aten.slice.Tensor(arg18_1, 1, 25544, 25576)
    where_126 = torch.ops.aten.where.self(view_109, full_default_125, slice_305)
    slice_306 = torch.ops.aten.slice.Tensor(arg18_1, 1, 26294, 26326)
    where_127 = torch.ops.aten.where.self(view_109, full_default_125, slice_306)
    return where_122,where_123,where_124,where_125,where_126,where_127


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
    'name': 'triton_poi_fused_slice_where_zeros_like_60',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
