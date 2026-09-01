import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg18_1):
    arg4_1 = view_723.shape[0]
    view_119 = view_723
    view_121 = view_723
    full_default_125 = torch.ops.aten.full.default([arg4_1, 32], 0, dtype=torch.float16, layout=torch.strided, device=torch.device(device), pin_memory=False)
    slice_346 = torch.ops.aten.slice.Tensor(arg18_1, 1, 22910, 22942)
    where_136 = torch.ops.aten.where.self(view_119, full_default_125, slice_346)
    slice_348 = torch.ops.aten.slice.Tensor(arg18_1, 1, 27147, 27179)
    where_137 = torch.ops.aten.where.self(view_121, full_default_125, slice_348)
    slice_349 = torch.ops.aten.slice.Tensor(arg18_1, 1, 23932, 23964)
    where_138 = torch.ops.aten.where.self(view_121, full_default_125, slice_349)
    slice_350 = torch.ops.aten.slice.Tensor(arg18_1, 1, 24778, 24810)
    where_139 = torch.ops.aten.where.self(view_121, full_default_125, slice_350)
    slice_351 = torch.ops.aten.slice.Tensor(arg18_1, 1, 25608, 25640)
    where_140 = torch.ops.aten.where.self(view_121, full_default_125, slice_351)
    slice_352 = torch.ops.aten.slice.Tensor(arg18_1, 1, 26358, 26390)
    where_141 = torch.ops.aten.where.self(view_121, full_default_125, slice_352)
    return where_136,where_137,where_138,where_139,where_140,where_141


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
    'name': 'triton_poi_fused_slice_where_zeros_like_175',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
