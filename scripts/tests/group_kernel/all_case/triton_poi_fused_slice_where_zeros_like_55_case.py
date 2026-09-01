import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg18_1):
    arg4_1 = view_723.shape[0]
    view_131 = view_723
    view_133 = view_723
    full_default_125 = torch.ops.aten.full.default([arg4_1, 32], 0, dtype=torch.float16, layout=torch.strided, device=torch.device(device), pin_memory=False)
    slice_392 = torch.ops.aten.slice.Tensor(arg18_1, 1, 22974, 23006)
    where_150 = torch.ops.aten.where.self(view_131, full_default_125, slice_392)
    slice_394 = torch.ops.aten.slice.Tensor(arg18_1, 1, 27211, 27243)
    where_151 = torch.ops.aten.where.self(view_133, full_default_125, slice_394)
    slice_395 = torch.ops.aten.slice.Tensor(arg18_1, 1, 23996, 24028)
    where_152 = torch.ops.aten.where.self(view_133, full_default_125, slice_395)
    slice_396 = torch.ops.aten.slice.Tensor(arg18_1, 1, 24842, 24874)
    where_153 = torch.ops.aten.where.self(view_133, full_default_125, slice_396)
    slice_397 = torch.ops.aten.slice.Tensor(arg18_1, 1, 25672, 25704)
    where_154 = torch.ops.aten.where.self(view_133, full_default_125, slice_397)
    slice_398 = torch.ops.aten.slice.Tensor(arg18_1, 1, 26422, 26454)
    where_155 = torch.ops.aten.where.self(view_133, full_default_125, slice_398)
    return where_150,where_151,where_152,where_153,where_154,where_155


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
    'name': 'triton_poi_fused_slice_where_zeros_like_55',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
