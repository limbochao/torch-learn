import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg18_1):
    arg4_1 = view_723.shape[0]
    view_113 = view_723
    view_115 = view_723
    full_default_125 = torch.ops.aten.full.default([arg4_1, 32], 0, dtype=torch.float16, layout=torch.strided, device=torch.device(device), pin_memory=False)
    slice_323 = torch.ops.aten.slice.Tensor(arg18_1, 1, 22878, 22910)
    where_129 = torch.ops.aten.where.self(view_113, full_default_125, slice_323)
    slice_325 = torch.ops.aten.slice.Tensor(arg18_1, 1, 27115, 27147)
    where_130 = torch.ops.aten.where.self(view_115, full_default_125, slice_325)
    slice_326 = torch.ops.aten.slice.Tensor(arg18_1, 1, 23900, 23932)
    where_131 = torch.ops.aten.where.self(view_115, full_default_125, slice_326)
    slice_327 = torch.ops.aten.slice.Tensor(arg18_1, 1, 24746, 24778)
    where_132 = torch.ops.aten.where.self(view_115, full_default_125, slice_327)
    slice_328 = torch.ops.aten.slice.Tensor(arg18_1, 1, 25576, 25608)
    where_133 = torch.ops.aten.where.self(view_115, full_default_125, slice_328)
    slice_329 = torch.ops.aten.slice.Tensor(arg18_1, 1, 26326, 26358)
    where_134 = torch.ops.aten.where.self(view_115, full_default_125, slice_329)
    return where_129,where_130,where_131,where_132,where_133,where_134


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
    'name': 'triton_poi_fused_slice_where_zeros_like_66',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
