import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg18_1):
    full_default_125 = torch.ops.aten.full.default([arg4_1, 32], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    slice_484 = torch.ops.aten.slice.Tensor(arg18_1, 1, 23134, 23166)
    where_178 = torch.ops.aten.where.self(view_155, full_default_125, slice_484)
    slice_486 = torch.ops.aten.slice.Tensor(arg18_1, 1, 27371, 27403)
    where_179 = torch.ops.aten.where.self(view_157, full_default_125, slice_486)
    slice_487 = torch.ops.aten.slice.Tensor(arg18_1, 1, 24156, 24188)
    where_180 = torch.ops.aten.where.self(view_157, full_default_125, slice_487)
    slice_488 = torch.ops.aten.slice.Tensor(arg18_1, 1, 25002, 25034)
    where_181 = torch.ops.aten.where.self(view_157, full_default_125, slice_488)
    slice_489 = torch.ops.aten.slice.Tensor(arg18_1, 1, 25832, 25864)
    where_182 = torch.ops.aten.where.self(view_157, full_default_125, slice_489)
    slice_490 = torch.ops.aten.slice.Tensor(arg18_1, 1, 26582, 26614)
    where_183 = torch.ops.aten.where.self(view_157, full_default_125, slice_490)
    return where_178,where_179,where_180,where_181,where_182,where_183


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
    'name': 'triton_poi_fused_slice_where_zeros_like_112',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
