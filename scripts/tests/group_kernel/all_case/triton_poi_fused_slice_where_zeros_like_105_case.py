import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg18_1):
    full_default_125 = torch.ops.aten.full.default([arg4_1, 32], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    slice_461 = torch.ops.aten.slice.Tensor(arg18_1, 1, 23102, 23134)
    where_171 = torch.ops.aten.where.self(view_149, full_default_125, slice_461)
    slice_463 = torch.ops.aten.slice.Tensor(arg18_1, 1, 27339, 27371)
    where_172 = torch.ops.aten.where.self(view_151, full_default_125, slice_463)
    slice_464 = torch.ops.aten.slice.Tensor(arg18_1, 1, 24124, 24156)
    where_173 = torch.ops.aten.where.self(view_151, full_default_125, slice_464)
    slice_465 = torch.ops.aten.slice.Tensor(arg18_1, 1, 24970, 25002)
    where_174 = torch.ops.aten.where.self(view_151, full_default_125, slice_465)
    slice_466 = torch.ops.aten.slice.Tensor(arg18_1, 1, 25800, 25832)
    where_175 = torch.ops.aten.where.self(view_151, full_default_125, slice_466)
    slice_467 = torch.ops.aten.slice.Tensor(arg18_1, 1, 26550, 26582)
    where_176 = torch.ops.aten.where.self(view_151, full_default_125, slice_467)
    return where_171,where_172,where_173,where_174,where_175,where_176


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
    'name': 'triton_poi_fused_slice_where_zeros_like_105',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
