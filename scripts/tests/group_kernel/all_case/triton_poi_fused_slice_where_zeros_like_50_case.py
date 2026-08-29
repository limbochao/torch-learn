import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg18_1):
    full_default_125 = torch.ops.aten.full.default([arg4_1, 32], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    slice_277 = torch.ops.aten.slice.Tensor(arg18_1, 1, 22814, 22846)
    where_115 = torch.ops.aten.where.self(view_101, full_default_125, slice_277)
    slice_279 = torch.ops.aten.slice.Tensor(arg18_1, 1, 27051, 27083)
    where_116 = torch.ops.aten.where.self(view_103, full_default_125, slice_279)
    slice_280 = torch.ops.aten.slice.Tensor(arg18_1, 1, 23836, 23868)
    where_117 = torch.ops.aten.where.self(view_103, full_default_125, slice_280)
    slice_281 = torch.ops.aten.slice.Tensor(arg18_1, 1, 24682, 24714)
    where_118 = torch.ops.aten.where.self(view_103, full_default_125, slice_281)
    slice_282 = torch.ops.aten.slice.Tensor(arg18_1, 1, 25512, 25544)
    where_119 = torch.ops.aten.where.self(view_103, full_default_125, slice_282)
    slice_283 = torch.ops.aten.slice.Tensor(arg18_1, 1, 26262, 26294)
    where_120 = torch.ops.aten.where.self(view_103, full_default_125, slice_283)
    return where_115,where_116,where_117,where_118,where_119,where_120


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
    'name': 'triton_poi_fused_slice_where_zeros_like_50',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
