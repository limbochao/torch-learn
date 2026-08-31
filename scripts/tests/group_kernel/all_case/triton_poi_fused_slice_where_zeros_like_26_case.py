import torch
from math import inf, nan
from cmath import nanj
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg18_1):
    arg4_1 = view_723.shape[0]
    view_40 = view_723
    full_default_46 = torch.ops.aten.full.default([arg4_1, 16], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    slice_111 = torch.ops.aten.slice.Tensor(arg18_1, 1, 47098, 47114)
    where_38 = torch.ops.aten.where.self(view_40, full_default_46, slice_111)
    slice_112 = torch.ops.aten.slice.Tensor(arg18_1, 1, 49061, 49077)
    where_39 = torch.ops.aten.where.self(view_40, full_default_46, slice_112)
    slice_113 = torch.ops.aten.slice.Tensor(arg18_1, 1, 46847, 46863)
    where_40 = torch.ops.aten.where.self(view_40, full_default_46, slice_113)
    slice_114 = torch.ops.aten.slice.Tensor(arg18_1, 1, 47119, 47135)
    where_41 = torch.ops.aten.where.self(view_40, full_default_46, slice_114)
    slice_115 = torch.ops.aten.slice.Tensor(arg18_1, 1, 47574, 47590)
    where_42 = torch.ops.aten.where.self(view_40, full_default_46, slice_115)
    slice_116 = torch.ops.aten.slice.Tensor(arg18_1, 1, 47553, 47569)
    where_43 = torch.ops.aten.where.self(view_40, full_default_46, slice_116)
    slice_117 = torch.ops.aten.slice.Tensor(arg18_1, 1, 47511, 47527)
    where_44 = torch.ops.aten.where.self(view_40, full_default_46, slice_117)
    slice_118 = torch.ops.aten.slice.Tensor(arg18_1, 1, 47532, 47548)
    where_45 = torch.ops.aten.where.self(view_40, full_default_46, slice_118)
    return where_38,where_39,where_40,where_41,where_42,where_43,where_44,where_45


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
    'name': 'triton_poi_fused_slice_where_zeros_like_26',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
