import torch
from math import inf, nan
from cmath import nanj
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg18_1):
    arg4_1 = view_723.shape[0]
    sym_size_int_875 = view_723.shape[0]
    sym_size_int_876 = view_723.shape[0]
    logical_not = view_723
    logical_not_1 = torch.ops.aten.logical_not.default(logical_not)
    expand_default_21 = torch.ops.aten.expand.default(logical_not_1, [sym_size_int_875, 128])
    slice_746 = torch.ops.aten.slice.Tensor(arg18_1, 1, 28076, 28204)
    slice_747 = torch.ops.aten.slice.Tensor(arg18_1, 1, 35842, 35970)
    where_218 = torch.ops.aten.where.self(expand_default_21, slice_746, slice_747)
    expand_default_22 = torch.ops.aten.expand.default(logical_not_1, [sym_size_int_876, 128])
    full_default_220 = torch.ops.aten.full.default([arg4_1, 128], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    slice_749 = torch.ops.aten.slice.Tensor(arg18_1, 1, 43554, 43682)
    where_219 = torch.ops.aten.where.self(logical_not, full_default_220, slice_749)
    slice_750 = torch.ops.aten.slice.Tensor(arg18_1, 1, 33380, 33508)
    where_220 = torch.ops.aten.where.self(expand_default_22, where_219, slice_750)
    return where_218,where_220


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
    'name': 'triton_poi_fused_logical_not_slice_where_zeros_159',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
