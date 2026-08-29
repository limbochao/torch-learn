import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg18_1):
    logical_not_1 = torch.ops.aten.logical_not.default(logical_not)
    expand_default_20 = torch.ops.aten.expand.default(logical_not_1, [sym_size_int_874, 112])
    slice_743 = torch.ops.aten.slice.Tensor(arg18_1, 1, 22385, 22497)
    slice_744 = torch.ops.aten.slice.Tensor(arg18_1, 1, 34648, 34760)
    where_217 = torch.ops.aten.where.self(expand_default_20, slice_743, slice_744)
    return where_217


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
    'name': 'triton_poi_fused_logical_not_slice_where_158',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
