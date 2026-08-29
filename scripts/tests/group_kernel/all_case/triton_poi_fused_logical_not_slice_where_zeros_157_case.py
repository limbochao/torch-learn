import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg18_1, arg871_1):
    logical_not_1 = torch.ops.aten.logical_not.default(logical_not)
    expand_default_18 = torch.ops.aten.expand.default(logical_not_1, [sym_size_int_872, 80])
    full_default_218 = torch.ops.aten.full.default([arg4_1, 80], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    slice_737 = torch.ops.aten.slice.Tensor(arg18_1, 1, 24220, 24300)
    where_213 = torch.ops.aten.where.self(logical_not, full_default_218, slice_737)
    slice_738 = torch.ops.aten.slice.Tensor(arg18_1, 1, 35098, 35178)
    where_214 = torch.ops.aten.where.self(expand_default_18, where_213, slice_738)
    expand_default_19 = torch.ops.aten.expand.default(logical_not_1, [sym_size_int_873, 80])
    slice_740 = torch.ops.aten.slice.Tensor(arg18_1, 1, 25066, 25146)
    where_215 = torch.ops.aten.where.self(logical_not, full_default_218, slice_740)
    slice_741 = torch.ops.aten.slice.Tensor(arg18_1, 1, 35470, 35550)
    where_216 = torch.ops.aten.where.self(expand_default_19, where_215, slice_741)
    where_234 = torch.ops.aten.where.self(view_447, full_default_218, arg871_1)
    return where_214,where_216,where_234


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
    arg871_1 = rand_strided(
        (s0, 80),
        (80, 1),
        device=device,
        dtype=torch.float16,
    )
    return (view_723, arg18_1, arg871_1), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (0,), 'args[2]': (0,)}


CASE = {
    'name': 'triton_poi_fused_logical_not_slice_where_zeros_157',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
