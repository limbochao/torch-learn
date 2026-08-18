import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg105_1, arg107_1):
    arg4_1 = view_723.shape[0]
    view_86 = view_723
    full_default_95 = torch.ops.aten.full.default([arg4_1, 8], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    where_87 = torch.ops.aten.where.self(view_86, full_default_95, arg105_1)
    where_89 = torch.ops.aten.where.self(view_86, full_default_95, arg107_1)
    return where_87,where_89


SAMPLE_BINDINGS = [
    {'s0': 64},
    {'s0': 112},
    {'s0': 160},
    {'s0': 208},
    {'s0': 256},
]


COMPILE_BINDINGS = [
    {'s0': 64},
    {'s0': 256},
]


def make_inputs(binding, device):
    s0 = binding['s0']
    view_723 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.bool,
    )
    arg105_1 = rand_strided(
        (s0, 8),
        (8, 1),
        device=device,
        dtype=torch.float16,
    )
    arg107_1 = rand_strided(
        (s0, 8),
        (8, 1),
        device=device,
        dtype=torch.float16,
    )
    return (view_723, arg105_1, arg107_1), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (0,), 'args[2]': (0,)}


CASE = {
    'name': 'triton_poi_fused_where_zeros_like_153',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
