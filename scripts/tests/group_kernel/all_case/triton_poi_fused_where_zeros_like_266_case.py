import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg1650_1):
    arg4_1 = view_723.shape[0]
    view_715 = view_723
    full_default_320 = torch.ops.aten.full.default([arg4_1, 22], 0, dtype=torch.float16, layout=torch.strided, device=torch.device(device), pin_memory=False)
    where_307 = torch.ops.aten.where.self(view_715, full_default_320, arg1650_1)
    return where_307


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
    arg1650_1 = rand_strided(
        (s0, 22),
        (22, 1),
        device=device,
        dtype=torch.float16,
    )
    return (view_723, arg1650_1), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (0,)}


CASE = {
    'name': 'triton_poi_fused_where_zeros_like_266',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
