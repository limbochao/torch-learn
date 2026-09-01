import torch
from math import inf, nan
from cmath import nanj
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(bitwise_or_6, arg33_1):
    arg4_1 = bitwise_or_6.shape[0]
    full_default_26 = torch.ops.aten.full.default([arg4_1, 256], 1.0013580322265625e-05, dtype=torch.float16, layout=torch.strided, device=torch.device(device), pin_memory=False)
    where_18 = torch.ops.aten.where.self(bitwise_or_6, full_default_26, arg33_1)
    return where_18


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
    bitwise_or_6 = rand_strided(
        (s0, 256),
        (256, 1),
        device=device,
        dtype=torch.bool,
    )
    arg33_1 = rand_strided(
        (s0, 256),
        (256, 1),
        device=device,
        dtype=torch.float16,
    )
    return (bitwise_or_6, arg33_1), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (0,)}


CASE = {
    'name': 'triton_poi_fused_mul_where_86',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
