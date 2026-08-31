import torch
from math import inf, nan
from cmath import nanj
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(cat_152):
    arg1339_1 = cat_152.shape[0]
    view_559 = torch.ops.aten.reshape.default(cat_152, [2, arg1339_1, 256, 64])
    sum_193 = torch.ops.aten.sum.dim_IntList(view_559, [0])
    return sum_193


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
    cat_152 = rand_strided(
        (2*s0, 256, 64),
        (16384, 64, 1),
        device=device,
        dtype=torch.float16,
    )
    return (cat_152,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_stack_sum_220',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
