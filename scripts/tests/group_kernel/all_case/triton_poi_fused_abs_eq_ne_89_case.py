import torch
from math import inf, nan
from cmath import nanj
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg20_1):
    ne_2 = torch.ops.aten.ne.Tensor(arg20_1, arg20_1)
    abs_2 = torch.ops.aten.abs.default(arg20_1)
    eq_176 = torch.ops.aten.eq.Scalar(abs_2, inf)
    return ne_2,eq_176


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
    arg20_1 = rand_strided(
        (s0, 32),
        (32, 1),
        device=device,
        dtype=torch.float16,
    )
    return (arg20_1,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_abs_eq_ne_89',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
