import torch
from math import inf, nan
from cmath import nanj
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg40_1):
    ne_8 = torch.ops.aten.ne.Tensor(arg40_1, arg40_1)
    abs_8 = torch.ops.aten.abs.default(arg40_1)
    eq_263 = torch.ops.aten.eq.Scalar(abs_8, inf)
    return ne_8,eq_263


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
    arg40_1 = rand_strided(
        (s0, 128),
        (128, 1),
        device=device,
        dtype=torch.float16,
    )
    return (arg40_1,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_abs_eq_ne_95',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
