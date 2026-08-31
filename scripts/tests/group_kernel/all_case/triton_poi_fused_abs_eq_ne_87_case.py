import torch
from math import inf, nan
from cmath import nanj
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg19_1):
    ne_1 = torch.ops.aten.ne.Tensor(arg19_1, arg19_1)
    abs_1 = torch.ops.aten.abs.default(arg19_1)
    eq_166 = torch.ops.aten.eq.Scalar(abs_1, inf)
    return ne_1,eq_166


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
    arg19_1 = rand_strided(
        (s0, 512),
        (512, 1),
        device=device,
        dtype=torch.float16,
    )
    return (arg19_1,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_abs_eq_ne_87',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
