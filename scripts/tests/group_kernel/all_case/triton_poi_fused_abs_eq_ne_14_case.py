import torch
from math import inf, nan
from cmath import nanj
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg28_1):
    ne_6 = torch.ops.aten.ne.Tensor(arg28_1, arg28_1)
    abs_6 = torch.ops.aten.abs.default(arg28_1)
    eq_221 = torch.ops.aten.eq.Scalar(abs_6, inf)
    return ne_6,eq_221


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
    arg28_1 = rand_strided(
        (s0, 328),
        (328, 1),
        device=device,
        dtype=torch.float16,
    )
    return (arg28_1,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_abs_eq_ne_14',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
