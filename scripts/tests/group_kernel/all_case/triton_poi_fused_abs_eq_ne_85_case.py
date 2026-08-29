import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg33_1):
    ne_7 = torch.ops.aten.ne.Tensor(arg33_1, arg33_1)
    abs_7 = torch.ops.aten.abs.default(arg33_1)
    eq_235 = torch.ops.aten.eq.Scalar(abs_7, inf)
    return ne_7,eq_235


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
    arg33_1 = rand_strided(
        (s0, 256),
        (256, 1),
        device=device,
        dtype=torch.float16,
    )
    return (arg33_1,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_abs_eq_ne_85',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
