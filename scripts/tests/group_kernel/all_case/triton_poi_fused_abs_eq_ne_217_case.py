import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg1340_1):
    ne_94 = torch.ops.aten.ne.Tensor(arg1340_1, arg1340_1)
    abs_10 = torch.ops.aten.abs.default(arg1340_1)
    eq_7064 = torch.ops.aten.eq.Scalar(abs_10, inf)
    return ne_94,eq_7064


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
    arg1340_1 = rand_strided(
        (s0, 16384),
        (16384, 1),
        device=device,
        dtype=torch.float16,
    )
    return (arg1340_1,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_abs_eq_ne_217',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
