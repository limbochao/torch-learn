import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg135_1):
    clone_6 = torch.ops.aten.clone.default(arg135_1)
    return clone_6


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
    arg135_1 = rand_strided(
        (s0, 576),
        (576, 1),
        device=device,
        dtype=torch.float16,
    )
    return (arg135_1,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_cat_183',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
