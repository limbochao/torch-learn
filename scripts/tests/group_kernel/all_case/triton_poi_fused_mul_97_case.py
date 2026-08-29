import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg45_1, addmm_7):
    mul_213 = torch.ops.aten.mul.Tensor(arg45_1, addmm_7)
    return mul_213


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
    arg45_1 = rand_strided(
        (1,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    addmm_7 = rand_strided(
        (s0, 64),
        (64, 1),
        device=device,
        dtype=torch.float16,
    )
    return (arg45_1, addmm_7), {}


DYNAMIC_DIMS = {'args[1]': (0,)}


CASE = {
    'name': 'triton_poi_fused_mul_97',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
