import torch
from torch._dynamo.testing import rand_strided


# Eager reference for f32[s0, s1] -> f32[s1, s0].
def eager_forward(x):
    return x.t() * 2.0 + 1.0


SAMPLE_BINDINGS = [
    {'s0': 64, 's1': 2048},
    {'s0': 112, 's1': 4096},
    {'s0': 160, 's1': 6144},
    {'s0': 208, 's1': 8192},
    {'s0': 256, 's1': 8192},
]


COMPILE_BINDINGS = [
    {'s0': 64, 's1': 2048},
]


def make_inputs(binding, device):
    s0, s1 = binding['s0'], binding['s1']
    x = rand_strided((s0, s1), (s1, 1), device=device, dtype=torch.float32)
    return (x,), {}


DYNAMIC_DIMS = {'args[0]': (0, 1)}


CASE = {
    'name': 'tc_kdp_051_transpose_mul_joint',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
