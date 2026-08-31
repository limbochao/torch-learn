import torch
from math import inf, nan
from cmath import nanj
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg925_1, arg926_1):
    arg124_1 = arg925_1.shape[0]
    view_459 = torch.ops.aten.reshape.default(arg925_1, [1, arg124_1, 16])
    squeeze_dims_41 = torch.ops.aten.squeeze.dims(view_459, [0])
    view_461 = torch.ops.aten.reshape.default(arg926_1, [1, arg124_1, 16])
    squeeze_dims_40 = torch.ops.aten.squeeze.dims(view_461, [0])
    mul_7195 = torch.ops.aten.mul.Tensor(squeeze_dims_41, squeeze_dims_40)
    add_10947 = torch.ops.aten.add.Tensor(squeeze_dims_41, squeeze_dims_40)
    return mul_7195,add_10947


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
    arg925_1 = rand_strided(
        (s0, 16),
        (16, 1),
        device=device,
        dtype=torch.float16,
    )
    arg926_1 = rand_strided(
        (s0, 16),
        (16, 1),
        device=device,
        dtype=torch.float16,
    )
    return (arg925_1, arg926_1), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (0,)}


CASE = {
    'name': 'triton_poi_fused_add_mul_stack_156',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
