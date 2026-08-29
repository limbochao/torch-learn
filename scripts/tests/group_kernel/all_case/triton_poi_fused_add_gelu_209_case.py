import torch
from torch._dynamo.testing import rand_strided


# Prefix static split axis case.
# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(bmm_34, arg1326_1):
    add_12693 = torch.ops.aten.add.Tensor(bmm_34, arg1326_1)
    mul_8378 = torch.ops.aten.mul.Tensor(add_12693, add_12693)
    mul_8379 = torch.ops.aten.mul.Tensor(mul_8378, add_12693)
    mul_8380 = torch.ops.aten.mul.Tensor(mul_8379, 0.044715)
    add_12698 = torch.ops.aten.add.Tensor(add_12693, mul_8380)
    mul_8381 = torch.ops.aten.mul.Tensor(add_12698, 1.5957691216057308)
    sigmoid_21 = torch.ops.aten.sigmoid.default(mul_8381)
    mul_8382 = torch.ops.aten.mul.Tensor(add_12693, sigmoid_21)
    return mul_8382


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
    bmm_34 = rand_strided(
        (16, s0, 160),
        (160*s0, 160, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1326_1 = rand_strided(
        (16, 1, 160),
        (160, 160, 1),
        device=device,
        dtype=torch.float16,
    )
    return (bmm_34, arg1326_1), {}


DYNAMIC_DIMS = {'args[0]': (1,)}


CASE = {
    'name': 'triton_poi_fused_add_gelu_209',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
