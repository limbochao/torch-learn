import torch
from torch._dynamo.testing import rand_strided


# Prefix static split axis case.
# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(bmm_18, arg1282_1):
    add_12251 = torch.ops.aten.add.Tensor(bmm_18, arg1282_1)
    mul_7914 = torch.ops.aten.mul.Tensor(add_12251, add_12251)
    mul_7915 = torch.ops.aten.mul.Tensor(mul_7914, add_12251)
    mul_7916 = torch.ops.aten.mul.Tensor(mul_7915, 0.044715)
    add_12256 = torch.ops.aten.add.Tensor(add_12251, mul_7916)
    mul_7917 = torch.ops.aten.mul.Tensor(add_12256, 1.5957691216057308)
    sigmoid_13 = torch.ops.aten.sigmoid.default(mul_7917)
    mul_7918 = torch.ops.aten.mul.Tensor(add_12251, sigmoid_13)
    return mul_7918


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
    bmm_18 = rand_strided(
        (80, s0, 960),
        (960*s0, 960, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1282_1 = rand_strided(
        (80, 1, 960),
        (960, 960, 1),
        device=device,
        dtype=torch.float16,
    )
    return (bmm_18, arg1282_1), {}


DYNAMIC_DIMS = {'args[0]': (1,)}


CASE = {
    'name': 'triton_poi_fused_add_gelu_198',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
