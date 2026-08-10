import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(bmm_32, arg1318_1, bmm_31, arg1316_1):
    add_12596 = torch.ops.aten.add.Tensor(bmm_32, arg1318_1)
    mul_8283 = torch.ops.aten.mul.Tensor(add_12596, add_12596)
    mul_8284 = torch.ops.aten.mul.Tensor(mul_8283, add_12596)
    mul_8285 = torch.ops.aten.mul.Tensor(mul_8284, 0.044715)
    add_12601 = torch.ops.aten.add.Tensor(add_12596, mul_8285)
    mul_8286 = torch.ops.aten.mul.Tensor(add_12601, 1.5957691216057308)
    sigmoid_20 = torch.ops.aten.sigmoid.default(mul_8286)
    mul_8287 = torch.ops.aten.mul.Tensor(add_12596, sigmoid_20)
    add_12580 = torch.ops.aten.add.Tensor(bmm_31, arg1316_1)
    tanh_5 = torch.ops.aten.tanh.default(add_12580)
    mul_8297 = torch.ops.aten.mul.Tensor(mul_8287, tanh_5)
    return mul_8297


SAMPLE_BINDINGS = [
    {'s0': 199},
    {'s0': 200},
    {'s0': 201},
]


COMPILE_BINDINGS = [
    {'s0': 200},
]


def make_inputs(binding, device):
    s0 = binding['s0']
    bmm_32 = rand_strided(
        (32, s0, 1280),
        (1280*s0, 1280, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1318_1 = rand_strided(
        (32, 1, 1280),
        (1280, 1280, 1),
        device=device,
        dtype=torch.float16,
    )
    bmm_31 = rand_strided(
        (32, s0, 1280),
        (1280*s0, 1280, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1316_1 = rand_strided(
        (32, 1, 1280),
        (1280, 1280, 1),
        device=device,
        dtype=torch.float16,
    )
    return (bmm_32, arg1318_1, bmm_31, arg1316_1), {}


DYNAMIC_DIMS = {'args[0]': (1,), 'args[2]': (1,)}


CASE = {
    'name': 'triton_poi_fused_add_gelu_mul_tanh_206',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
