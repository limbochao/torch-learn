import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(bmm_38, arg1334_1, bmm_37, arg1332_1):
    add_12746 = torch.ops.aten.add.Tensor(bmm_38, arg1334_1)
    mul_8446 = torch.ops.aten.mul.Tensor(add_12746, add_12746)
    mul_8447 = torch.ops.aten.mul.Tensor(mul_8446, add_12746)
    mul_8448 = torch.ops.aten.mul.Tensor(mul_8447, 0.044715)
    add_12751 = torch.ops.aten.add.Tensor(add_12746, mul_8448)
    mul_8449 = torch.ops.aten.mul.Tensor(add_12751, 1.5957691216057308)
    sigmoid_23 = torch.ops.aten.sigmoid.default(mul_8449)
    mul_8450 = torch.ops.aten.mul.Tensor(add_12746, sigmoid_23)
    add_12730 = torch.ops.aten.add.Tensor(bmm_37, arg1332_1)
    tanh_7 = torch.ops.aten.tanh.default(add_12730)
    mul_8460 = torch.ops.aten.mul.Tensor(mul_8450, tanh_7)
    return mul_8460


SAMPLE_BINDINGS = [
    {'s0': 400},
    {'s0': 200},
    {'s0': 256},
]


COMPILE_BINDINGS = [
    {'s0': 400},
]


def make_inputs(binding, device):
    s0 = binding['s0']
    bmm_38 = rand_strided(
        (16, s0, 1280),
        (1280*s0, 1280, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1334_1 = rand_strided(
        (16, 1, 1280),
        (1280, 1280, 1),
        device=device,
        dtype=torch.float16,
    )
    bmm_37 = rand_strided(
        (16, s0, 1280),
        (1280*s0, 1280, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1332_1 = rand_strided(
        (16, 1, 1280),
        (1280, 1280, 1),
        device=device,
        dtype=torch.float16,
    )
    return (bmm_38, arg1334_1, bmm_37, arg1332_1), {}


DYNAMIC_DIMS = {'args[0]': (1,), 'args[2]': (1,)}


CASE = {
    'name': 'triton_poi_fused_add_gelu_mul_tanh_212',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
