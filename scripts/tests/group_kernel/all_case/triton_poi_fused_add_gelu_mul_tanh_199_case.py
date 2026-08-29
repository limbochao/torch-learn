import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(bmm_20, arg1286_1, bmm_19, arg1284_1):
    add_12281 = torch.ops.aten.add.Tensor(bmm_20, arg1286_1)
    mul_7951 = torch.ops.aten.mul.Tensor(add_12281, add_12281)
    mul_7952 = torch.ops.aten.mul.Tensor(mul_7951, add_12281)
    mul_7953 = torch.ops.aten.mul.Tensor(mul_7952, 0.044715)
    add_12286 = torch.ops.aten.add.Tensor(add_12281, mul_7953)
    mul_7954 = torch.ops.aten.mul.Tensor(add_12286, 1.5957691216057308)
    sigmoid_14 = torch.ops.aten.sigmoid.default(mul_7954)
    mul_7955 = torch.ops.aten.mul.Tensor(add_12281, sigmoid_14)
    add_12265 = torch.ops.aten.add.Tensor(bmm_19, arg1284_1)
    tanh_1 = torch.ops.aten.tanh.default(add_12265)
    mul_7965 = torch.ops.aten.mul.Tensor(mul_7955, tanh_1)
    return mul_7965


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
    bmm_20 = rand_strided(
        (80, s0, 1280),
        (1280*s0, 1280, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1286_1 = rand_strided(
        (80, 1, 1280),
        (1280, 1280, 1),
        device=device,
        dtype=torch.float16,
    )
    bmm_19 = rand_strided(
        (80, s0, 1280),
        (1280*s0, 1280, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1284_1 = rand_strided(
        (80, 1, 1280),
        (1280, 1280, 1),
        device=device,
        dtype=torch.float16,
    )
    return (bmm_20, arg1286_1, bmm_19, arg1284_1), {}


DYNAMIC_DIMS = {'args[0]': (1,), 'args[2]': (1,)}


CASE = {
    'name': 'triton_poi_fused_add_gelu_mul_tanh_199',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
