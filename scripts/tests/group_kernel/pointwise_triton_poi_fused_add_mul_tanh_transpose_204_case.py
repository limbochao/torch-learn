import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(getitem_355, bmm_29, arg1312_1):
    permute_787 = torch.ops.aten.permute.default(getitem_355, [1, 0, 2])
    add_12557 = torch.ops.aten.add.Tensor(bmm_29, arg1312_1)
    tanh_4 = torch.ops.aten.tanh.default(add_12557)
    mul_8267 = torch.ops.aten.mul.Tensor(permute_787, tanh_4)
    return mul_8267


SAMPLE_BINDINGS = [
    {'s0': 400},
    {'s0': 200},
    {'s0': 256},
    {'s0': 512},
]


COMPILE_BINDINGS = [
    {'s0': 400},
]


def make_inputs(binding, device):
    s0 = binding['s0']
    getitem_355 = rand_strided(
        (s0, 32, 1600),
        (51200, 1600, 1),
        device=device,
        dtype=torch.float16,
    )
    bmm_29 = rand_strided(
        (32, s0, 1600),
        (1600*s0, 1600, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1312_1 = rand_strided(
        (32, 1, 1600),
        (1600, 1600, 1),
        device=device,
        dtype=torch.float16,
    )
    return (getitem_355, bmm_29, arg1312_1), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (1,)}


CASE = {
    'name': 'pointwise_triton_poi_fused_add_mul_tanh_transpose_204',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
