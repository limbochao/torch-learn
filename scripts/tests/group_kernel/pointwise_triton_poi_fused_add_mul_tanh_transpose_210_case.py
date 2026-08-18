import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(getitem_361, bmm_35, arg1328_1):
    permute_790 = torch.ops.aten.permute.default(getitem_361, [1, 0, 2])
    add_12707 = torch.ops.aten.add.Tensor(bmm_35, arg1328_1)
    tanh_6 = torch.ops.aten.tanh.default(add_12707)
    mul_8430 = torch.ops.aten.mul.Tensor(permute_790, tanh_6)
    return mul_8430


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
    getitem_361 = rand_strided(
        (s0, 16, 1280),
        (20480, 1280, 1),
        device=device,
        dtype=torch.float16,
    )
    bmm_35 = rand_strided(
        (16, s0, 1280),
        (1280*s0, 1280, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1328_1 = rand_strided(
        (16, 1, 1280),
        (1280, 1280, 1),
        device=device,
        dtype=torch.float16,
    )
    return (getitem_361, bmm_35, arg1328_1), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (1,)}


CASE = {
    'name': 'pointwise_triton_poi_fused_add_mul_tanh_transpose_210',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
