import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(convert_element_type_76, mm_1, arg883_1, add_10656):
    add_10663 = torch.ops.aten.add.Tensor(mm_1, arg883_1)
    mul_7037 = torch.ops.aten.mul.Tensor(convert_element_type_76, add_10663)
    add_10670 = torch.ops.aten.add.Tensor(mul_7037, add_10656)
    return add_10670


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
    convert_element_type_76 = rand_strided(
        (s0, 744),
        (744, 1),
        device=device,
        dtype=torch.float16,
    )
    mm_1 = rand_strided(
        (s0, 744),
        (744, 1),
        device=device,
        dtype=torch.float16,
    )
    arg883_1 = rand_strided(
        (1, 744),
        (744, 1),
        device=device,
        dtype=torch.float16,
    )
    add_10656 = rand_strided(
        (s0, 744),
        (744, 1),
        device=device,
        dtype=torch.float16,
    )
    return (convert_element_type_76, mm_1, arg883_1, add_10656), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (0,), 'args[3]': (0,)}


CASE = {
    'name': 'triton_poi_fused_add_mul_166',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
