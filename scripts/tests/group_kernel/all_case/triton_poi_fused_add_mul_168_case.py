import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(convert_element_type_82, mm_2, arg889_1):
    add_10710 = torch.ops.aten.add.Tensor(mm_2, arg889_1)
    mul_7058 = torch.ops.aten.mul.Tensor(convert_element_type_82, add_10710)
    add_10717 = torch.ops.aten.add.Tensor(mul_7058, convert_element_type_82)
    return add_10717


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
    convert_element_type_82 = rand_strided(
        (s0, 704),
        (704, 1),
        device=device,
        dtype=torch.float16,
    )
    mm_2 = rand_strided(
        (s0, 704),
        (704, 1),
        device=device,
        dtype=torch.float16,
    )
    arg889_1 = rand_strided(
        (1, 704),
        (704, 1),
        device=device,
        dtype=torch.float16,
    )
    return (convert_element_type_82, mm_2, arg889_1), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (0,)}


CASE = {
    'name': 'triton_poi_fused_add_mul_168',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
