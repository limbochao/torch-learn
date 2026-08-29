import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(convert_element_type_76, mm, arg881_1):
    add_10649 = torch.ops.aten.add.Tensor(mm, arg881_1)
    mul_7028 = torch.ops.aten.mul.Tensor(convert_element_type_76, add_10649)
    add_10656 = torch.ops.aten.add.Tensor(mul_7028, convert_element_type_76)
    return add_10656


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
    mm = rand_strided(
        (s0, 744),
        (744, 1),
        device=device,
        dtype=torch.float16,
    )
    arg881_1 = rand_strided(
        (1, 744),
        (744, 1),
        device=device,
        dtype=torch.float16,
    )
    return (convert_element_type_76, mm, arg881_1), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (0,)}


CASE = {
    'name': 'triton_poi_fused_add_mul_165',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
