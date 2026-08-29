import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(where_251, mm_5, add_tensor_239_fused_relu):
    add_11058 = torch.ops.aten.add.Tensor(where_251, mm_5)
    mul_7264 = torch.ops.aten.mul.Tensor(add_11058, add_tensor_239_fused_relu)
    return mul_7264


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
    where_251 = rand_strided(
        (s0, 256),
        (256, 1),
        device=device,
        dtype=torch.float16,
    )
    mm_5 = rand_strided(
        (s0, 256),
        (256, 1),
        device=device,
        dtype=torch.float16,
    )
    add_tensor_239_fused_relu = rand_strided(
        (s0, 256),
        (256, 1),
        device=device,
        dtype=torch.float16,
    )
    return (where_251, mm_5, add_tensor_239_fused_relu), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (0,), 'args[2]': (0,)}


CASE = {
    'name': 'triton_poi_fused_add_mul_94',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
