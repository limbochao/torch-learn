import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(bmm_39, arg1336_1):
    add_12772 = torch.ops.aten.add.Tensor(bmm_39, arg1336_1)
    return add_12772


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
    bmm_39 = rand_strided(
        (16, s0, 640),
        (640*s0, 640, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1336_1 = rand_strided(
        (16, 1, 640),
        (640, 640, 1),
        device=device,
        dtype=torch.float16,
    )
    return (bmm_39, arg1336_1), {}


DYNAMIC_DIMS = {'args[0]': (1,)}


CASE = {
    'name': 'pointwise_non_first_split_triton_poi_fused_add_213',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
