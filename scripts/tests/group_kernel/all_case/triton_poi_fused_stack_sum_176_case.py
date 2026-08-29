import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(cat_114):
    view_285 = torch.ops.aten.reshape.default(cat_114, [6, arg124_1, 64, 128])
    sum_151 = torch.ops.aten.sum.dim_IntList(view_285, [0])
    return sum_151


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
    cat_114 = rand_strided(
        (6*s0, 64, 128),
        (8192, 128, 1),
        device=device,
        dtype=torch.float16,
    )
    return (cat_114,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_stack_sum_176',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
