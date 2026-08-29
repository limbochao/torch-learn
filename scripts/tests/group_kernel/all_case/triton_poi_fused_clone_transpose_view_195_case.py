import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(cat_150):
    view_476 = torch.ops.aten.reshape.default(cat_150, [-1, 80, 80, 8])
    permute_780 = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3])
    clone_50 = torch.ops.aten.clone.default(permute_780, memory_format=torch.contiguous_format)
    return clone_50


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
    cat_150 = rand_strided(
        (s0, 51200),
        (51200, 1),
        device=device,
        dtype=torch.float16,
    )
    return (cat_150,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_clone_transpose_view_195',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
