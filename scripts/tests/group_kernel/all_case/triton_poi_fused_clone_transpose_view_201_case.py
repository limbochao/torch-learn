import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(getitem_346):
    view_496 = torch.ops.aten.reshape.default(getitem_346, [-1, 80, 80, 8])
    permute_783 = torch.ops.aten.permute.default(view_496, [0, 2, 1, 3])
    clone_55 = torch.ops.aten.clone.default(permute_783, memory_format=torch.contiguous_format)
    return clone_55


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
    getitem_346 = rand_strided(
        (s0, 80, 640),
        (51200, 640, 1),
        device=device,
        dtype=torch.float16,
    )
    return (getitem_346,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_clone_transpose_view_201',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
