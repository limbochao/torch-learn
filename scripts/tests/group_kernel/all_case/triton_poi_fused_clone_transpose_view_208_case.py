import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(getitem_358):
    view_536 = torch.ops.aten.reshape.default(getitem_358, [-1, 32, 16, 40])
    permute_789 = torch.ops.aten.permute.default(view_536, [0, 2, 1, 3])
    clone_65 = torch.ops.aten.clone.default(permute_789, memory_format=torch.contiguous_format)
    return clone_65


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
    getitem_358 = rand_strided(
        (s0, 32, 640),
        (20480, 640, 1),
        device=device,
        dtype=torch.float16,
    )
    return (getitem_358,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_clone_transpose_view_208',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
