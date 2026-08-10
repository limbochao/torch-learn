import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(cat_150, getitem_352):
    # arg124_1 is omitted from the Graph fragment placeholders; it is cat_150's dynamic leading size.
    view_475 = torch.ops.aten.reshape.default(cat_150, [cat_150.shape[0], 80, 640])
    add_12496 = torch.ops.aten.add.Tensor(view_475, getitem_352)
    view_516 = torch.ops.aten.reshape.default(add_12496, [-1, 80, 32, 20])
    permute_786 = torch.ops.aten.permute.default(view_516, [0, 2, 1, 3])
    clone_60 = torch.ops.aten.clone.default(permute_786, memory_format=torch.contiguous_format)
    return clone_60


SAMPLE_BINDINGS = [
    {'s0': 199},
    {'s0': 200},
    {'s0': 201},
]


COMPILE_BINDINGS = [
    {'s0': 200},
]


def make_inputs(binding, device):
    s0 = binding['s0']
    cat_150 = rand_strided(
        (s0, 51200),
        (51200, 1),
        device=device,
        dtype=torch.float16,
    )
    getitem_352 = rand_strided(
        (s0, 80, 640),
        (51200, 640, 1),
        device=device,
        dtype=torch.float16,
    )
    return (cat_150, getitem_352), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (0,)}


CASE = {
    'name': 'triton_poi_fused_add_clone_stack_transpose_vie_202',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
