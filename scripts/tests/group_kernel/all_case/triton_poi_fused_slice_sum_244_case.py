import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(sum_151):
    slice_682 = torch.ops.aten.slice.Tensor(sum_151, 1, 0, 2)
    sum_152 = torch.ops.aten.sum.dim_IntList(slice_682, [1])
    slice_683 = torch.ops.aten.slice.Tensor(sum_151, 1, 0, 4)
    sum_153 = torch.ops.aten.sum.dim_IntList(slice_683, [1])
    return sum_152,sum_153


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
    sum_151 = rand_strided(
        (s0, 64, 128),
        (8192, 128, 1),
        device=device,
        dtype=torch.float16,
    )
    return (sum_151,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_slice_sum_244',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
