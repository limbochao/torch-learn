import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(index_7):
    slice_623 = torch.ops.aten.slice.Tensor(index_7, 1, 0, 2)
    sum_115 = torch.ops.aten.sum.dim_IntList(slice_623, [1])
    slice_624 = torch.ops.aten.slice.Tensor(index_7, 1, 0, 4)
    sum_116 = torch.ops.aten.sum.dim_IntList(slice_624, [1])
    return sum_115,sum_116


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
    index_7 = rand_strided(
        (s0, 64, 64),
        (4096, 64, 1),
        device=device,
        dtype=torch.float16,
    )
    return (index_7,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_slice_sum_126',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
