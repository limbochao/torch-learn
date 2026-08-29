import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg383_1):
    slice_599 = torch.ops.aten.slice.Tensor(arg383_1, 2, 1, 66)
    slice_600 = torch.ops.aten.slice.Tensor(slice_599, 2, 0, 64)
    slice_604 = torch.ops.aten.slice.Tensor(slice_600, 1, 0, 2)
    sum_105 = torch.ops.aten.sum.dim_IntList(slice_604, [1])
    slice_605 = torch.ops.aten.slice.Tensor(slice_600, 1, 0, 4)
    sum_106 = torch.ops.aten.sum.dim_IntList(slice_605, [1])
    return sum_105,sum_106


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
    arg383_1 = rand_strided(
        (s0, 80, 66),
        (5280, 66, 1),
        device=device,
        dtype=torch.float16,
    )
    return (arg383_1,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_slice_sum_150',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
