import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg323_1):
    slice_569 = torch.ops.aten.slice.Tensor(arg323_1, 2, 1, 34)
    slice_570 = torch.ops.aten.slice.Tensor(slice_569, 2, 0, 32)
    slice_574 = torch.ops.aten.slice.Tensor(slice_570, 1, 0, 2)
    sum_75 = torch.ops.aten.sum.dim_IntList(slice_574, [1])
    slice_575 = torch.ops.aten.slice.Tensor(slice_570, 1, 0, 4)
    sum_76 = torch.ops.aten.sum.dim_IntList(slice_575, [1])
    return sum_75,sum_76


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
    arg323_1 = rand_strided(
        (s0, 64, 34),
        (2176, 34, 1),
        device=device,
        dtype=torch.float16,
    )
    return (arg323_1,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_slice_sum_117',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
