import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg263_1):
    slice_532 = torch.ops.aten.slice.Tensor(arg263_1, 2, 1, 130)
    slice_533 = torch.ops.aten.slice.Tensor(slice_532, 2, 0, 128)
    slice_537 = torch.ops.aten.slice.Tensor(slice_533, 1, 0, 2)
    sum_45 = torch.ops.aten.sum.dim_IntList(slice_537, [1])
    slice_538 = torch.ops.aten.slice.Tensor(slice_533, 1, 0, 4)
    sum_46 = torch.ops.aten.sum.dim_IntList(slice_538, [1])
    return sum_45,sum_46


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
    arg263_1 = rand_strided(
        (s0, 32, 130),
        (4160, 130, 1),
        device=device,
        dtype=torch.float16,
    )
    return (arg263_1,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_slice_sum_142',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
