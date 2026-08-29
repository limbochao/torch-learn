import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg283_1):
    slice_541 = torch.ops.aten.slice.Tensor(arg283_1, 2, 1, 130)
    slice_542 = torch.ops.aten.slice.Tensor(slice_541, 2, 0, 128)
    slice_546 = torch.ops.aten.slice.Tensor(slice_542, 1, 0, 2)
    sum_55 = torch.ops.aten.sum.dim_IntList(slice_546, [1])
    slice_547 = torch.ops.aten.slice.Tensor(slice_542, 1, 0, 4)
    sum_56 = torch.ops.aten.sum.dim_IntList(slice_547, [1])
    return sum_55,sum_56


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
    arg283_1 = rand_strided(
        (s0, 64, 130),
        (8320, 130, 1),
        device=device,
        dtype=torch.float16,
    )
    return (arg283_1,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_slice_sum_111',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
