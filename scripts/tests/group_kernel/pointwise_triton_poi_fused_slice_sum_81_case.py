import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(index_13):
    slice_659 = torch.ops.aten.slice.Tensor(index_13, 1, 0, 2)
    sum_135 = torch.ops.aten.sum.dim_IntList(slice_659, [1])
    slice_660 = torch.ops.aten.slice.Tensor(index_13, 1, 0, 4)
    sum_136 = torch.ops.aten.sum.dim_IntList(slice_660, [1])
    return sum_135,sum_136


SAMPLE_BINDINGS = [
    {'s0': 64},
    {'s0': 112},
    {'s0': 160},
    {'s0': 208},
    {'s0': 256},
]


COMPILE_BINDINGS = [
    {'s0': 64},
    {'s0': 256},
]


def make_inputs(binding, device):
    s0 = binding['s0']
    index_13 = rand_strided(
        (s0, 64, 32),
        (2048, 32, 1),
        device=device,
        dtype=torch.float16,
    )
    return (index_13,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'pointwise_triton_poi_fused_slice_sum_81',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
