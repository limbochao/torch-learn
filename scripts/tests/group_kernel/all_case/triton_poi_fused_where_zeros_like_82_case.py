import torch
from math import inf, nan
from cmath import nanj
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(logical_or_12, cat_113):
    arg4_1 = logical_or_12.shape[0]
    full_default_200 = torch.ops.aten.full.default([arg4_1, 320], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    where_201 = torch.ops.aten.where.self(logical_or_12, full_default_200, cat_113)
    return where_201


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
    logical_or_12 = rand_strided(
        (s0, 1),
        (1, s0),
        device=device,
        dtype=torch.bool,
    )
    cat_113 = rand_strided(
        (s0, 320),
        (320, 1),
        device=device,
        dtype=torch.float16,
    )
    return (logical_or_12, cat_113), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (0,)}


CASE = {
    'name': 'triton_poi_fused_where_zeros_like_82',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
