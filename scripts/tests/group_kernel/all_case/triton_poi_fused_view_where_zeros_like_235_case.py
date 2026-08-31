import torch
from math import inf, nan
from cmath import nanj
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(logical_or_12, softcap_39):
    arg4_1 = logical_or_12.shape[0]
    full_default_259 = torch.ops.aten.full.default([arg4_1, 1024], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    view_592 = torch.ops.aten.reshape.default(softcap_39, [-1, 1024])
    where_256 = torch.ops.aten.where.self(logical_or_12, full_default_259, view_592)
    return where_256


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
    softcap_39 = rand_strided(
        (2*s0, 512),
        (512, 1),
        device=device,
        dtype=torch.float16,
    )
    return (logical_or_12, softcap_39), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (0,)}


CASE = {
    'name': 'triton_poi_fused_view_where_zeros_like_235',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
