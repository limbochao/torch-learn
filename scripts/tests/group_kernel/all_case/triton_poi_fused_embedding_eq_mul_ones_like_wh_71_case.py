import torch
from math import inf, nan
from cmath import nanj
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(where_7, arg141_1):
    arg4_1 = where_7.shape[0]
    full_default_17 = torch.ops.aten.full.default([arg4_1, 1], 3, dtype=torch.int64, layout=torch.strided, device=torch.device(device), pin_memory=False)
    full_default_16 = torch.ops.aten.full.default([arg4_1, 1], 2, dtype=torch.int64, layout=torch.strided, device=torch.device(device), pin_memory=False)
    full_default_13 = torch.ops.aten.full.default([arg4_1, 1], 0, dtype=torch.int64, layout=torch.strided, device=torch.device(device), pin_memory=False)
    full_default_14 = torch.ops.aten.full.default([arg4_1, 1], 1, dtype=torch.int64, layout=torch.strided, device=torch.device(device), pin_memory=False)
    eq_1885 = torch.ops.aten.eq.Scalar(where_7, 9999)
    full_default_122 = torch.ops.aten.full.default([arg4_1, 1], 5, dtype=torch.int64, layout=torch.strided, device=torch.device(device), pin_memory=False)
    eq_1878 = torch.ops.aten.eq.Scalar(where_7, 9998)
    full_default_121 = torch.ops.aten.full.default([arg4_1, 1], 4, dtype=torch.int64, layout=torch.strided, device=torch.device(device), pin_memory=False)
    eq_1871 = torch.ops.aten.eq.Scalar(where_7, 3)
    eq_1864 = torch.ops.aten.eq.Scalar(where_7, 2)
    eq_1858 = torch.ops.aten.eq.Scalar(where_7, 1)
    where_108 = torch.ops.aten.where.self(eq_1858, full_default_14, full_default_13)
    where_109 = torch.ops.aten.where.self(eq_1864, full_default_16, where_108)
    where_110 = torch.ops.aten.where.self(eq_1871, full_default_17, where_109)
    where_111 = torch.ops.aten.where.self(eq_1878, full_default_121, where_110)
    where_112 = torch.ops.aten.where.self(eq_1885, full_default_122, where_111)
    embedding_1 = torch.ops.aten.embedding.default(arg141_1, where_112)
    return embedding_1


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
    where_7 = rand_strided(
        (s0, 1),
        (1, s0),
        device=device,
        dtype=torch.int64,
    )
    arg141_1 = rand_strided(
        (7, 128),
        (128, 1),
        device=device,
        dtype=torch.float16,
    )
    return (where_7, arg141_1), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_embedding_eq_mul_ones_like_wh_71',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
