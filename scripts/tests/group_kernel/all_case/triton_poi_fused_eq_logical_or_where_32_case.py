import torch
from math import inf, nan
from cmath import nanj
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(where_7, add_tensor_291_fused_relu, add_tensor_290_fused_relu):
    sym_size_int_854 = where_7.shape[0]
    eq_3958 = torch.ops.aten.eq.Scalar(where_7, 9998)
    eq_3960 = torch.ops.aten.eq.Scalar(where_7, 3)
    logical_or_30 = torch.ops.aten.logical_or.default(eq_3958, eq_3960)
    expand_default = torch.ops.aten.expand.default(logical_or_30, [sym_size_int_854, 512])
    where_203 = torch.ops.aten.where.self(expand_default, add_tensor_291_fused_relu, add_tensor_290_fused_relu)
    return where_203


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
    add_tensor_291_fused_relu = rand_strided(
        (s0, 512),
        (512, 1),
        device=device,
        dtype=torch.float16,
    )
    add_tensor_290_fused_relu = rand_strided(
        (s0, 512),
        (512, 1),
        device=device,
        dtype=torch.float16,
    )
    return (where_7, add_tensor_291_fused_relu, add_tensor_290_fused_relu), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (0,), 'args[2]': (0,)}


CASE = {
    'name': 'triton_poi_fused_eq_logical_or_where_32',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
