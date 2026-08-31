import torch
from math import inf, nan
from cmath import nanj
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg6_1, arg7_1, arg5_1, add_tensor_16_fused_relu, add_tensor_14_fused_relu):
    arg4_1 = arg6_1.shape[0]
    eq_16 = torch.ops.aten.eq.Scalar(arg6_1, 96)
    eq_11 = torch.ops.aten.eq.Scalar(arg6_1, 412)
    eq_13 = torch.ops.aten.eq.Scalar(arg7_1, 102)
    logical_and = torch.ops.aten.logical_and.default(eq_11, eq_13)
    logical_or_2 = torch.ops.aten.logical_or.default(eq_16, logical_and)
    full_default_7 = torch.ops.aten.full.default([arg4_1, 1], 9998, dtype=torch.int64, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    where = torch.ops.aten.where.self(logical_or_2, full_default_7, arg5_1)
    full_default_13 = torch.ops.aten.full.default([arg4_1, 1], 0, dtype=torch.int64, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    eq_55 = torch.ops.aten.eq.Tensor(where, full_default_13)
    where_313 = torch.ops.aten.where.self(eq_55, add_tensor_16_fused_relu, add_tensor_14_fused_relu)
    return where_313


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
    arg6_1 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.int64,
    )
    arg7_1 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.int64,
    )
    arg5_1 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.int64,
    )
    add_tensor_16_fused_relu = rand_strided(
        (s0, 64),
        (64, 1),
        device=device,
        dtype=torch.float16,
    )
    add_tensor_14_fused_relu = rand_strided(
        (s0, 64),
        (64, 1),
        device=device,
        dtype=torch.float16,
    )
    return (arg6_1, arg7_1, arg5_1, add_tensor_16_fused_relu, add_tensor_14_fused_relu), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (0,), 'args[2]': (0,), 'args[3]': (0,), 'args[4]': (0,)}


CASE = {
    'name': 'triton_poi_fused_eq_logical_and_logical_or_mul_269',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
