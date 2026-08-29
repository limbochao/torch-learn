import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg1643_1, cat_210, arg1647_1, arg1646_1):
    view_701 = torch.ops.aten.reshape.default(arg1643_1, [1, arg124_1, 8])
    squeeze_dims_13 = torch.ops.aten.squeeze.dims(view_701, [0])
    view_707 = torch.ops.aten.reshape.default(arg1646_1, [1, arg124_1, 8])
    squeeze_dims_12 = torch.ops.aten.squeeze.dims(view_707, [0])
    mul_10400 = torch.ops.aten.mul.Tensor(squeeze_dims_13, squeeze_dims_12)
    view_710 = torch.ops.aten.reshape.default(arg1647_1, [1, arg124_1, 8])
    squeeze_dims_11 = torch.ops.aten.squeeze.dims(view_710, [0])
    mul_10403 = torch.ops.aten.mul.Tensor(squeeze_dims_13, squeeze_dims_11)
    mul_10406 = torch.ops.aten.mul.Tensor(squeeze_dims_11, squeeze_dims_12)
    view_704 = torch.ops.aten.reshape.default(cat_210, [2, arg124_1, 8])
    sum_260 = torch.ops.aten.sum.dim_IntList(view_704, [0])
    mul_10389 = torch.ops.aten.mul.Tensor(squeeze_dims_13, sum_260)
    mul_10392 = torch.ops.aten.mul.Tensor(squeeze_dims_13, squeeze_dims_11)
    mul_10395 = torch.ops.aten.mul.Tensor(squeeze_dims_11, sum_260)
    return mul_10389,mul_10395,mul_10400,mul_10403,mul_10392,mul_10406


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
    arg1643_1 = rand_strided(
        (s0, 8),
        (8, 1),
        device=device,
        dtype=torch.float16,
    )
    cat_210 = rand_strided(
        (2*s0, 8),
        (8, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1647_1 = rand_strided(
        (s0, 8),
        (8, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1646_1 = rand_strided(
        (s0, 8),
        (8, 1),
        device=device,
        dtype=torch.float16,
    )
    return (arg1643_1, cat_210, arg1647_1, arg1646_1), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (0,), 'args[2]': (0,), 'args[3]': (0,)}


CASE = {
    'name': 'triton_poi_fused_mul_stack_sum_265',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
