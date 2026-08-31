import torch
from math import inf, nan
from cmath import nanj
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg39_1, cat_1, arg38_1):
    arg124_1 = cat_1.shape[0]
    view_7 = torch.ops.aten.reshape.default(arg38_1, [1, arg124_1, 256])
    squeeze_dims_53 = torch.ops.aten.squeeze.dims(view_7, [0])
    mul_185 = torch.ops.aten.mul.Tensor(cat_1, squeeze_dims_53)
    mul_188 = torch.ops.aten.mul.Tensor(arg39_1, mul_185)
    return mul_188


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
    arg39_1 = rand_strided(
        (1,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    cat_1 = rand_strided(
        (s0, 256),
        (256, 1),
        device=device,
        dtype=torch.float16,
    )
    arg38_1 = rand_strided(
        (s0, 256),
        (256, 1),
        device=device,
        dtype=torch.float16,
    )
    return (arg39_1, cat_1, arg38_1), {}


DYNAMIC_DIMS = {'args[1]': (0,), 'args[2]': (0,)}


CASE = {
    'name': 'triton_poi_fused_mul_stack_91',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
