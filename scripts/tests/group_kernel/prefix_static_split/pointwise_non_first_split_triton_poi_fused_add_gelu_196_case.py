import torch
from torch._dynamo.testing import rand_strided


# Prefix static split axis case.
# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(bmm_16, arg1278_1):
    add_12228 = torch.ops.aten.add.Tensor(bmm_16, arg1278_1)
    mul_7883 = torch.ops.aten.mul.Tensor(add_12228, add_12228)
    mul_7884 = torch.ops.aten.mul.Tensor(mul_7883, add_12228)
    mul_7885 = torch.ops.aten.mul.Tensor(mul_7884, 0.044715)
    add_12233 = torch.ops.aten.add.Tensor(add_12228, mul_7885)
    mul_7886 = torch.ops.aten.mul.Tensor(add_12233, 1.5957691216057308)
    sigmoid_12 = torch.ops.aten.sigmoid.default(mul_7886)
    mul_7887 = torch.ops.aten.mul.Tensor(add_12228, sigmoid_12)
    return mul_7887


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
    bmm_16 = rand_strided(
        (80, s0, 80),
        (80*s0, 80, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1278_1 = rand_strided(
        (80, 1, 80),
        (80, 80, 1),
        device=device,
        dtype=torch.float16,
    )
    return (bmm_16, arg1278_1), {}


DYNAMIC_DIMS = {'args[0]': (1,)}


CASE = {
    'name': 'pointwise_non_first_split_triton_poi_fused_add_gelu_196',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
