import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(bmm_30, arg1314_1):
    add_12566 = torch.ops.aten.add.Tensor(bmm_30, arg1314_1)
    mul_8246 = torch.ops.aten.mul.Tensor(add_12566, add_12566)
    mul_8247 = torch.ops.aten.mul.Tensor(mul_8246, add_12566)
    mul_8248 = torch.ops.aten.mul.Tensor(mul_8247, 0.044715)
    add_12571 = torch.ops.aten.add.Tensor(add_12566, mul_8248)
    mul_8249 = torch.ops.aten.mul.Tensor(add_12571, 1.5957691216057308)
    sigmoid_19 = torch.ops.aten.sigmoid.default(mul_8249)
    mul_8250 = torch.ops.aten.mul.Tensor(add_12566, sigmoid_19)
    return mul_8250


SAMPLE_BINDINGS = [
    {'s0': 400},
    {'s0': 200},
    {'s0': 256},
    {'s0': 512},
]


COMPILE_BINDINGS = [
    {'s0': 400},
]


def make_inputs(binding, device):
    s0 = binding['s0']
    bmm_30 = rand_strided(
        (32, s0, 960),
        (960*s0, 960, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1314_1 = rand_strided(
        (32, 1, 960),
        (960, 960, 1),
        device=device,
        dtype=torch.float16,
    )
    return (bmm_30, arg1314_1), {}


DYNAMIC_DIMS = {'args[0]': (1,)}


CASE = {
    'name': 'pointwise_triton_poi_fused_add_gelu_205',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
