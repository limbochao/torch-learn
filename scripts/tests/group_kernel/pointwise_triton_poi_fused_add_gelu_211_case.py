import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(bmm_36, arg1330_1):
    add_12716 = torch.ops.aten.add.Tensor(bmm_36, arg1330_1)
    mul_8409 = torch.ops.aten.mul.Tensor(add_12716, add_12716)
    mul_8410 = torch.ops.aten.mul.Tensor(mul_8409, add_12716)
    mul_8411 = torch.ops.aten.mul.Tensor(mul_8410, 0.044715)
    add_12721 = torch.ops.aten.add.Tensor(add_12716, mul_8411)
    mul_8412 = torch.ops.aten.mul.Tensor(add_12721, 1.5957691216057308)
    sigmoid_22 = torch.ops.aten.sigmoid.default(mul_8412)
    mul_8413 = torch.ops.aten.mul.Tensor(add_12716, sigmoid_22)
    return mul_8413


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
    bmm_36 = rand_strided(
        (16, s0, 960),
        (960*s0, 960, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1330_1 = rand_strided(
        (16, 1, 960),
        (960, 960, 1),
        device=device,
        dtype=torch.float16,
    )
    return (bmm_36, arg1330_1), {}


DYNAMIC_DIMS = {'args[0]': (1,)}


CASE = {
    'name': 'pointwise_triton_poi_fused_add_gelu_211',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
