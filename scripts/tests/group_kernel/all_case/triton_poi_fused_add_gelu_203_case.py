import torch
from torch._dynamo.testing import rand_strided


# Prefix static split axis case.
# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(bmm_28, arg1310_1):
    add_12543 = torch.ops.aten.add.Tensor(bmm_28, arg1310_1)
    mul_8215 = torch.ops.aten.mul.Tensor(add_12543, add_12543)
    mul_8216 = torch.ops.aten.mul.Tensor(mul_8215, add_12543)
    mul_8217 = torch.ops.aten.mul.Tensor(mul_8216, 0.044715)
    add_12548 = torch.ops.aten.add.Tensor(add_12543, mul_8217)
    mul_8218 = torch.ops.aten.mul.Tensor(add_12548, 1.5957691216057308)
    sigmoid_18 = torch.ops.aten.sigmoid.default(mul_8218)
    mul_8219 = torch.ops.aten.mul.Tensor(add_12543, sigmoid_18)
    return mul_8219


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
    bmm_28 = rand_strided(
        (32, s0, 200),
        (200*s0, 200, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1310_1 = rand_strided(
        (32, 1, 200),
        (200, 200, 1),
        device=device,
        dtype=torch.float16,
    )
    return (bmm_28, arg1310_1), {}


DYNAMIC_DIMS = {'args[0]': (1,)}


CASE = {
    'name': 'triton_poi_fused_add_gelu_203',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
