import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(getitem_343, bmm_17, arg1280_1):
    permute_781 = torch.ops.aten.permute.default(getitem_343, [1, 0, 2])
    add_12242 = torch.ops.aten.add.Tensor(bmm_17, arg1280_1)
    tanh = torch.ops.aten.tanh.default(add_12242)
    mul_7935 = torch.ops.aten.mul.Tensor(permute_781, tanh)
    return mul_7935


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
    getitem_343 = rand_strided(
        (s0, 80, 640),
        (51200, 640, 1),
        device=device,
        dtype=torch.float16,
    )
    bmm_17 = rand_strided(
        (80, s0, 640),
        (640*s0, 640, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1280_1 = rand_strided(
        (80, 1, 640),
        (640, 640, 1),
        device=device,
        dtype=torch.float16,
    )
    return (getitem_343, bmm_17, arg1280_1), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (1,)}


CASE = {
    'name': 'triton_poi_fused_add_mul_tanh_transpose_197',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
