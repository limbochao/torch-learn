import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(getitem_343, bmm_21, arg1288_1):
    add_12307 = torch.ops.aten.add.Tensor(bmm_21, arg1288_1)
    permute_782 = torch.ops.aten.permute.default(add_12307, [1, 0, 2])
    add_12324 = torch.ops.aten.add.Tensor(getitem_343, permute_782)
    return add_12324


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
    getitem_343 = rand_strided(
        (s0, 80, 640),
        (51200, 640, 1),
        device=device,
        dtype=torch.float16,
    )
    bmm_21 = rand_strided(
        (80, s0, 640),
        (640*s0, 640, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1288_1 = rand_strided(
        (80, 1, 640),
        (640, 640, 1),
        device=device,
        dtype=torch.float16,
    )
    return (getitem_343, bmm_21, arg1288_1), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (1,)}


CASE = {
    'name': 'pointwise_triton_poi_fused_add_transpose_200',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
