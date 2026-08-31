import torch
from math import inf, nan
from cmath import nanj
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg1460_1, mm_default_85, softcap_38):
    add_tensor_85 = torch.ops.aten.add.Tensor(arg1460_1, mm_default_85)
    add_13550 = torch.ops.aten.add.Tensor(add_tensor_85, softcap_38)
    softcap_39 = add_13550
    return softcap_39


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
    arg1460_1 = rand_strided(
        (512,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    mm_default_85 = rand_strided(
        (2*s0, 512),
        (512, 1),
        device=device,
        dtype=torch.float16,
    )
    softcap_38 = rand_strided(
        (2*s0, 512),
        (512, 1),
        device=device,
        dtype=torch.float16,
    )
    return (arg1460_1, mm_default_85, softcap_38), {}


DYNAMIC_DIMS = {'args[1]': (0,), 'args[2]': (0,)}


CASE = {
    'name': 'triton_poi_fused_add_addmm_softcap_234',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
