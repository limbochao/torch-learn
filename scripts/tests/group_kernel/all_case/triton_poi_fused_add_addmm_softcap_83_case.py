import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg675_1, mm_default_243, softcap_14):
    add_tensor_243 = torch.ops.aten.add.Tensor(arg675_1, mm_default_243)
    add_8719 = torch.ops.aten.add.Tensor(add_tensor_243, softcap_14)
    softcap_15 = torch.ops.qianchuan_triton.softcap.default(add_8719, 50.0)
    return buf884


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
    arg675_1 = rand_strided(
        (512,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    mm_default_243 = rand_strided(
        (s0, 512),
        (512, 1),
        device=device,
        dtype=torch.float16,
    )
    softcap_14 = rand_strided(
        (s0, 512),
        (512, 1),
        device=device,
        dtype=torch.float16,
    )
    return (arg675_1, mm_default_243, softcap_14), {}


DYNAMIC_DIMS = {'args[1]': (0,), 'args[2]': (0,)}


CASE = {
    'name': 'triton_poi_fused_add_addmm_softcap_83',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
