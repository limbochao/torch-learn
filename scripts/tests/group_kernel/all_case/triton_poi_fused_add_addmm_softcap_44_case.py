import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg521_1, mm_default_288, getitem, nonzero):
    add_tensor_288 = torch.ops.aten.add.Tensor(arg521_1, mm_default_288)
    add_7870 = torch.ops.aten.add.Tensor(add_tensor_288, getitem)
    softcap = torch.ops.qianchuan_triton.softcap.default(add_7870, 50.0)
    return buf156


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
    arg521_1 = rand_strided(
        (512,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    mm_default_288 = rand_strided(
        (s0, 512),
        (512, 1),
        device=device,
        dtype=torch.float16,
    )
    getitem = rand_strided(
        (s0, 512),
        (512, 1),
        device=device,
        dtype=torch.float16,
    )
    nonzero = rand_strided(
        (1,),
        (1,),
        device=device,
        dtype=torch.int64,
    )
    return (arg521_1, mm_default_288, getitem, nonzero), {}


DYNAMIC_DIMS = {'args[1]': (0,), 'args[2]': (0,)}


CASE = {
    'name': 'triton_poi_fused_add_addmm_softcap_44',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
