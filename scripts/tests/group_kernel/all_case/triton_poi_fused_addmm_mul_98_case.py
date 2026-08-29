import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg47_1, mm_default_237, arg49_1, mm_default_236):
    add_tensor_237 = torch.ops.aten.add.Tensor(arg47_1, mm_default_237)
    add_tensor_236 = torch.ops.aten.add.Tensor(arg49_1, mm_default_236)
    mul_988 = torch.ops.aten.mul.Tensor(add_tensor_237, add_tensor_236)
    return mul_988


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
    arg47_1 = rand_strided(
        (256,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    mm_default_237 = rand_strided(
        (s0, 256),
        (256, 1),
        device=device,
        dtype=torch.float16,
    )
    arg49_1 = rand_strided(
        (256,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    mm_default_236 = rand_strided(
        (s0, 256),
        (256, 1),
        device=device,
        dtype=torch.float16,
    )
    return (arg47_1, mm_default_237, arg49_1, mm_default_236), {}


DYNAMIC_DIMS = {'args[1]': (0,), 'args[3]': (0,)}


CASE = {
    'name': 'triton_poi_fused_addmm_mul_98',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
