import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(cat_196, arg1610_1, mm_default_64):
    add_tensor_64 = torch.ops.aten.add.Tensor(arg1610_1, mm_default_64)
    sigmoid_29 = torch.ops.aten.sigmoid.default(add_tensor_64)
    mul_10173 = torch.ops.aten.mul.Tensor(sigmoid_29, 2.0)
    mul_10178 = torch.ops.aten.mul.Tensor(cat_196, mul_10173)
    sum_245 = torch.ops.aten.sum.dim_IntList(mul_10178, [1])
    return sum_245


SAMPLE_BINDINGS = [
    {'s0': 64},
    {'s0': 112},
    {'s0': 160},
    {'s0': 208},
    {'s0': 256},
]


COMPILE_BINDINGS = [
    {'s0': 64},
    {'s0': 256},
]


def make_inputs(binding, device):
    s0 = binding['s0']
    cat_196 = rand_strided(
        (s0, 2),
        (2, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1610_1 = rand_strided(
        (2,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    mm_default_64 = rand_strided(
        (s0, 2),
        (2, 1),
        device=device,
        dtype=torch.float16,
    )
    return (cat_196, arg1610_1, mm_default_64), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[2]': (0,)}


CASE = {
    'name': 'triton_poi_fused_addmm_mul_sigmoid_sum_255',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
