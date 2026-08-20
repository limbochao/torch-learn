import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(bmm_7, arg393_1, mm_default_139, where_149, add_6357):
    unsqueeze_30 = torch.ops.aten.unsqueeze.default(where_149, 1)
    expand_23 = torch.ops.aten.expand.default(unsqueeze_30, [-1, 80, -1])
    add_tensor_139 = torch.ops.aten.add.Tensor(arg393_1, mm_default_139)
    unsqueeze_29 = torch.ops.aten.unsqueeze.default(add_tensor_139, 1)
    add_6357 = torch.ops.aten.add.Tensor(bmm_7, unsqueeze_29)
    sub_2087 = torch.ops.aten.sub.Tensor(expand_23, add_6357)
    mul_3611 = torch.ops.aten.mul.Tensor(expand_23, add_6357)
    return add_6357,sub_2087,mul_3611


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
    bmm_7 = rand_strided(
        (s0, 80, 32),
        (2560, 32, 1),
        device=device,
        dtype=torch.float16,
    )
    arg393_1 = rand_strided(
        (32,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    mm_default_139 = rand_strided(
        (s0, 32),
        (32, 1),
        device=device,
        dtype=torch.float16,
    )
    where_149 = rand_strided(
        (s0, 32),
        (32, 1),
        device=device,
        dtype=torch.float16,
    )
    add_6357 = rand_strided(
        (s0, 80, 32),
        (2560, 32, 1),
        device=device,
        dtype=torch.float16,
    )
    return (bmm_7, arg393_1, mm_default_139, where_149, add_6357), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[2]': (0,), 'args[3]': (0,), 'args[4]': (0,)}


CASE = {
    'name': 'pointwise_triton_poi_fused_add_addmm_expand_mul_sub_unsq_144',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
