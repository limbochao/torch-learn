import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(bmm_1, arg273_1, mm_default_155, where_170, add_5036):
    unsqueeze_6 = torch.ops.aten.unsqueeze.default(where_170, 1)
    expand_5 = torch.ops.aten.expand.default(unsqueeze_6, [-1, 32, -1])
    add_tensor_155 = torch.ops.aten.add.Tensor(arg273_1, mm_default_155)
    unsqueeze_5 = torch.ops.aten.unsqueeze.default(add_tensor_155, 1)
    add_5036 = torch.ops.aten.add.Tensor(bmm_1, unsqueeze_5)
    sub_1713 = torch.ops.aten.sub.Tensor(expand_5, add_5036)
    mul_2715 = torch.ops.aten.mul.Tensor(expand_5, add_5036)
    return add_5036,sub_1713,mul_2715


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
    bmm_1 = rand_strided(
        (s0, 32, 32),
        (1024, 32, 1),
        device=device,
        dtype=torch.float16,
    )
    arg273_1 = rand_strided(
        (32,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    mm_default_155 = rand_strided(
        (s0, 32),
        (32, 1),
        device=device,
        dtype=torch.float16,
    )
    where_170 = rand_strided(
        (s0, 32),
        (32, 1),
        device=device,
        dtype=torch.float16,
    )
    add_5036 = rand_strided(
        (s0, 32, 32),
        (1024, 32, 1),
        device=device,
        dtype=torch.float16,
    )
    return (bmm_1, arg273_1, mm_default_155, where_170, add_5036), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[2]': (0,), 'args[3]': (0,), 'args[4]': (0,)}


CASE = {
    'name': 'triton_poi_fused_add_addmm_expand_mul_sub_unsq_137',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
