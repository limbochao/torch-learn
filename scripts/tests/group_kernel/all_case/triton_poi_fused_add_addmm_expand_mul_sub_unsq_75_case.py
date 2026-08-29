import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(bmm_10, arg454_1, mm_default_247, where_121, add_7135):
    unsqueeze_42 = torch.ops.aten.unsqueeze.default(where_121, 1)
    expand_32 = torch.ops.aten.expand.default(unsqueeze_42, [-1, 64, -1])
    add_tensor_247 = torch.ops.aten.add.Tensor(arg454_1, mm_default_247)
    unsqueeze_41 = torch.ops.aten.unsqueeze.default(add_tensor_247, 1)
    add_7135 = torch.ops.aten.add.Tensor(bmm_10, unsqueeze_41)
    sub_2307 = torch.ops.aten.sub.Tensor(expand_32, add_7135)
    mul_4133 = torch.ops.aten.mul.Tensor(expand_32, add_7135)
    return add_7135,sub_2307,mul_4133


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
    bmm_10 = rand_strided(
        (s0, 64, 32),
        (2048, 32, 1),
        device=device,
        dtype=torch.float16,
    )
    arg454_1 = rand_strided(
        (32,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    mm_default_247 = rand_strided(
        (s0, 32),
        (32, 1),
        device=device,
        dtype=torch.float16,
    )
    where_121 = rand_strided(
        (s0, 32),
        (32, 1),
        device=device,
        dtype=torch.float16,
    )
    add_7135 = rand_strided(
        (s0, 64, 32),
        (2048, 32, 1),
        device=device,
        dtype=torch.float16,
    )
    return (bmm_10, arg454_1, mm_default_247, where_121, add_7135), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[2]': (0,), 'args[3]': (0,), 'args[4]': (0,)}


CASE = {
    'name': 'triton_poi_fused_add_addmm_expand_mul_sub_unsq_75',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
