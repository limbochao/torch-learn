import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(bmm_8, arg413_1, mm_default_199, where_163, add_6606):
    unsqueeze_34 = torch.ops.aten.unsqueeze.default(where_163, 1)
    expand_26 = torch.ops.aten.expand.default(unsqueeze_34, [-1, 64, -1])
    add_tensor_199 = torch.ops.aten.add.Tensor(arg413_1, mm_default_199)
    unsqueeze_33 = torch.ops.aten.unsqueeze.default(add_tensor_199, 1)
    add_6606 = torch.ops.aten.add.Tensor(bmm_8, unsqueeze_33)
    sub_2157 = torch.ops.aten.sub.Tensor(expand_26, add_6606)
    mul_3775 = torch.ops.aten.mul.Tensor(expand_26, add_6606)
    return add_6606,sub_2157,mul_3775


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
    bmm_8 = rand_strided(
        (s0, 64, 64),
        (4096, 64, 1),
        device=device,
        dtype=torch.float16,
    )
    arg413_1 = rand_strided(
        (64,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    mm_default_199 = rand_strided(
        (s0, 64),
        (64, 1),
        device=device,
        dtype=torch.float16,
    )
    where_163 = rand_strided(
        (s0, 64),
        (64, 1),
        device=device,
        dtype=torch.float16,
    )
    add_6606 = rand_strided(
        (s0, 64, 64),
        (4096, 64, 1),
        device=device,
        dtype=torch.float16,
    )
    return (bmm_8, arg413_1, mm_default_199, where_163, add_6606), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[2]': (0,), 'args[3]': (0,), 'args[4]': (0,)}


CASE = {
    'name': 'pointwise_triton_poi_fused_add_addmm_expand_mul_sub_unsq_120',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
