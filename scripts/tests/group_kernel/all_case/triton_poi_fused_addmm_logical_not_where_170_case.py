import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg885_1, mm_default_123, arg893_1, mm_default_122):
    logical_not_1 = torch.ops.aten.logical_not.default(logical_not)
    expand_default_23 = torch.ops.aten.expand.default(logical_not_1, [sym_size_int_877, 128])
    add_tensor_123 = torch.ops.aten.add.Tensor(arg885_1, mm_default_123)
    add_tensor_122 = torch.ops.aten.add.Tensor(arg893_1, mm_default_122)
    where_239 = torch.ops.aten.where.self(expand_default_23, add_tensor_123, add_tensor_122)
    return where_239


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
    view_723 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.bool,
    )
    arg885_1 = rand_strided(
        (128,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    mm_default_123 = rand_strided(
        (s0, 128),
        (128, 1),
        device=device,
        dtype=torch.float16,
    )
    arg893_1 = rand_strided(
        (128,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    mm_default_122 = rand_strided(
        (s0, 128),
        (128, 1),
        device=device,
        dtype=torch.float16,
    )
    return (view_723, arg885_1, mm_default_123, arg893_1, mm_default_122), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[2]': (0,), 'args[4]': (0,)}


CASE = {
    'name': 'triton_poi_fused_addmm_logical_not_where_170',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
