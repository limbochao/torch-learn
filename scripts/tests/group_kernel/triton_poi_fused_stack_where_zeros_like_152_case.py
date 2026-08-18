import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_87, arg944_1, arg945_1, arg946_1):
    arg4_1 = view_87.shape[0]
    arg124_1 = view_87.shape[0]
    logical_not_2 = view_87
    sym_size_int_869 = view_87.shape[0]
    sym_size_int_870 = view_87.shape[0]
    sym_size_int_871 = view_87.shape[0]
    full_default_46 = torch.ops.aten.full.default([arg4_1, 16], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    expand_default_15 = torch.ops.aten.expand.default(logical_not_2, [sym_size_int_869, 16])
    view_468 = torch.ops.aten.reshape.default(arg944_1, [1, arg124_1, 16])
    squeeze_dims_42 = torch.ops.aten.squeeze.dims(view_468, [0])
    where_252 = torch.ops.aten.where.self(expand_default_15, full_default_46, squeeze_dims_42)
    expand_default_16 = torch.ops.aten.expand.default(logical_not_2, [sym_size_int_870, 16])
    where_253 = torch.ops.aten.where.self(expand_default_16, arg945_1, full_default_46)
    expand_default_17 = torch.ops.aten.expand.default(logical_not_2, [sym_size_int_871, 16])
    where_254 = torch.ops.aten.where.self(expand_default_17, full_default_46, arg946_1)
    return where_252,where_253,where_254


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
    view_87 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.bool,
    )
    arg944_1 = rand_strided(
        (s0, 16),
        (16, 1),
        device=device,
        dtype=torch.float16,
    )
    arg945_1 = rand_strided(
        (s0, 16),
        (16, 1),
        device=device,
        dtype=torch.float16,
    )
    arg946_1 = rand_strided(
        (s0, 16),
        (16, 1),
        device=device,
        dtype=torch.float16,
    )
    return (view_87, arg944_1, arg945_1, arg946_1), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (0,), 'args[2]': (0,), 'args[3]': (0,)}


CASE = {
    'name': 'triton_poi_fused_stack_where_zeros_like_152',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
