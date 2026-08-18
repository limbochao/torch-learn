import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1):
    arg4_1 = view_723.shape[0]
    view_456 = view_723
    full_default_94 = torch.ops.aten.full.default([arg4_1, 4], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    where_241 = torch.ops.aten.where.self(view_456, full_default_94, arg903_1)
    where_242 = torch.ops.aten.where.self(view_456, full_default_94, arg904_1)
    where_243 = torch.ops.aten.where.self(view_456, full_default_94, arg905_1)
    where_244 = torch.ops.aten.where.self(view_456, full_default_94, arg906_1)
    where_245 = torch.ops.aten.where.self(view_456, full_default_94, arg907_1)
    where_246 = torch.ops.aten.where.self(view_456, full_default_94, arg908_1)
    where_247 = torch.ops.aten.where.self(view_456, full_default_94, arg909_1)
    return where_241,where_242,where_243,where_244,where_245,where_246,where_247


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
    view_723 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.bool,
    )
    arg903_1 = rand_strided(
        (s0, 4),
        (4, 1),
        device=device,
        dtype=torch.float16,
    )
    arg904_1 = rand_strided(
        (s0, 4),
        (4, 1),
        device=device,
        dtype=torch.float16,
    )
    arg905_1 = rand_strided(
        (s0, 4),
        (4, 1),
        device=device,
        dtype=torch.float16,
    )
    arg906_1 = rand_strided(
        (s0, 4),
        (4, 1),
        device=device,
        dtype=torch.float16,
    )
    arg907_1 = rand_strided(
        (s0, 4),
        (4, 1),
        device=device,
        dtype=torch.float16,
    )
    arg908_1 = rand_strided(
        (s0, 4),
        (4, 1),
        device=device,
        dtype=torch.float16,
    )
    arg909_1 = rand_strided(
        (s0, 4),
        (4, 1),
        device=device,
        dtype=torch.float16,
    )
    return (view_723, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1), {}


DYNAMIC_DIMS = {'args[0]': (0,),
 'args[1]': (0,),
 'args[2]': (0,),
 'args[3]': (0,),
 'args[4]': (0,),
 'args[5]': (0,),
 'args[6]': (0,),
 'args[7]': (0,)}


CASE = {
    'name': 'pointwise_triton_poi_fused_where_zeros_like_192',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
