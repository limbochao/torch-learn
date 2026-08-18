import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg108_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1):
    arg4_1 = view_723.shape[0]
    view_86 = view_723
    full_default_94 = torch.ops.aten.full.default([arg4_1, 4], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    where_90 = torch.ops.aten.where.self(view_86, full_default_94, arg108_1)
    where_95 = torch.ops.aten.where.self(view_86, full_default_94, arg113_1)
    where_96 = torch.ops.aten.where.self(view_86, full_default_94, arg114_1)
    where_97 = torch.ops.aten.where.self(view_86, full_default_94, arg115_1)
    where_98 = torch.ops.aten.where.self(view_86, full_default_94, arg116_1)
    where_99 = torch.ops.aten.where.self(view_86, full_default_94, arg117_1)
    where_100 = torch.ops.aten.where.self(view_86, full_default_94, arg118_1)
    where_101 = torch.ops.aten.where.self(view_86, full_default_94, arg119_1)
    return where_90,where_95,where_96,where_97,where_98,where_99,where_100,where_101


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
    arg108_1 = rand_strided(
        (s0, 4),
        (4, 1),
        device=device,
        dtype=torch.float16,
    )
    arg113_1 = rand_strided(
        (s0, 4),
        (4, 1),
        device=device,
        dtype=torch.float16,
    )
    arg114_1 = rand_strided(
        (s0, 4),
        (4, 1),
        device=device,
        dtype=torch.float16,
    )
    arg115_1 = rand_strided(
        (s0, 4),
        (4, 1),
        device=device,
        dtype=torch.float16,
    )
    arg116_1 = rand_strided(
        (s0, 4),
        (4, 1),
        device=device,
        dtype=torch.float16,
    )
    arg117_1 = rand_strided(
        (s0, 4),
        (4, 1),
        device=device,
        dtype=torch.float16,
    )
    arg118_1 = rand_strided(
        (s0, 4),
        (4, 1),
        device=device,
        dtype=torch.float16,
    )
    arg119_1 = rand_strided(
        (s0, 4),
        (4, 1),
        device=device,
        dtype=torch.float16,
    )
    return (view_723, arg108_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1), {}


DYNAMIC_DIMS = {'args[0]': (0,),
 'args[1]': (0,),
 'args[2]': (0,),
 'args[3]': (0,),
 'args[4]': (0,),
 'args[5]': (0,),
 'args[6]': (0,),
 'args[7]': (0,),
 'args[8]': (0,)}


CASE = {
    'name': 'pointwise_triton_poi_fused_where_zeros_like_154',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
