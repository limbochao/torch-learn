import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(where_7, arg160_1, mm_default_252, arg156_1, mm_default_250):
    sym_size_int_859 = where_7.shape[0]
    eq_2011 = torch.ops.aten.eq.Scalar(where_7, 9998)
    eq_2013 = torch.ops.aten.eq.Scalar(where_7, 3)
    logical_or_18 = torch.ops.aten.logical_or.default(eq_2011, eq_2013)
    expand_default_5 = torch.ops.aten.expand.default(logical_or_18, [sym_size_int_859, 32])
    add_tensor_252 = torch.ops.aten.add.Tensor(arg160_1, mm_default_252)
    add_tensor_250 = torch.ops.aten.add.Tensor(arg156_1, mm_default_250)
    where_121 = torch.ops.aten.where.self(expand_default_5, add_tensor_252, add_tensor_250)
    return where_121


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
    where_7 = rand_strided(
        (s0, 1),
        (1, s0),
        device=device,
        dtype=torch.int64,
    )
    arg160_1 = rand_strided(
        (32,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    mm_default_252 = rand_strided(
        (s0, 32),
        (32, 1),
        device=device,
        dtype=torch.float16,
    )
    arg156_1 = rand_strided(
        (32,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    mm_default_250 = rand_strided(
        (s0, 32),
        (32, 1),
        device=device,
        dtype=torch.float16,
    )
    return (where_7, arg160_1, mm_default_252, arg156_1, mm_default_250), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[2]': (0,), 'args[4]': (0,)}


CASE = {
    'name': 'pointwise_triton_poi_fused_addmm_eq_logical_or_where_70',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
