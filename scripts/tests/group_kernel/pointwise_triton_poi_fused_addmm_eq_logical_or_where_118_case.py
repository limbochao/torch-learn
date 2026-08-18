import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(where_7, arg208_1, mm_default_204, arg204_1, mm_default_202):
    sym_size_int_862 = where_7.shape[0]
    eq_2527 = torch.ops.aten.eq.Scalar(where_7, 9998)
    eq_2529 = torch.ops.aten.eq.Scalar(where_7, 3)
    logical_or_24 = torch.ops.aten.logical_or.default(eq_2527, eq_2529)
    expand_default_8 = torch.ops.aten.expand.default(logical_or_24, [sym_size_int_862, 64])
    add_tensor_204 = torch.ops.aten.add.Tensor(arg208_1, mm_default_204)
    add_tensor_202 = torch.ops.aten.add.Tensor(arg204_1, mm_default_202)
    where_163 = torch.ops.aten.where.self(expand_default_8, add_tensor_204, add_tensor_202)
    return where_163


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
    arg208_1 = rand_strided(
        (64,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    mm_default_204 = rand_strided(
        (s0, 64),
        (64, 1),
        device=device,
        dtype=torch.float16,
    )
    arg204_1 = rand_strided(
        (64,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    mm_default_202 = rand_strided(
        (s0, 64),
        (64, 1),
        device=device,
        dtype=torch.float16,
    )
    return (where_7, arg208_1, mm_default_204, arg204_1, mm_default_202), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[2]': (0,), 'args[4]': (0,)}


CASE = {
    'name': 'pointwise_triton_poi_fused_addmm_eq_logical_or_where_118',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
