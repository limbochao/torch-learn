import torch
from math import inf, nan
from cmath import nanj
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(where_7, arg465_1, mm_default_83, arg463_1, mm_default_82):
    sym_size_int_879 = where_7.shape[0]
    eq_3818 = torch.ops.aten.eq.Scalar(where_7, 9998)
    eq_3820 = torch.ops.aten.eq.Scalar(where_7, 3)
    logical_or_29 = torch.ops.aten.logical_or.default(eq_3818, eq_3820)
    expand_default_25 = torch.ops.aten.expand.default(logical_or_29, [sym_size_int_879, 256])
    add_tensor_83 = torch.ops.aten.add.Tensor(arg465_1, mm_default_83)
    add_tensor_82 = torch.ops.aten.add.Tensor(arg463_1, mm_default_82)
    where_202 = torch.ops.aten.where.self(expand_default_25, add_tensor_83, add_tensor_82)
    return where_202


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
    where_7 = rand_strided(
        (s0, 1),
        (1, s0),
        device=device,
        dtype=torch.int64,
    )
    arg465_1 = rand_strided(
        (256,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    mm_default_83 = rand_strided(
        (s0, 256),
        (256, 1),
        device=device,
        dtype=torch.float16,
    )
    arg463_1 = rand_strided(
        (256,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    mm_default_82 = rand_strided(
        (s0, 256),
        (256, 1),
        device=device,
        dtype=torch.float16,
    )
    return (where_7, arg465_1, mm_default_83, arg463_1, mm_default_82), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[2]': (0,), 'args[4]': (0,)}


CASE = {
    'name': 'triton_poi_fused_addmm_eq_logical_or_where_243',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
