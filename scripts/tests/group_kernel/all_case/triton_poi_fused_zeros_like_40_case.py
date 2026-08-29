import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(nonzero):
    full_206 = torch.ops.aten.full.default([sym_size_int_352], False, dtype=torch.bool, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    return index_put_1


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
    nonzero = rand_strided(
        (1,),
        (1,),
        device=device,
        dtype=torch.int64,
    )
    return (nonzero,), {}


DYNAMIC_DIMS = {}


CASE = {
    'name': 'triton_poi_fused_zeros_like_40',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
