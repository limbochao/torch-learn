import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(isnan_5, convert_element_type_195):
    arg4_1 = isnan_5.shape[0]
    full_default_201 = torch.ops.aten.full.default([arg4_1], 0, dtype=torch.float16, layout=torch.strided, device=torch.device(device), pin_memory=False)
    where_296 = torch.ops.aten.where.self(isnan_5, full_default_201, convert_element_type_195)
    full_default_309 = torch.ops.aten.full.default([], -2.197265625, dtype=torch.float16, layout=torch.strided, device=torch.device('cpu'), pin_memory=False)
    add_15514 = torch.ops.aten.add.Tensor(where_296, full_default_309)
    return add_15514


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
    isnan_5 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.bool,
    )
    convert_element_type_195 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    return (isnan_5, convert_element_type_195), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (0,)}


CASE = {
    'name': 'triton_poi_fused_add_neg_where_zeros_257',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
