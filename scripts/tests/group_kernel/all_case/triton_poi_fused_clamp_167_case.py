import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(cat_130):
    convert_element_type_81 = torch.ops.prims.convert_element_type.default(cat_130, torch.float32)
    clamp_min_7 = torch.ops.aten.clamp_min.default(convert_element_type_81, -20)
    clamp_max_7 = torch.ops.aten.clamp_max.default(clamp_min_7, 20)
    convert_element_type_82 = torch.ops.prims.convert_element_type.default(clamp_max_7, torch.float16)
    return convert_element_type_82


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
    cat_130 = rand_strided(
        (s0, 704),
        (704, 1),
        device=device,
        dtype=torch.float16,
    )
    return (cat_130,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_clamp_167',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
