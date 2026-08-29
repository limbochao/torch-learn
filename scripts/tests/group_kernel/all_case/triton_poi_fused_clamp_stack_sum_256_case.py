import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(cat_197):
    view_690 = torch.ops.aten.reshape.default(cat_197, [4, arg124_1])
    sum_246 = torch.ops.aten.sum.dim_IntList(view_690, [0])
    convert_element_type_194 = torch.ops.prims.convert_element_type.default(sum_246, torch.float32)
    clamp_min_21 = torch.ops.aten.clamp_min.default(convert_element_type_194, -15)
    clamp_max_19 = torch.ops.aten.clamp_max.default(clamp_min_21, 15)
    convert_element_type_195 = torch.ops.prims.convert_element_type.default(clamp_max_19, torch.float16)
    return convert_element_type_195


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
    cat_197 = rand_strided(
        (4*s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    return (cat_197,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_clamp_stack_sum_256',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
