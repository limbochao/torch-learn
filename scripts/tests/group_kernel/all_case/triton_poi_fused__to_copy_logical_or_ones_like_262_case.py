import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(
    arg3_1,
    arg2_1,
    convert_element_type_2,
    arg1_1,
    convert_element_type_1,
    arg9_1,
    arg8_1,
    arg10_1,
):
    arg0_1 = arg3_1.shape[0]
    convert_element_type_2 = torch.ops.prims.convert_element_type.default(
        arg3_1, torch.bool
    )
    convert_element_type = torch.ops.prims.convert_element_type.default(
        arg1_1, torch.bool
    )
    convert_element_type_1 = torch.ops.prims.convert_element_type.default(
        arg2_1, torch.bool
    )
    logical_or = torch.ops.aten.logical_or.default(
        convert_element_type, convert_element_type_1
    )
    logical_or_1 = torch.ops.aten.logical_or.default(
        convert_element_type_2, logical_or
    )
    full_default_11 = torch.ops.aten.full.default(
        [arg0_1, 1],
        1,
        dtype=torch.float16,
        layout=torch.strided,
        device=torch.device(device),
        pin_memory=False,
    )
    where_3 = torch.ops.aten.where.self(logical_or_1, arg9_1, full_default_11)
    where_1 = torch.ops.aten.where.self(logical_or_1, arg8_1, full_default_11)
    full_default_10 = torch.ops.aten.full.default(
        [arg0_1, 1],
        0,
        dtype=torch.float16,
        layout=torch.strided,
        device=torch.device(device),
        pin_memory=False,
    )
    where_2 = torch.ops.aten.where.self(logical_or_1, arg8_1, full_default_10)
    where_4 = torch.ops.aten.where.self(
        convert_element_type_2, arg10_1, full_default_11
    )
    return (
        convert_element_type_2,
        convert_element_type_1,
        where_3,
        where_1,
        where_2,
        where_4,
    )


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
    arg3_1 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.int64,
    )
    arg2_1 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.int64,
    )
    convert_element_type_2 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.bool,
    )
    arg1_1 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.int64,
    )
    convert_element_type_1 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.bool,
    )
    arg9_1 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    arg8_1 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    arg10_1 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    return (
        (
            arg3_1,
            arg2_1,
            convert_element_type_2,
            arg1_1,
            convert_element_type_1,
            arg9_1,
            arg8_1,
            arg10_1,
        ),
        {},
    )


DYNAMIC_DIMS = {
    'args[0]': (0,),
    'args[1]': (0,),
    'args[2]': (0,),
    'args[3]': (0,),
    'args[4]': (0,),
    'args[5]': (0,),
    'args[6]': (0,),
    'args[7]': (0,),
}


CASE = {
    'name': 'triton_poi_fused__to_copy_logical_or_ones_like_262',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
