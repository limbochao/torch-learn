import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg18_1):
    arg4_1 = view_723.shape[0]
    view_15 = view_723
    full_default_95 = torch.ops.aten.full.default([arg4_1, 8], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    slice_32 = torch.ops.aten.slice.Tensor(arg18_1, 1, 27573, 27581)
    where_20 = torch.ops.aten.where.self(view_15, full_default_95, slice_32)
    slice_33 = torch.ops.aten.slice.Tensor(arg18_1, 1, 26772, 26780)
    where_21 = torch.ops.aten.where.self(view_15, full_default_95, slice_33)
    slice_34 = torch.ops.aten.slice.Tensor(arg18_1, 1, 23504, 23512)
    where_22 = torch.ops.aten.where.self(view_15, full_default_95, slice_34)
    slice_35 = torch.ops.aten.slice.Tensor(arg18_1, 1, 23313, 23321)
    where_23 = torch.ops.aten.where.self(view_15, full_default_95, slice_35)
    slice_36 = torch.ops.aten.slice.Tensor(arg18_1, 1, 23369, 23377)
    where_24 = torch.ops.aten.where.self(view_15, full_default_95, slice_36)
    slice_37 = torch.ops.aten.slice.Tensor(arg18_1, 1, 23425, 23433)
    where_25 = torch.ops.aten.where.self(view_15, full_default_95, slice_37)
    slice_38 = torch.ops.aten.slice.Tensor(arg18_1, 1, 23465, 23473)
    where_26 = torch.ops.aten.where.self(view_15, full_default_95, slice_38)
    slice_39 = torch.ops.aten.slice.Tensor(arg18_1, 1, 23567, 23575)
    where_27 = torch.ops.aten.where.self(view_15, full_default_95, slice_39)
    slice_40 = torch.ops.aten.slice.Tensor(arg18_1, 1, 23596, 23604)
    where_28 = torch.ops.aten.where.self(view_15, full_default_95, slice_40)
    slice_41 = torch.ops.aten.slice.Tensor(arg18_1, 1, 24442, 24450)
    where_29 = torch.ops.aten.where.self(view_15, full_default_95, slice_41)
    slice_42 = torch.ops.aten.slice.Tensor(arg18_1, 1, 25272, 25280)
    where_30 = torch.ops.aten.where.self(view_15, full_default_95, slice_42)
    return where_20,where_21,where_22,where_23,where_24,where_25,where_26,where_27,where_28,where_29,where_30


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
    arg18_1 = rand_strided(
        (s0, 69876),
        (69876, 1),
        device=device,
        dtype=torch.float16,
    )
    return (view_723, arg18_1), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (0,)}


CASE = {
    'name': 'pointwise_triton_poi_fused_slice_where_zeros_like_180',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
