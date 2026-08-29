import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg18_1):
    arg4_1 = view_723.shape[0]
    view_18 = view_723
    full_default_46 = torch.ops.aten.full.default([arg4_1, 16], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    slice_51 = torch.ops.aten.slice.Tensor(arg18_1, 1, 17868, 17884)
    where_31 = torch.ops.aten.where.self(view_18, full_default_46, slice_51)
    slice_52 = torch.ops.aten.slice.Tensor(arg18_1, 1, 31506, 31522)
    where_32 = torch.ops.aten.where.self(view_18, full_default_46, slice_52)
    slice_53 = torch.ops.aten.slice.Tensor(arg18_1, 1, 28338, 28354)
    where_33 = torch.ops.aten.where.self(view_18, full_default_46, slice_53)
    slice_54 = torch.ops.aten.slice.Tensor(arg18_1, 1, 29184, 29200)
    where_34 = torch.ops.aten.where.self(view_18, full_default_46, slice_54)
    slice_55 = torch.ops.aten.slice.Tensor(arg18_1, 1, 30014, 30030)
    where_35 = torch.ops.aten.where.self(view_18, full_default_46, slice_55)
    slice_56 = torch.ops.aten.slice.Tensor(arg18_1, 1, 30772, 30788)
    where_36 = torch.ops.aten.where.self(view_18, full_default_46, slice_56)
    return where_31,where_32,where_33,where_34,where_35,where_36


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
    'name': 'triton_poi_fused_slice_where_zeros_like_99',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
