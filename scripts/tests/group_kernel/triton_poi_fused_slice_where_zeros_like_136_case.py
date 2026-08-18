import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg18_1):
    arg4_1 = view_723.shape[0]
    view_143 = view_723
    view_145 = view_723
    full_default_125 = torch.ops.aten.full.default([arg4_1, 32], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    slice_438 = torch.ops.aten.slice.Tensor(arg18_1, 1, 23070, 23102)
    where_164 = torch.ops.aten.where.self(view_143, full_default_125, slice_438)
    slice_440 = torch.ops.aten.slice.Tensor(arg18_1, 1, 27307, 27339)
    where_165 = torch.ops.aten.where.self(view_145, full_default_125, slice_440)
    slice_441 = torch.ops.aten.slice.Tensor(arg18_1, 1, 24092, 24124)
    where_166 = torch.ops.aten.where.self(view_145, full_default_125, slice_441)
    slice_442 = torch.ops.aten.slice.Tensor(arg18_1, 1, 24938, 24970)
    where_167 = torch.ops.aten.where.self(view_145, full_default_125, slice_442)
    slice_443 = torch.ops.aten.slice.Tensor(arg18_1, 1, 25768, 25800)
    where_168 = torch.ops.aten.where.self(view_145, full_default_125, slice_443)
    slice_444 = torch.ops.aten.slice.Tensor(arg18_1, 1, 26518, 26550)
    where_169 = torch.ops.aten.where.self(view_145, full_default_125, slice_444)
    return where_164,where_165,where_166,where_167,where_168,where_169


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
    'name': 'triton_poi_fused_slice_where_zeros_like_136',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
