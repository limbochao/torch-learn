import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(view_723, arg18_1):
    arg4_1 = view_723.shape[0]
    view_40 = view_723
    view_42 = view_723
    full_default_46 = torch.ops.aten.full.default([arg4_1, 16], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    slice_187 = torch.ops.aten.slice.Tensor(arg18_1, 1, 46397, 46413)
    where_68 = torch.ops.aten.where.self(view_40, full_default_46, slice_187)
    slice_188 = torch.ops.aten.slice.Tensor(arg18_1, 1, 46418, 46434)
    where_69 = torch.ops.aten.where.self(view_40, full_default_46, slice_188)
    slice_189 = torch.ops.aten.slice.Tensor(arg18_1, 1, 46439, 46455)
    where_70 = torch.ops.aten.where.self(view_40, full_default_46, slice_189)
    slice_190 = torch.ops.aten.slice.Tensor(arg18_1, 1, 46376, 46392)
    where_71 = torch.ops.aten.where.self(view_40, full_default_46, slice_190)
    slice_191 = torch.ops.aten.slice.Tensor(arg18_1, 1, 46481, 46497)
    where_72 = torch.ops.aten.where.self(view_40, full_default_46, slice_191)
    slice_192 = torch.ops.aten.slice.Tensor(arg18_1, 1, 46502, 46518)
    where_73 = torch.ops.aten.where.self(view_40, full_default_46, slice_192)
    slice_193 = torch.ops.aten.slice.Tensor(arg18_1, 1, 46523, 46539)
    where_74 = torch.ops.aten.where.self(view_40, full_default_46, slice_193)
    slice_194 = torch.ops.aten.slice.Tensor(arg18_1, 1, 46460, 46476)
    where_75 = torch.ops.aten.where.self(view_40, full_default_46, slice_194)
    slice_198 = torch.ops.aten.slice.Tensor(arg18_1, 1, 46826, 46842)
    where_76 = torch.ops.aten.where.self(view_40, full_default_46, slice_198)
    slice_199 = torch.ops.aten.slice.Tensor(arg18_1, 1, 47035, 47051)
    where_77 = torch.ops.aten.where.self(view_40, full_default_46, slice_199)
    slice_201 = torch.ops.aten.slice.Tensor(arg18_1, 1, 47469, 47485)
    where_78 = torch.ops.aten.where.self(view_40, full_default_46, slice_201)
    slice_202 = torch.ops.aten.slice.Tensor(arg18_1, 1, 47490, 47506)
    where_79 = torch.ops.aten.where.self(view_40, full_default_46, slice_202)
    slice_204 = torch.ops.aten.slice.Tensor(arg18_1, 1, 47056, 47072)
    where_80 = torch.ops.aten.where.self(view_40, full_default_46, slice_204)
    slice_205 = torch.ops.aten.slice.Tensor(arg18_1, 1, 47014, 47030)
    where_81 = torch.ops.aten.where.self(view_40, full_default_46, slice_205)
    slice_206 = torch.ops.aten.slice.Tensor(arg18_1, 1, 47077, 47093)
    where_82 = torch.ops.aten.where.self(view_40, full_default_46, slice_206)
    slice_207 = torch.ops.aten.slice.Tensor(arg18_1, 1, 8732, 8748)
    where_83 = torch.ops.aten.where.self(view_42, full_default_46, slice_207)
    return where_68,where_69,where_70,where_71,where_72,where_73,where_74,where_75,where_76,where_77,where_78,where_79,where_80,where_81,where_82,where_83


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
    'name': 'pointwise_triton_poi_fused_slice_where_zeros_like_29',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
