import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg13_1, arg1555_1, mm_default_62, neg, where_265):
    arg4_1 = arg13_1.shape[0]
    full_default_13 = torch.ops.aten.full.default([arg4_1, 1], 0, dtype=torch.int64, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    full_default_14 = torch.ops.aten.full.default([arg4_1, 1], 1, dtype=torch.int64, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    le = torch.ops.aten.le.Scalar(arg13_1, 0)
    where_8 = torch.ops.aten.where.self(le, full_default_13, full_default_14)
    full_default_221 = torch.ops.aten.full.default([arg4_1, 1], 0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    eq_8102 = torch.ops.aten.eq.Scalar(where_8, 1)
    add_tensor_62 = torch.ops.aten.add.Tensor(arg1555_1, mm_default_62)
    sub_4770 = torch.ops.aten.sub.Tensor(0.0, add_tensor_62)
    exp = torch.ops.aten.exp.default(sub_4770)
    add_14555 = torch.ops.aten.add.Tensor(exp, 1)
    log = torch.ops.aten.log.default(add_14555)
    neg = torch.ops.aten.neg.default(log)
    where_265 = torch.ops.aten.where.self(eq_8102, neg, full_default_221)
    squeeze_271 = torch.ops.aten.squeeze.dim(where_265, 1)
    full_default_306 = torch.ops.aten.full.default([arg4_1], -11.0, dtype=torch.float16, layout=torch.strided, device=torch.device('npu:0'), pin_memory=False)
    squeeze_272 = torch.ops.aten.squeeze.dim(neg, 1)
    isinf = torch.ops.aten.isinf.default(squeeze_272)
    eq_8463 = torch.ops.aten.eq.Tensor(squeeze_272, squeeze_271)
    bitwise_and = torch.ops.aten.bitwise_and.Tensor(isinf, eq_8463)
    le_713 = torch.ops.aten.le.Tensor(squeeze_272, full_default_306)
    bitwise_and_1 = torch.ops.aten.bitwise_and.Tensor(eq_8463, le_713)
    return where_265,neg,bitwise_and,bitwise_and_1


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
    arg13_1 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.int64,
    )
    arg1555_1 = rand_strided(
        (1,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    mm_default_62 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    neg = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    where_265 = rand_strided(
        (s0, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    return (arg13_1, arg1555_1, mm_default_62, neg, where_265), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[2]': (0,), 'args[3]': (0,), 'args[4]': (0,)}


CASE = {
    'name': 'triton_poi_fused_add_addmm_bitwise_and_eq_exp__253',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
