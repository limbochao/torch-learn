import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(arg1481_1, arg140_1, mm_default_18, arg1485_1):
    add_tensor_18 = torch.ops.aten.add.Tensor(arg140_1, mm_default_18)
    unsqueeze_67 = torch.ops.aten.unsqueeze.default(arg1481_1, 0)
    expand_100 = torch.ops.aten.expand.default(unsqueeze_67, [arg124_1, -1, -1])
    slice_771 = torch.ops.aten.slice.Tensor(add_tensor_18, 1, 0, 256)
    unsqueeze_66 = torch.ops.aten.unsqueeze.default(slice_771, -1)
    add_13722 = torch.ops.aten.add.Tensor(expand_100, unsqueeze_66)
    unsqueeze_70 = torch.ops.aten.unsqueeze.default(arg1485_1, 0)
    expand_103 = torch.ops.aten.expand.default(unsqueeze_70, [arg124_1, -1, -1])
    slice_773 = torch.ops.aten.slice.Tensor(add_tensor_18, 1, 0, 256)
    unsqueeze_69 = torch.ops.aten.unsqueeze.default(slice_773, -1)
    add_13764 = torch.ops.aten.add.Tensor(expand_103, unsqueeze_69)
    return add_13722,add_13764


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
    arg1481_1 = rand_strided(
        (256, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    arg140_1 = rand_strided(
        (257,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    mm_default_18 = rand_strided(
        (s0, 257),
        (257, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1485_1 = rand_strided(
        (256, 1),
        (1, 1),
        device=device,
        dtype=torch.float16,
    )
    return (arg1481_1, arg140_1, mm_default_18, arg1485_1), {}


DYNAMIC_DIMS = {'args[2]': (0,)}


CASE = {
    'name': 'triton_poi_fused_add_addmm_expand_slice_unsque_270',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
