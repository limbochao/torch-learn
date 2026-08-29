import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(bmm_40, arg140_1, mm_default_18, arg1474_1):
    add_tensor_18 = torch.ops.aten.add.Tensor(arg140_1, mm_default_18)
    squeeze_220 = torch.ops.aten.squeeze.dim(bmm_40, 2)
    slice_768 = torch.ops.aten.slice.Tensor(add_tensor_18, 1, 256, 9223372036854775807)
    add_13643 = torch.ops.aten.add.Tensor(slice_768, arg1474_1)
    add_13658 = torch.ops.aten.add.Tensor(squeeze_220, add_13643)
    return add_13658


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
    bmm_40 = rand_strided(
        (s0, 1, 1),
        (1, 1, 1),
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
    arg1474_1 = rand_strided(
        (1,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    return (bmm_40, arg140_1, mm_default_18, arg1474_1), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[2]': (0,)}


CASE = {
    'name': 'triton_poi_fused_add_addmm_slice_squeeze_264',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
