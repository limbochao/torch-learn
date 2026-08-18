import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(bmm_42, arg140_1, mm_default_18, arg1482_1, bmm_43, arg1486_1):
    add_tensor_18 = torch.ops.aten.add.Tensor(arg140_1, mm_default_18)
    squeeze_222 = torch.ops.aten.squeeze.dim(bmm_42, 2)
    slice_772 = torch.ops.aten.slice.Tensor(add_tensor_18, 1, 256, 9223372036854775807)
    add_13727 = torch.ops.aten.add.Tensor(slice_772, arg1482_1)
    add_13742 = torch.ops.aten.add.Tensor(squeeze_222, add_13727)
    squeeze_223 = torch.ops.aten.squeeze.dim(bmm_43, 2)
    slice_774 = torch.ops.aten.slice.Tensor(add_tensor_18, 1, 256, 9223372036854775807)
    add_13769 = torch.ops.aten.add.Tensor(slice_774, arg1486_1)
    add_13784 = torch.ops.aten.add.Tensor(squeeze_223, add_13769)
    return add_13742,add_13784


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
    bmm_42 = rand_strided(
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
    arg1482_1 = rand_strided(
        (1,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    bmm_43 = rand_strided(
        (s0, 1, 1),
        (1, 1, 1),
        device=device,
        dtype=torch.float16,
    )
    arg1486_1 = rand_strided(
        (1,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    return (bmm_42, arg140_1, mm_default_18, arg1482_1, bmm_43, arg1486_1), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[2]': (0,), 'args[4]': (0,)}


CASE = {
    'name': 'triton_poi_fused_add_addmm_slice_squeeze_271',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
