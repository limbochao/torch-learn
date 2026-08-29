import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(index_22):
    slice_695 = torch.ops.aten.slice.Tensor(index_22, 0, 1, 9223372036854775807)
    slice_696 = torch.ops.aten.slice.Tensor(index_22, 0, 0, -1)
    sub_2651 = torch.ops.aten.sub.Tensor(slice_695, slice_696)
    max_2 = torch.ops.aten.max.default(sub_2651)
    return max_2


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
    index_22 = rand_strided(
        (s0 + 1,),
        (1,),
        device=device,
        dtype=torch.int64,
    )
    return (index_22,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_red_fused_max_slice_sub_46',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
