import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(ascend_seq_tensor_concat_2, nonzero):
    eq_4079 = torch.ops.aten.eq.Scalar(ascend_seq_tensor_concat_2, 0)
    return eq_4079


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
    ascend_seq_tensor_concat_2 = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.float16,
    )
    nonzero = rand_strided(
        (1,),
        (1,),
        device=device,
        dtype=torch.int64,
    )
    return (ascend_seq_tensor_concat_2, nonzero), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'triton_poi_fused_eq_39',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
