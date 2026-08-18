import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from the Triton store pattern.
def eager_forward(size_token):
    s0 = size_token.shape[0]
    return torch.arange(
        1,
        s0 + 1,
        dtype=torch.int64,
        device=size_token.device,
    ).repeat_interleave(2)


SAMPLE_BINDINGS = [
    {'s0': 400},
    {'s0': 200},
    {'s0': 256},
    {'s0': 512},
]


COMPILE_BINDINGS = [
    {'s0': 400},
]


def make_inputs(binding, device):
    s0 = binding['s0']
    size_token = rand_strided(
        (s0,),
        (1,),
        device=device,
        dtype=torch.int64,
    )
    return (size_token,), {}


DYNAMIC_DIMS = {'args[0]': (0,)}


CASE = {
    'name': 'pointwise_triton_poi_fused_add_arange_ascend_seq_tensor__226',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
