import torch
from torch._dynamo.testing import rand_strided


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(repeat_interleave_6, arg403_1, arg405_1):
    slice_611 = torch.ops.aten.slice.Tensor(arg403_1, 2, 1, 66)
    slice_615 = torch.ops.aten.slice.Tensor(slice_611, 2, 64, 9223372036854775807)
    index_6 = torch.ops.aten.index.Tensor(slice_615, [repeat_interleave_6])
    slice_617 = torch.ops.aten.slice.Tensor(arg405_1, 2, 64, 9223372036854775807)
    mul_6912 = torch.ops.aten.mul.Tensor(index_6, slice_617)
    squeeze_186 = torch.ops.aten.squeeze.dims(mul_6912, [2])
    sum_181 = torch.ops.aten.sum.dim_IntList(squeeze_186, [1], True)
    return sum_181


SAMPLE_BINDINGS = [
    {'s0': 199, 's100': 7},
    {'s0': 200, 's100': 6},
    {'s0': 200, 's100': 7},
    {'s0': 200, 's100': 8},
    {'s0': 201, 's100': 7},
]


COMPILE_BINDINGS = [
    {'s0': 200, 's100': 7},
]


def make_inputs(binding, device):
    s0 = binding['s0']
    s100 = binding['s100']
    # This tensor indexes arg403_1's leading dimension, so random values must stay in range.
    repeat_interleave_6 = torch.randint(
        0,
        s100,
        (s0,),
        device=device,
        dtype=torch.int32,
    )
    arg403_1 = rand_strided(
        (s100, 4800, 66),
        (316800, 66, 1),
        device=device,
        dtype=torch.float16,
    )
    arg405_1 = rand_strided(
        (1, 4800, 65),
        (312000, 65, 1),
        device=device,
        dtype=torch.float16,
    )
    return (repeat_interleave_6, arg403_1, arg405_1), {}


DYNAMIC_DIMS = {'args[0]': (0,), 'args[1]': (0,)}


CASE = {
    'name': 'reduction_triton_red_fused_mul_repeat_interleave_slice_s_9',
    'forward': eager_forward,
    'make_inputs': make_inputs,
    'sample_bindings': SAMPLE_BINDINGS,
    'compile_bindings': COMPILE_BINDINGS,
    'dynamic_dims': DYNAMIC_DIMS,
}
