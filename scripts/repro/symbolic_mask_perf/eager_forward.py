"""Eager reference for ``triton_poi_fused_add_gelu_split_with_sizes_229``."""

import os

import torch


BATCH = 32
INPUT_FEATURES = 1160
OUTPUT_FEATURES = 200


def eager_forward(bmm_26, arg1288_1):
    """Reconstruct the eager computation represented by ``new_kernel.py``."""
    add_14751 = torch.ops.aten.add.Tensor(bmm_26, arg1288_1)
    getitem_470 = torch.ops.aten.split_with_sizes.default(add_14751, [200, 960], -1)[0]
    mul_9678 = torch.ops.aten.mul.Tensor(getitem_470, getitem_470)
    mul_9679 = torch.ops.aten.mul.Tensor(mul_9678, getitem_470)
    mul_9680 = torch.ops.aten.mul.Tensor(mul_9679, 0.044715)
    add_14764 = torch.ops.aten.add.Tensor(getitem_470, mul_9680)
    mul_9681 = torch.ops.aten.mul.Tensor(add_14764, 1.5957691216057308)
    sigmoid_18 = torch.ops.aten.sigmoid.default(mul_9681)
    mul_9682 = torch.ops.aten.mul.Tensor(getitem_470, sigmoid_18)
    return mul_9682


def make_inputs(s0=256, device='npu:0'):
    """Construct inputs with the shape and stride recorded in ``new_kernel.py``."""
    bmm_26 = torch.randn(
        (BATCH, s0, INPUT_FEATURES),
        device=device,
        dtype=torch.float16,
    )
    arg1288_1 = torch.randn(
        (BATCH, 1, INPUT_FEATURES),
        device=device,
        dtype=torch.float16,
    )
    return bmm_26, arg1288_1


def run_compiled_eager(bmm_26, arg1288_1):
    """Mark the symbolic sequence dimension and invoke the compiled reference."""
    torch._dynamo.mark_dynamic(bmm_26, 1)
    compiled_eager_forward = torch.compile(eager_forward, dynamic=None)
    return compiled_eager_forward(bmm_26, arg1288_1)


def main():
    s0 = int(os.getenv('S0', '256'))
    device = os.getenv('DEVICE', 'npu:0')
    torch.manual_seed(0)
    bmm_26, arg1288_1 = make_inputs(s0, device)
    eager_result = eager_forward(bmm_26, arg1288_1)
    compiled_result = run_compiled_eager(bmm_26, arg1288_1)
    assert tuple(eager_result.shape) == (BATCH, s0, OUTPUT_FEATURES)
    assert tuple(compiled_result.shape) == (BATCH, s0, OUTPUT_FEATURES)
    print(f's0={s0}, device={device}')
    print(
        f'eager_result: shape={tuple(eager_result.shape)}, '
        f'stride={eager_result.stride()}, dtype={eager_result.dtype}'
    )
    print(
        f'compiled_result: shape={tuple(compiled_result.shape)}, '
        f'stride={compiled_result.stride()}, dtype={compiled_result.dtype}'
    )


if __name__ == '__main__':
    main()
