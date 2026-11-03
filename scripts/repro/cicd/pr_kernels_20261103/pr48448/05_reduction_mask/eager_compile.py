"""PR !48448: unbacked split length followed by persistent sum with a missing tail mask.

Source: test_unbacked_symints.py::test_split_with_sizes; original sizes=(7,16,9).
Keep the CPU sizes tensor, tolist(), and scalar sum in the original test graph.
The paired problem_dsl.py is the actual failing pre-PR capture from 2026-10-29.
Inputs are synthetic, seed=0; this file has no repository runtime dependencies.
"""

import argparse

import torch


def eager_forward(value, sizes):
    lengths = sizes.tolist()
    parts = torch.split(value, lengths)
    total = lengths[0] + lengths[1] + lengths[2]
    return parts[0].sum(), total


def make_inputs(device):
    return torch.randn(32, dtype=torch.float32, device=device), torch.tensor((7, 16, 9), dtype=torch.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--dynamic', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.device.split(':', 1)[0] == 'npu':
        import torch_npu

        print(f'torch_npu={torch_npu.__version__}', flush=True)
    device = torch.device(args.device)
    torch.manual_seed(0)
    print(f'torch={torch.__version__} device={device} dynamic={args.dynamic} inputs=synthetic seed=0')
    if device.type in ('cuda', 'npu'):
        import triton

        print(f'hardware={getattr(torch, device.type).get_device_name(device)} triton={triton.__version__}')
    inputs = make_inputs(device)
    for name, value in zip(('value', 'sizes'), inputs):
        print(
            f'input={name} shape={tuple(value.shape)} stride={value.stride()} dtype={value.dtype} '
            f'device={value.device} offset={value.storage_offset()}'
        )
    expected = eager_forward(*inputs)
    assert torch.isfinite(expected[0]).item()
    assert expected[1] == 32
    print(f'PASS eager sum={expected[0].item()} total={expected[1]} sizes={inputs[1].tolist()}', flush=True)
    if args.compile:
        # Preserve data-dependent (unbacked) lengths instead of specializing them as Python constants.
        with torch._dynamo.config.patch(capture_scalar_outputs=True):
            compiled = torch.compile(eager_forward, backend='inductor', fullgraph=True, dynamic=args.dynamic)
            actual = compiled(*inputs)
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
        print('PASS compiled_vs_eager rtol=0.002 atol=0.002', flush=True)


if __name__ == '__main__':
    main()
