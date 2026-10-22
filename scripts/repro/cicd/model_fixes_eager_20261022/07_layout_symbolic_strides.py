"""symbolic strides: NFNet-derived dynamic training graph; run both shapes in one process."""
import argparse

import torch
import torch.nn.functional as F


def eager_forward(value, weight):
    value = F.conv2d(value, weight, stride=32)
    return value * value.mean((-2, -1), keepdim=True).sigmoid()


def make_inputs(device, batch, height):
    value = torch.randn(batch, 16, height, height, device=device, requires_grad=True)
    weight = torch.randn(16, 16, 1, 1, device=device, requires_grad=True)
    return value, weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action="store_true", help="compile forward and execute its backward")
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    if torch.device(args.device).type == "npu":
        import torch_npu  # noqa: F401
    print(f"torch={torch.__version__} cuda={torch.version.cuda} device={args.device} dynamic={args.dynamic}")
    if torch.device(args.device).type == "cuda":
        import triton
        torch.backends.cudnn.allow_tf32 = False
        print(f"gpu={torch.cuda.get_device_name()} triton={triton.__version__}")
    compiled = None
    if args.compile:
        compiled = torch.compile(eager_forward, backend="inductor", fullgraph=True, dynamic=args.dynamic)
    for batch, height in ((2, 224), (3, 256)):
        inputs = make_inputs(args.device, batch, height)
        for value in inputs:
            print(f"input shape={tuple(value.shape)} stride={value.stride()} dtype={value.dtype}")
        expected = eager_forward(*inputs)
        gradient = torch.randn_like(expected)
        expected_grads = torch.autograd.grad(expected, inputs, gradient)
        for value in (expected, *expected_grads):
            assert torch.isfinite(value).all().item(), "non-finite eager result"
        print(f"PASS eager_forward_backward batch={batch} height={height}", flush=True)
        if compiled is not None:
            actual = compiled(*inputs)
            actual_grads = torch.autograd.grad(actual, inputs, gradient)
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
            for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-3, atol=2e-3)
            print(f"PASS compiled_forward_backward batch={batch} height={height}", flush=True)


if __name__ == "__main__":
    main()
