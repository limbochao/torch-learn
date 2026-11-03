"""PR !48475: NFNet-derived training graph; run both shapes and backward with one compiled callable. Numerical success
alone does not prove that the fallback and regrouping path ran.

Standalone ATen graph with synthetic inputs, seed=0.
Default: eager; --compile: fresh Inductor codegen and numerical comparison.
NPU registration occurs only for --device npu[:index].
See ../../README.md for provenance; no repository files are runtime dependencies.
"""
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


def enable_index_debug():
    # Observe real caught failures; do not force exceptions or alter fusion.
    import logging
    from torch_npu._inductor.config import log

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[index-fallback] %(message)s"))
    prefixes = ("index codegen for node", "linear codegen for node", "linear benchmark_fused_nodes")
    handler.addFilter(lambda record: record.getMessage().startswith(prefixes))
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action="store_true", help="compile forward and execute its backward")
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--index-debug", action="store_true", help="show real caught index errors")
    args = parser.parse_args()
    if args.index_debug and (not args.device.startswith("npu") or not args.compile):
        parser.error("--index-debug requires --device npu[:index] --compile")
    if args.device.split(":", 1)[0] == "npu":
        import torch_npu
        print(f"torch_npu={torch_npu.__version__}", flush=True)
        if args.index_debug:
            enable_index_debug()
    torch.manual_seed(0)
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
            print(
                f"input shape={tuple(value.shape)} stride={value.stride()} "
                f"dtype={value.dtype} device={value.device} offset={value.storage_offset()}"
            )
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
