"""05a_tiling_pooling: extracted kernel ATen graph; see README.md for provenance and comparison points."""
import torch


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(
    mul_tensor_12,
):
    avg_pool2d = torch.ops.aten.avg_pool2d.default(mul_tensor_12, [2, 2], [2, 2], [0, 0], True, False)
    return avg_pool2d


DEFAULT_DYNAMIC = True
INPUT_NAMES = (
    'mul_tensor_12',
)


def make_inputs(device, batch=128, height=288):
    s21 = batch
    s65 = height
    specs = (
        (
            (s21, 256, (s65 - 1) // 4 + 1, (s65 - 1) // 4 + 1),
            (256 * ((s65 - 1) // 4 + 1) ** 2, ((s65 - 1) // 4 + 1) ** 2, (s65 - 1) // 4 + 1, 1),
            torch.float32,
        ),
    )
    inputs = []
    for shape, stride, dtype in specs:
        value = torch.empty_strided(shape, stride, dtype=dtype, device=device)
        if dtype.is_floating_point:
            value.normal_()
        else:
            value.zero_()
        inputs.append(value)
    return tuple(inputs)


def main():
    import argparse
    from torch.utils import _pytree

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action="store_true", help="compile the extracted ATen graph with Inductor")
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=DEFAULT_DYNAMIC)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--height", type=int, default=288)
    args = parser.parse_args()
    torch.manual_seed(0)
    if torch.device(args.device).type == "npu":
        import torch_npu  # noqa: F401
    print(f"torch={torch.__version__} cuda={torch.version.cuda} device={args.device} dynamic={args.dynamic}")
    if torch.device(args.device).type == "cuda":
        import triton
        print(f"gpu={torch.cuda.get_device_name()} triton={triton.__version__}")
    inputs = make_inputs(args.device, args.batch, args.height)
    for name, value in zip(INPUT_NAMES, inputs):
        print(f"input={name} shape={tuple(value.shape)} stride={value.stride()} dtype={value.dtype}")
    expected = eager_forward(*inputs)
    leaves = _pytree.tree_leaves(expected)
    for value in leaves:
        assert torch.isfinite(value).all().item(), "non-finite eager result"
    print(f"PASS eager outputs={[tuple(value.shape) for value in leaves]}", flush=True)
    if args.compile:
        compiled = torch.compile(eager_forward, backend="inductor", fullgraph=True, dynamic=args.dynamic)
        actual = compiled(*inputs)
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
        print("PASS compiled_vs_eager", flush=True)


if __name__ == "__main__":
    main()
