"""03_negative_index: extracted kernel ATen graph; see README.md for provenance and comparison points."""
import torch


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(
    arg1_1,
    arg0_1,
):
    index = torch.ops.aten.index.Tensor(arg0_1, [arg1_1])
    add = torch.ops.aten.add.Tensor(index, 1)
    return add


DEFAULT_DYNAMIC = False
INPUT_NAMES = (
    'arg1_1',
    'arg0_1',
)


def make_inputs(device):
    specs = (
        ((6,), (1,), torch.int64),
        ((8, 16), (16, 1), torch.float32),
    )
    inputs = []
    for shape, stride, dtype in specs:
        value = torch.empty_strided(shape, stride, dtype=dtype, device=device)
        if dtype.is_floating_point:
            value.normal_()
        else:
            value.zero_()
        inputs.append(value)
    inputs[0].copy_(torch.tensor([-8, -3, -1, 0, 4, 7], device=device))
    return tuple(inputs)


def main():
    import argparse
    from torch.utils import _pytree

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action="store_true", help="compile the extracted ATen graph with Inductor")
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=DEFAULT_DYNAMIC)
    args = parser.parse_args()
    torch.manual_seed(0)
    if torch.device(args.device).type == "npu":
        import torch_npu  # noqa: F401
    print(f"torch={torch.__version__} cuda={torch.version.cuda} device={args.device} dynamic={args.dynamic}")
    if torch.device(args.device).type == "cuda":
        import triton
        print(f"gpu={torch.cuda.get_device_name()} triton={triton.__version__}")
    inputs = make_inputs(args.device)
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
