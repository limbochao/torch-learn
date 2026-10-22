"""01_reduction_mask: extracted kernel ATen graph; see README.md for provenance and comparison points."""
import torch


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(
    arg8_1,
    mm_default_35,
    arg0_1,
    arg1_1,
    arg2_1,
    arg10_1,
    arg11_1,
):
    embedding = torch.ops.aten.embedding.default(arg1_1, arg0_1)
    iota = torch.ops.prims.iota.default(
        1024, start=0, step=1, dtype=torch.int32, device=arg0_1.device, requires_grad=False,
    )
    unsqueeze = torch.ops.aten.unsqueeze.default(iota, 0)
    embedding_1 = torch.ops.aten.embedding.default(arg2_1, unsqueeze)
    add = torch.ops.aten.add.Tensor(embedding, embedding_1)
    add_tensor_35 = torch.ops.aten.add.Tensor(arg8_1, mm_default_35)
    add_1_replacement = torch.ops.aten.add.Tensor(add_tensor_35, add)
    native_layer_norm_1 = torch.ops.aten.native_layer_norm.default(add_1_replacement, [768], arg10_1, arg11_1, 1e-05)
    return add_1_replacement,native_layer_norm_1[0]


DEFAULT_DYNAMIC = False
INPUT_NAMES = (
    'arg8_1',
    'mm_default_35',
    'arg0_1',
    'arg1_1',
    'arg2_1',
    'arg10_1',
    'arg11_1',
)


def make_inputs(device):
    specs = (
        ((768,), (1,), torch.float32),
        ((1024, 768), (768, 1), torch.float32),
        ((1, 1024), (1024, 1), torch.int64),
        ((50257, 768), (768, 1), torch.float32),
        ((1024, 768), (768, 1), torch.float32),
        ((768,), (1,), torch.float32),
        ((768,), (1,), torch.float32),
    )
    inputs = []
    for shape, stride, dtype in specs:
        value = torch.empty_strided(shape, stride, dtype=dtype, device=device)
        if dtype.is_floating_point:
            value.normal_()
        else:
            value.zero_()
        inputs.append(value)
    inputs[2].random_(0, 50257)
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
