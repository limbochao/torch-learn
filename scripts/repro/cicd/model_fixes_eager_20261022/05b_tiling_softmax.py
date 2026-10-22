"""05b_tiling_softmax: extracted kernel ATen graph; see README.md for provenance and comparison points."""
import torch


# Eager reference reconstructed from Inductor Graph fragment metadata.
def eager_forward(
    bmm,
    any_dims_11,
):
    primals_1 = torch.ops.aten.sym_size.int(any_dims_11, 0)
    primals_2 = torch.ops.aten.sym_size.int(bmm, 1)
    view_13 = torch.ops.aten.reshape.default(bmm, [primals_1, 64, primals_2, primals_2])
    div_tensor_22 = torch.ops.aten.div.Tensor(view_13, 8.0)
    amax_default_22 = torch.ops.aten.amax.default(view_13, [-1], True)
    sub_tensor_22 = torch.ops.aten.sub.Tensor(view_13, amax_default_22)
    div_tensor_23 = torch.ops.aten.div.Tensor(sub_tensor_22, 8.0)
    amax_default_23 = torch.ops.aten.amax.default(div_tensor_22, [-1], True)
    sub_tensor_23 = torch.ops.aten.sub.Tensor(div_tensor_22, amax_default_23)
    logical_not_default_23 = torch.ops.aten.logical_not.default(any_dims_11)
    where_self_11 = torch.ops.aten.where.self(logical_not_default_23, div_tensor_23, sub_tensor_23)
    exp = torch.ops.aten.exp.default(where_self_11)
    sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1 = torch.ops.aten.div.Tensor(exp, sum_1)
    return div_1


DEFAULT_DYNAMIC = True
INPUT_NAMES = (
    'bmm',
    'any_dims_11',
)


def make_inputs(device, batch=4, sequence=512):
    s87 = batch
    s40 = sequence
    specs = (
        ((64*s87, s40, s40), (s40**2, s40, 1), torch.float32),
        ((s87, 64, s40, 1), (64*s40, s40, 1, 1), torch.bool),
    )
    inputs = []
    for shape, stride, dtype in specs:
        value = torch.empty_strided(shape, stride, dtype=dtype, device=device)
        if dtype.is_floating_point:
            value.normal_()
        else:
            value.zero_()
        inputs.append(value)
    inputs[1].copy_(torch.randint(0, 2, inputs[1].shape, device=device).bool())
    return tuple(inputs)


def main():
    import argparse
    from torch.utils import _pytree

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action="store_true", help="compile the extracted ATen graph with Inductor")
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=DEFAULT_DYNAMIC)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--sequence", type=int, default=512)
    args = parser.parse_args()
    torch.manual_seed(0)
    if torch.device(args.device).type == "npu":
        import torch_npu  # noqa: F401
    print(f"torch={torch.__version__} cuda={torch.version.cuda} device={args.device} dynamic={args.dynamic}")
    if torch.device(args.device).type == "cuda":
        import triton
        print(f"gpu={torch.cuda.get_device_name()} triton={triton.__version__}")
    inputs = make_inputs(args.device, args.batch, args.sequence)
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
