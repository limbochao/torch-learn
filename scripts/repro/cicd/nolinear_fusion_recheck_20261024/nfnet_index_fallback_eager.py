"""NFNet index-codegen investigation: original backward subgraph or the recorded training repro.

Single-file runner; only --device npu imports torch_npu. Inputs are seeded synthetic values.
The original case executes an already decomposed backward graph, without another backward.
The training case preserves conv stride=32 and runs forward/backward for both recorded shapes.
"""

import argparse
import logging

import torch
import torch.nn.functional as F


def eager_forward(
    getitem_114,
    getitem_132,
    add_1604,
    convolution_71,
    convolution_73,
    convert_element_type_default_171,
    convert_element_type_default_169,
    convert_element_type_default_167,
):
    # Original NFNet backward FX nodes, corresponding to scheduler op59/op60/op61.
    mul_tensor_100 = torch.ops.aten.mul.Tensor(getitem_132, convert_element_type_default_171)
    sigmoid_16 = torch.ops.aten.sigmoid.default(add_1604)
    mul_2032 = torch.ops.aten.mul.Tensor(mul_tensor_100, sigmoid_16)
    sub_866 = torch.ops.aten.sub.Tensor(1, sigmoid_16)
    mul_2033 = torch.ops.aten.mul.Tensor(add_1604, sub_866)
    add_1788 = torch.ops.aten.add.Tensor(mul_2033, 1)
    mul_2034 = torch.ops.aten.mul.Tensor(mul_2032, add_1788)
    add_1789 = torch.ops.aten.add.Tensor(getitem_114, mul_2034)
    mul_tensor_101 = torch.ops.aten.mul.Tensor(add_1789, convert_element_type_default_169)
    mul_tensor_102 = torch.ops.aten.mul.Tensor(mul_tensor_101, convert_element_type_default_167)
    mul_2037 = torch.ops.aten.mul.Tensor(mul_tensor_102, convolution_71)
    sigmoid_10 = torch.ops.aten.sigmoid.default(convolution_73)
    sum_20 = torch.ops.aten.sum.dim_IntList(mul_2037, [2, 3], True)
    sub_867 = torch.ops.aten.sub.Tensor(1, sigmoid_10)
    mul_2039 = torch.ops.aten.mul.Tensor(sigmoid_10, sub_867)
    mul_2040 = torch.ops.aten.mul.Tensor(sum_20, mul_2039)
    # buf59 also has an external user (op71); buf60 is internal; buf61 escapes.
    return mul_tensor_102, mul_2040


def training_forward(value, weight):
    # Exact forward of the small graph recorded in the 2026-10-24 parent control.
    value = F.conv2d(value, weight, stride=32)
    return value * value.mean((-2, -1), keepdim=True).sigmoid()


def make_original_inputs(device, batch, height):
    spatial = (height - 1) // 32 + 1
    shape = (batch, 1536, spatial, spatial)
    spatial_inputs = tuple(torch.randn(shape, device=device) for _ in range(4))
    gate = torch.randn(batch, 1536, 1, 1, device=device)
    # Original metadata has three independent CPU float32 scalar tensors.
    # Their captured values are unavailable; these nonzero constants are synthetic.
    scales = tuple(torch.tensor(value, dtype=torch.float32) for value in (1.25, 0.2, 0.75))
    return (*spatial_inputs, gate, *scales)


def make_training_inputs(device, batch, height):
    value = torch.randn(batch, 16, height, height, device=device, requires_grad=True)
    weight = torch.randn(16, 16, 1, 1, device=device, requires_grad=True)
    return value, weight


def enable_index_debug():
    # Existing combined-scheduler logs include exc_info=True BEFORE the fallback.
    # Enable only those records; leave codegen, fusion, and exception behavior intact.
    from torch_npu._inductor.config import log

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("[index-fallback] %(message)s"))
    handler.addFilter(
        lambda record: record.getMessage().startswith(
            ("index codegen for node", "linear benchmark_fused_nodes")
        )
    )
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)


def check_finite(values):
    for value in values:
        assert torch.isfinite(value).all().item(), "non-finite result"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("original", "training"), default="original")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--index-debug", action="store_true", help="show the caught NPU index exception")
    args = parser.parse_args()
    is_npu = args.device.split(":", 1)[0] == "npu"
    if args.index_debug and (not is_npu or not args.compile):
        parser.error("--index-debug requires --device npu[:index] --compile")
    if is_npu:
        import torch_npu

        print(f"torch_npu={torch_npu.__version__}", flush=True)
        if args.index_debug:
            enable_index_debug()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = False
        print(f"gpu={torch.cuda.get_device_name(device)}", flush=True)
    if device.type in ("cuda", "npu"):
        import triton

        print(f"triton={triton.__version__}", flush=True)
    torch.manual_seed(0)
    print(
        f"torch={torch.__version__} device={device} case={args.case} "
        f"dynamic={args.dynamic} compile={args.compile} inputs=synthetic seed=0",
        flush=True,
    )
    forward = eager_forward if args.case == "original" else training_forward
    make_inputs = make_original_inputs if args.case == "original" else make_training_inputs
    # One callable reused across both sizes, including the training backward.
    compiled = None
    if args.compile:
        compiled = torch.compile(forward, backend="inductor", fullgraph=True, dynamic=args.dynamic)
    shapes = ((128, 224), (3, 256)) if args.case == "original" else ((2, 224), (3, 256))
    for batch, height in shapes:
        print(f"BEGIN batch={batch} height={height}", flush=True)
        inputs = make_inputs(device, batch, height)
        for index, value in enumerate(inputs):
            print(
                f"input[{index}] shape={tuple(value.shape)} stride={value.stride()} "
                f"dtype={value.dtype} device={value.device} offset={value.storage_offset()}",
                flush=True,
            )
        if args.compile and args.case == "original":
            # Original channel count is static even though batch/spatial sizes vary.
            for value in inputs[:5]:
                torch._dynamo.mark_static(value, 1)
        expected = forward(*inputs)
        if args.case == "training":
            gradient = torch.randn_like(expected)
            expected_grads = torch.autograd.grad(expected, inputs, gradient)
            check_finite((expected, *expected_grads))
        else:
            check_finite(expected)
        print("PASS eager" + (" forward/backward" if args.case == "training" else " backward subgraph"), flush=True)
        if compiled is None:
            continue
        print("BEGIN compiled forward", flush=True)
        actual = compiled(*inputs)
        if args.case == "training":
            print("BEGIN compiled backward", flush=True)
            actual_grads = torch.autograd.grad(actual, inputs, gradient)
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
            for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-3, atol=2e-3)
        else:
            torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
        print(f"PASS compiled/eager batch={batch} height={height}", flush=True)
    print("Numerical checks complete; inspect index-fallback logs to establish path coverage.", flush=True)


if __name__ == "__main__":
    main()
