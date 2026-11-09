"""Compare NPU ND x 2D matmul decompositions and retain generated-code evidence."""

import argparse
import functools
import importlib.util
import json
import math
import operator
from pathlib import Path
import statistics
import sys
import time

import torch
import torch_npu


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", default="generic,always,view")
    parser.add_argument("--layouts", default="contiguous,row_slice,col_slice,batch_slice,transpose,expand")
    parser.add_argument("--sizes", default="32,128,256,256")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--rhs-transpose", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--weight-grad", action="store_true")
    parser.add_argument("--capture-calls", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=9)
    return parser.parse_args()


def can_fold_without_copy(tensor):
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    if guard_or_false(tensor.numel() == 0):
        return True
    expected_stride = None
    for size, stride in zip(reversed(tensor.shape[:-1]), reversed(tensor.stride()[:-1])):
        if guard_or_false(size == 1):
            continue
        if expected_stride is not None and not guard_or_false(stride == expected_stride):
            return False
        expected_stride = stride * size
    return True


def always_fold(left, right):
    from torch._decomp.decompositions import matmul

    if left.device.type == "npu" and left.dim() >= 3 and right.dim() == 2:
        rows = functools.reduce(operator.mul, left.shape[:-1], 1)
        return torch.ops.aten._unsafe_view.default(
            torch.mm(left.reshape(rows, left.shape[-1]), right),
            list(left.shape[:-1]) + [right.shape[-1]],
        )
    return matmul(left, right)


def view_fold(left, right):
    from torch._decomp.decompositions import matmul

    if left.device.type == "npu" and left.dim() >= 3 and right.dim() == 2 and can_fold_without_copy(left):
        return always_fold(left, right)
    return matmul(left, right)


def make_inputs(layout, sizes, dtype, rhs_transpose, weight_grad):
    batch, rows, inner, columns = sizes

    def random(shape):
        return torch.randn(shape, dtype=dtype, device="npu") / math.sqrt(max(inner, 1))

    if layout == "contiguous":
        left = random((batch, rows, inner))
    elif layout == "row_slice":
        left = random((batch, rows * 2, inner))[:, ::2, :]
    elif layout == "col_slice":
        left = random((batch, rows, inner * 2))[:, :, ::2]
    elif layout == "batch_slice":
        left = random((batch * 2, rows, inner))[::2]
    elif layout == "eta_slice":
        left = random((batch, 100, inner))[:, :rows, :]
    elif layout == "transpose":
        left = random((batch, inner, rows)).transpose(-1, -2)
    elif layout == "column_major":
        left = random((inner, batch, rows)).permute(1, 2, 0)
    elif layout == "expand":
        left = random((1, rows, inner)).expand(batch, rows, inner)
    elif layout == "full_expand":
        left = random((1, 1, inner)).expand(batch, rows, inner)
    elif layout == "batch_permute":
        left = random((2, batch, rows, inner)).transpose(0, 1)
    elif layout == "channels_last":
        left = random((batch, rows, inner, 4)).permute(0, 3, 1, 2)
    elif layout == "singleton":
        left = random((batch, rows, inner)).unsqueeze(1)
    else:
        raise ValueError(layout)
    right = random((columns, inner)).t() if rhs_transpose else random((inner, columns))
    right.requires_grad_(weight_grad)
    return left, right


def matmul(left, right):
    return torch.matmul(left, right)


def graph_latency(function, left, right, capture_calls, repeats):
    for _ in range(5):
        function(left, right)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(capture_calls):
            output = function(left, right)
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    measurements = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        measurements.append(start.elapsed_time(end) * 1000 / capture_calls)
    return {"median_us": statistics.median(measurements), "samples_us": measurements}


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.source:
        if "torch_npu._inductor" in sys.modules:
            raise RuntimeError("Load the compiler source before importing torch_npu._inductor")
        source_file = args.source.resolve() / "torch_npu" / "_inductor" / "__init__.py"
        spec = importlib.util.spec_from_file_location("torch_npu._inductor", source_file)
        compiler_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = compiler_module
        torch_npu._inductor = compiler_module
        spec.loader.exec_module(compiler_module)
    from torch_npu import _inductor as npu_inductor
    from torch._decomp.decompositions import matmul as generic_matmul
    from torch._inductor import config
    from torch._inductor.decomposition import decompositions, fast_random_decomps
    from torch._inductor.utils import run_and_get_code
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch_npu._inductor.decomposition import _matmul_triton

    config.fx_graph_cache = False
    torch.manual_seed(20260905)
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    variants = {
        "generic": generic_matmul, "always": always_fold, "view": view_fold,
        "installed": _matmul_triton, "upstream": None,
    }
    print(json.dumps({
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "compiler": npu_inductor.__file__, "device": torch.npu.get_device_name(),
        "dynamic": args.dynamic, "dtype": args.dtype, "weight_grad": args.weight_grad,
        "max_autotune": config.max_autotune, "gemm_backends": config.max_autotune_gemm_backends,
    }), flush=True)
    sizes = tuple(int(value) for value in args.sizes.split(","))
    result_path = args.output / "results.jsonl"
    original_decomposition = decompositions.get(torch.ops.aten.matmul.default)
    failures = 0
    try:
        for layout in args.layouts.split(","):
            left, right = make_inputs(layout, sizes, getattr(torch, args.dtype), args.rhs_transpose, args.weight_grad)
            with torch.no_grad():
                expected = matmul(left, right)
            folded = left.reshape(math.prod(left.shape[:-1]), left.shape[-1])
            metadata = {
                "layout": layout, "left_shape": list(left.shape), "left_stride": list(left.stride()),
                "right_shape": list(right.shape), "right_stride": list(right.stride()),
                "reshape_aliases": folded.untyped_storage().data_ptr() == left.untyped_storage().data_ptr(),
                "view_eligible": can_fold_without_copy(left),
            }
            del folded
            for variant in args.variants.split(","):
                torch._dynamo.reset()
                decomposition = variants[variant]
                if decomposition is None:
                    decompositions.pop(torch.ops.aten.matmul.default, None)
                    decomposition = matmul
                else:
                    decompositions[torch.ops.aten.matmul.default] = decomposition
                fast_random_decomps.cache_clear()
                case_name = f"{layout}_{variant}"
                record = dict(metadata, variant=variant)
                try:
                    graph_module = make_fx(
                        lambda left, right: decomposition(left, right), tracing_mode="fake"
                    )(left, right)
                    (args.output / f"{case_name}_fx.py").write_text(graph_module.code)
                    record["fx_ops"] = [
                        str(node.target) for node in graph_module.graph.nodes if node.op == "call_function"
                    ]
                    compiled = torch.compile(matmul, fullgraph=True, dynamic=args.dynamic)
                    started = time.perf_counter()
                    actual, codes = run_and_get_code(compiled, left, right)
                    torch.npu.synchronize()
                    record["compile_seconds"] = time.perf_counter() - started
                    for index, code in enumerate(codes):
                        (args.output / f"{case_name}_code_{index}.py").write_text(code)
                    tolerance = 0.03 if args.dtype == "bfloat16" else 0.003
                    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                    record["correct"] = True
                    record.update(graph_latency(compiled, left, right, args.capture_calls, args.repeats))
                except Exception as error:
                    import traceback

                    failures += 1
                    record["error"] = f"{type(error).__name__}: {error}"
                    (args.output / f"{case_name}_error.txt").write_text(traceback.format_exc())
                with result_path.open("a") as result_file:
                    result_file.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
    finally:
        if original_decomposition is None:
            decompositions.pop(torch.ops.aten.matmul.default, None)
        else:
            decompositions[torch.ops.aten.matmul.default] = original_decomposition
        fast_random_decomps.cache_clear()
    if failures:
        raise SystemExit(f"{failures} benchmark cases failed; see {result_path}")


if __name__ == "__main__":
    main()
