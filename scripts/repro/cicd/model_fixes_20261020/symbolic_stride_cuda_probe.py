"""Probe symbolic stride modulo in installed PyTorch, without importing torch_npu.

Metadata: python symbolic_stride_cuda_probe.py --mode metadata
CUDA:     python symbolic_stride_cuda_probe.py --mode all --device cuda
Control:  python symbolic_stride_cuda_probe.py --mode compile --device cuda --patch-pythonmod
Run CUDA controls in separate processes with separate TORCHINDUCTOR_CACHE_DIR values.
"""

import argparse
import contextlib
import inspect
import json
from types import SimpleNamespace

import sympy
import torch
import torch.nn.functional as F
from torch._dynamo.source import LocalSource
from torch._inductor import config
from torch._inductor.ir import Layout
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.utils._sympy.functions import FloorDiv, PythonMod
from torch.utils._sympy.numbers import int_oo


def report(**fields):
    print(json.dumps(fields, ensure_ascii=False, default=str), flush=True)


def pythonmod_compare(left, right):
    """Candidate from c74c45c960; installed only with --patch-pythonmod in compile mode."""
    sizevars = V.graph.sizevars
    if sizevars.guard_or_false(sympy.Eq(right, 0)):
        return True
    if sizevars.guard_or_false(sympy.Eq(left, 0)):
        return False
    return sizevars.guard_or_false(sympy.Ge(left, right)) or sizevars.guard_or_false(
        sympy.Eq(PythonMod(left, right), 0)
    )


def symbolic_pair(expanded, reverse):
    # Fresh ShapeEnv for each observation: guards from one probe cannot help another.
    env = ShapeEnv()
    height = env.create_symbol(256, LocalSource("height"), dynamic_dim=DimDynamic.DYNAMIC)
    env.constrain_symbol_range(height, 33, int_oo)
    side = FloorDiv(height - 1, 32) + 1
    square = sympy.expand(side**2) if expanded else side**2
    left, right = (square, side) if reverse else (side, square)
    return env, left, right


def metadata_probe():
    native = getattr(Layout, "_stride_expr_ge_or_false", None)
    failures = 0
    for expanded in (False, True):
        for reverse in (False, True):
            case = f"{'expanded' if expanded else 'factored'}_{'reverse' if reverse else 'forward'}"
            for method in ("percent_expr", "pythonmod_expr", "native_layout", "pythonmod_candidate"):
                if method == "native_layout" and native is None:
                    report(stage="metadata", case=case, method=method, status="SKIP", reason="helper absent")
                    continue
                env, left, right = symbolic_pair(expanded, reverse)
                fields = dict(stage="metadata", case=case, method=method, left=str(left), right=str(right))
                try:
                    with V.set_graph_handler(SimpleNamespace(sizevars=SizeVarAllocator(env))):
                        if method.endswith("expr"):
                            remainder = left % right if method == "percent_expr" else PythonMod(left, right)
                            fields["remainder"] = str(remainder)
                            fields["negative_powers"] = [
                                str(power) for power in remainder.atoms(sympy.Pow) if power.exp.is_negative
                            ]
                            actual = bool(env.evaluate_expr(sympy.Eq(remainder, 0)))
                        else:
                            compare = native if method == "native_layout" else pythonmod_compare
                            actual = bool(compare(left, right))
                    assert actual == reverse, (actual, reverse)
                    report(**fields, status="PASS", value=actual, guards=len(env.guards))
                except Exception as error:
                    reproduced = (
                        method in ("percent_expr", "native_layout")
                        and type(error).__name__ == "ValueRangeError"
                        and "Invalid ranges [0:-1]" in str(error)
                    )
                    failures += not reproduced
                    report(
                        **fields, status="REPRODUCED" if reproduced else "ERROR",
                        error_type=type(error).__name__, error=str(error).splitlines()[0],
                    )
    # Existing zero/order/divisibility behavior of the candidate, independent of symbolic simplification.
    for left, right, expected in ((0, 0, True), (0, 8, False), (8, 0, True), (8, 8, True),
                                  (4, 8, False), (8, 4, True)):
        with V.set_graph_handler(SimpleNamespace(sizevars=SizeVarAllocator())):
            actual = bool(pythonmod_compare(left, right))
        assert actual == expected, (left, right, actual, expected)
    report(stage="metadata", case="candidate_static_controls", status="PASS", cases=6)
    return failures


def forward(value, weight):
    value = F.conv2d(value, weight, stride=32)
    return value * value.mean((-2, -1), keepdim=True).sigmoid()


@contextlib.contextmanager
def observe_layout(use_pythonmod):
    # Preserve the staticmethod descriptor, not just its extracted function.
    descriptor = inspect.getattr_static(Layout, "_stride_expr_ge_or_false", None)
    native = getattr(Layout, "_stride_expr_ge_or_false", None)
    counts = {"calls": 0, "errors": 0}
    if native is None:
        if use_pythonmod:
            raise RuntimeError("Cannot patch: this PyTorch has no _stride_expr_ge_or_false")
        report(stage="compile", event="observer", status="SKIP", reason="helper absent in this version")
        yield counts
        return
    compare = pythonmod_compare if use_pythonmod else native

    def observed(left, right):
        counts["calls"] += 1
        try:
            result = compare(left, right)
        except Exception as error:
            counts["errors"] += 1
            report(stage="compile", event="layout_error", left=str(left), right=str(right),
                   error_type=type(error).__name__, error=str(error).splitlines()[0])
            raise
        if counts["calls"] <= 8:
            report(stage="compile", event="layout_compare", left=str(left), right=str(right), value=bool(result))
        return result

    Layout._stride_expr_ge_or_false = staticmethod(observed)
    try:
        yield counts
    finally:
        Layout._stride_expr_ge_or_false = descriptor


def compile_probe(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        report(stage="compile", status="SKIP", reason="CUDA unavailable; no CUDA result")
        return 0
    if device.type not in ("cuda", "cpu"):
        raise ValueError("Use cuda for device validation or cpu for a harness smoke check")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report(stage="compile", gpu=torch.cuda.get_device_name(device))
    torch.manual_seed(0)
    # Avoid silent eager fallback; backend errors must remain visible to this probe.
    torch._dynamo.config.suppress_errors = False
    overrides = {"layout_optimization": False} if args.disable_layout_optimization else {}
    failures = 0
    with config.patch(overrides), observe_layout(args.patch_pythonmod) as counts:
        compiled = torch.compile(forward, backend="inductor", fullgraph=True, dynamic=True)
        for batch, height in ((2, 224), (3, 256)):
            value = torch.randn(batch, 16, height, height, device=device, requires_grad=True)
            weight = torch.randn(16, 16, 1, 1, device=device, requires_grad=True)
            report(stage="compile", event="input", batch=batch, height=height,
                   shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype))
            try:
                expected = forward(value, weight)
                gradient = torch.randn_like(expected)
                expected_grads = torch.autograd.grad(expected, (value, weight), gradient)
                actual = compiled(value, weight)
                actual_grads = torch.autograd.grad(actual, (value, weight), gradient)
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-3, atol=2e-3)
                report(stage="compile", status="PASS", batch=batch, height=height,
                       max_abs_forward=(actual - expected).abs().max().item(),
                       max_abs_grads=[(a - b).abs().max().item() for a, b in zip(actual_grads, expected_grads)])
            except Exception as error:
                failures += 1
                report(stage="compile", status="ERROR", batch=batch, height=height,
                       error_type=type(error).__name__, error=str(error).splitlines()[0])
                if args.traceback:
                    import traceback
                    traceback.print_exc()
                break
        report(stage="compile", event="layout_summary", **counts,
               patch_pythonmod=args.patch_pythonmod, disable_layout_optimization=args.disable_layout_optimization)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("metadata", "compile", "all"), default="metadata")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patch-pythonmod", action="store_true", help="process-local control, compile mode only")
    parser.add_argument("--disable-layout-optimization", action="store_true")
    parser.add_argument("--traceback", action="store_true", help="include full compile errors")
    args = parser.parse_args()
    helper = getattr(Layout, "_stride_expr_ge_or_false", None)
    report(stage="environment", torch=torch.__version__, torch_git=torch.version.git_version,
           sympy=sympy.__version__, cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
           native_helper=helper is not None, npu_patch_marker=getattr(helper, "_torch_npu_integer_strides", False))
    failures = metadata_probe() if args.mode in ("metadata", "all") else 0
    if args.mode in ("compile", "all"):
        failures += compile_probe(args)
    report(stage="summary", unexpected_failures=failures,
           note="REPRODUCED is an observed baseline bug; SKIP is not a validation pass")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
