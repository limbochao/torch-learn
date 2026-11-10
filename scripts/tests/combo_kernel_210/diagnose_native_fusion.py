"""Trace native foreach Combo decisions in one process without patching installed source files."""

import argparse
import functools
import json
from pathlib import Path
import sys
import traceback

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch_npu.utils import _dynamo

_dynamo.use_jit_script = True
_dynamo.patch_has_triton()
torch.cuda.get_device_capability = lambda *args, **kwargs: (10, 0)
import torch_npu.testing
import torch_npu._inductor
from torch._inductor import config, metrics
from torch._inductor.codegen.common import BackendFeature
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import run_and_get_code
from torch_npu._inductor.codegen.npu_combined_scheduling import NPUCombinedScheduling
from torch_npu._inductor.codegen.scheduling import NPUTritonScheduling
from torch_npu._inductor.codegen.triton_combo_kernel import NPUComboKernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enable-foreach", action="store_true", help="Diagnostic process-local capability control")
    parser.add_argument("--benchmark-test-root", type=Path, help="Run the two original native 2D benchmark methods")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.npu.set_device(0)
    events = []
    current = ""

    def record(event, **fields):
        events.append({"case": current, "event": event, **fields})
        (args.output / "events.json").write_text(json.dumps(events, indent=2) + "\n")

    before = BackendFeature.FOREACH in NPUTritonScheduling.get_backend_features(torch.device("npu"))
    if args.enable_foreach:
        NPUTritonScheduling.backend_features.add(BackendFeature.FOREACH)
    record("backend_capability", foreach_before=before,
           foreach_after=BackendFeature.FOREACH in NPUTritonScheduling.get_backend_features(torch.device("npu")))
    original_has_feature = GraphLowering.has_feature

    def has_feature(self, device, feature):
        result = original_has_feature(self, device, feature)
        if feature == BackendFeature.FOREACH:
            record("foreach_gate", device=str(device), enabled=result)
        return result

    GraphLowering.has_feature = has_feature

    def trace_method(cls, name):
        original = getattr(cls, name)

        @functools.wraps(original)
        def traced(self, *positional, **keywords):
            label = cls.__name__ + "." + name
            record("enter", method=label, kernel_count=metrics.generated_kernel_count)
            try:
                result = original(self, *positional, **keywords)
            except Exception:
                record("exception", method=label, traceback=traceback.format_exc())
                if name == "make_ttir_for_check":
                    destination = args.output / (current + "_rejected_combo.py")
                    destination.write_text(positional[0])
                raise
            record("exit", method=label, kernel_count=metrics.generated_kernel_count)
            return result

        setattr(cls, name, traced)

    for cls, method in [
        (NPUCombinedScheduling, "codegen_combo_kernel"),
        (NPUCombinedScheduling, "codegen_node"),
        (NPUTritonScheduling, "codegen_combo_kernel"),
        (NPUTritonScheduling, "make_ttir_for_check"),
    ]:
        trace_method(cls, method)
    original_partition = NPUComboKernel.horizontal_partition

    def partition(*positional, **keywords):
        nodes = keywords.get("nodes", positional[0] if positional else None)
        kernel_map = keywords["kernel_map"]
        items = []
        for node in nodes:
            kernel = kernel_map[node]
            items.append({"node": node.get_name(), "kernel_type": str(kernel.npu_kernel_type),
                          "inside_reduction": kernel.inside_reduction,
                          "persistent_reduction": kernel.persistent_reduction,
                          "axes": [
                              {"name": axis.name, "length": str(axis.length),
                               "split": axis.is_split_axis, "tiling": axis.is_tiling_axis,
                               "no_loop": axis.is_no_loop_axis, "reduction": axis.is_reduction}
                              for axis in kernel.sorted_axis]})
        result = original_partition(*positional, **keywords)
        record("horizontal_partition", inputs=items,
               partitions=[[node.get_name() for node in group] for group in result])
        return result

    NPUComboKernel.horizontal_partition = staticmethod(partition)
    results = []
    if args.benchmark_test_root:
        sys.path.insert(0, str(args.benchmark_test_root.resolve()))
        import test_combo_kernels

        for cls, method in [
            (test_combo_kernels.ComboKernelBenchmarkTests, "test_2d_blocking_benchmark"),
            (test_combo_kernels.ComboKernelDynamicShapesTests, "test_dynamic_shapes_2d_blocking"),
        ]:
            current = method
            test = cls(method)
            test.setUp()
            try:
                getattr(test, method)()
                result = {"case": current, "result": "passed", "kernel_count": metrics.generated_kernel_count}
            except Exception:
                result = {"case": current, "error": traceback.format_exc(),
                          "kernel_count": metrics.generated_kernel_count}
            finally:
                test.tearDown()
            results.append(result)
            record("result", **{key: value for key, value in result.items() if key != "case"})
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2), flush=True)
        return
    for current, shapes, combo in [
        ("native_abs_mixed_shape", [(10, 10), (20, 20)], False),
        ("native_abs_same_shape", [(10, 10), (10, 10)], False),
        ("explicit_combo_abs", [(10, 10), (20, 20)], True),
    ]:
        torch._dynamo.reset()
        torch.manual_seed(1234)
        inputs = [torch.randn(shape, device="npu") for shape in shapes]

        def fn(a, b):
            return torch._foreach_abs([a, b])

        expected = fn(*inputs)
        metrics.reset()
        record("start", shapes=shapes, combo_kernels=combo)
        try:
            with config.patch(combo_kernels=combo, force_disable_caches=True):
                actual, sources = run_and_get_code(torch.compile(fn, fullgraph=True), *inputs)
            torch.npu.synchronize()
            torch.testing.assert_close(actual, expected)
            for index, source in enumerate(sources):
                (args.output / f"{current}_wrapper_{index}.py").write_text(source)
            result = {"case": current, "numerics": "passed", "kernel_count": metrics.generated_kernel_count,
                      "wrapper_sources": len(sources)}
        except Exception:
            result = {"case": current, "error": traceback.format_exc()}
        results.append(result)
        record("result", **{key: value for key, value in result.items() if key != "case"})
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
