"""Replay MobileBERT NoNorm parameter reductions; diagnostics only, no installed-source patch."""

import argparse
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    # Physical device is checked before importing a device backend.
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2":
        raise RuntimeError("This investigation requires ASCEND_RT_VISIBLE_DEVICES=2 (physical card 2).")
    args.output.mkdir(parents=True, exist_ok=True)

    import torch
    import torch_npu
    import triton

    root = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(root / "scripts/tools"))
    from autotune_tiling import BestTilingRecorder

    torch.npu.set_device(0)
    torch.manual_seed(20261103)
    cpu = [torch.randn(128, 128, 128, dtype=torch.float32) for _ in range(3)]
    inputs = [value.npu() for value in cpu]
    reference = ((cpu[0].double() + cpu[1].double()).sum((0, 1), keepdim=True),
                 ((cpu[0].double() + cpu[1].double()) * cpu[2].double()).sum((0, 1), keepdim=True))

    def eager(a, b, c):
        value = a + b
        return value.sum((0, 1), keepdim=True), (value * c).sum((0, 1), keepdim=True)

    result = {
        "physical_device": 2,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "triton": triton.__version__,
        "shape": list(inputs[0].shape),
        "stride": list(inputs[0].stride()),
        "dtype": str(inputs[0].dtype),
        "cases": {},
    }

    def check(outputs):
        errors = []
        for actual, expected in zip(outputs, reference):
            actual = actual.cpu().double()
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-3)
            errors.append(float((actual - expected).abs().max()))
        return errors

    def bench(name, fn, outputs):
        for _ in range(5):
            fn()
        torch.npu.synchronize()
        check_result = check(outputs())
        # Capture repeated calls so Event measurements do not include Python launch gaps.
        graph = torch.npu.NPUGraph()
        count = 64
        with torch.npu.graph(graph):
            for _ in range(count):
                fn()
        torch.npu.synchronize()
        samples = []
        for _ in range(7):
            begin = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) * 1000 / count)
        result["cases"][name] = {
            "graph_event_us_per_call": statistics.median(samples),
            "samples_us": samples,
            "max_abs_error_vs_cpu_fp64": check_result,
        }
        print(name, result["cases"][name], flush=True)
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    recorder = BestTilingRecorder("npu")
    recorder.install()
    recorder.start_capture()
    artifact = root / "artifacts/cicd/performance/mobilebert-reduction-20261103/original_kernel.py"
    spec = importlib.util.spec_from_file_location("mobilebert_original", artifact)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = module.triton_red_fused_add_mul_sum_view_11
    out = [torch.empty((1, 1, 128), device="npu") for _ in range(2)]
    stream = torch_npu._C._npu_getCurrentRawStreamNoWait(0)

    def original():
        kernel.run(*inputs, *out, 128, 16384, stream=stream)

    original()
    result["original_selected_configs"] = recorder.stop_capture()
    result["original_launcher_config"] = repr(kernel.best_launcher.config)
    recorder.uninstall()
    bench("original", original, lambda: out)
    bench("eager_same_boundary", lambda: eager(*inputs), lambda: eager(*inputs))

    if args.regenerate:
        torch._inductor.config.compile_threads = 1
        torch._inductor.config.trace.enabled = True
        recorder.install()
        recorder.start_capture()
        compiled = torch.compile(eager, fullgraph=True)
        compiled(*inputs)
        result["regenerated_selected_configs"] = recorder.stop_capture()
        recorder.uninstall()
        bench("regenerated", lambda: compiled(*inputs), lambda: compiled(*inputs))

    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
