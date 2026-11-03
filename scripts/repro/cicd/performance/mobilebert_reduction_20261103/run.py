"""Replay MobileBERT NoNorm parameter reductions; diagnostics only, no installed-source patch."""

import argparse
import copy
import csv
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
    parser.add_argument("--experiments", action="store_true")
    parser.add_argument("--profile", action="store_true")
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
        graph_output_check = None
        if name == "original" or name.startswith(("same_dsl_", "two_stage_")):
            # Prove the captured graph writes outputs, rather than timing an empty capture.
            for value in out:
                value.fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
            graph_output_check = check(out)
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
            "graph_replay_max_abs_error": graph_output_check,
        }
        profile_names = {"original", "eager_same_boundary", "same_dsl_x16_xs16_rs256",
                         "two_stage_chunk128_xs32_rs128", "explicit_stages_codegen"}
        if args.profile and name in profile_names:
            target = args.output / "profiles" / name
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(target)),
            ):
                for _ in range(10):
                    fn()
                torch.npu.synchronize()
            files = list(target.rglob("kernel_details.csv"))
            if len(files) != 1:
                raise RuntimeError(f"Expected one kernel_details.csv, found {files}")
            rows = list(csv.DictReader(files[0].open()))
            groups = {}
            for row in rows:
                key = row["Name"]
                entry = groups.setdefault(key, {"count": 0, "total_us": 0.0})
                entry["count"] += 1
                entry["total_us"] += float(row["Duration(us)"])
            result["cases"][name]["profile"] = {
                "calls": 10,
                "device_us_per_call": sum(item["total_us"] for item in groups.values()) / 10,
                "kernels": groups,
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
    def original():
        # Graph capture switches the current stream; never cache its pointer before capture.
        stream = torch_npu._C._npu_getCurrentRawStreamNoWait(0)
        kernel.run(*inputs, *out, 128, 16384, stream=stream)

    original()
    result["original_selected_configs"] = recorder.stop_capture()
    result["original_launcher_config"] = repr(kernel.best_launcher.config)
    from torch_npu._inductor.runtime.triton_heuristics import triton_config_npu_index

    candidates = triton_config_npu_index(
        {"x0": 128, "r1": 16384}, copy.deepcopy(kernel.inductor_meta),
        copy.deepcopy(kernel.triton_meta), is_reduction=True)
    result["generated_candidates"] = [
        {**cfg.kwargs, "num_warps": cfg.num_warps, "num_stages": cfg.num_stages} for cfg in candidates]
    recorder.uninstall()
    bench("original", original, lambda: out)
    bench("eager_same_boundary", lambda: eager(*inputs), lambda: eager(*inputs))

    if args.experiments:
        from split_kernels import partial, finish

        selected = result["original_selected_configs"][0]["selected_config"]
        mode = selected["compile_mode"]
        options = {"num_warps": selected["num_warps"], "num_stages": selected["num_stages"],
                   "compile_mode": mode, "multibuffer": selected["multibuffer"]}

        def save_binary(name, binary):
            target = args.output / "ir" / name
            target.mkdir(parents=True, exist_ok=True)
            for key, value in binary.asm.items():
                if isinstance(value, str):
                    (target / ("kernel." + key)).write_text(value)
            (target / "metadata.json").write_text(json.dumps(binary.metadata._asdict(), indent=2, default=str))

        tiles = [(3, 1, 4096), (4, 4, 1024), (8, 8, 512), (16, 16, 256), (32, 32, 128)]
        for block, xsub, rsub in tiles:
            name = f"same_dsl_x{block}_xs{xsub}_rs{rsub}"

            def fixed():
                return kernel.fn[(triton.cdiv(128, block),)](
                    *inputs, *out, 128, 16384, block, X0BLOCK_SUB=xsub, R1BLOCK_SUB=rsub, **options)

            binary = fixed()
            save_binary(name, binary)
            bench(name, fixed, lambda: out)

        for chunk in [128, 512, 2048]:
            parts = triton.cdiv(16384, chunk)
            buffers = [torch.empty((parts, 128), device="npu") for _ in range(2)]
            name = f"two_stage_chunk{chunk}_xs32_rs128"

            def split():
                first = partial[(4, parts)](*inputs, *buffers, chunk, XBLOCK=32, RBLOCK=128, **options)
                last = finish[(4,)](*buffers, *out, parts, XBLOCK=32, RBLOCK=triton.next_power_of_2(parts),
                                   **options)
                return first, last

            first, last = split()
            save_binary(name + "_partial", first)
            save_binary(name + "_finish", last)
            bench(name, split, lambda: out)

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

        if args.experiments:
            def explicit_stages(a, b, c):
                value = a + b
                return (value.sum(1, keepdim=True).sum(0, keepdim=True),
                        (value * c).sum(1, keepdim=True).sum(0, keepdim=True))

            recorder.install()
            recorder.start_capture()
            staged = torch.compile(explicit_stages, fullgraph=True)
            staged(*inputs)
            result["explicit_stages_selected_configs"] = recorder.stop_capture()
            recorder.uninstall()
            bench("explicit_stages_codegen", lambda: staged(*inputs), lambda: staged(*inputs))

    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
