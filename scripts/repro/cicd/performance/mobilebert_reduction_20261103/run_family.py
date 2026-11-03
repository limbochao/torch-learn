"""Replay all requested remaining reductions, without modifying framework sources."""

import argparse
import copy
import csv
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

NAMES = {
    0: "add_sum_view_0",
    7: "mul_sum_view_7",
    9: "mul_sum_view_9",
    11: "add_mul_sum_view_11",
    13: "add_mul_sum_view_13",
    20: "add_mul_sum_view_20",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-device", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--kinds", type=int, nargs="+", default=[9, 7, 13, 20, 0], choices=list(NAMES))
    parser.add_argument("--verify-compile-path", action="store_true")
    args = parser.parse_args()
    if 11 in args.kinds and not args.verify_compile_path:
        parser.error("The _11 cross-check requires --verify-compile-path; use run.py for its full experiment.")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.physical_device):
        raise RuntimeError("Visible physical device does not match --physical-device")
    import torch
    import torch_npu
    import triton
    from family_kernels import partial, finish
    from torch_npu._inductor.runtime.triton_heuristics import triton_config_npu_index

    root = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(root / "scripts/tools"))
    from autotune_tiling import BestTilingRecorder

    args.output.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0)
    torch.manual_seed(20261103)
    torch.set_num_threads(8)
    torch._inductor.config.compile_threads = 1
    torch._inductor.config.trace.enabled = True
    result = {
        "physical_device": args.physical_device,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "triton": triton.__version__,
        "kernels": {},
    }

    def save():
        (args.output / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")

    for kind in args.kinds:
        width = 30522 if kind == 0 else 512 if kind == 20 else 128
        dual = kind in (7, 11, 13, 20)
        input_count = {0: 2, 7: 2, 9: 1, 11: 3, 13: 6, 20: 5}[kind]
        cpu = [torch.randn((width,) if kind == 13 and i >= 4 else (128, 128, width)) for i in range(input_count)]
        inputs = [v.npu() for v in cpu]

        def expressions(values):
            a = values[0]
            if kind in (0, 11, 13, 20):
                a = a + values[1]
            if kind == 20:
                a = (a + values[2]) + values[3]
            if kind == 7:
                return a, a * values[1]
            if kind == 11:
                return a, a * values[2]
            if kind == 13:
                return a, a * (values[2] + (values[3] * values[4] + values[5]))
            if kind == 20:
                return a, a * values[4]
            return (a,)

        # Preserve FP32 pointwise semantics, then accumulate an independent FP64 reference.
        # Chunk the large vocabulary case to avoid multi-GB FP64 intermediates.
        reference = [torch.zeros((1, 1, width), dtype=torch.float64) for _ in range(2 if dual else 1)]
        for start in range(0, 128, 8):
            values = [v if v.ndim == 1 else v[start : start + 8] for v in cpu]
            for index, value in enumerate(expressions(values)):
                reference[index] += value.double().sum((0, 1), keepdim=True)
        del cpu, values, value

        def eager(*values):
            return tuple(v.sum((0, 1), keepdim=True) for v in expressions(values))

        def staged(*values):
            return tuple(v.sum(1, keepdim=True).sum(0, keepdim=True) for v in expressions(values))

        data = {
            "shape": [128, 128, width],
            "inputs": [{"shape": list(v.shape), "stride": list(v.stride()), "dtype": str(v.dtype)} for v in inputs],
            "cases": {},
        }
        result["kernels"][NAMES[kind]] = data

        def check(values):
            errors = []
            for actual, expected in zip(values, reference):
                actual = actual.cpu().double()
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-3)
                errors.append(float((actual - expected).abs().max()))
            return errors

        def bench(name, fn, fixed_outputs=None):
            outputs = None
            for _ in range(5):
                outputs = fn()
            torch.npu.synchronize()
            errors = check(fixed_outputs if fixed_outputs is not None else outputs)
            # Capture only 4 calls for the vocabulary case: Eager materializes a 2-GB add.
            count = 4 if kind == 0 else 32
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(count):
                    outputs = fn()
            torch.npu.synchronize()
            captured_outputs = fixed_outputs if fixed_outputs is not None else outputs
            for v in captured_outputs:
                v.fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
            graph_errors = check(captured_outputs)
            samples = []
            for _ in range(7):
                begin = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                begin.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end) * 1000 / count)
            entry = {
                "event_median_us": statistics.median(samples),
                "event_samples_us": samples,
                "max_abs_error": errors,
                "graph_output_error": graph_errors,
                "capture_calls": count,
            }
            data["cases"][name] = entry
            target = args.output / "profiler" / NAMES[kind] / name
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(target)),
            ):
                for _ in range(10):
                    fn()
                torch.npu.synchronize()
            files = list(target.rglob("kernel_details.csv"))
            if len(files) != 1:
                raise RuntimeError(f"Expected one profiler CSV: {files}")
            groups = {}
            for row in csv.DictReader(files[0].open()):
                item = groups.setdefault(row["Name"], {"count": 0, "total_us": 0.0, "samples_us": []})
                item["count"] += 1
                item["total_us"] += float(row["Duration(us)"])
                item["samples_us"].append(float(row["Duration(us)"]))
            entry["profile"] = {
                "calls": 10,
                "kernels": groups,
                "device_us_per_call": sum(v["total_us"] for v in groups.values()) / 10,
            }
            print(NAMES[kind], name, entry, flush=True)
            save()

        recorder = BestTilingRecorder("npu")
        recorder.install()
        recorder.start_capture()
        path = root / "artifacts/cicd/performance/mobilebert-reduction-20261103/family" / (NAMES[kind] + ".py")
        if kind == 11:
            path = path.parent.parent / "original_kernel.py"
        spec = importlib.util.spec_from_file_location("original_" + str(kind), path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        kernel = getattr(module, "triton_red_fused_" + NAMES[kind])
        out = [torch.empty((1, 1, width), device="npu") for _ in reference]

        def original():
            stream = torch_npu._C._npu_getCurrentRawStreamNoWait(0)
            kernel.run(*inputs, *out, width, 16384, stream=stream)

        original()
        data["selected_configs"] = recorder.stop_capture()
        recorder.uninstall()
        data["candidates"] = [
            dict(cfg.kwargs)
            for cfg in triton_config_npu_index(
                {"x0": width, "r1": 16384},
                copy.deepcopy(kernel.inductor_meta),
                copy.deepcopy(kernel.triton_meta),
                is_reduction=True,
            )
        ]
        bench("original", original, out)
        bench("eager", lambda: eager(*inputs))
        selected = data["selected_configs"][0]["selected_config"]
        options = {k: selected[k] for k in ["num_warps", "num_stages", "compile_mode", "multibuffer"]}

        def save_binary(name, binary):
            target = args.output / "ir" / NAMES[kind] / name
            target.mkdir(parents=True, exist_ok=True)
            for key in ["source", "ttir", "ttadapter"]:
                if key in binary.asm:
                    (target / ("kernel." + key)).write_text(binary.asm[key])
            (target / "metadata.json").write_text(json.dumps(binary.metadata._asdict(), indent=2, default=str))

        # The launcher keeps the binary in its generated function scope unless store_cubin is enabled.
        save_binary("original", kernel.best_launcher.__globals__["bin"])
        if args.verify_compile_path:
            selected_tile = (selected["X0BLOCK"], selected["X0BLOCK_SUB"], selected["R1BLOCK_SUB"])
            matched_tiles = [selected_tile, (16, 16, 256), (32, 32, 128), (64, 64, 64)]
            if kind == 0:
                matched_tiles = [selected_tile, (128, 128, 32), (256, 256, 16), (512, 512, 8)]
            for index, (xb, xs, rb) in enumerate(matched_tiles):
                cfg = copy.deepcopy(kernel.best_candidate_config)
                cfg.kwargs.update(X0BLOCK=xb, X0BLOCK_SUB=xs, R1BLOCK_SUB=rb, split_blocks=[xb])
                compiled_result = kernel._precompile_config(cfg)
                launcher = compiled_result.make_launcher()
                name = "matched_selected" if index == 0 else f"matched_x{xb}_r{rb}"

                def matched():
                    stream = torch_npu._C._npu_getCurrentRawStreamNoWait(0)
                    return launcher(*inputs, *out, width, 16384, xb, stream=stream)

                save_binary(name, compiled_result.kernel)
                bench(name, matched, out)

        tiles = [(16, 256), (32, 128), (64, 64)]
        if kind == 0:
            tiles += [(128, 32), (256, 16), (512, 8), (1024, 4)]
        for xb, rb in ([] if args.verify_compile_path else tiles):
            name = f"same_dsl_x{xb}_r{rb}"

            def fixed():
                return kernel.fn[(triton.cdiv(width, xb),)](
                    *inputs, *out, width, 16384, xb, X0BLOCK_SUB=xb, R1BLOCK_SUB=rb, **options
                )

            binary = fixed()
            save_binary(name, binary)
            bench(name, fixed, out)

        padded_inputs = inputs + [inputs[0]] * (6 - len(inputs))
        padded_outputs = out + out[:1] if not dual else out
        for chunk in ([] if args.verify_compile_path else [128, 512, 2048]):
            parts = triton.cdiv(16384, chunk)
            buffers = [torch.empty((parts, width), device="npu") for _ in range(2 if dual else 1)]
            padded_buffers = buffers + buffers[:1] if not dual else buffers
            xb = 128 if kind == 0 else 32
            name = f"two_stage_chunk{chunk}"

            def split():
                first = partial[(triton.cdiv(width, xb), parts)](
                    *padded_inputs, *padded_buffers, width, kind, chunk, xb, 128, **options
                )
                last = finish[(triton.cdiv(width, xb),)](
                    *padded_buffers, *padded_outputs, width, dual, parts, xb, triton.next_power_of_2(parts), **options
                )
                return first, last

            first, last = split()
            save_binary(name + "_partial", first)
            save_binary(name + "_finish", last)
            bench(name, split, out)

        recorder.install()
        recorder.start_capture()
        compiled = torch.compile(staged, fullgraph=True)
        compiled(*inputs)
        data["staged_selected_configs"] = recorder.stop_capture()
        recorder.uninstall()
        bench("aten_explicit_stages", lambda: compiled(*inputs))
        save()
        del compiled, inputs, out, padded_inputs, padded_outputs
        if not args.verify_compile_path:
            del buffers, padded_buffers
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
