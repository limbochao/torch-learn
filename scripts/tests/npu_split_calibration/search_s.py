"""Explore logical reduction factors and retain their code/tiling/resource evidence.

This is an experiment, not a product selector or a shape-to-S policy table.
The installed NPU eligibility checks and upstream multilayer construction stay
active. Only the original reduction's Choices result is replaced for a probe.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
from unittest.mock import patch

from common import file_hash, profiler_path, recorder_path, write_json
from hooks import make_recorder
from profiling import benchmark


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0, help="logical device in the visible set")
    return parser.parse_args()


def compiler_evidence(recorder):
    """Augment the shared recorder without changing its selection or launcher."""
    original = recorder.record_launch

    def record(tuner, args, grouped):
        original(tuner, args, grouped)
        row = recorder.launches[-1]
        launcher = getattr(tuner, "best_launcher", None)
        if launcher is None:
            launchers = getattr(tuner, "launchers", ())
            launcher = launchers[0] if len(launchers) == 1 else None
        binary = getattr(launcher, "bin", None)
        if binary is None:
            # make_launcher creates a Python function whose globals hold bin.
            binary = getattr(launcher, "__globals__", {}).get("bin")
        metadata = getattr(binary, "metadata", None)
        if hasattr(metadata, "_asdict"):
            metadata = metadata._asdict()
        elif metadata is not None and not isinstance(metadata, dict):
            metadata = vars(metadata)
        row["compiler_metadata"] = metadata
        required_bits = (metadata or {}).get("required_ub_bits", 0)
        row["required_ub_bytes"] = (required_bits + 7) // 8 if required_bits > 0 else None
        row["ub_evidence"] = "compiler report" if required_bits > 0 else "not reported; zero is not zero usage"
        row["binary_hash"] = getattr(binary, "hash", None)
        row["grid_stage"] = "Inductor logical launch grid before Triton runtime transforms"
        row["inductor_metadata"] = {
            key: value for key, value in tuner.inductor_meta.items() if key in (
                "axis_names", "axis_static_values", "split_axis", "tiling_axis", "low_dims",
                "npu_kernel_type", "reduction_hint", "requires_no_linear_block_remap",
                "runtime_block_arg_names", "max_autotune", "backend_hash",
            )
        }
        row["heuristic_type"] = str(getattr(tuner, "heuristic_type", None))

    recorder.record_launch = record


def add_profiled_blocks(result, out):
    """Attach CANN's launch block count using already validated task offsets.

    This distinguishes the device dispatch from Inductor's logical grid. A
    dispatched block count is still not a measurement of concurrent residency.
    """
    captures = {}
    blocks = {}
    for sample in result["samples"]:
        name = sample["profiler_csv"]
        if name not in captures:
            with (out / name).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            start_column = next(k for k in ("Start Time(us)", "Task Start Time(us)") if k in rows[0])
            rows.sort(key=lambda row: float(row[start_column].strip()))
            captures[name] = rows
        tasks = captures[name][sample["task_offset"]:sample["task_offset"] + sample["task_count"]]
        values = blocks.setdefault(sample["mode"], [set() for _ in sample["kernels"]])
        for index, observed in enumerate(values):
            observed.update((task.get("Block Num", ""), task.get("Mix Block Num", ""))
                            for task in tasks[index::len(values)])
    for mode, values in blocks.items():
        for launch, observed in zip(result["strategies"][mode]["launches"], values):
            launch["profiled_blocks"] = [dict(block_num=b, mix_block_num=m) for b, m in sorted(observed)]


def run_case(case, out, recorder, device, base_options):
    import torch
    from torch._inductor import config
    from torch._inductor.choices import InductorChoices
    from torch._inductor.ir import Reduction
    from torch._inductor.utils import run_and_get_code
    from torch_npu._inductor import choices, ir

    x, r = case["X"], case["R"]
    if x < 1 or r < 2 or case["layout"] not in ("inner", "outer", "stride2"):
        raise ValueError(f"unsupported case: {case}")
    splits = list(dict.fromkeys([1, *case["splits"]]))
    if any(not isinstance(s, int) or not 1 <= s < r for s in splits):
        raise ValueError("each S must be an integer in [1, R); S=R is a degenerate experiment")
    dtype = getattr(torch, case["dtype"])
    if not dtype.is_floating_point or case["op"] not in ("sum", "amax", "amin", "prod"):
        raise ValueError("this exploration currently supports floating ordinary reductions")
    case_dir = out / case["id"]
    case_dir.mkdir(exist_ok=False)
    torch._dynamo.reset()
    generator = torch.Generator().manual_seed(case.get("seed", 20261128))
    step = 2 if case["layout"] == "stride2" else 1
    host = torch.randn(x * r * step, generator=generator).to(dtype)
    host = host.view(r, x).t() if case["layout"] == "outer" else host.view(x, r * step)[:, ::step]
    value = host.to(device)
    # .to preserves transposes but may compact a sliced input: recreate stride2.
    if step == 2:
        storage = torch.empty((x, r * step), dtype=dtype, device=device)
        value = storage[:, ::step]
        value.copy_(host)
    reference = getattr(host.double(), case["op"])(1).to(dtype)
    if not torch.isfinite(reference).all():
        raise ValueError("nonfinite reference; choose input distribution suitable for this reduction")
    options = dict(base_options, max_autotune=case.get("max_autotune", True))
    result = dict(case=case, shape=list(value.shape), stride=list(value.stride()),
                  options=options, strategies={}, measurements={}, profiles=[], status="RUNNING")
    modes = ["current", *(f"s{s}" for s in splits)]
    forwards = {}
    signature = inspect.signature(ir.num_splits)
    multilayer_signature = inspect.signature(Reduction.create_multilayer)
    for mode in modes:
        namespace = {}
        exec(compile(f"def fn(a):\n    return a.{case['op']}(1)\n", f"{case['id']}-{mode}.py", "exec"), namespace)
        compiled_forwards, decisions, factors, multilayers = [], [], [], []

        def backend(gm, inputs, **kwargs):
            forward = torch._inductor.compile(gm, inputs)
            compiled_forwards.append(forward)
            return forward

        def choose(*args, **kwargs):
            bound = signature.bind(*args, **kwargs).arguments
            hint, s = ir.num_splits(*args, **kwargs)
            decisions.append(dict(ranges=str(bound["ranges"]), reduction_ranges=str(bound["reduction_ranges"]),
                                  hint=str(hint), S=int(s)))
            return hint, s

        def factor(device, reduction_numel_hint, numel_hint, inner_reduction):
            if mode == "current":
                s = choices.reduction_split_factor(device, reduction_numel_hint, numel_hint, inner_reduction)
            else:
                s = int(mode[1:]) if (numel_hint, reduction_numel_hint) == (x, r) else 1
            factors.append(dict(X=int(numel_hint), R=int(reduction_numel_hint), S=int(s), inner=inner_reduction))
            return s

        original_multilayer = Reduction.create_multilayer

        def observe_multilayer(*args, **kwargs):
            values = multilayer_signature.bind(*args, **kwargs).arguments
            multilayers.append(dict(S=int(values["split"]), ranges=str(values["ranges"]),
                                    reduction_ranges=str(values["reduction_ranges"])))
            return original_multilayer(*args, **kwargs)

        start = time.monotonic()
        with config.patch(options), patch.object(Reduction, "num_splits", staticmethod(choose)), patch.object(
            InductorChoices, "reduction_split_factor", staticmethod(factor)
        ), patch.object(Reduction, "create_multilayer", staticmethod(observe_multilayer)):
            compiled = torch.compile(namespace["fn"], backend=backend, fullgraph=True, dynamic=False)
            actual, code = run_and_get_code(compiled, value)
        compile_seconds = time.monotonic() - start
        if len(compiled_forwards) != 1:
            raise AssertionError(f"expected one compiled graph, got {len(compiled_forwards)}")
        forward = compiled_forwards[0]
        tolerance = {torch.float32: (2e-4, 5e-4), torch.float16: (3e-3, 3e-3),
                     torch.bfloat16: (2e-2, 2e-2)}[dtype]
        torch.testing.assert_close(actual.cpu(), reference, rtol=tolerance[0], atol=tolerance[1])
        torch.testing.assert_close(forward(value)[0], actual, rtol=0, atol=0)
        if mode.startswith("s") and int(mode[1:]) > 1:
            if not multilayers or multilayers[0]["S"] != int(mode[1:]):
                raise AssertionError(f"requested factor was not used: {mode}, {multilayers}")
        recorder.launches = []
        recorder.start_capture()
        try:
            forward(value)
        finally:
            tiling = recorder.stop_capture()
        if not recorder.launches or any("grid_error" in launch for launch in recorder.launches):
            raise AssertionError(f"missing launch evidence: {recorder.launches}")
        text = "\n".join(code)
        generated = case_dir / f"{mode}.py"
        generated.write_text(text, encoding="utf-8")
        result["strategies"][mode] = dict(
            decisions=decisions, factors=factors, multilayers=multilayers, tiling=tiling,
            launches=recorder.launches, source_sha256=hashlib.sha256(text.encode()).hexdigest(),
            generated_code=generated.relative_to(out).as_posix(), compile_seconds=compile_seconds,
            accuracy=dict(passed=True, max_abs_error=float((actual.cpu().double() - reference.double()).abs().max())),
        )
        forwards[mode] = SimpleNamespace(compiled=forward)
        write_json(case_dir / "result.json", result)
        print("COMPILED", case["id"], mode, round(compile_seconds, 2), flush=True)
    settings = dict(modes=modes, warmup=10, sessions=2, rounds=3, repeats=30)
    with patch.object(Reduction, "num_splits", side_effect=AssertionError("recompile during timing")):
        samples, captures = benchmark(forwards, [(value,)], result["strategies"], settings,
                                      seed=50315, directory=case_dir / "profiles", case_root=out)
    result.update(samples=samples, profiles=captures, settings=settings, status="PASS")
    add_profiled_blocks(result, out)
    for mode in modes:
        selected = [s for s in samples if s["mode"] == mode]
        timings = [s["kernel_us"] for s in selected]
        per_kernel = [[k["mean_us"] for k in s["kernels"]] for s in selected]
        result["measurements"][mode] = dict(median_us=statistics.median(timings), batches_us=timings,
                                            per_kernel_batches_us=per_kernel)
    write_json(case_dir / "result.json", result)
    print("MEASURED", case["id"], {m: round(v["median_us"], 4) for m, v in result["measurements"].items()}, flush=True)
    return result


def main():
    args = parse_args()
    out = args.out.resolve()
    repo = Path(__file__).resolve().parents[3]
    if out == repo or repo in out.parents:
        raise ValueError("raw evidence must be written outside the torch-learn repository")
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(out / "inductor_cache"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(out / "triton_cache"))
    import torch
    import torch_npu
    import triton
    from torch_npu._inductor import choices, config as npu_config, ir

    torch.npu.set_device(args.device)
    choices.patch_inductor_choices()
    ir.patch_num_splits()
    options = dict(split_reductions=True, min_num_split=0, fx_graph_cache=False, compile_threads=1,
                   autotune_local_cache=False, **{"triton.cooperative_reductions": False,
                   "triton.force_cooperative_reductions": False, "triton.multi_kernel": 0,
                   "triton.cudagraphs": False, "triton.mix_order_reduction": False})
    plan = json.loads(args.cases_json.read_text(encoding="utf-8"))
    ids = [case["id"] for case in plan["cases"]]
    if len(ids) != len(set(ids)) or any(Path(name).name != name or name in ("", ".", "..") for name in ids):
        raise ValueError("case IDs must be unique safe directory names")
    source_files = [Path(choices.__file__), Path(ir.__file__), Path(npu_config.__file__), Path(__file__),
                    recorder_path(), profiler_path()]
    source_files.extend(Path(__file__).parent / name for name in ("profiling.py", "hooks.py", "common.py"))
    source_files.extend(Path(torch_npu.__file__).parent / name for name in (
        "_inductor/runtime/tile_generator.py", "_inductor/runtime/triton_heuristics.py",
        "_inductor/codegen/triton.py", "_inductor/codegen/split_tiling.py"))
    source_files.extend(Path(triton.__file__).parent / name for name in (
        "backends/ascend/compiler.py", "backends/ascend/utils.py", "backends/ascend/driver.py"))
    report = dict(plan=plan, device=torch.npu.get_device_name(), torch=torch.__version__,
                  torch_npu=torch_npu.__version__, triton=triton.__version__,
                  device_properties=str(torch.npu.get_device_properties()),
                  vector_cores=npu_config._obtain_and_limit_cube_vector_core_num()[2],
                  backend_ub_budget=npu_config.ub_size,
                  source_hashes={str(path): file_hash(path) for path in source_files},
                  environment={key: os.environ.get(key) for key in (
                      "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_HOME_PATH", "NPU_DEVICE_LIMIT",
                      "FASTAUTOTUNE", "FAST_RUN_WITH_MAX_TILING_NUM", "TORCHINDUCTOR_NPU_BACKEND",
                      "ENABLE_PRINT_UB_BITS")},
                  options=options, method="hot; 2 sessions x 3 paired rounds x 30 forwards; all device CSV rows",
                  status="RUNNING", cases=[])
    recorder = make_recorder()
    compiler_evidence(recorder)
    recorder.install()
    try:
        for case in plan["cases"]:
            report["cases"].append(run_case(case, out, recorder, torch.device("npu", args.device), options))
            write_json(out / "strategy-sweep.json", report)
        report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        recorder.uninstall()
        write_json(out / "strategy-sweep.json", report)


if __name__ == "__main__":
    main()
