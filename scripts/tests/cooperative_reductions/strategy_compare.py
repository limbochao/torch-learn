"""Compare actual CUDA Inductor plain, split, cooperative, and staged reduction graphs."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import traceback


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def make_function(name):
    import torch

    if name == "sum":
        return lambda a, b: (a + b).sum(-1)
    if name == "rmsnorm":
        def rmsnorm(a, b):
            z = a + b
            return z * torch.rsqrt((z * z).mean(-1, keepdim=True) + 1e-5)
        return rmsnorm
    if name == "softmax":
        return lambda a, b: torch.softmax(a + b, dim=-1)
    if name == "chain":
        def chain(a, b):
            z = a + b
            for _ in range(4):
                z = z + torch.softmax(z, dim=-1)
            return z
        return chain
    if name == "mixed":
        return lambda a, b: ((a + b).sum(), (a - b).sum(-1))
    raise ValueError(name)


def compare(actual, expected, name):
    import torch

    actuals = actual if isinstance(actual, tuple) else (actual,)
    expecteds = expected if isinstance(expected, tuple) else (expected,)
    atol = 0.02 if name in ("sum", "mixed") else (1e-8 if name == "softmax" else 3e-5)
    rtol = 3e-4
    mismatches, maximum = 0, 0.0
    for result, reference in zip(actuals, expecteds, strict=True):
        result, reference = result.detach().cpu().double(), reference.detach().cpu().double()
        close = torch.isclose(result, reference, atol=atol, rtol=rtol, equal_nan=False)
        mismatches += int((~close).sum())
        error = (result - reference).abs()
        maximum = max(maximum, float(error.max()))
    return {"mismatched_values": mismatches, "max_abs_error": maximum, "atol": atol, "rtol": rtol}


def timing(torch, invoke, args):
    for _ in range(5):
        invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(args.batch):
            invoke()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (args.batch * args.replays))
    return {"median_us": statistics.median(samples), "min_us": min(samples),
            "max_us": max(samples), "samples_us": samples, "batch": args.batch,
            "replays": args.replays, "method": "CUDA Graph + device Events"}


def worker(args):
    import torch
    import triton
    from torch._inductor import config, metrics
    from torch._inductor.utils import run_and_get_code

    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.manual_seed(1731)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    record = {"workload": args.workloads, "strategy": args.strategies, "X": args.rows,
              "R": args.rnumels, "status": "RUNNING"}
    write_json(out / "active.json", record)
    sources = []
    try:
        rows, rnumel = int(args.rows), int(args.rnumels)
        a = torch.randn((rows, rnumel), device="cuda", dtype=torch.float32)
        b = torch.randn_like(a)
        fn = make_function(args.workloads)
        strategy = args.strategies
        options = {"split_reductions": strategy != "plain",
                   "triton.cooperative_reductions": strategy in ("cooperative", "hybrid"),
                   "triton.force_cooperative_reductions": False,
                   "triton.multi_kernel": 0, "fx_graph_cache": False, "autotune_local_cache": False}
        record["config"] = options
        record["versions"] = {"torch": torch.__version__, "torch_commit": torch.version.git_version,
                              "triton": triton.__version__, "cuda": torch.version.cuda}
        prop = torch.cuda.get_device_properties(0)
        record["device"] = {"name": prop.name, "uuid": str(prop.uuid), "sm_count": prop.multi_processor_count}
        for module in ("torch._inductor.choices", "torch._inductor.ir", "torch._inductor.codegen.triton"):
            path = Path(inspect.getfile(__import__(module, fromlist=["__file__"])))
            record.setdefault("source_sha256", {})[module] = hashlib.sha256(path.read_bytes()).hexdigest()
        with config.patch(options):
            metrics.reset()
            begin = time.monotonic()
            if strategy == "hybrid":
                assert args.workloads == "sum" and rnumel % args.groups == 0

                def partial(a, b):
                    return (a + b).reshape(rows, args.groups, rnumel // args.groups).sum(-1)

                partial_fn = torch.compile(partial, fullgraph=True, dynamic=False)
                temp, code = run_and_get_code(partial_fn, a, b)
                sources.extend(code)
                finish_fn = torch.compile(lambda t: t.sum(-1), fullgraph=True, dynamic=False)
                _, code = run_and_get_code(finish_fn, temp)
                sources.extend(code)
                invoke = lambda: finish_fn(partial_fn(a, b))
                record["outer_groups"] = args.groups
            else:
                compiled = torch.compile(fn, fullgraph=True, dynamic=False)
                _, sources = run_and_get_code(compiled, a, b)
                invoke = lambda: compiled(a, b)
            torch.cuda.synchronize()
            record["compile_wall_s"] = time.monotonic() - begin
            record["generated_kernel_count"] = metrics.generated_kernel_count
            text = "\n".join(sources)
            for index, source in enumerate(sources):
                (out / f"output_code_{index}.py").write_text(source)
            record["source_summary"] = {
                "cooperative_decorators": text.count("@triton_heuristics.cooperative_reduction("),
                "barrier_calls": text.count("triton_helpers.x_grid_barrier("),
                "run_calls": len(re.findall(r"\w+\.run\(", text.split("def benchmark_compiled_module")[0])),
                "rsplit_literals": re.findall(r"['\"]RSPLIT['\"]\s*:\s*(\d+)", text)}
            record["accuracy"] = []
            for epoch in range(3):
                if epoch:
                    a.normal_()
                    b.normal_()
                reference = fn(a, b)
                record["accuracy"].append(compare(invoke(), reference, args.workloads))
                torch.cuda.synchronize()
            record["timing"] = timing(torch, invoke, args)
            record["post_accuracy"] = compare(invoke(), fn(a, b), args.workloads)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                invoke()
                torch.cuda.synchronize()
            events = [e for e in prof.events() if str(e.device_type).endswith("CUDA")]
            record["device_events"] = [{"name": e.name, "duration_us": e.device_time_total} for e in events]
            record["status"] = "PASS" if not any(
                item["mismatched_values"] for item in [*record["accuracy"], record["post_accuracy"]]) else "WRONG"
    except Exception:
        record.update(status="ERROR", error=traceback.format_exc())
    write_json(out / "result.json", record)
    print(json.dumps(record, ensure_ascii=False, default=str), flush=True)
    return 0 if record["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workloads", default="sum,rmsnorm,softmax,chain")
    parser.add_argument("--rows", default="1,4")
    parser.add_argument("--rnumels", default="32768,1048576")
    parser.add_argument("--strategies", default="plain,split,cooperative")
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--case-timeout", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "manifest.json", {"argv": sys.argv, "time_utc": datetime.now(timezone.utc).isoformat(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")})
    results = []
    for workload in args.workloads.split(","):
        for rows in args.rows.split(","):
            for rnumel in args.rnumels.split(","):
                for strategy in args.strategies.split(","):
                    if strategy == "hybrid" and workload != "sum":
                        continue
                    case = args.out / f"{workload}-x{rows}-r{rnumel}-{strategy}"
                    case.mkdir(parents=True, exist_ok=True)
                    if args.resume and (case / "result.json").exists():
                        record = json.loads((case / "result.json").read_text())
                    else:
                        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--out", str(case),
                                   "--workloads", workload, "--rows", rows, "--rnumels", rnumel,
                                   "--strategies", strategy, "--groups", str(args.groups), "--batch", str(args.batch),
                                   "--replays", str(args.replays), "--samples", str(args.samples)]
                        with (case / "run.log").open("w") as stream:
                            try:
                                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                                           timeout=args.case_timeout)
                                if (case / "result.json").exists():
                                    record = json.loads((case / "result.json").read_text())
                                else:
                                    record = {"status": "PROCESS_ERROR", "returncode": completed.returncode}
                            except subprocess.TimeoutExpired:
                                record = {"status": "TIMEOUT"}
                    record.update(workload=workload, strategy=strategy, X=int(rows), R=int(rnumel))
                    results.append(record)
                    write_json(args.out / "results.json", results)
                    print(json.dumps({k: record.get(k) for k in (
                        "workload", "strategy", "X", "R", "status", "timing", "source_summary")}), flush=True)
    print(dict(Counter(r["status"] for r in results)), flush=True)
    return 0 if all(r["status"] == "PASS" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
