"""Scan fp32 sum/amax and RS candidates on CUDA or A5, keeping isolated failure records."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

import barrier_suite as suite


DEFAULT_R = ",".join(str(4096 * 2**index) for index in range(9))


def check(torch, actual, expected, operation):
    actual = actual.detach().cpu().double()
    matching = actual == expected if operation == "amax" else torch.isclose(
        actual, expected, rtol=2e-4, atol=0.02)
    error = (actual - expected).abs()
    finite = torch.isfinite(error)
    return {"checked_values": actual.numel(), "mismatched_values": int((~matching).sum()),
            "nonfinite_values": int((~torch.isfinite(actual)).sum()),
            "max_finite_abs_error": float(error[finite].max()) if bool(finite.any()) else None,
            "atol": 0 if operation == "amax" else 0.02, "rtol": 0 if operation == "amax" else 2e-4}


def device_time(args, api, invoke, checkpoint):
    start, end = api.Event(enable_timing=True), api.Event(enable_timing=True)
    start.record()
    invoke()
    end.record()
    end.synchronize()
    pilot = start.elapsed_time(end) * 1000
    checkpoint("timing_pilot", {"pilot_us": pilot})
    slow = pilot >= 5000
    warmups = 1 if slow else 5
    for _ in range(warmups):
        invoke()
    api.synchronize()
    budget = max(1, int(args.max_sample_ms * 1000 / max(pilot, 1)))
    batch = min(args.batch, budget)
    replays = min(args.replays, max(1, budget // batch))
    graph = api.CUDAGraph() if args.backend == "cuda" else api.NPUGraph()
    with api.graph(graph):
        for _ in range(batch):
            invoke()
    graph_warmups = 1 if slow else 3
    for _ in range(graph_warmups):
        graph.replay()
    api.synchronize()
    samples = []
    for _ in range(args.samples):
        start, end = api.Event(enable_timing=True), api.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (batch * replays))
        checkpoint("timing_samples", {"samples_us": samples, "batch": batch, "replays": replays})
    return {"median_us": statistics.median(samples), "min_us": min(samples), "max_us": max(samples),
            "samples_us": samples, "batch": batch, "replays": replays, "sample_count": len(samples),
            "pilot_us": pilot, "warmups": warmups, "graph_warmups": graph_warmups,
            "max_sample_ms": args.max_sample_ms, "method": "Graph + device Events"}


def point_key(record):
    return tuple(record[field] for field in ("operation", "X", "R", "kind", "RS"))


def worker(args):
    args.mapping = "auto"
    args.phase = "matrix"
    args.kinds = [args.kind]
    torch, triton, api, args.options = suite.load_stack(args)
    torch.set_num_threads(4)
    info = suite.manifest(args, api)
    info.update(driver=Path(__file__).name, driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                torch_commit=torch.version.git_version, npu_device_limit=os.getenv("NPU_DEVICE_LIMIT"),
                scope="Handwritten (a+b).sum/amax; fixed BLOCK; all-read combine; barrier retained at RS=1")
    if args.backend == "npu":
        from triton.backends.ascend.driver import NPUUtils

        info["effective_core_count"] = NPUUtils().get_device_properties(0)["num_vectorcore"]
        info["runtime_device_limit"] = api.get_device_limit(0)
        info["runtime_stream_limit"] = api.get_stream_limit(api.current_stream())
        info["device_name"] = str(api.get_device_properties(0).name)
    else:
        info["effective_core_count"] = info["core_count"]
    suite.write_json(args.out / "manifest.json", info)
    exchange, reduce = suite.build_kernels()
    rows, count = int(args.rows), int(args.rnumels)
    operation = args.operations
    splits_list = list(map(int, args.splits.split(",")))
    _, output, expected, initial = suite.make_reduction(
        args, reduce, rows, count, 1, args.kind, operation=operation)
    left, right = initial[:2]
    records = json.loads((args.out / "candidates.json").read_text()) if (
        args.out / "candidates.json").exists() else []
    saved_left = saved_right = None
    for splits in splits_list:
        record = {"backend": args.backend, "device_name": info["device_name"],
                  "device_uuid": info.get("device_uuid"), "physical_device": args.physical_device,
                  "operation": operation, "dtype": "float32", "X": rows, "R": count, "RS": splits,
                  "kind": args.kind, "block": args.block, "grid": [splits, rows],
                  "core_count": info["core_count"], "effective_core_count": info["effective_core_count"],
                  "over_core": rows * splits > info["effective_core_count"],
                  "status": "RUNNING", "worker_pid": os.getpid(), "accuracy": []}

        def checkpoint(stage, extra=None):
            record["stage"] = stage
            if extra:
                record.update(extra)
            suite.write_json(args.out / "active.json", record)

        try:
            checkpoint("prepare")
            workspace = torch.full((rows, splits), float("nan"), device=args.backend)
            sem = torch.zeros(rows, dtype=torch.int32, device=args.backend)
            output.fill_(float("nan"))
            prepared = (output, expected, (left, right, workspace, sem))
            invoke, _, _, _ = suite.make_reduction(
                args, reduce, rows, count, splits, args.kind, operation=operation, prepared=prepared)
            checkpoint("compile_launch")
            compiled = invoke()
            checkpoint("first_synchronize")
            api.synchronize()
            record["accuracy"].append(check(torch, output, expected, operation))
            metadata = suite.metadata_dict(compiled)
            fields = ("auto_blockify_enabled", "auto_blockify_size", "program_grid_mapping_applied",
                      "program_grid_transforms", "launch_cooperative_grid", "compile_mode", "num_warps")
            record["metadata"] = {field: metadata.get(field) for field in fields}
            record["ir_sha256"] = {stage: hashlib.sha256(source.encode()).hexdigest()
                                   for stage in ("ttir", "ttadapter", "ptx")
                                   if isinstance(source := compiled.asm.get(stage), str)}
            checkpoint("timing")
            record.update(device_time(args, api, invoke, checkpoint))
            record["post_accuracy"] = check(torch, output, expected, operation)
            if saved_left is None:
                saved_left, saved_right = left.clone(), right.clone()
            for epoch in range(2):
                generator = torch.Generator().manual_seed(args.seed + epoch + 1)
                new_left = torch.randn((rows, count), generator=generator)
                new_right = torch.randn((rows, count), generator=generator)
                values = (new_left + new_right).double()
                reference = values.amax(-1) if operation == "amax" else values.sum(-1)
                left.copy_(new_left)
                right.copy_(new_right)
                output.fill_(float("nan"))
                checkpoint("changing_input")
                invoke()
                api.synchronize()
                record["accuracy"].append(check(torch, output, reference, operation))
                del new_left, new_right, values
            left.copy_(saved_left)
            right.copy_(saved_right)
            checkpoint("exchange_compile_launch")
            try:
                exchange_call, exchange_out, exchange_sem, exchange_buffers = suite.make_exchange(
                    args, exchange, splits, rows, 1, args.kind, phases=args.phases, epoch=11, skew=True)
                exchange_call()
                checkpoint("integer_exchange")
                api.synchronize()
                actual = exchange_out.cpu()
                wanted = suite.exchange_expected(splits, rows, args.phases, 11, True)
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
                if args.kind != "original":
                    assert bool((exchange_sem.cpu() == 123).all()), "Debug helper modified semaphore"
                record["exchange"] = {"max_abs_error": 0, "checked_values": actual.numel(),
                                      "final_sem": exchange_sem.cpu().tolist()}
            except AssertionError as error:
                record["exchange"] = {"error": str(error)}
            bad = any(item["mismatched_values"] for item in [*record["accuracy"], record["post_accuracy"]])
            bad = bad or "error" in record["exchange"]
            record["status"] = "WRONG" if bad else "PASS"
            checkpoint("complete")
        except Exception:
            record.update(status="ERROR", error=traceback.format_exc())
            records.append(record)
            suite.write_json(args.out / "candidates.json", records)
            print(json.dumps({key: record.get(key) for key in
                              ("operation", "X", "R", "RS", "kind", "status", "median_us")}), flush=True)
            # Start a fresh context after a launch/synchronization exception.
            return 1
        records.append(record)
        suite.write_json(args.out / "candidates.json", records)
        print(json.dumps({key: record.get(key) for key in
                          ("operation", "X", "R", "RS", "kind", "status", "median_us")}), flush=True)
    return 0


def collect_group(args, operation, rows, count, kind):
    directory = args.out / f"{operation}-x{rows}-r{count}-{kind}"
    directory.mkdir(exist_ok=True)
    result_path = directory / "candidates.json"
    previous = json.loads(result_path.read_text()) if args.resume and result_path.exists() else []
    if result_path.exists() and not args.resume:
        raise ValueError(f"{result_path} already exists; use --resume or a new output directory")
    records = previous
    while pending := [s for s in args.splits if s not in {r["RS"] for r in records}]:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--backend", args.backend,
                   "--out", str(directory), "--operations", operation, "--rows", str(rows),
                   "--rnumels", str(count), "--kind", kind, "--splits", ",".join(map(str, pending))]
        for name in ("physical_device", "block", "seed", "phases", "batch", "replays", "samples",
                     "op_timeout_ms", "max_sample_ms", "cuda_barrier"):
            if (value := getattr(args, name)) is not None:
                command.extend(["--" + name.replace("_", "-"), str(value)])
        attempt = len(records)
        environment = dict(os.environ)
        environment.setdefault("TRITON_CACHE_DIR", str(args.out / "triton_cache"))
        with (directory / f"worker-{attempt}.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=environment)
            started = time.monotonic()
            current, point_started, active, timeout = None, started, {}, False
            progress = None
            try:
                while process.poll() is None:
                    time.sleep(0.2)
                    active_path = directory / "active.json"
                    if active_path.exists():
                        candidate = json.loads(active_path.read_text())
                        if candidate.get("worker_pid") == process.pid:
                            active = candidate
                            current = active["RS"]
                    now = time.monotonic()
                    marker = (current, active.get("stage"), active.get("status"),
                              len(active.get("samples_us", [])), len(active.get("accuracy", [])))
                    if marker != progress:
                        progress, point_started = marker, now
                    compiling = active.get("stage") in ("compile_launch", "exchange_compile_launch")
                    limit = args.init_timeout if current is None or compiling else args.point_timeout
                    if active.get("status") in ("PASS", "WRONG", "ERROR"):
                        # Completed measurements remain valid even if backend teardown stalls.
                        limit = 15
                    if now - point_started > limit:
                        timeout = True
                        break
            finally:
                # Ctrl-C and unexpected parent exceptions must not leave an executing worker behind.
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
        records = json.loads(result_path.read_text()) if result_path.exists() else records
        if process.returncode and not active and not any(r["RS"] in pending for r in records):
            raise RuntimeError(f"worker initialization failed; see {directory / f'worker-{attempt}.log'}")
        if (timeout or process.returncode) and any(r["RS"] == current for r in records):
            if timeout:
                for record in records:
                    if record["RS"] == current:
                        record["worker_teardown_terminated"] = True
                suite.write_json(result_path, records)
        elif timeout or process.returncode:
            failed = active or {"operation": operation, "X": rows, "R": count, "kind": kind,
                                "RS": pending[0], "backend": args.backend, "dtype": "float32"}
            failed.update(status="TIMEOUT" if timeout else "PROCESS_ERROR", worker_exit_code=process.returncode,
                          point_timeout_seconds=limit, worker_log=f"worker-{attempt}.log")
            records.append(failed)
            suite.write_json(result_path, records)
        elif process.returncode == 0 and not any(r["RS"] == pending[0] for r in records):
            raise RuntimeError(f"worker returned no records: {directory}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "npu"), default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--physical-device", type=int)
    parser.add_argument("--operations", default="sum,amax")
    parser.add_argument("--rows", default="1,7,56")
    parser.add_argument("--rnumels", default=DEFAULT_R)
    parser.add_argument("--splits", help="default: 1,2,4,8,16,32,48,56; CUDA also includes 64")
    parser.add_argument("--kinds", help="NPU defaults to original,debug; CUDA uses original")
    parser.add_argument("--kind", choices=("original", "debug"), help=argparse.SUPPRESS)
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=213)
    parser.add_argument("--phases", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--max-sample-ms", type=float, default=50)
    parser.add_argument("--op-timeout-ms", type=int, default=10000)
    parser.add_argument("--point-timeout", type=float, default=60,
                        help="wall seconds without progress in execution/checking; compilation uses --init-timeout")
    parser.add_argument("--init-timeout", type=float, default=180)
    parser.add_argument("--cuda-barrier", choices=("snapshot", "installed"), default="snapshot")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.splits = args.splits or ("1,2,4,8,16,32,48,56" + (",64" if args.backend == "cuda" else ""))
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.backend == "npu" and args.physical_device is None:
        parser.error("NPU requires --physical-device")
    for field in ("block", "phases", "batch", "replays", "samples", "max_sample_ms", "point_timeout",
                  "init_timeout", "op_timeout_ms"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.worker:
        return worker(args)
    args.operations = args.operations.split(",")
    if set(args.operations) - {"sum", "amax"}:
        parser.error("--operations supports sum,amax")
    for field in ("rows", "rnumels", "splits"):
        values = list(dict.fromkeys(map(int, getattr(args, field).split(","))))
        if not values or min(values) < 1:
            parser.error(f"--{field} requires positive integers")
        setattr(args, field, values)
    args.kinds = (args.kinds or ("original,debug" if args.backend == "npu" else "original")).split(",")
    if set(args.kinds) - ({"original", "debug"} if args.backend == "npu" else {"original"}):
        parser.error("CUDA supports original; NPU supports original,debug")
    run = {"argv": sys.argv, "backend": args.backend, "operations": args.operations, "rows": args.rows,
           "rnumels": args.rnumels, "splits": args.splits, "kinds": args.kinds, "dtype": "float32",
           "block": args.block, "seed": args.seed, "batch": args.batch, "replays": args.replays,
           "samples": args.samples, "phases": args.phases, "max_sample_ms": args.max_sample_ms,
           "point_timeout_seconds": args.point_timeout, "script_sha256": hashlib.sha256(
               Path(__file__).read_bytes()).hexdigest(), "unix_time": time.time()}
    old_path = args.out / "run.json"
    if args.resume and old_path.exists():
        old = json.loads(old_path.read_text())
        if any(old[key] != run[key] for key in ("backend", "block", "seed", "batch", "replays", "samples")):
            parser.error("--resume changes the measurement configuration; use a new output directory")
    suite.write_json(old_path, run)
    records = []
    for operation in args.operations:
        for rows in args.rows:
            for count in args.rnumels:
                for kind in args.kinds:
                    group = collect_group(args, operation, rows, count, kind)
                    records.extend(group)
                    suite.write_json(args.out / "candidates.json", records)
                    best = min((r for r in group if r["status"] == "PASS"),
                               key=lambda r: r["median_us"], default=None)
                    print(json.dumps({"operation": operation, "X": rows, "R": count, "kind": kind,
                                      "counts": dict(Counter(r["status"] for r in group)),
                                      "best_RS": best["RS"] if best else None,
                                      "best_us": best["median_us"] if best else None}), flush=True)
    print(dict(Counter(r["status"] for r in records)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
