"""Compare barriers across output occupancy, split count, and effective core limits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

import barrier_suite as suite


def build_plain_kernel():
    @triton.jit
    def plain(A, B, Out, R: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        acc = tl.full((BLOCK,), 0, tl.float32)
        for begin in range(0, R, BLOCK):
            index = begin + offsets
            a = tl.load(A + row * R + index, index < R, other=0).to(tl.float32)
            b = tl.load(B + row * R + index, index < R, other=0).to(tl.float32)
            acc += a + b
        tl.store(Out + row, tl.sum(acc, 0))

    return plain


def compare(actual, expected, exact=False):
    actual, expected = actual.cpu(), expected.cpu()
    matching = actual == expected if exact else torch.isclose(actual.double(), expected, rtol=2e-4, atol=2e-2)
    bad = (~matching).nonzero()
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    difference = (actual.double() - expected.double()).abs()

    def value(number):
        item = number.item()
        return item if math.isfinite(item) else str(item)

    return {"checked_values": actual.numel(), "mismatched_values": len(bad),
            "nonfinite_values": int((~torch.isfinite(actual)).sum()),
            "max_finite_abs_error": difference[finite].max().item() if bool(finite.any()) else None,
            "examples": [{"index": index.tolist(), "actual": value(actual[tuple(index.tolist())]),
                          "expected": value(expected[tuple(index.tolist())])} for index in bad[:8]],
            "rtol": 0 if exact else 2e-4, "atol": 0 if exact else 2e-2}


def mapping_metadata(compiled):
    metadata = suite.metadata_dict(compiled)
    fields = ("auto_blockify_enabled", "auto_blockify_size", "program_grid_mapping_applied",
              "program_grid_transforms", "launch_cooperative_grid", "compile_mode", "mix_mode")
    return {key: metadata.get(key) for key in fields}


def isolated_case(args, record):
    launch_manifest = record["run_manifest"]
    directory = args.out / "cases" / f"x{record['X']}-r{record['R']}-s{record['S']}-{record['kind']}"
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--backend", args.backend,
               "--out", str(directory), "--rows", str(record["X"]), "--rnumels", str(record["R"]),
               "--splits", str(record["S"]), "--only-kind", record["kind"], "--case-worker",
               "--allow-over-core", "--cuda-barrier", args.cuda_barrier]
    for name in ("physical_device", "block", "seed", "phases", "batch", "replays", "samples", "op_timeout_ms"):
        value = getattr(args, name)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    if args.time_wrong_results:
        command.append("--time-wrong-results")
    if args.accuracy_repeats:
        command.extend(["--accuracy-repeats", str(args.accuracy_repeats)])
    if args.save_kernels:
        command.append("--save-kernels")
    if args.max_sample_ms is not None:
        command.extend(["--max-sample-ms", str(args.max_sample_ms)])
    # Reuse compiled kernels, while keeping each risky launch in its own device context.
    environment = dict(os.environ, TRITON_CACHE_DIR=str(args.out / "triton_cache"))
    with (directory / "run.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=args.case_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    result_path = directory / "output-axis.json"
    checkpoint = directory / "active-case.json"
    if result_path.exists():
        record.update(json.loads(result_path.read_text())[-1])
    elif checkpoint.exists():
        record.update(json.loads(checkpoint.read_text()))
    record.update(isolated=True, run_manifest=launch_manifest, worker_exit_code=process.returncode,
                  worker_manifest=str((directory / "manifest.json").relative_to(args.out)),
                  worker_log=str((directory / "run.log").relative_to(args.out)))
    if timed_out:
        record.update(status="FAIL_TIMEOUT", timeout_seconds=args.case_timeout)
    elif process.returncode and record.get("status", "RUNNING") in ("RUNNING", "PASS"):
        record.update(status="FAIL_PROCESS")
    elif "status" not in record or record["status"] == "RUNNING":
        record.update(status="FAIL_PROCESS")
    return record


def measure_case(args, api, exchange, reduce, plain, record, prepared):
    output, expected, buffers = prepared
    left, right, workspace, sem = buffers
    rows, count, splits, kind = (record[key] for key in ("X", "R", "S", "kind"))

    def checkpoint(stage):
        record["execution_stage"] = stage
        suite.write_json(args.out / "active-case.json", record)

    record.update(status="RUNNING", timing_status="NOT_MEASURED")
    try:
        checkpoint("sum_prepare")
        workspace.fill_(float("nan"))
        output.fill_(float("nan"))
        sem.zero_()
        api.synchronize()
        if kind == "plain":
            def invoke():
                return plain[(rows,)](left, right, output, R=count, BLOCK=args.block, **args.options)
        else:
            invoke, _, _, _ = suite.make_reduction(args, reduce, rows, count, splits, kind, prepared=prepared)
        checkpoint("sum_launch")
        compiled = invoke()
        api.synchronize()
        record["accuracy"] = compare(output, expected)
        record["metadata"] = mapping_metadata(compiled)
        stem = f"x{rows}-r{count}-s{splits}-{kind}"
        if args.save_kernels:
            suite.save_kernel(args, compiled, "sum-" + stem)
        if args.time_wrong_results or not record["accuracy"]["mismatched_values"]:
            checkpoint("sum_timing")
            record.update(suite.device_time(args, api, invoke), timing_status="MEASURED")
            record["post_accuracy"] = compare(output, expected)
        if args.accuracy_repeats:
            record["changing_input_accuracy"] = []
            saved_left, saved_right = left.clone(), right.clone()
            try:
                for epoch in range(args.accuracy_repeats):
                    generator = torch.Generator().manual_seed(args.seed + epoch + 1)
                    new_left = torch.randn((rows, count), generator=generator, dtype=torch.float32)
                    new_right = torch.randn((rows, count), generator=generator, dtype=torch.float32)
                    reference = (new_left + new_right).double().sum(-1)
                    left.copy_(new_left)
                    right.copy_(new_right)
                    output.fill_(float("nan"))
                    # Keep partials and semaphore from the preceding call to expose stale reads.
                    checkpoint("changing_input_launch")
                    invoke()
                    api.synchronize()
                    record["changing_input_accuracy"].append(dict(compare(output, reference), epoch=epoch))
            finally:
                left.copy_(saved_left)
                right.copy_(saved_right)
                api.synchronize()
        checkpoint("sum_complete")
        if kind != "plain":
            call, exchange_out, exchange_sem, _ = suite.make_exchange(
                args, exchange, splits, rows, 1, kind, phases=args.phases, epoch=11, skew=True)
            checkpoint("exchange_launch")
            exchange_compiled = call()
            api.synchronize()
            reference = suite.exchange_expected(splits, rows, args.phases, 11, True)
            record["exchange"] = compare(exchange_out, reference, exact=True)
            record["exchange"]["phases"] = args.phases
            if kind != "original":
                assert bool((exchange_sem.cpu() == 123).all())
            if args.save_kernels:
                suite.save_kernel(args, exchange_compiled, "exchange-" + stem)
            elif count == args.rnumels[0] and (rows, splits, kind) in (
                (7, 8, "native"), (7, 16, "native"), (56, 2, "native"),
                (56, 16, "native"), (56, 16, "debug"), (56, 1, "original")):
                stem = f"x{rows}-s{splits}-{kind}"
                suite.save_kernel(args, compiled, "sum-" + stem)
                suite.save_kernel(args, exchange_compiled, "exchange-" + stem)
        failed = any(record.get(key, {}).get("mismatched_values", 0)
                     for key in ("accuracy", "post_accuracy", "exchange"))
        failed = failed or any(item["mismatched_values"] for item in record.get("changing_input_accuracy", []))
        record["status"] = "FAIL_WRONG_RESULT" if failed else "PASS"
        record["timing_reference_only"] = bool(failed or record["diagnostic_over_core"])
        record["buffer_addresses"] = [t.data_ptr() for t in buffers] + [output.data_ptr()]
        checkpoint("complete")
    except Exception:
        record.update(status="FAIL_EXECUTION", error=traceback.format_exc())
        suite.write_json(args.out / "active-case.json", record)
    return record


def run(args, api, info):
    exchange, reduce = suite.build_kernels()
    plain = build_plain_kernel()
    records = args.previous_records
    completed = {tuple(record[key] for key in ("X", "R", "S", "kind")) for record in records}
    for rows in args.rows:
        for count in args.rnumels:
            _, output, expected, initial = suite.make_reduction(args, reduce, rows, count, 1, "original")
            left, right = initial[:2]
            for splits in args.splits:
                workspace = torch.full((rows, splits), float("nan"), device=args.backend)
                sem = torch.zeros(rows, dtype=torch.int32, device=args.backend)
                prepared = (output, expected, (left, right, workspace, sem))
                kinds = [args.only_kind] if args.only_kind else (["plain"] if splits == 1 else []) + args.kinds
                shift = args.splits.index(splits) % len(kinds)
                for kind in kinds[shift:] + kinds[:shift]:
                    key = (rows, count, splits, kind)
                    if key in completed:
                        continue
                    record = {"backend": args.backend, "device_name": info.get("device_name", "NPU"),
                              "X": rows, "R": count, "S": splits, "kind": kind,
                              "output_axis_full": rows == info["core_count"],
                              "program_count": rows * splits, "core_count": info["core_count"],
                              "physical_core_count": info.get("physical_core_count", info["core_count"]),
                              "npu_device_limit": info.get("npu_device_limit"),
                              "diagnostic_over_core": rows * splits > info["core_count"],
                              "block": args.block, "dtype": "float32", "run_manifest": args.run_manifest}
                    if kind == "original" and record["diagnostic_over_core"] and not args.allow_over_core:
                        record.update(status="SKIP_CORE_BUDGET",
                                      reason="Upstream cooperative candidates bound the grid by the SM budget")
                    elif record["diagnostic_over_core"] and not args.case_worker:
                        record = isolated_case(args, record)
                    else:
                        record = measure_case(args, api, exchange, reduce, plain, record, prepared)
                    records.append(record)
                    completed.add(key)
                    suite.write_json(args.out / "output-axis.json", records)
                    print(json.dumps({key: record[key] for key in
                                      ("X", "R", "S", "kind", "status")}
                                     | {"median_us": record.get("median_us")}), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("npu", "cuda"), default="cuda")
    parser.add_argument("--physical-device", type=int, help="required for NPU; optional CUDA physical-ID annotation")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", default="auto", help="comma-separated X; auto uses 7,56,SMs on CUDA, 7,cores on NPU")
    parser.add_argument("--cuda-barrier", choices=("snapshot", "installed"), default="snapshot",
                        help="CUDA helper: fixed v2.13.0 snapshot or installed Inductor helper")
    parser.add_argument("--rnumels", default="32768,1048593")
    parser.add_argument("--splits", default="1,2,4,8,16,32,48,56")
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=213)
    parser.add_argument("--phases", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--op-timeout-ms", type=int, default=10000)
    parser.add_argument("--allow-over-core", action="store_true",
                        help="launch over-budget original cases in isolated subprocesses")
    parser.add_argument("--time-wrong-results", action="store_true", help="retain timings when sum accuracy fails")
    parser.add_argument("--accuracy-repeats", type=int, default=0,
                        help="check new random inputs with the same workspace after timing")
    parser.add_argument("--save-kernels", action="store_true", help="save every compiled sum/exchange and metadata")
    parser.add_argument("--case-timeout", type=float, default=120, help="wall seconds per isolated over-budget case")
    parser.add_argument("--max-sample-ms", type=float, help="reduce graph repetitions for very slow kernels")
    parser.add_argument("--resume", action="store_true", help="retain completed cases and run remaining configurations")
    parser.add_argument("--only-kind", choices=("plain", "original", "native", "debug"))
    parser.add_argument("--case-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    args.previous_records = []
    previous_path = args.out / "output-axis.json"
    previous_manifest = args.out / "manifest.json"
    if args.resume and previous_path.exists():
        previous_info = json.loads(previous_manifest.read_text())
        if previous_info["backend"] != args.backend:
            parser.error("--resume backend differs from the existing run")
        if args.backend == "npu" and previous_info.get("npu_device_limit") != os.getenv("NPU_DEVICE_LIMIT"):
            parser.error("--resume NPU_DEVICE_LIMIT differs from the existing run; use a separate output directory")
        stamp = previous_info["time_utc"].replace(":", "").replace("+", "_")
        manifest_name = f"manifest-run-{stamp}.json"
        suite.write_json(args.out / manifest_name, previous_info)
        args.previous_records = json.loads(previous_path.read_text())
        args.previous_records = [record for record in args.previous_records
                                 if not (args.allow_over_core and record["status"] == "SKIP_CORE_BUDGET")
                                 and not (args.time_wrong_results and record["status"] == "FAIL_WRONG_RESULT"
                                          and "median_us" not in record)]
        for record in args.previous_records:
            record.setdefault("run_manifest", manifest_name)
    if args.backend == "npu" and args.physical_device is None:
        parser.error("--physical-device is required for NPU mapping verification")
    if args.backend == "cuda" and args.only_kind in ("native", "debug"):
        parser.error("CUDA supports plain/original only")
    if args.case_timeout <= 0:
        parser.error("--case-timeout must be positive")
    if args.max_sample_ms is not None and args.max_sample_ms <= 0:
        parser.error("--max-sample-ms must be positive")
    for field in ("rnumels", "splits") + (() if args.rows == "auto" else ("rows",)):
        try:
            setattr(args, field, list(dict.fromkeys(map(int, getattr(args, field).split(",")))))
        except ValueError:
            parser.error(f"--{field} requires comma-separated positive integers")
        if not getattr(args, field) or min(getattr(args, field)) < 1:
            parser.error(f"--{field} requires positive integers")
    for field in ("block", "phases", "batch", "replays", "samples"):
        if getattr(args, field) < 1:
            parser.error(f"--{field} must be positive")
    if args.accuracy_repeats < 0:
        parser.error("--accuracy-repeats must be nonnegative")
    args.kinds = ["original", "native", "debug"] if args.backend == "npu" else ["original"]
    args.mapping, args.phase = "auto", "output-axis"
    torch_module, triton_module, api, args.options = suite.load_stack(args)
    globals().update(torch=torch_module, triton=triton_module, tl=suite.tl)
    info = suite.manifest(args, api)
    if args.backend == "npu":
        from triton.backends.ascend.driver import NPUUtils

        info["physical_core_count"] = info["core_count"]
        info["npu_device_limit"] = os.environ.get("NPU_DEVICE_LIMIT")
        info["triton_device_properties"] = NPUUtils().get_device_properties(0)
        info["core_count"] = info["triton_device_properties"]["num_vectorcore"]
        info["runtime_device_limit"] = api.get_device_limit(0)
        info["runtime_stream_limit"] = api.get_stream_limit(api.current_stream())
    if args.rows == "auto":
        args.rows = sorted({7, info["core_count"]} | ({56} if args.backend == "cuda" else set()))
    info["rows"] = args.rows
    info["rnumels"], info["splits"] = args.rnumels, args.splits
    info["block"], info["seed"], info["phases"] = args.block, args.seed, args.phases
    info["accuracy_repeats"], info["save_kernels"] = args.accuracy_repeats, args.save_kernels
    info["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    info["driver"] = Path(__file__).name
    info["scope"] = "Output-axis occupancy; over-budget/incorrect timings are diagnostic references"
    info["allow_over_core"], info["time_wrong_results"] = args.allow_over_core, args.time_wrong_results
    info["case_timeout_seconds"] = args.case_timeout
    info["max_sample_ms"] = args.max_sample_ms
    info["resumed_records"] = len(args.previous_records)
    stamp = info["time_utc"].replace(":", "").replace("+", "_")
    args.run_manifest = f"manifest-run-{stamp}.json"
    bundle = Path(__file__).with_name("bundle_manifest.json")
    if bundle.exists():
        info["bundle_source_commit"] = json.loads(bundle.read_text())["source_commit"]
        info["bundle_manifest_sha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
    suite.write_json(args.out / "manifest.json", info)
    suite.write_json(args.out / args.run_manifest, info)
    print(json.dumps({key: info.get(key) for key in
                      ("backend", "device_name", "device_uuid", "core_count", "rows", "cuda_barrier")}), flush=True)
    records = run(args, api, info)
    print(json.dumps({"counts": {status: sum(r["status"] == status for r in records)
                                 for status in sorted({r["status"] for r in records})}}), flush=True)
    if args.backend == "cuda" and any(r["status"] != "PASS" and not r["status"].startswith("SKIP_") for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
