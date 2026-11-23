"""Run compiled CUDA/NPU reductions and tuned NPU DSL, with one serial queue per device."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace

import barrier_suite as suite
import reduction_matrix as matrix


HERE = Path(__file__).resolve().parent
SCHEMA = "cooperative-compile-matrix-v1"
STOP = threading.Event()
PRINT_LOCK = threading.Lock()


def message(text):
    with PRINT_LOCK:
        print(text, flush=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numbers(value):
    result = list(dict.fromkeys(int(item) for item in value.split(",")))
    if not result or min(result) <= 0:
        raise ValueError("lists must contain positive integers")
    return result


def tool_path():
    bundled = HERE / "autotune_tiling.py"
    return bundled if bundled.exists() else HERE.parents[1] / "tools" / "autotune_tiling.py"


def source_hashes():
    paths = [HERE / name for name in ("run_matrix.py", "reduction_matrix.py", "barrier_suite.py",
                                      "npu_triton_helpers.py", "cuda_triton_helpers.py")]
    paths.append(tool_path())
    return {path.name: sha256(path) for path in paths}


def device_environment(backend, selector, cache):
    environment = dict(os.environ)
    variable = "CUDA_VISIBLE_DEVICES" if backend == "cuda" else "ASCEND_RT_VISIBLE_DEVICES"
    environment[variable] = str(selector)
    environment["TRITON_CACHE_DIR"] = str(cache / "triton")
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    environment.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    environment.setdefault("OMP_NUM_THREADS", "4")
    return environment


def initialize(backend):
    import torch

    if backend == "npu":
        import torch_npu  # noqa: F401

    api = getattr(torch, backend)
    if not api.is_available():
        raise RuntimeError(f"{backend} is unavailable in this Python environment")
    api.set_device(0)
    torch.set_num_threads(4)
    return torch, api


def inventory(job):
    import torch
    from torch._inductor import config

    for name in ("cooperative_reductions", "force_cooperative_reductions"):
        if not hasattr(config.triton, name):
            raise RuntimeError(f"This PyTorch build lacks triton.{name}; use a compatible PyTorch 2.13 build")

    if job["backend"] == "cuda":
        visible = []
        for index in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(index)
            visible.append({"logical_device": index, "uuid": str(prop.uuid), "name": prop.name,
                            "core_count": prop.multi_processor_count})
        selected = []
        for selector in job["selectors"]:
            if selector.isdecimal():
                index = int(selector)
                if index >= len(visible):
                    raise ValueError(f"CUDA logical device {index} is outside visible range 0..{len(visible)-1}")
                device = dict(visible[index])
            else:
                matches = [device for device in visible if device["uuid"] == selector]
                if not matches:
                    raise ValueError(f"UUID is not currently visible: {selector}")
                device = dict(matches[0])
            torch.cuda.set_device(device["logical_device"])
            device["free_bytes"], device["total_bytes"] = torch.cuda.mem_get_info()
            device.update(selector=device["uuid"], requested_selector=selector, physical_device=None)
            selected.append(device)
        if not selected:
            raise RuntimeError("No CUDA devices selected")
        return selected
    torch, api = initialize("npu")
    import ctypes
    from triton.backends.ascend.driver import NPUUtils

    physical = ctypes.c_int32(-1)
    query = ctypes.CDLL("libascendcl.so").aclrtGetPhyDevIdByUserDevId
    query.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
    query.restype = ctypes.c_int32
    status = query(0, ctypes.byref(physical))
    if status or str(physical.value) != job["selectors"][0]:
        raise RuntimeError(f"NPU physical mapping mismatch: status={status}, physical={physical.value}")
    prop = api.get_device_properties(0)
    return [{"selector": str(physical.value), "physical_device": physical.value,
             "uuid": f"npu-physical-{physical.value}", "name": str(prop.name),
             "core_count": prop.vector_core_num,
             "effective_core_count": NPUUtils().get_device_properties(0)["num_vectorcore"],
             "device_limit": api.get_device_limit(0), "stream_limit": api.get_stream_limit(api.current_stream())}]


def compile_worker(job, out, checkpoint):
    torch, api = initialize(job["backend"])
    if job["backend"] == "npu":
        api.set_op_timeout_ms(job["op_timeout_ms"])
    import triton
    from torch._inductor import config, metrics
    from torch._inductor.utils import run_and_get_code

    sys.path.insert(0, str(tool_path().parent))
    from autotune_tiling import BestTilingRecorder, _config_payload

    class Recorder(BestTilingRecorder):
        def __init__(self):
            super().__init__(job["backend"])
            self.artifacts = []
            self.saved = set()

        def save(self, autotuner):
            if id(autotuner) in self.saved:
                return
            launcher = getattr(autotuner, "best_launcher", None)
            if launcher is None:
                launchers = getattr(autotuner, "launchers", ())
                launcher = launchers[0] if len(launchers) == 1 else None
            kernel = getattr(launcher, "bin", None)
            if kernel is None:
                return
            self.saved.add(id(autotuner))
            name = f"kernel-{len(self.artifacts)}"
            suite.save_kernel(SimpleNamespace(out=out), kernel, name)
            fn = getattr(autotuner, "fn", None)
            source = getattr(fn, "src", None)
            if isinstance(source, str):
                (out / (name + ".py")).write_text(source)
            results = getattr(autotuner, "compile_results", ()) or ()
            self.artifacts.append({"name": name, "compiled_config_count": len(results),
                                   "compiled_configs": [_config_payload(item.config) for item in results],
                                   "metadata": suite.metadata_dict(kernel)})

        def _record_standard(self, autotuner):
            super()._record_standard(autotuner)
            self.save(autotuner)

        def _record_grouped(self, autotuner, args):
            super()._record_grouped(autotuner, args)
            self.save(autotuner)

    operation, rows, count = job["operation"], job["X"], job["R"]
    kind = job["kind"]
    cooperative = kind in ("cooperative", "natural")
    options = {"split_reductions": False, "triton.cooperative_reductions": cooperative,
               "triton.force_cooperative_reductions": kind == "cooperative", "max_autotune": True,
               "triton.cudagraphs": False, "triton.multi_kernel": 0,
               "fx_graph_cache": False, "autotune_local_cache": False}
    record = {key: job[key] for key in ("backend", "operation", "X", "R", "kind", "device")}
    record.update(entry="inductor", dtype="float32", config=options,
                  force_cooperative=kind == "cooperative", status="RUNNING",
                  versions={"torch": torch.__version__, "torch_commit": torch.version.git_version,
                            "triton": triton.__version__, "cuda": torch.version.cuda}, accuracy=[])
    for name in ("torch._inductor.choices", "torch._inductor.ir", "torch._inductor.codegen.triton",
                 *( ("torch_npu._inductor.codegen.triton", "torch_npu._inductor.runtime.triton_heuristics")
                    if job["backend"] == "npu" else ())):
        module = importlib.import_module(name)
        record.setdefault("source_sha256", {})[name] = sha256(inspect.getfile(module))
    generator = torch.Generator().manual_seed(job["seed"])
    left_cpu = torch.randn((rows, count), generator=generator)
    right_cpu = torch.randn((rows, count), generator=generator)
    values = (left_cpu + right_cpu).double()
    reference = values.amax(-1) if operation == "amax" else values.sum(-1)
    left, right = left_cpu.to(job["backend"]), right_cpu.to(job["backend"])
    del left_cpu, right_cpu, values

    def fn(a, b):
        return (a + b).amax(-1) if operation == "amax" else (a + b).sum(-1)

    recorder = Recorder()
    recorder.install()
    recorder.start_capture()
    try:
        with config.patch(options), torch.no_grad():
            metrics.reset()
            checkpoint("compile_launch")
            begin = time.monotonic()
            compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False)
            first, sources = run_and_get_code(compiled, left, right)
            api.synchronize()
            record["compile_autotune_s"] = time.monotonic() - begin
            for index, source in enumerate(sources):
                (out / f"output_code_{index}.py").write_text(source)
            record["selected_configs"] = recorder.stop_capture()
            recorder.uninstall()
            record["kernels"] = recorder.artifacts
            record["generated_kernel_count"] = metrics.generated_kernel_count
            text = "\n".join(sources)
            record["source_summary"] = {
                "cooperative_decorators": text.count("@triton_heuristics.cooperative_reduction("),
                "barrier_calls": text.count("triton_helpers.x_grid_barrier("),
                "run_calls": len(re.findall(r"\w+\.run\(", text.split("def benchmark_compiled_module")[0]))}
            splits = [item["selected_config"].get("RSPLIT") for item in record["selected_configs"]]
            splits = [int(value) for value in splits if value is not None]
            record["RS"] = max(splits) if splits else 1
            has_coop = bool(record["source_summary"]["cooperative_decorators"])
            record["actual_path"] = ("cooperative" if has_coop and record["RS"] > 1 else
                                     "unsplit" if has_coop else "ordinary" if record["kernels"] else "external")
            record["accuracy"].append(matrix.check(torch, first, reference, operation))
            last_output = [first]

            def invoke():
                last_output[0] = compiled(left, right)
                return last_output[0]

            settings = SimpleNamespace(backend=job["backend"], **job["timing"])
            checkpoint("timing")
            record.update(matrix.device_time(settings, api, invoke, checkpoint))
            # invoke() last ran during capture: this tensor is written by the last Graph replay.
            record["post_accuracy"] = matrix.check(torch, last_output[0], reference, operation)
            for epoch in range(2):
                checkpoint("changing_input", {"epoch": epoch})
                generator = torch.Generator().manual_seed(job["seed"] + epoch + 1)
                a = torch.randn((rows, count), generator=generator)
                b = torch.randn((rows, count), generator=generator)
                value = (a + b).double()
                expected = value.amax(-1) if operation == "amax" else value.sum(-1)
                left.copy_(a)
                right.copy_(b)
                result = invoke()
                api.synchronize()
                record["accuracy"].append(matrix.check(torch, result, expected, operation))
            checks = [*record["accuracy"], record["post_accuracy"]]
            record["status"] = "WRONG" if any(item["mismatched_values"] for item in checks) else "PASS"
    finally:
        recorder.uninstall()
    return record


def internal_worker(args):
    job = json.loads(args.job.read_text())
    out = args.job.parent

    def checkpoint(stage, extra=None):
        suite.write_json(out / "active.json", {"stage": stage, **(extra or {})})

    try:
        checkpoint("initialize")
        if args.worker == "inventory":
            record = {"status": "PASS", "devices": inventory(job)}
        else:
            record = compile_worker(job, out, checkpoint)
    except Exception:
        record = {"status": "ERROR", "error": traceback.format_exc()}
        print(record["error"], flush=True)
    suite.write_json(out / "result.json", record)
    return 0 if record["status"] == "PASS" else 1


def launch_job(args, directory, job, environment, mode="compile"):
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if args.resume and result_path.exists():
        return json.loads(result_path.read_text())
    for name in ("result.json", "active.json"):
        (directory / name).unlink(missing_ok=True)
    suite.write_json(directory / "job.json", job)
    command = [sys.executable, "-u", str(HERE / "run_matrix.py"), "--worker", mode,
               "--job", str(directory / "job.json")]
    with (directory / "run.log").open("w") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        changed = time.monotonic()
        marker = None
        timed_out = False
        try:
            while process.poll() is None:
                if STOP.is_set():
                    raise InterruptedError("matrix execution interrupted")
                active_path = directory / "active.json"
                active = json.loads(active_path.read_text()) if active_path.exists() else {}
                current = json.dumps(active, sort_keys=True)
                if current != marker:
                    marker, changed = current, time.monotonic()
                compiling = active.get("stage", "initialize") in ("initialize", "compile_launch")
                limit = args.init_timeout if compiling else args.point_timeout
                if result_path.exists():
                    limit = 15
                if time.monotonic() - changed > limit:
                    timed_out = True
                    break
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    if result_path.exists():
        return json.loads(result_path.read_text())
    record = {"status": "TIMEOUT" if timed_out else "PROCESS_ERROR",
              "worker_exit_code": process.returncode, "error": f"See {directory / 'run.log'}"}
    suite.write_json(result_path, record)
    return record


def compiled_case(args, directory, device, environment, operation, rows, count, kind):
    job = {"backend": args.backend, "operation": operation, "X": rows, "R": count,
           "kind": kind, "device": device, "seed": args.seed,
           "op_timeout_ms": args.op_timeout_ms,
           "timing": {name: getattr(args, name) for name in
                      ("batch", "replays", "samples", "max_sample_ms")}}
    result = launch_job(args, directory, job, environment)
    result = {**job, **result, "artifact_dir": str(directory.relative_to(args.out))}
    result.pop("timing", None)
    return result


def dsl_group(args, directory, device, environment, operation, rows, count, kind, block, splits, samples):
    directory.mkdir(parents=True, exist_ok=True)
    settings = SimpleNamespace(backend="npu", out=directory, physical_device=device["physical_device"],
                               block=block, splits=splits, seed=args.seed, phases=4, batch=args.batch,
                               replays=args.replays, samples=samples, max_sample_ms=args.max_sample_ms,
                               op_timeout_ms=args.op_timeout_ms, cuda_barrier="installed", resume=True,
                               init_timeout=args.init_timeout, point_timeout=args.point_timeout,
                               save_kernels=True, worker_environment=environment, stop_event=STOP)
    result = matrix.collect_group(settings, operation, rows, count, kind)
    for item in result:
        item.update(entry="dsl", device=device, actual_path="cooperative" if item["RS"] > 1 else "unsplit",
                    artifact_dir=str((directory / f"{operation}-x{rows}-r{count}-{kind}").relative_to(args.out)))
        metadata = item.get("metadata", {})
        single_wave = rows * item["RS"] <= item.get("effective_core_count", device["effective_core_count"])
        item["single_wave"] = single_wave
        item["integration_candidate"] = bool(kind == "debug" and single_wave and item["RS"] > 1 and
                                              item["status"] == "PASS" and
                                              metadata.get("auto_blockify_enabled") is True)
        item["integration_note"] = ("A5 SIMD single-wave candidate; backend contract still required" if
                                    item["integration_candidate"] else
                                    "No R partition" if item["RS"] == 1 else
                                    "Semaphore mapping/residency needs confirmation" if kind == "original" else
                                    "Outside first-stage contract or checks failed")
    return result


def aggregate(records, kind, device, operation, rows, count):
    base = {"kind": kind, "device": device, "operation": operation, "X": rows, "R": count,
            "backend": records[0].get("backend") if records else None, "rounds": records}
    if not records or any(record["status"] != "PASS" for record in records):
        return {**base, "status": "NO_VALID_CONFIG"}
    samples = [record["median_us"] for record in records]
    result = {**records[-1], **base, "status": "PASS", "median_us": statistics.median(samples),
              "min_us": min(record["min_us"] for record in records),
              "max_us": max(record["max_us"] for record in records), "round_medians_us": samples}
    result["configs_by_round"] = [record.get("selected_configs", []) for record in records]
    result["selection_varied"] = len({json.dumps(value, sort_keys=True) for value in result["configs_by_round"]}) > 1
    return result


def run_shape(args, device, environment, operation, rows, count):
    directory = args.out / "cases" / f"{operation}-x{rows}-r{count}"
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / "summary.json"
    if args.resume and final.exists():
        return json.loads(final.read_text())
    message(f"[{device['selector']}] {operation} X={rows} R={count}")
    rounds = {"plain": []}
    raw = []
    natural = []
    if args.backend == "cuda":
        rounds["cooperative"] = []
        for repeat in range(args.confirm_rounds):
            kinds = ["plain", "cooperative"] if repeat % 2 == 0 else ["cooperative", "plain"]
            for kind in kinds:
                record = compiled_case(args, directory / f"round-{repeat}" / kind, device, environment,
                                       operation, rows, count, kind)
                rounds[kind].append(record)
                raw.append(record)
        if not args.no_natural:
            natural.append(compiled_case(args, directory / "natural", device, environment,
                                         operation, rows, count, "natural"))
    else:
        # Save a compiled baseline before exploring DSL configurations.
        first = compiled_case(args, directory / "round-0" / "plain", device, environment,
                              operation, rows, count, "plain")
        rounds["plain"].append(first)
        raw.append(first)
        shortlists = {}
        for kind in ("original", "debug"):
            candidates = []
            for block in args.blocks:
                candidates.extend(dsl_group(args, directory / "search" / f"block-{block}", device, environment,
                                            operation, rows, count, kind, block, args.splits, args.tune_samples))
            raw.extend(candidates)
            correct = sorted((item for item in candidates if item["status"] == "PASS" and item["RS"] > 1),
                             key=lambda item: item["median_us"])
            shortlists[kind] = correct[:args.topk]
            legal = next((item for item in correct if item.get("integration_candidate")), None)
            if legal and not any((item["RS"], item["block"]) == (legal["RS"], legal["block"])
                                 for item in shortlists[kind]):
                shortlists[kind].append(legal)
        candidate_rounds = {(kind, item["block"], item["RS"]): []
                            for kind, items in shortlists.items() for item in items}
        for repeat in range(args.confirm_rounds):
            if repeat:
                record = compiled_case(args, directory / f"round-{repeat}" / "plain", device, environment,
                                       operation, rows, count, "plain")
                rounds["plain"].append(record)
                raw.append(record)
            kinds = ["original", "debug"] if repeat % 2 == 0 else ["debug", "original"]
            for kind in kinds:
                for item in shortlists[kind]:
                    folder = directory / f"round-{repeat}" / f"block-{item['block']}-rs{item['RS']}"
                    record = dsl_group(args, folder, device, environment, operation, rows, count, kind,
                                       item["block"], [item["RS"]], args.samples)[0]
                    candidate_rounds[kind, item["block"], item["RS"]].append(record)
                    raw.append(record)
        for kind in ("original", "debug"):
            confirmed = [aggregate(items, kind, device, operation, rows, count)
                         for (name, _, _), items in candidate_rounds.items() if name == kind]
            passing = [item for item in confirmed if item["status"] == "PASS"]
            chosen = min(passing, key=lambda item: item["median_us"]) if passing else aggregate(
                [], kind, device, operation, rows, count)
            chosen.update(backend="npu", entry="dsl")
            legal = [item for item in passing if item.get("integration_candidate")]
            winner = min(legal, key=lambda item: item["median_us"]) if legal else None
            chosen["best_integration_config"] = ({key: winner.get(key) for key in
                ("RS", "block", "median_us", "status", "artifact_dir", "integration_note")} if winner else None)
            chosen["block_search_boundary"] = chosen.get("block") in (min(args.blocks), max(args.blocks))
            rounds[kind] = chosen
    selected = []
    for kind, items in rounds.items():
        record = aggregate(items, kind, device, operation, rows, count) if isinstance(items, list) else items
        record.update(backend=args.backend, kind=kind)
        if kind == "cooperative" and record["status"] == "PASS" and record.get("actual_path") != "cooperative":
            record["status"] = "NOT_COOPERATIVE"
        selected.append(record)
    plain = next(item for item in selected if item["kind"] == "plain")
    for item in selected:
        item["plain_us"] = plain.get("median_us") if plain["status"] == "PASS" else None
        item["speedup_vs_plain"] = (item["plain_us"] / item["median_us"] if item["status"] == "PASS" and
                                    item["plain_us"] is not None else None)
    summary = {"selected": selected, "candidates": raw, "natural": natural}
    suite.write_json(final, summary)
    message(f"[{device['selector']}] done {operation} X={rows} R={count}: " +
            ", ".join(f"{item['kind']}={item['status']}" for item in selected))
    return summary


def collect(args):
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / ".run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = {name: getattr(args, name) for name in (
        "backend", "devices", "operations", "rows", "rnumels", "splits", "blocks", "seed", "batch",
        "replays", "samples", "tune_samples", "topk", "confirm_rounds", "max_sample_ms", "no_natural",
        "op_timeout_ms", "init_timeout", "point_timeout")}
    config.update(schema=SCHEMA, source_sha256=source_hashes(),
                  npu_device_limit=os.getenv("NPU_DEVICE_LIMIT"),
                  npu_backend=os.getenv("TORCHINDUCTOR_NPU_BACKEND"))
    manifest_path = args.out / "run.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if old and not args.resume:
        raise ValueError(f"{manifest_path} exists; use --resume or a new --out directory")
    if old and old["config"] != config:
        raise ValueError("Resume configuration/source changed; use a new --out directory")
    devices = []
    probe_args = SimpleNamespace(**vars(args))
    probe_args.resume = False
    if args.backend == "cuda":
        record = launch_job(probe_args, args.out / "inventory", {"backend": "cuda", "selectors": args.devices},
                            dict(os.environ), mode="inventory")
        if record["status"] != "PASS":
            raise RuntimeError(record)
        devices = record["devices"]
    else:
        visible = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "")
        if visible and any(item not in visible.split(",") for item in args.devices):
            raise ValueError("NPU --devices are physical IDs and must be in ASCEND_RT_VISIBLE_DEVICES if set")
        for selector in args.devices:
            environment = device_environment("npu", selector, args.out / "cache" / selector)
            record = launch_job(probe_args, args.out / f"inventory-{selector}",
                                {"backend": "npu", "selectors": [selector]}, environment, mode="inventory")
            if record["status"] != "PASS":
                raise RuntimeError(record)
            devices.extend(record["devices"])
    if len({item["uuid"] for item in devices}) != len(devices):
        raise ValueError("Duplicate physical devices after resolving --devices")
    def identities(items):
        return [{key: item.get(key) for key in ("uuid", "selector", "name", "core_count", "effective_core_count")}
                for item in items]

    if old and identities(old["devices"]) != identities(devices):
        raise ValueError("Device inventory changed; use a new --out directory")
    suite.write_json(manifest_path, {"config": config, "devices": devices,
                                   "time_utc": old["time_utc"] if old else datetime.now(timezone.utc).isoformat()})
    for device in devices:
        message(f"Device: {device['name']} selector={device['selector']} cores={device['core_count']}")
    shapes = [(op, rows, count) for op in args.operations for rows in args.rows for count in args.rnumels]
    collected = {}
    result_lock = threading.Lock()

    def save():
        for name in ("selected", "candidates", "natural"):
            data = [item for shape in sorted(collected) for item in collected[shape][name]]
            suite.write_json(args.out / ("results.json" if name == "selected" else name + ".json"), data)

    def queue(rank, device):
        environment = device_environment(args.backend, device["selector"], args.out / "cache" / str(rank))
        try:
            for shape in shapes[rank::len(devices)]:
                if STOP.is_set():
                    break
                summary = run_shape(args, device, environment, *shape)
                with result_lock:
                    collected[shape] = summary
                    save()
        except BaseException:
            STOP.set()
            raise

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: STOP.set())
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(queue, rank, device) for rank, device in enumerate(devices)]
        for future in futures:
            try:
                future.result()
            except BaseException:
                STOP.set()
                raise
    if STOP.is_set():
        return 130
    records = [item for value in collected.values() for item in value["selected"]]
    message(f"Selected results: {dict(Counter(item['status'] for item in records))}")
    command = [sys.executable, str(HERE / "plot_compile_matrix.py"), "--input", str(args.out),
               "--out", str(args.out / "figures")]
    if args.no_plot:
        command.append("--csv-only")
    subprocess.run(command, check=True)
    return 0 if all(item["status"] == "PASS" for item in records) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "npu"), default="cuda")
    parser.add_argument("--devices", help="CUDA: current logical indices or full UUIDs; NPU: physical IDs")
    parser.add_argument("--out", type=Path, default=Path("cooperative-compile-results"))
    parser.add_argument("--operations", default="sum,amax")
    parser.add_argument("--rows")
    parser.add_argument("--rnumels")
    parser.add_argument("--splits")
    parser.add_argument("--blocks")
    parser.add_argument("--seed", type=int, default=213)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--tune-samples", type=int, default=3)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--confirm-rounds", type=int)
    parser.add_argument("--max-sample-ms", type=float, default=50)
    parser.add_argument("--op-timeout-ms", type=int, default=10000)
    parser.add_argument("--point-timeout", type=float, default=30)
    parser.add_argument("--init-timeout", type=float, default=300)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-natural", action="store_true")
    parser.add_argument("--no-plot", action="store_true", help="Still produce CSV; render SVG later on another machine")
    parser.add_argument("--worker", choices=("compile", "inventory"), help=argparse.SUPPRESS)
    parser.add_argument("--job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return internal_worker(args)
    if not args.devices:
        parser.error("--devices is required; devices are never selected implicitly")
    args.devices = [item.strip() for item in args.devices.split(",")]
    if any(not item or args.backend == "npu" and not item.isdecimal() for item in args.devices):
        parser.error("--devices must contain CUDA indices/UUIDs, or NPU physical numbers")
    args.operations = list(dict.fromkeys(args.operations.split(",")))
    if not set(args.operations) <= {"sum", "amax"}:
        parser.error("--operations must contain sum and/or amax")
    args.rows = numbers(args.rows or ("1" if args.smoke else "1,7,56"))
    args.rnumels = numbers(args.rnumels or ("4096,32768" if args.smoke else matrix.DEFAULT_R))
    args.splits = numbers(args.splits or ("1,2,8" if args.smoke else "1,2,4,8,16,32,48,56"))
    args.blocks = numbers(args.blocks or ("512,2048" if args.smoke else "256,512,1024,2048,4096,8192"))
    if any(block & (block - 1) for block in args.blocks):
        parser.error("--blocks must be powers of two")
    args.topk = args.topk if args.topk is not None else (1 if args.smoke else 3)
    args.confirm_rounds = args.confirm_rounds if args.confirm_rounds is not None else (1 if args.smoke else 3)
    for name in ("batch", "replays", "samples", "tune_samples", "topk", "confirm_rounds", "max_sample_ms",
                 "op_timeout_ms", "point_timeout", "init_timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not args.no_plot:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            parser.error("Install matplotlib>=3.8 or use --no-plot and render on another machine")
    args.out = args.out.resolve()
    try:
        return collect(args)
    except (Exception, KeyboardInterrupt) as error:
        STOP.set()
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 130 if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
