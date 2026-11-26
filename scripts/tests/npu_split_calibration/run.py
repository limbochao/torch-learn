#!/usr/bin/env python3
"""Portable NPU split-reduction calibration: plan, run/resume, analyze."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from analyze import summarize
from profiling import validate_captures
from cases import CONTROLS, DTYPE_BYTES, FLOAT_DTYPES, FUSIONS, LAYOUTS, OPS, PRIMITIVES, PRESETS, estimated_working_bytes, make_case, matrix, shapes
from common import (HERE, collection_complete, collection_selection, external_output,
                    METRIC, TIMING_METHOD, measurement_settings, percentage, profiler_path, read_json, recorder_path, require_profiler_settings, source_hashes,
                    validate_collection, write_json)
from policy import DEFAULTS, digest


def csv_items(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def device_ids(value):
    cards = csv_items(value)
    if not cards or any(not card.isascii() or not card.isdigit() for card in cards):
        raise argparse.ArgumentTypeError("use comma-separated nonnegative physical device IDs")
    cards = [str(int(card)) for card in cards]
    if len(set(cards)) != len(cards):
        raise argparse.ArgumentTypeError("physical device IDs must be distinct")
    return ",".join(cards)


def percent_range(value):
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("use START,END percentages, e.g. --range 0,25")
    try:
        start, end = (percentage(part.strip()) for part in parts)
        collection_selection([], start, end)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return start, end


def normalize_args(p, args):
    """Canonicalize public and legacy options before planning or saving identity."""
    if args.percent_range is not None:
        if args.start_percent is not None or args.end_percent is not None:
            p.error("use --range START,END without --start-percent/--end-percent")
        args.start_percent, args.end_percent = args.percent_range
    else:
        args.start_percent = args.start_percent if args.start_percent is not None else "0"
        args.end_percent = args.end_percent if args.end_percent is not None else "100"
    del args.percent_range
    if args.visible_devices and args.device != 0:
        p.error("--devices already selects physical cards; do not combine it with a nonzero legacy --device")
    return args


def balanced_assignments(items, slots):
    return {slot: items[index::len(slots)] for index, slot in enumerate(slots)}


def child_env(directory):
    env = os.environ.copy()
    env.update(TORCHINDUCTOR_COMPILE_THREADS="1", TORCH_WARM_POOL="0", PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1", TORCHINDUCTOR_FORCE_DISABLE_CACHES="1",
               TORCHINDUCTOR_CACHE_DIR=str(directory / "cache" / "inductor"),
               TRITON_CACHE_DIR=str(directory / "cache" / "triton"),
               TORCH_COMPILE_DEBUG_DIR=str(directory / "debug"))
    env.setdefault("TORCHINDUCTOR_NPU_BACKEND", "default")
    return env


def launch(command, directory, timeout, *, visible_device=None, cancel=None):
    directory.mkdir(parents=True, exist_ok=True)
    env = child_env(directory)
    if visible_device is not None:
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(visible_device)
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=os.name == "posix")
        start = time.monotonic()
        try:
            while process.poll() is None:
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("run interrupted; terminating this case process group")
                if time.monotonic() - start > timeout:
                    raise subprocess.TimeoutExpired(command, timeout)
                time.sleep(0.5)
            return process.returncode
        except (KeyboardInterrupt, InterruptedError, subprocess.TimeoutExpired) as exc:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            write_json(directory / "error.json", dict(status="TIMEOUT" if isinstance(exc, subprocess.TimeoutExpired) else "INTERRUPTED", error=str(exc)))
            if isinstance(exc, KeyboardInterrupt):
                raise
            return 124


def collect_cases(cases, out, settings, table, environment, identity, args):
    """One subprocess per selected card; each comparison stays on the same card."""
    pending = []
    for index, case in enumerate(cases, 1):
        directory = out / "cases" / case["id"]
        result_path = directory / "result.json"
        if args.resume and result_path.exists() and not (directory / "error.json").exists():
            previous = read_json(result_path)
            expected_probes = len(shapes(case)) * len(settings["cache_modes"]) * len(settings["executions"])
            if (previous.get("status") == "PASS" and previous.get("case") == case
                    and previous.get("table_digest") == digest(table)
                    and measurement_settings(previous.get("settings", {})) == measurement_settings(settings)
                    and previous.get("source_hashes") == identity["source_hashes"]
                    and previous.get("environment") == environment and len(previous.get("probes", [])) == expected_probes
                    and all(len(probe.get("samples", [])) == settings["sessions"] * settings["rounds"] * 2 * len(settings["modes"])
                            for probe in previous["probes"])):
                try:
                    for probe in previous["probes"]:
                        validate_captures(probe, settings, directory)
                except (ValueError, KeyError, OSError):
                    pass  # Missing/invalid CSV is an incomplete case, never a resumable PASS.
                else:
                    continue
        pending.append((index, case, directory))

    slots = csv_items(args.visible_devices) if args.visible_devices else [None]
    cancel = threading.Event()
    pool = ThreadPoolExecutor(max_workers=len(slots))
    running, failed, stop = {}, False, False
    assignments = balanced_assignments(pending, slots)
    queues = {slot: iter(items) for slot, items in assignments.items()}
    shared_queue = iter(pending)
    write_json(out / "schedule.json", dict(schedule=args.schedule, pending=len(pending),
               skipped=len(cases) - len(pending), cards=[dict(card=slot if slot is not None else args.device,
               case_ids=[item[1]["id"] for item in assignments[slot]] if args.schedule == "balanced" else None)
               for slot in slots]))

    def dispatch(slot):
        item = next(queues[slot] if args.schedule == "balanced" else shared_queue, None)
        if item is None:
            return
        index, case, directory = item
        if directory.exists():
            attempt = 1
            archive = out / "attempts" / f"{case['id']}.{attempt}"
            while archive.exists():
                attempt += 1
                archive = out / "attempts" / f"{case['id']}.{attempt}"
            if out not in directory.resolve().parents or out not in archive.resolve().parents:
                raise ValueError("retry archive path escaped output directory")
            archive.parent.mkdir(parents=True, exist_ok=True)
            directory.rename(archive)
        directory.mkdir(parents=True, exist_ok=True)
        job = dict(case=case, settings=settings, table=table, environment=environment, source_hashes=identity["source_hashes"])
        write_json(directory / "job.json", job)
        print(f"[{index}/{len(cases)}] card={slot if slot is not None else args.device} {case['id']}", flush=True)
        command = [sys.executable, str(HERE / "worker.py"), "--job", str(directory / "job.json"), "--out", str(directory)]
        future = pool.submit(launch, command, directory, args.timeout, visible_device=slot, cancel=cancel)
        running[future] = (slot, index, case, directory)

    try:
        for slot in slots:
            dispatch(slot)
        while running:
            write_json(out / "active.json", dict(status="RUNNING", total=len(cases),
                       active=[dict(card=slot, index=index, case_id=case["id"]) for slot, index, case, _ in running.values()]))
            done, _ = wait(running, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in done:
                slot, index, case, directory = running.pop(future)
                code = future.result()
                result_path = directory / "result.json"
                passed = code == 0 and result_path.exists() and read_json(result_path).get("status") == "PASS"
                failed |= not passed
                stop |= not passed and args.fail_fast
                print(f"  {'PASS' if passed else 'FAILED/INCOMPLETE'}; {directory / 'worker.log'}", flush=True)
                if not stop:
                    dispatch(slot)
    finally:
        cancel.set()
        pool.shutdown(wait=True, cancel_futures=True)
    return failed


def validate_case(case):
    if case["X"] < 1 or case["R"] < 1 or case["dtype"] not in DTYPE_BYTES:
        raise ValueError(f"invalid case: {case}")
    if case["layout"] not in LAYOUTS:
        raise ValueError(f"unknown layout: {case['layout']}")
    if case["op"] not in OPS:
        raise ValueError(f"unknown op: {case['op']}")
    if (case["op"], case["dtype"]) not in PRIMITIVES and not (case["op"] in (*FUSIONS, *CONTROLS) and case["dtype"] in FLOAT_DTYPES):
        raise ValueError(f"unsupported op/dtype pair: {case['op']}/{case['dtype']}")
    if case["op"] in CONTROLS and case["layout"] != "contiguous":
        raise ValueError("control forwards currently require contiguous layout")
    if case["op"] == "softmax" and case["layout"] == "multi_axis":
        raise ValueError("softmax reduces one axis; multi_axis R would not describe its reduction")
    if case["role"] not in ("smoke", "calibration", "holdout", "dynamic", "control"):
        raise ValueError("unknown case role")
    if case.get("dynamic_axes") and case["layout"] != "contiguous":
        raise ValueError("dynamic custom cases currently require contiguous [X,R] inputs")
    if any(axis not in (0, 1) for axis in case.get("dynamic_axes", [])):
        raise ValueError("dynamic_axes must contain only 0/1")
    if case.get("dynamic_axes") and any(case[("X", "R")[axis]] <= 1 for axis in case["dynamic_axes"]):
        raise ValueError("dynamic compile seeds must avoid specialized dimensions 0/1")
    if case.get("runtime_shapes"):
        for shape in case["runtime_shapes"]:
            if len(shape) != 2 or any(not isinstance(value, int) or value < 1 for value in shape):
                raise ValueError("runtime_shapes entries must be positive [X,R] integers")
            if any(shape[axis] != case[("X", "R")[axis]] for axis in (0, 1) if axis not in case.get("dynamic_axes", [])):
                raise ValueError("runtime_shapes changes an axis not declared in dynamic_axes")


def select_cases(args, cores, *, with_collection=False):
    if args.cases_json:
        raw = read_json(args.cases_json)
        if isinstance(raw, dict) and "collection" in raw:
            # A shared plan/manifest defines the FULL universe, even if the
            # planning command selected a range or excluded large cases.
            raw = dict(cases=validate_collection(raw))
        cases = [make_case(**{{"X": "x", "R": "r"}.get(key, key): value for key, value in case.items() if key != "id"})
                 for case in (raw["cases"] if isinstance(raw, dict) else raw)]
    else:
        cases = []
        for preset in csv_items(args.preset):
            if preset not in PRESETS:
                raise ValueError(f"unknown preset: {preset}")
            cases.extend(matrix(preset, cores, args.around, {key: getattr(args, key) for key in DEFAULTS}))
    for case in cases:
        validate_case(case)
    cases = list({case["id"]: case for case in cases}.values())
    for field, values in (("op", args.ops), ("dtype", args.dtypes), ("layout", args.layouts)):
        if values:
            cases = [case for case in cases if case[field] in csv_items(values)]
    if args.x:
        cases = [case for case in cases if case["X"] in args.x]
    if args.r:
        cases = [case for case in cases if case["R"] in args.r]
    if args.match:
        cases = [case for case in cases if args.match in case["id"]]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError("no cases selected")
    selected, collection = collection_selection(cases, args.start_percent, args.end_percent,
                                                args.shards, args.shard_index)
    return (selected, collection) if with_collection else selected


def budget_cases(cases, settings):
    included, excluded = [], []
    for case in cases:
        peak = max(estimated_working_bytes(case, x, r, settings)
                   for x, r in shapes(case) + [(case["X"], case["R"])])
        if peak > settings["max_working_mib"] * 1024**2:
            excluded.append(dict(case=case, reason="PLANNED_MEMORY_BUDGET", estimated_bytes=peak))
        else:
            included.append(case)
    return included, excluded


def parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog="Default collection: coarse,boundary,coverage; ordinary_ir; direct profiler kernel durations x hot,rotate; "
               "65536 MiB per-case planning budget. Override these only for a different experiment.",
    )
    p.add_argument("--out", type=Path, required=True, help="raw output directory outside the source/repository")
    devices = p.add_mutually_exclusive_group()
    devices.add_argument("--devices", dest="visible_devices", type=device_ids,
                         help="physical cards, e.g. 0,1,2,3; omitted: use the first currently visible device")
    devices.add_argument("--visible-devices", dest="visible_devices", type=device_ids, help=argparse.SUPPRESS)
    p.add_argument("--range", dest="percent_range", type=percent_range, metavar="START,END",
                   help="case percentage interval, e.g. 0,25 or 12.5,50; default 0,100")
    p.add_argument("--resume", action="store_true", help="skip completed cases only when manifest and code match")
    p.add_argument("--plan", action="store_true", help="write the matrix only; requires no torch/NPU")
    p.add_argument("--cores", type=int, help="plan uses this C; a real run checks it against backend effective cores")
    p.add_argument("--device", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--start-percent", type=percentage, help=argparse.SUPPRESS)
    p.add_argument("--end-percent", type=percentage, help=argparse.SUPPRESS)
    overrides = p.add_argument_group("optional experiment overrides")
    overrides.add_argument("--preset", default="coarse,boundary,coverage", help="comma-separated: " + ",".join(PRESETS))
    overrides.add_argument("--schedule", choices=("balanced", "queue"), default="balanced",
                   help="balanced: round-robin case counts; queue: next case goes to the first idle card")
    overrides.add_argument("--lowering-profile", choices=("installed", "ordinary_ir"), default="ordinary_ir",
                   help="ordinary_ir restores process-local community prod/any/xor lowering; calibrated table applies only to this profile")
    overrides.add_argument("--modes", default="plain,candidate", help="plain,candidate,policy")
    overrides.add_argument("--policy", choices=("installed", "table"), default="installed")
    overrides.add_argument("--table", type=Path)
    overrides.add_argument("--execution", choices=("direct",), default="direct", help=argparse.SUPPRESS)
    overrides.add_argument("--cache", default="hot,rotate", help="hot,rotate")
    overrides.add_argument("--rotate-buffers", type=int, default=4)
    overrides.add_argument("--sessions", type=int, default=3)
    overrides.add_argument("--rounds", type=int, default=7)
    overrides.add_argument("--repeats", type=int, default=20, help="forwards per profiler batch; default 20")
    overrides.add_argument("--warmup", type=int, default=10)
    overrides.add_argument("--seed", type=int, default=50315)
    overrides.add_argument("--max-working-mib", type=int, default=65536,
                         help="per-case estimated working-set limit in MiB, not a memory reservation; default 65536 (64 GiB)")
    overrides.add_argument("--timeout", type=int, default=1800, help="seconds per case/preflight")
    overrides.add_argument("--max-autotune", action=argparse.BooleanOptionalAction, default=True)
    overrides.add_argument("--outputs-per-program", type=int, default=DEFAULTS["outputs_per_program"])
    overrides.add_argument("--target-waves", type=int, default=DEFAULTS["target_waves"])
    overrides.add_argument("--min-chunk", type=int, default=DEFAULTS["min_chunk"])
    overrides.add_argument("--max-s", type=int, default=DEFAULTS["max_s"])
    overrides.add_argument("--relative-margin", type=float, default=0.08)
    overrides.add_argument("--absolute-margin-us", type=float, default=0.2)
    overrides.add_argument("--metric", choices=(METRIC,), default=METRIC, help=argparse.SUPPRESS)
    overrides.add_argument("--max-r-gap", type=float, default=1.25, help="maximum R ratio between passing interval samples")
    overrides.add_argument("--boundary-r-gap", type=float, default=1.05, help="refine gain/tiling transitions to this R ratio")
    overrides.add_argument("--cases-json", type=Path)
    overrides.add_argument("--around", type=int, nargs="+", default=[])
    overrides.add_argument("--ops")
    overrides.add_argument("--dtypes")
    overrides.add_argument("--layouts")
    overrides.add_argument("--x", type=int, nargs="+")
    overrides.add_argument("--r", type=int, nargs="+")
    overrides.add_argument("--match")
    overrides.add_argument("--limit", type=int, help="limit the filtered universe BEFORE percentage/shard selection")
    overrides.add_argument("--shards", type=int, default=1)
    overrides.add_argument("--shard-index", type=int, default=0)
    overrides.add_argument("--extend", action="store_true", help="with --resume, append cases while preserving the old manifest domain")
    overrides.add_argument("--fail-fast", action="store_true")
    return p


def main():
    # nohup may inherit SIGINT=ignored; restore orderly case-group shutdown.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, signal.default_int_handler)
        if os.name == "posix":
            signal.signal(signal.SIGTERM, signal.default_int_handler)
    p = parser()
    args = normalize_args(p, p.parse_args())
    if not (args.shards > 0 and 0 <= args.shard_index < args.shards):
        p.error("invalid shard index/count")
    try:
        collection_selection([], args.start_percent, args.end_percent, args.shards, args.shard_index)
    except ValueError as exc:
        p.error(str(exc))
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    partitioned = (args.start_percent, args.end_percent) != ("0", "100") or args.shards != 1
    if args.extend and partitioned:
        p.error("percentage/shard collections are fixed; use a new output directory for a new case universe")
    for name in ("sessions", "rounds", "repeats", "warmup", "rotate_buffers", "max_working_mib", "timeout",
                 "outputs_per_program", "target_waves", "min_chunk"):
        if getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    if args.max_s < 0 or args.relative_margin < 0 or args.absolute_margin_us < 0:
        p.error("max-s and margins must be nonnegative")
    if not (1 < args.boundary_r_gap <= args.max_r_gap) or not all(map(math.isfinite,
            (args.relative_margin, args.absolute_margin_us, args.max_r_gap, args.boundary_r_gap))):
        p.error("require finite margins and 1 < boundary-r-gap <= max-r-gap")
    if args.extend and not args.resume:
        p.error("--extend requires --resume")
    modes, executions, cache_modes = map(csv_items, (args.modes, args.execution, args.cache))
    if "plain" not in modes or len(set(modes)) != len(modes) or len(modes) < 2 or set(modes) - {"plain", "candidate", "policy"}:
        p.error("modes must include plain and candidate and/or policy")
    if not executions or set(executions) - {"direct"} or not cache_modes or set(cache_modes) - {"hot", "rotate"}:
        p.error("invalid execution/cache modes")
    if len(set(executions)) != len(executions) or len(set(cache_modes)) != len(cache_modes):
        p.error("execution/cache entries must be unique")
    if "rotate" in cache_modes and "direct" in executions and args.repeats % args.rotate_buffers:
        p.error("repeats must be a multiple of rotate-buffers so all direct inputs rotate evenly")
    if args.policy == "table" and "policy" in modes and not args.table:
        p.error("--policy table requires --table")
    out = external_output(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:
        p.error("output is not empty; use a new directory or --resume with identical arguments")
    if (out / "manifest.json").exists():
        require_profiler_settings(read_json(out / "manifest.json")["settings"])
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(device=args.device, modes=modes, policy=args.policy, executions=executions, cache_modes=cache_modes,
                    rotate_buffers=args.rotate_buffers, sessions=args.sessions, rounds=args.rounds,
                    repeats=args.repeats, warmup=args.warmup, seed=args.seed, max_working_mib=args.max_working_mib,
                    timing_method=TIMING_METHOD, lowering_profile=args.lowering_profile,
                    max_autotune=args.max_autotune,
                    max_r_gap=args.max_r_gap, boundary_r_gap=args.boundary_r_gap,
                    criteria=dict(metric=args.metric, relative_margin=args.relative_margin, absolute_margin_us=args.absolute_margin_us),
                    selector={key: getattr(args, key) for key in DEFAULTS})
    print(f"Collection: preset={args.preset if not args.cases_json else args.cases_json}, "
          f"range={args.start_percent},{args.end_percent}%, "
          f"devices={args.visible_devices or 'visible logical ' + str(args.device)}, "
          f"lowering={args.lowering_profile}, execution={args.execution}, cache={args.cache}, "
          f"memory_budget={args.max_working_mib} MiB", flush=True)
    table = read_json(args.table) if args.table else {}
    if args.policy == "table" and "policy" in modes:
        criteria = dict(metric=args.metric, relative_margin=args.relative_margin, absolute_margin_us=args.absolute_margin_us)
        if table.get("criteria") != criteria:
            p.error("table acceptance criteria differ; pass its --metric/--relative-margin/--absolute-margin-us explicitly")
    if args.plan:
        if not args.cores or args.cores < 1:
            p.error("--plan requires --cores (actual effective Vector cores on the target)")
        selected, collection = select_cases(args, args.cores, with_collection=True)
        cases, excluded = budget_cases(selected, settings)
        slots = csv_items(args.visible_devices) if args.visible_devices else [str(args.device)]
        assignments = balanced_assignments(cases, slots)
        write_json(out / "plan.json", dict(cases=cases, excluded_cases=excluded, settings=settings, expected_cores=args.cores,
                   collection=collection, schedule=args.schedule,
                   cards=[dict(card=slot, count=len(items) if args.schedule == "balanced" else None,
                               case_ids=[case["id"] for case in items] if args.schedule == "balanced" else None)
                          for slot, items in assignments.items()]))
        print(f"Planned {len(cases)} cases, {len(excluded)} excluded by memory budget, C={args.cores}: {out / 'plan.json'}")
        print(f"Universe={len(collection['universe'])}, partition={collection['partition']}, cards={slots}, schedule={args.schedule}")
        return
    probe = out / "preflight"
    probe_kwargs = {"visible_device": csv_items(args.visible_devices)[0]} if args.visible_devices else {}
    code = launch([sys.executable, str(HERE / "worker.py"), "--probe", "--out", str(probe), "--device", str(args.device),
                   "--lowering-profile", args.lowering_profile],
                  probe, args.timeout, **probe_kwargs)
    if code:
        p.exit(1, f"NPU preflight failed; inspect {probe / 'worker.log'}\n")
    environment = read_json(probe / "environment.json")["fingerprint"]
    if args.visible_devices:
        card_environments = {csv_items(args.visible_devices)[0]: read_json(probe / "environment.json")}
        for card in csv_items(args.visible_devices):
            if card in card_environments:
                continue
            card_probe = out / "preflight" / f"card-{card}"
            code = launch([sys.executable, str(HERE / "worker.py"), "--probe", "--out", str(card_probe), "--device", "0",
                           "--lowering-profile", args.lowering_profile],
                          card_probe, args.timeout, visible_device=card)
            if code:
                p.exit(1, f"NPU card {card} preflight failed; inspect {card_probe / 'worker.log'}\n")
            info = read_json(card_probe / "environment.json")
            if info["fingerprint"] != environment:
                p.error("selected cards have different hardware/software fingerprints; cannot share an admission table")
            environment = info["fingerprint"]
            card_environments[card] = info
        write_json(out / "preflight" / "devices.json", card_environments)
    cores = environment["vector_cores"]
    if args.cores is not None and args.cores != cores:
        p.error(f"--cores={args.cores} differs from effective backend cores={cores}")
    selected, collection = select_cases(args, cores, with_collection=True)
    cases, excluded = budget_cases(selected, settings)
    identity = dict(schema="npu-split-manifest-v3", cases=cases, excluded_cases=excluded, settings=settings, table=table,
                    initial_case_ids=[case["id"] for case in cases] + [entry["case"]["id"] for entry in excluded],
                    environment=environment, source_hashes=source_hashes(), visible_devices=args.visible_devices,
                    collection=collection, schedule=args.schedule)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if "collection" not in old:
            p.error("old manifest has no collection metadata; resume it with its original source snapshot")
        identity["initial_case_ids"] = old["initial_case_ids"]
        if args.extend and not collection_complete(old) and not old.get("merge"):
            p.error("cannot extend a partial collection; merge the ranges first")
        fixed = [key for key in identity if not args.extend or key not in ("cases", "excluded_cases", "collection")]
        if args.extend and old.get("merge"):
            # Offline merging has no live device assignment. Bind pending work
            # to the requested local cards after checking their fingerprints.
            fixed = [key for key in fixed if key not in ("visible_devices", "schedule")]
        if old.get("merge"):
            identity["merge"] = old["merge"]
        if args.extend and measurement_settings(old["settings"]) == measurement_settings(settings):
            if settings["max_working_mib"] < old["settings"]["max_working_mib"]:
                p.error("an extension may raise the memory budget, not shrink the existing domain")
            fixed.remove("settings")
        if any(old.get(key) != identity[key] for key in fixed):
            p.error("resume manifest/code/environment mismatch; use a new output directory")
        if args.extend:
            cases = list({case["id"]: case for case in [*old["cases"], *cases]}.values())
            excluded = list({entry["case"]["id"]: entry for entry in [*old.get("excluded_cases", []), *excluded]}.values())
            included_ids = {case["id"] for case in cases}
            excluded = [entry for entry in excluded if entry["case"]["id"] not in included_ids]
            universe = list({case["id"]: case for case in [*old["collection"]["universe"], *collection["universe"]]}.values())
            _, collection = collection_selection(universe)
            identity.update(cases=cases, excluded_cases=excluded, collection=collection)
            if any(old.get(key) != value for key, value in identity.items()):
                history = out / "manifest.history"
                history.mkdir(exist_ok=True)
                write_json(history / f"{len(list(history.glob('*.json'))):04d}.json", old)
                write_json(manifest_path, dict(identity, created_utc=old["created_utc"]))
    else:
        write_json(manifest_path, dict(identity, created_utc=datetime.now(timezone.utc).isoformat()))
        source_dir = out / "source"
        source_dir.mkdir(exist_ok=True)
        for path in [*HERE.glob("*.py"), HERE / "README.md", recorder_path(), profiler_path()]:
            (source_dir / path.name).write_bytes(path.read_bytes())
    print(f"{len(cases)} cases, {len(excluded)} excluded by memory budget, C={cores}, modes={modes}, execution={executions}, cache={cache_modes}", flush=True)
    failed = False
    try:
        failed = collect_cases(cases, out, settings, table, environment, identity, args)
    finally:
        rows, failures = summarize(out, args.relative_margin, args.absolute_margin_us, args.metric)
        write_json(out / "active.json", dict(status="COMPLETE" if not failures else "INCOMPLETE", failed_cases=len(failures)))
    if failed or failures:
        raise SystemExit(1)
    if args.policy == "table" and "policy" in modes and collection_complete(identity):
        if not read_json(out / "holdout.audit.json")["passed"]:
            print("Frozen policy lacks complete passing evidence for every rule/context; inspect holdout.audit.json.", flush=True)
            raise SystemExit(2)


if __name__ == "__main__":
    main()
