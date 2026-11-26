"""Direct NPU profiling; reuse the repository collector and CSV reader."""

from __future__ import annotations

import math
from pathlib import Path
import random
import sys

from common import METRIC, TIMING_METHOD, file_hash, profiler_path, write_json

sys.path.insert(0, str(profiler_path().parent))
from npu_profiler import ProfileResultParser, TorchNpuProfiler


def expected_kernels(metadata):
    patterns = {mode: [launch["kernel_name"] for launch in info["launches"]]
                for mode, info in metadata.items()}
    if any(not names or not all(names) for names in patterns.values()):
        raise ValueError("profiling requires the complete inspected kernel launch sequence")
    return patterns


def paired_schedule(modes, session, rounds, repeats, seed):
    rng = random.Random(seed + session)
    plan = []
    for round_id in range(rounds):
        order = list(modes)
        rng.shuffle(order)
        for position, mode in enumerate(order + order[::-1]):
            plan.append(dict(session=session, round=round_id, position=position,
                             mode=mode, forwards=repeats))
    return plan


def parse_capture(directory, plan, patterns, case_root):
    """Consume every device CSV row, validating counts and the serial launch order.

    Host annotations and device timestamps need not share a clock. Each batch is
    synchronized before the next one; attribution uses its known launch order.
    Only per-task Duration(us) is summed, never gaps or start/end differences.
    """
    directory, case_root = Path(directory), Path(case_root)
    paths = list(directory.rglob("kernel_details.csv"))
    if len(paths) != 1:
        raise ValueError(f"expected one kernel_details.csv in {directory}, found {len(paths)}")
    csv_path = paths[0]
    rows = ProfileResultParser(csv_path).kernel_rows()
    tasks = []
    for index, row in enumerate(rows):
        name = next((row[k].strip() for k in ProfileResultParser.KERNEL_NAME_COLUMNS if row.get(k)), "")
        duration_column = next((k for k in ProfileResultParser.KERNEL_DURATION_COLUMNS if row.get(k)), None)
        start_column = next((k for k in ("Start Time(us)", "Task Start Time(us)") if row.get(k)), None)
        if not name or duration_column is None or start_column is None:
            raise ValueError(f"missing device kernel name/duration/start in CSV row {index}")
        duration, start = float(row[duration_column]), float(row[start_column])
        if not math.isfinite(duration) or duration <= 0 or not math.isfinite(start):
            raise ValueError(f"invalid device duration/timestamp in CSV row {index}")
        tasks.append(dict(name=name, duration_us=duration, start_us=start, csv_row=index,
                          device=row.get("Device_id", row.get("Device ID", "")),
                          stream=row.get("Stream ID", row.get("Stream Id", "")),
                          duration_column=duration_column))
    if not tasks:
        raise ValueError("profile contains no device kernels")
    if len({(t["device"], t["stream"]) for t in tasks}) != 1:
        raise ValueError("expected one device/stream; cannot attribute concurrent kernels by serial order")
    stream_evidence = "csv" if tasks[0]["stream"] else "CSV omits stream; direct caller and inspected launch order are serial"
    tasks.sort(key=lambda item: (item["start_us"], item["csv_row"]))
    expected_count = sum(entry["forwards"] * len(patterns[entry["mode"]]) for entry in plan)
    if len(tasks) != expected_count:
        raise ValueError(f"profile task count {len(tasks)} != inspected launches {expected_count}; no rows discarded")
    relative = csv_path.relative_to(case_root).as_posix()
    samples, offset = [], 0
    for entry in plan:
        pattern = patterns[entry["mode"]]
        count = entry["forwards"] * len(pattern)
        selected = tasks[offset:offset + count]
        if [task["name"] for task in selected] != pattern * entry["forwards"]:
            raise ValueError(f"profile kernel sequence differs from inspected launches: {entry}")
        kernels = []
        for slot, name in enumerate(pattern):
            durations = [task["duration_us"] for task in selected[slot::len(pattern)]]
            kernels.append(dict(launch_index=slot, name=name, count=len(durations),
                                total_us=sum(durations), mean_us=sum(durations) / len(durations)))
        samples.append(dict(**entry, timing_method=TIMING_METHOD,
                            kernel_us=sum(t["duration_us"] for t in selected) / entry["forwards"],
                            kernels=kernels, profiler_csv=relative, task_offset=offset, task_count=count))
        offset += count
    return samples, dict(session=plan[0]["session"], csv=relative, sha256=file_hash(csv_path),
                         task_count=len(tasks), duration_columns=sorted({t["duration_column"] for t in tasks}),
                         device=tasks[0]["device"], stream=tasks[0]["stream"],
                         stream_evidence=stream_evidence,
                         attribution="checked serial launch sequence; all device rows consumed")


def benchmark(strategies, inputs, metadata, settings, seed, directory, case_root):
    import torch

    patterns = expected_kernels(metadata)
    # Compile/autotune, allocations, input copies and validation occur outside capture.
    for strategy in strategies.values():
        for index in range(settings["warmup"]):
            output = strategy.compiled(*inputs[index % len(inputs)])
    torch.npu.synchronize()
    samples, captures = [], []
    for session in range(settings["sessions"]):
        capture_dir = directory / f"session-{session:02d}"
        capture_dir.mkdir(parents=True, exist_ok=False)
        plan = paired_schedule(settings["modes"], session, settings["rounds"], settings["repeats"], seed)
        write_json(capture_dir / "plan.json", dict(timing_method=TIMING_METHOD, plan=plan, patterns=patterns))
        collector = TorchNpuProfiler(capture_dir, wait=0, warmup=0, active=len(plan), repeat=1,
                                     profiler_level=0, activities=("CPU", "NPU"),
                                     record_shapes=False, profile_memory=False, with_stack=False)
        print(f"PROFILE {capture_dir}: {len(plan)} batches x {settings['repeats']} forwards", flush=True)
        with collector.profile() as prof:
            for entry in plan:
                for index in range(entry["forwards"]):
                    output = strategies[entry["mode"]].compiled(*inputs[index % len(inputs)])
                torch.npu.synchronize()  # Defines order only; never used as a duration.
                prof.step()
        current, capture = parse_capture(capture_dir, plan, patterns, case_root)
        samples.extend(current)
        captures.append(capture)
        write_json(capture_dir / "kernel_samples.json", dict(samples=current, capture=capture))
    del output
    return samples, captures


def validate_captures(probe, settings, case_root):
    """Re-read portable raw CSV evidence before analysis/resume/merge admission."""
    patterns = expected_kernels(probe["metadata"])
    captures = probe.get("profiles", [])
    if [c.get("session") for c in captures] != list(range(settings["sessions"])):
        raise ValueError("missing or duplicate profiler sessions")
    reconstructed = []
    for capture in captures:
        relative = Path(capture["csv"])
        root = Path(case_root).resolve()
        path = (root / relative).resolve()
        if relative.is_absolute() or root not in path.parents:
            raise ValueError("profiler CSV path escaped case directory")
        if not path.is_file() or file_hash(path) != capture["sha256"]:
            raise ValueError("missing or altered profiler CSV")
        seed = settings["seed"] + probe["probe_id"]
        plan = paired_schedule(settings["modes"], capture["session"], settings["rounds"], settings["repeats"], seed)
        samples, recomputed = parse_capture(path.parent, plan, patterns, root)
        if recomputed != capture:
            raise ValueError("profiler capture identity differs")
        reconstructed.extend(samples)
    if reconstructed != probe.get("samples"):
        raise ValueError("stored samples differ from raw profiler kernel durations")


def sample_evidence(probe, settings):
    if probe.get("execution") != "direct":
        return "INVALID_EXECUTION"
    for sample in probe.get("samples", []):
        value = sample.get(METRIC, 0)
        if sample.get("timing_method") != TIMING_METHOD or not math.isfinite(value) or value <= 0:
            return "INVALID_PROFILER_TIMING"
        if sample.get("forwards") != settings["repeats"]:
            return "INVALID_REPETITION_CONTRACT"
        kernels = sample.get("kernels", [])
        names = [launch["kernel_name"] for launch in probe["metadata"][sample["mode"]]["launches"]]
        if [k["name"] for k in kernels] != names or any(k["count"] != sample["forwards"] for k in kernels):
            return "INVALID_KERNEL_COUNTS"
        if not math.isclose(sum(k["mean_us"] for k in kernels), value, rel_tol=1e-10, abs_tol=1e-10):
            return "INVALID_KERNEL_SUM"
    return None
