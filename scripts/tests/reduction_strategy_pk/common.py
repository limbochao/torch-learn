"""Shared eager semantics, inputs, checks, and profiler-only device timing."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import statistics


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def case_id(case):
    return f"{case['op']}-x{case['X']}-r{case['R']}-{case['dtype']}-{case['layout']}"


def cases():
    result = []

    def add(op, x, r, dtype="float32", layout="contiguous", group="core"):
        item = dict(op=op, X=x, R=r, dtype=dtype, layout=layout, group=group)
        item["id"] = case_id(item)
        if not any(old["id"] == item["id"] for old in result):
            result.append(item)

    for op in ("sum", "amax", "rmsnorm", "softmax"):
        for x in (1, 7, 14):
            for r in (256, 1024, 4096, 8192, 16384, 32768, 65536, 131072,
                      262144, 524288, 1048576, 2097152, 4194304):
                add(op, x, r)
    for dtype in ("float16", "bfloat16"):
        for op in ("sum", "rmsnorm", "softmax"):
            for x in (1, 7, 14):
                add(op, x, 262144, dtype, group="dtype")
    for layout in ("stride2", "transpose"):
        for op in ("sum", "amax"):
            for x in (1, 7, 14):
                add(op, x, 262144, layout=layout, group="layout")
    for x, r in ((1, 32768), (7, 229376), (14, 2097152)):
        for delta in (-1, 1):
            add("sum", x, r + delta, group="tail")
    for x in (28, 56):
        for r in (32768, 1048576):
            add("sum", x, r, group="output_control")
    for x in (1, 7, 14):
        add("sum_amax", x, 262144, group="fusion")
    assert len(result) == 199
    return result


def eager_function(op):
    import torch

    def forward(a, b):
        z = a.float() + b.float()
        if op == "sum":
            return z.sum(-1)
        if op == "amax":
            return z.amax(-1)
        if op == "rmsnorm":
            return z * torch.rsqrt((z * z).mean(-1, keepdim=True) + 1e-5)
        if op == "softmax":
            return torch.softmax(z, dim=-1)
        if op == "sum_amax":
            return z.sum(-1), z.amax(-1)
        raise ValueError(op)

    return forward


def cpu_inputs(case, seed):
    import torch

    generator = torch.Generator().manual_seed(seed)
    dtype = getattr(torch, case["dtype"])
    return tuple(torch.randn((case["X"], case["R"]), generator=generator).to(dtype) for _ in range(2))


def device_inputs(case, values, backend):
    import torch

    x, r = case["X"], case["R"]
    stride = (r, 1) if case["layout"] == "contiguous" else (2 * r, 2) if case["layout"] == "stride2" else (1, x)
    tensors = tuple(torch.empty_strided((x, r), stride, device=backend, dtype=value.dtype) for value in values)
    for tensor, value in zip(tensors, values):
        tensor.copy_(value)
    return tensors


def reference(case, values):
    z = values[0].float() + values[1].float()
    if case["op"] == "sum":
        return z.double().sum(-1)
    if case["op"] == "amax":
        return z.amax(-1)
    if case["op"] == "sum_amax":
        return z.double().sum(-1), z.amax(-1)
    return eager_function(case["op"])(*values)


def compare(actual, expected, op):
    import torch

    actuals = actual if isinstance(actual, (tuple, list)) else (actual,)
    expecteds = expected if isinstance(expected, (tuple, list)) else (expected,)
    checks = []
    for index, (value, target) in enumerate(zip(actuals, expecteds, strict=True)):
        exact = op == "amax" or (op == "sum_amax" and index == 1)
        atol = 0 if exact else 0.02 if op in ("sum", "sum_amax") else 1e-8 if op == "softmax" else 3e-5
        rtol = 0 if exact else 3e-4
        value, target = value.detach().cpu().double(), target.detach().cpu().double()
        close = torch.isclose(value, target, atol=atol, rtol=rtol, equal_nan=False)
        error = (value - target).abs()
        checks.append(dict(values=value.numel(), mismatched=int((~close).sum()),
                           nonfinite=int((~torch.isfinite(value)).sum()),
                           max_abs_error=float(error.max()), atol=atol, rtol=rtol))
    return dict(passed=all(item["mismatched"] == 0 for item in checks), outputs=checks)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_profile(backend, directory, batch):
    """Attribute CUDA by range, NPU by checked serial order; durations stay device-only."""
    directory = Path(directory)
    if backend == "cuda":
        trace_path = directory / "trace.json"
    else:
        paths = list(directory.rglob("trace_view.json"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected one NPU trace, found {len(paths)}")
        trace_path = paths[0]
    trace = json.loads(trace_path.read_text())
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    spans = []
    for event in events:
        if (event.get("ph") == "X" and event.get("name", "").startswith("pk:")
                and event.get("cat") in ("user_annotation", "cpu_op")):
            start, duration = float(event["ts"]), float(event["dur"])
            spans.append((start, start + duration, event["name"][3:]))
    spans.sort()
    if not spans:
        raise RuntimeError("No labeled profiler ranges")
    device_events = []
    primer_spans = [(float(event["ts"]), float(event["ts"]) + float(event["dur"])) for event in events
                   if event.get("ph") == "X" and event.get("name") == "pk-primer"
                   and event.get("cat") in ("user_annotation", "cpu_op")]
    primer_events = []
    if backend == "cuda":
        for event in events:
            if event.get("ph") == "X" and event.get("cat") in ("kernel", "gpu_memset", "gpu_memcpy"):
                device_events.append(dict(name=event["name"], start_us=float(event["ts"]),
                                          duration_us=float(event["dur"]), category=event["cat"],
                                          args=event.get("args", {})))
    else:
        paths = list(directory.rglob("kernel_details.csv"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected one kernel_details.csv, found {len(paths)}")
        with paths[0].open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                device_events.append(dict(name=row["Name"], start_us=float(row["Start Time(us)"].strip()),
                                          duration_us=float(row["Duration(us)"]), category=row["Type"],
                                          device=row["Device_id"], stream=row.get("Stream ID"), block=row["Block Num"]))
    assigned = defaultdict(list)
    unassigned = []
    if backend == "npu":
        # CANN Level0 host/device clocks have an observed offset. All calls are on
        # one stream and explicitly synchronized between labeled batches. Consume
        # the complete device sequence, validating exact counts and kernel names.
        device_events.sort(key=lambda event: event["start_us"])
        counts = {}
        unknown = []
        for _, _, label in spans:
            candidate = label.rsplit(":", 1)[0]
            if candidate.startswith("split-"):
                counts[label] = 2 * batch
            elif candidate.startswith("cooperative-"):
                counts[label] = batch
            elif candidate == "plain":
                unknown.append(label)
            else:
                raise RuntimeError(f"Unknown NPU candidate kind {candidate}")
        remaining = len(device_events) - sum(counts.values())
        if unknown:
            if remaining <= 0 or remaining % (len(unknown) * batch):
                raise RuntimeError("NPU compiled baseline event count is inconsistent")
            for label in unknown:
                counts[label] = remaining // len(unknown)
        elif remaining:
            raise RuntimeError(f"NPU device event count differs from the launch schedule by {remaining}")
        offset = 0
        baseline_pattern = None
        for _, _, label in spans:
            selected = device_events[offset:offset + counts[label]]
            offset += counts[label]
            pattern = [event["name"] for event in selected]
            candidate = label.rsplit(":", 1)[0]
            if candidate.startswith("split-") and pattern != ["partial", "finish"] * batch:
                raise RuntimeError(f"NPU split kernel sequence mismatch: {label}: {pattern}")
            if candidate.startswith("cooperative-") and pattern != ["cooperative"] * batch:
                raise RuntimeError(f"NPU cooperative kernel sequence mismatch: {label}: {pattern}")
            if candidate == "plain":
                one = pattern[:len(pattern) // batch]
                if one * batch != pattern or baseline_pattern is not None and one != baseline_pattern:
                    raise RuntimeError("NPU baseline kernel sequence changed between calls")
                baseline_pattern = one
            assigned[label] = selected
        if offset != len(device_events):
            raise RuntimeError("Unconsumed NPU device events")
    else:
        for event in device_events:
            # End tolerance only accounts for timestamp representation/alignment.
            matches = [label for start, end, label in spans if start - 2 <= event["start_us"]
                       and event["start_us"] + event["duration_us"] <= end + 2]
            if len(matches) != 1:
                if any(start - 2 <= event["start_us"] and event["start_us"] + event["duration_us"] <= end + 2
                       for start, end in primer_spans):
                    primer_events.append(event)
                else:
                    unassigned.append(event)
            else:
                assigned[matches[0]].append(event)
    samples = defaultdict(list)
    rows = []
    for _, _, label in spans:
        candidate, repeat = label.rsplit(":", 1)
        selected = sorted(assigned[label], key=lambda item: item["start_us"])
        if not selected or len(selected) % batch:
            raise RuntimeError(f"Unusable event count for {label}: {len(selected)} / batch {batch}")
        counts = Counter(item["name"] for item in selected)
        if any(count % batch for count in counts.values()):
            raise RuntimeError(f"Kernel pattern is not repeated exactly {batch} times: {label}: {counts}")
        total = sum(item["duration_us"] for item in selected) / batch
        samples[candidate].append(total)
        rows.append(dict(label=label, repeat=int(repeat), device_us=total, events_per_call=len(selected) // batch,
                         names_per_call={name: count // batch for name, count in counts.items()}, events=selected))
    if unassigned:
        raise RuntimeError(f"{len(unassigned)} device events fall outside the synchronized labels")
    write_json(directory / "device_events.json", dict(backend=backend, batch=batch, ranges=rows,
               excluded_profiler_primer=primer_events,
               attribution="validated serial kernel sequence" if backend == "npu" else "synchronized CPU ranges"))
    return {candidate: dict(median_us=statistics.median(values), min_us=min(values), max_us=max(values),
                            samples_us=values, batch=batch, method="profiler device duration sum / calls",
                            events_per_call=next(r["events_per_call"] for r in rows if r["label"].startswith(candidate + ":")))
            for candidate, values in samples.items()}


def profile_candidates(backend, invocations, directory, rounds=3, batch=5, validation=None):
    import torch

    api = getattr(torch, backend)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Compiles, allocations, eager reference, warmup and output copies are outside capture.
    for invoke in invocations.values():
        for _ in range(4):
            invoke()
    api.synchronize()
    use_graph = os.getenv("PK_PROFILE_GRAPH") == "1"
    graphs = {}
    graph_outputs = {}
    graph_checks = defaultdict(list)

    def check_graph(label, phase):
        if validation is not None:
            check = validation(label, graph_outputs[label])
            graph_checks[label].append(dict(phase=phase, **check))
            write_json(directory / "graph_checks.json", graph_checks)
            if not check["passed"]:
                raise RuntimeError(f"Graph replay output mismatch: {label}, {phase}")

    if use_graph:
        for label, invoke in invocations.items():
            graph = api.NPUGraph() if backend == "npu" else api.CUDAGraph()
            with api.graph(graph):
                for _ in range(batch):
                    graph_outputs[label] = invoke()
            for _ in range(3):
                graph.replay()
            api.synchronize()
            graphs[label] = graph
            check_graph(label, "before_profile")
    if backend == "cuda":
        context = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA])
    else:
        import torch_npu

        context = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory)),
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level0),
            record_shapes=False, with_stack=False)
    order = list(invocations)
    random.Random(213).shuffle(order)
    with context as prof:
        if backend == "cuda":
            # CUPTI can omit the first sub-microsecond fill after restarting
            # collection. Keep a separate, explicitly excluded primer range;
            # the measured ranges must still contain every expected operation.
            with torch.profiler.record_function("pk-primer"):
                for label in order:
                    if use_graph:
                        graphs[label].replay()
                    else:
                        invocations[label]()
                api.synchronize()
            prof.step()
        for repeat in range(rounds):
            for label in order if repeat % 2 == 0 else reversed(order):
                with torch.profiler.record_function(f"pk:{label}:{repeat}"):
                    if use_graph:
                        graphs[label].replay()
                    else:
                        for _ in range(batch):
                            invocations[label]()
                    api.synchronize()
                prof.step()
    if backend == "cuda":
        prof.export_chrome_trace(str(directory / "trace.json"))
    parsed = parse_profile(backend, directory, batch)
    if use_graph:
        # Verify the tensor written by this exact captured graph, not a fresh
        # direct invocation. Replay separately because NPU DSL candidates share
        # scratch/output buffers and would otherwise overwrite each other.
        for label, graph in graphs.items():
            graph.replay()
            api.synchronize()
            check_graph(label, "after_profile")
    for record in parsed.values():
        record["submission"] = backend.upper() + " Graph replay" if use_graph else "direct callable"
    return parsed
