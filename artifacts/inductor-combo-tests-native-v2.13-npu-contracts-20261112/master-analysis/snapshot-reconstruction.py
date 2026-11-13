"""Derive address-free peak evidence from locally retained NPU allocator snapshots.

This is an evidence derivation, not a maintained test runner. The raw snapshots
are deliberately retained outside the repository. Only trusted task snapshots
should be supplied because pickle loads executable Python objects.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stack_frames(frames):
    result = []
    for frame in frames:
        filename = frame["filename"]
        if "/site-packages/" in filename:
            filename = filename.split("/site-packages/", 1)[1]
        elif "/combo-master-analysis-20261113/" in filename:
            filename = filename.split("/combo-master-analysis-20261113/", 1)[1]
        elif "/inductor_npu/" in filename:
            filename = "inductor_npu/" + filename.split("/inductor_npu/", 1)[1]
        else:
            filename = Path(filename).name
        result.append({"name": frame["name"], "file": filename, "line": frame["line"]})
    return result


def reconstruct(path):
    raw = path.read_bytes()
    snapshot = pickle.loads(raw)
    assert len(snapshot["device_traces"]) == 1
    trace = snapshot["device_traces"][0]
    live = {
        block["address"]: (block["size"], block.get("frames", []))
        for segment in snapshot["segments"]
        for block in segment["blocks"]
        if block["state"] == "active_allocated"
    }
    allocated = sum(size for size, _ in live.values())
    final_allocated = peak = allocated
    peak_live = live.copy()
    peak_index = len(trace)
    for index, event in reversed(list(enumerate(trace))):
        if event["action"] == "alloc":
            previous = live.pop(event["addr"], None)
            assert previous is not None
            assert previous[0] == ((event["size"] + 511) // 512) * 512
            allocated -= previous[0]
        elif event["action"] == "free_requested":
            assert event["addr"] not in live
            # Allocated bytes leave at free_requested, before free_completed.
            # Trace sizes can be requested sizes, whereas active blocks are rounded.
            size = ((event["size"] + 511) // 512) * 512
            live[event["addr"]] = (size, event.get("frames", []))
            allocated += size
        if allocated > peak:
            peak = allocated
            peak_live = live.copy()
            peak_index = index

    grouped = defaultdict(lambda: {"bytes": 0, "count": 0, "allocations": []})
    stack_groups = defaultdict(lambda: {"count": 0, "allocated_bytes_each": 0, "stack": []})
    for size, frames in peak_live.values():
        if not frames:
            category = "allocated_before_probe"
        elif any(frame["name"] == "get_empty_cache_for_benchmark" for frame in frames):
            category = "autotune_clear_cache_tensor"
        elif any(frame["name"] == "clone_args" for frame in frames):
            category = "mutated_argument_clones"
        else:
            category = "generated_wrapper_and_aten_allocations"
        converted = stack_frames(frames)
        key = (category, size, json.dumps(converted, sort_keys=True))
        stack_groups[key]["count"] += 1
        stack_groups[key]["allocated_bytes_each"] = size
        stack_groups[key]["stack"] = converted
        grouped[category]["bytes"] += size
        grouped[category]["count"] += 1
    for (category, _, _), value in stack_groups.items():
        grouped[category]["allocations"].append(value)
    for value in grouped.values():
        value["allocations"].sort(key=lambda item: (-item["allocated_bytes_each"], str(item["stack"])))
    assert sum(value["bytes"] for value in grouped.values()) == peak
    return {
        "raw_snapshot_name": path.name,
        "raw_snapshot_sha256": sha(raw),
        "raw_snapshot_bytes": len(raw),
        "trace_entries": len(trace),
        "trace_action_counts": dict(Counter(item["action"] for item in trace)),
        "history_limit": 100000,
        "history_is_ring_buffer": len(trace) == 100000,
        "reconstruction": "reverse from final active_allocated blocks; invert alloc/free_requested",
        "free_completed_changes_allocated_bytes": False,
        "alignment_bytes": 512,
        "end_allocated_bytes": final_allocated,
        "reconstructed_peak_bytes": peak,
        "peak_trace_index": peak_index,
        "live_allocations_at_peak": len(peak_live),
        "categories": dict(grouped),
        "no_addresses_exported": True,
        "scope": "peak inside retained trace, cross-checked against max_memory_allocated",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.input_dir / "wrn-fresh-result.json").read_text())
    by_label = {item["label"]: item for item in report["records"]}
    output = []
    for label in ("disabled", "tight"):
        for phase in ("first", "warm"):
            record = reconstruct(args.input_dir / f"fresh-{label}-{phase}-snapshot.pkl")
            observed = by_label[label]
            record["configuration"] = label
            record["phase"] = phase
            record["observed_max_memory_allocated"] = observed[f"{phase}_call_peak"]
            record["probe_start_allocated"] = observed["start_allocated"]
            assert record["reconstructed_peak_bytes"] == record["observed_max_memory_allocated"]
            assert record["end_allocated_bytes"] == observed[f"{phase}_call_current"]
            assert record["categories"]["allocated_before_probe"]["bytes"] == observed["start_allocated"]
            record["matches_observed_peak"] = True
            output.append(record)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps([{key: row[key] for key in ("configuration", "phase", "reconstructed_peak_bytes")}
                      for row in output]))


if __name__ == "__main__":
    main()
