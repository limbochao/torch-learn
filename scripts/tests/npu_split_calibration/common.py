"""Small standard-library helpers shared by the calibration suite."""

from __future__ import annotations

import hashlib
import json
import os
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

from policy import digest


HERE = Path(__file__).resolve().parent
TIMING_METHOD = "npu-profiler-kernel-sum-v1"
METRIC = "kernel_us"


def timing_contract(settings):
    return dict(method=TIMING_METHOD, metric=METRIC, csv="kernel_details.csv",
                profiler_level=0, profiler_warmup_steps=0, execution="direct", repeats=settings["repeats"],
                rotate_buffers=settings["rotate_buffers"])


def require_profiler_settings(settings):
    if (settings.get("timing_method") != TIMING_METHOD or settings.get("executions") != ["direct"]
            or settings.get("criteria", {}).get("metric") != METRIC):
        raise ValueError("Event/Graph timing data is invalid for calibration; collect a new profiler run in a new directory")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    files = {path.name: file_hash(path) for path in HERE.glob("*.py")}
    recorder = recorder_path()
    files["autotune_tiling.py"] = file_hash(recorder)
    files["npu_profiler.py"] = file_hash(profiler_path())
    return files


def measurement_settings(settings):
    # The memory budget only admits/excludes cases before measurement. Raising
    # it cannot change an already completed case's tensors, code or timing.
    return {key: value for key, value in settings.items() if key != "max_working_mib"}


def percentage(value):
    """Canonical decimal text, so CLI boundaries do not depend on float rounding."""
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("percentage must be a finite decimal in [0, 100]") from exc
    if not number.is_finite() or not 0 <= number <= 100:
        raise ValueError("percentage must be a finite decimal in [0, 100]")
    text = format(number, "f")
    return (text.rstrip("0").rstrip(".") if "." in text else text) if number else "0"


def collection_selection(universe, start="0", end="100", shards=1, shard_index=0):
    """Select before memory exclusions; adjacent percentages share an exact boundary."""
    start, end = percentage(start), percentage(end)
    if not Fraction(start) < Fraction(end):
        raise ValueError("require 0 <= start-percent < end-percent <= 100")
    if shards < 1 or not 0 <= shard_index < shards:
        raise ValueError("invalid shard index/count")
    if shards != 1 and (start, end) != ("0", "100"):
        raise ValueError("percentage ranges cannot be combined with --shards")
    total = len(universe)
    first, stop = (int(Fraction(p) * total // 100) for p in (start, end))
    if shards == 1:
        selected = universe[first:stop]
        partition = dict(kind="percent", start_percent=start, end_percent=end,
                         start_index=first, end_index=stop)
    else:
        selected = universe[shard_index::shards]
        partition = dict(kind="shard", shards=shards, shard_index=shard_index)
    return selected, dict(schema="npu-split-collection-v1", universe=universe,
                          universe_digest=digest(universe), partition=partition,
                          selected_case_ids=[case["id"] for case in selected])


def collection_complete(manifest):
    collection = manifest.get("collection")
    if collection is None:
        return True  # Preserve analysis of older, unsliced manifests.
    return set(collection["selected_case_ids"]) == {case["id"] for case in collection["universe"]}


def validate_collection(manifest):
    """Reject altered manifests before offline merging or analyzing a new collection."""
    collection = manifest.get("collection", {})
    if collection.get("schema") != "npu-split-collection-v1":
        raise ValueError("merge requires collection metadata from the percentage-capable runner")
    universe = collection["universe"]
    if not universe or collection["universe_digest"] != digest(universe):
        raise ValueError("collection universe is empty or its digest differs")
    by_id = {case["id"]: case for case in universe}
    if any(not isinstance(key, str) or key in ("", ".", "..") or "/" in key or "\\" in key for key in by_id):
        raise ValueError("case IDs must be safe directory names")
    if len(by_id) != len(universe):
        raise ValueError("duplicate case IDs in collection universe")
    selected = collection["selected_case_ids"]
    if len(set(selected)) != len(selected) or set(selected) - by_id.keys():
        raise ValueError("invalid selected case IDs")
    partition = collection["partition"]
    if partition["kind"] == "percent":
        _, expected = collection_selection(universe, partition["start_percent"], partition["end_percent"])
    elif partition["kind"] == "shard":
        _, expected = collection_selection(universe, shards=partition["shards"], shard_index=partition["shard_index"])
    elif partition["kind"] == "merged":
        expected = collection
    else:
        raise ValueError("unknown collection partition")
    if expected != collection:
        raise ValueError("collection boundaries or selected cases differ from their definition")
    declared = [*manifest["cases"], *(item["case"] for item in manifest["excluded_cases"])]
    if len({case["id"] for case in declared}) != len(declared):
        raise ValueError("duplicate included/excluded case IDs")
    expected_ids = set(by_id) if partition["kind"] == "merged" else set(selected)
    if {case["id"] for case in declared} != expected_ids:
        raise ValueError("manifest cases differ from the collection domain")
    if any(by_id.get(case["id"]) != case for case in declared):
        raise ValueError("case definition differs from the collection universe")
    return universe


def recorder_path():
    # Recorded source snapshots keep the shared recorder alongside this module.
    snapshot = HERE / "autotune_tiling.py"
    return snapshot if snapshot.exists() else HERE.parents[1] / "tools" / "autotune_tiling.py"


def profiler_path():
    snapshot = HERE / "npu_profiler.py"
    return snapshot if snapshot.exists() else HERE.parents[1] / "tools" / "npu_profiler.py"


def external_output(path):
    path = Path(path).expanduser().resolve()
    for parent in HERE.parents:
        if (parent / ".git").exists() and (path == parent or parent in path.parents):
            raise ValueError(f"raw results must be outside repository: {parent}")
    if path == HERE or HERE in path.parents:
        raise ValueError("raw results must be outside the script directory")
    return path
