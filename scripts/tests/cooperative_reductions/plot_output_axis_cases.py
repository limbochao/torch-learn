"""Merge NPU/CUDA output-axis results into SVG/PNG, matching the exact X/R shape."""

import argparse
import csv
import json
import math
from pathlib import Path

from plot_barrier_suite import export
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SERIES = (("npu", "original", "NPU: upstream semaphore", "#ba3d32", "o"),
          ("npu", "native", "NPU: all_vector", "#087f8c", "s"),
          ("npu", "debug", "NPU: debug + all_vector + debug", "#8860ac", "^"),
          ("cuda", "original", "CUDA: upstream semaphore", "#3569b0", "D"))
COLORS = {"npu": "#555555", "cuda": "#3569b0"}


def load_results(path, backend):
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a nonempty output-axis.json list")
    manifest_path = path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    keys, normalized = set(), []
    for record in records:
        if record.get("backend", backend) != backend or manifest.get("backend", backend) != backend:
            raise ValueError(f"{path}: backend does not match --{backend}")
        for field in ("X", "R", "S", "core_count", "kind", "status"):
            if field not in record:
                raise ValueError(f"{path}: missing {field}; use output_axis_cases.py results")
        if min(record[field] for field in ("X", "R", "S", "core_count")) <= 0:
            raise ValueError(f"{path}: X/R/S/core_count must be positive")
        key = tuple(record[field] for field in ("X", "R", "S", "kind"))
        if key in keys:
            raise ValueError(f"{path}: duplicate configuration {key}")
        keys.add(key)
        if record["status"] == "PASS":
            times = [record[field] for field in ("min_us", "median_us", "max_us")]
            if not all(math.isfinite(value) and value > 0 for value in times) or times != sorted(times):
                raise ValueError(f"{path}: invalid PASS timings for {key}")
        normalized.append(dict(record, backend=backend,
                               device_name=record.get("device_name", manifest.get("device_name", backend.upper()))))
    if len({record["core_count"] for record in normalized}) != 1:
        raise ValueError(f"{path}: mixed devices/core counts; supply one GPU run at a time")
    return normalized, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", "--data", dest="npu", type=Path, help="NPU output-axis.json (--data is an alias)")
    parser.add_argument("--cuda", type=Path, help="CUDA output-axis.json; optional until GPU measurements exist")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.npu and not args.cuda:
        parser.error("provide --npu and/or --cuda")
    records, manifests = [], {}
    for backend in ("npu", "cuda"):
        path = getattr(args, backend)
        if path:
            loaded, manifests[backend] = load_results(path, backend)
            records.extend(loaded)
    for x, count in {(r["X"], r["R"]) for r in records}:
        settings = {(r.get("dtype"), r.get("block")) for r in records if (r["X"], r["R"]) == (x, count)}
        if len(settings) > 1:
            raise ValueError(f"X={x}, R={count}: dtype/BLOCK differ between inputs")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = sorted({r["X"] for r in records})
    counts = sorted({r["R"] for r in records})
    splits = sorted({r["S"] for r in records})
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(len(rows), len(counts), figsize=(7 * len(counts), 4 * len(rows) + 1.8), squeeze=False,
                                constrained_layout=True)
    for i, x in enumerate(rows):
        for j, count in enumerate(counts):
            axis = axes[i, j]
            selected = [r for r in records if r["X"] == x and r["R"] == count]
            if not selected:
                axis.axis("off")
                continue
            for index, (backend, kind, label, color, marker) in enumerate(SERIES):
                series = {r["S"]: r for r in selected if r["backend"] == backend and r["kind"] == kind}
                if not series:
                    continue
                values = [series[s]["median_us"] if series.get(s, {}).get("status") == "PASS"
                          else float("nan") for s in splits]
                axis.plot(splits, values, label=label, color=color, marker=marker, markersize=4)
                limits = [[series[s][key] if series.get(s, {}).get("status") == "PASS" else float("nan")
                           for s in splits] for key in ("min_us", "max_us")]
                axis.fill_between(splits, *limits, color=color, alpha=0.1)
                failed = [s for s, r in series.items() if r["status"] != "PASS" and not r["status"].startswith("SKIP_")]
                axis.plot(failed, [0.025 + index * 0.025] * len(failed),
                          transform=axis.get_xaxis_transform(), linestyle="none", marker="x", color=color)
            occupancy = []
            for backend in manifests:
                device_records = [r for r in selected if r["backend"] == backend]
                label = "CUDA" if backend == "cuda" else "NPU"
                if not device_records:
                    occupancy.append(f"{label}: no matching data")
                    continue
                cores = device_records[0]["core_count"]
                state = "full" if x == cores else "underfilled" if x < cores else "more outputs than cores"
                occupancy.append(f"{label}: {state} ({cores} {'SMs' if backend == 'cuda' else 'cores'})")
                baseline = next((r for r in device_records if r["kind"] == "plain" and r["status"] == "PASS"), None)
                if baseline:
                    axis.axhline(baseline["median_us"], label=f"{label}: ordinary reduction (S=1)",
                                 color=COLORS[backend], linestyle=":" if backend == "npu" else "-.")
                    axis.axhspan(baseline["min_us"], baseline["max_us"], color=COLORS[backend], alpha=0.07)
                limit = cores / x
                if limit < max(splits):
                    if backend == "npu" or "npu" not in manifests:
                        axis.axvspan(max(limit, min(splits)), max(splits), color="#777777", alpha=0.07)
                    if limit >= min(splits):
                        axis.axvline(limit, color=COLORS[backend], linewidth=0.8, linestyle="--")
            axis.set_title(f"X={x}, R={count:,}\n" + "; ".join(occupancy), fontsize=9.5)
            axis.set_xlabel("R-axis split count S")
            axis.set_ylabel("Device time per kernel (us; log scale)")
            ticks = splits if len(splits) <= 5 else sorted({splits[0], splits[-1]} | set(splits) & {8, 16, 32, 48, 64})
            axis.set_xticks(ticks)
            axis.set_yscale("log")
            axis.grid(True, alpha=0.2, which="both")
    legend = {}
    for axis in axes.flat:
        handles, labels = axis.get_legend_handles_labels()
        legend.update(zip(labels, handles))
    labels, handles = list(legend), list(legend.values())
    handles.append(Line2D([], [], marker="x", linestyle="none", color="black"))
    labels.append("Error; no timing")
    figure.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    device = manifests.get("cuda", {}).get("device_name", "CUDA")
    comparison = f"NPU / {device}" if args.npu and args.cuda else device if args.cuda else "NPU; CUDA data pending"
    gray = "NPU over-core diagnostics" if args.npu else "CUDA over-SM budget (skipped)"
    figure.suptitle(f"Output-axis occupancy and cooperative split count: {comparison}\n"
                    f"Median with min/max bands; gray: {gray}; vertical dashed lines: device grid budgets", fontsize=12)
    export(figure, args.out, "output-axis-occupancy")
    plt.close(figure)
    fields = ("backend", "device_name", "core_count", "X", "R", "S", "kind", "dtype", "block", "status",
              "diagnostic_over_core", "median_us", "min_us", "max_us")
    with (args.out / "output-axis.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
