"""Select the fastest correct RS per shape and rebuild comparison CSV/SVG without a device."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

from plot_barrier_suite import export
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import LogLocator, NullLocator, StrMethodFormatter


COLORS = ("#ba3d32", "#087f8c", "#3569b0", "#8860ac", "#b87520")


def label_r(count):
    return f"{count // 1048576}M" if count >= 1048576 and count % 1048576 == 0 else f"{count // 1024}K"


def load(path, backend, index, gpu_count):
    records = json.loads(path.read_text())
    run = json.loads(path.with_name("run.json").read_text())
    if run["backend"] != backend:
        raise ValueError(f"{path}: wrong backend")
    keys = set()
    for record in records:
        key = tuple(record[field] for field in ("operation", "X", "R", "kind", "RS"))
        if key in keys:
            raise ValueError(f"{path}: duplicate candidate {key}")
        keys.add(key)
        if record["status"] == "PASS":
            values = [record[field] for field in ("min_us", "median_us", "max_us")]
            if values != sorted(values) or not all(math.isfinite(v) and v > 0 for v in values):
                raise ValueError(f"{path}: invalid timing {key}")
            checks = [*record["accuracy"], record["post_accuracy"]]
            if any(item["mismatched_values"] for item in checks) or "error" in record["exchange"]:
                raise ValueError(f"{path}: invalid PASS accuracy {key}")
    device = next((r["device_name"] for r in records if "device_name" in r), backend.upper())
    result, series = [], []
    for kind in run["kinds"]:
        scenario = f"npu-{kind}" if backend == "npu" else ("cuda" if gpu_count == 1 else f"cuda-{index + 1}")
        label = f"A5 {kind}" if backend == "npu" else device.replace("NVIDIA GeForce ", "")
        if backend == "cuda" and gpu_count > 1:
            label += f" (run {index + 1})"
        series.append((scenario, label))
        for operation in run["operations"]:
            for rows in run["rows"]:
                for count in run["rnumels"]:
                    selected = [r for r in records if (r["operation"], r["X"], r["R"], r["kind"]) == (
                        operation, rows, count, kind)]
                    valid = [r for r in selected if r["status"] == "PASS"]
                    best = min(valid, key=lambda r: r["median_us"], default={})
                    status = "PASS" if best else "NO_VALID_RS"
                    if {r["RS"] for r in selected} != set(run["splits"]):
                        status = "INCOMPLETE"
                    counts = Counter(r["status"] for r in selected)
                    result.append({"scenario": scenario, "device_name": device, "backend": backend,
                                   "kind": kind, "operation": operation, "dtype": "float32", "X": rows,
                                   "R": count, "best_RS": best.get("RS", ""), "median_us": best.get("median_us", ""),
                                   "min_us": best.get("min_us", ""), "max_us": best.get("max_us", ""),
                                   "status": status, "candidate_count": len(selected), "pass_count": len(valid),
                                   "wrong_count": counts["WRONG"], "error_count": counts["ERROR"] + counts[
                                       "PROCESS_ERROR"], "timeout_count": counts["TIMEOUT"],
                                   "over_core": best.get("over_core", ""), "block": run["block"],
                                   "RS_candidates": "|".join(map(str, run["splits"])), "source": str(path)})
    return result, series


def draw_times(records, series, out):
    operations = [op for op in ("sum", "amax") if any(r["operation"] == op for r in records)]
    row_counts = sorted({r["X"] for r in records})
    counts = sorted({r["R"] for r in records})
    figure, axes = plt.subplots(len(operations), len(row_counts),
                                figsize=(5.2 * len(row_counts), 4.1 * len(operations)),
                                squeeze=False, constrained_layout=True)
    for i, operation in enumerate(operations):
        for j, rows in enumerate(row_counts):
            axis = axes[i, j]
            for n, (scenario, label) in enumerate(series):
                selected = sorted((r for r in records if (r["scenario"], r["operation"], r["X"], r["status"]) == (
                    scenario, operation, rows, "PASS")), key=lambda r: r["R"])
                axis.plot([r["R"] for r in selected], [r["median_us"] for r in selected],
                          marker="o", markersize=4, color=COLORS[n % len(COLORS)], label=label)
                axis.fill_between([r["R"] for r in selected], [r["min_us"] for r in selected],
                                  [r["max_us"] for r in selected], color=COLORS[n % len(COLORS)], alpha=0.1)
            axis.set_title(f"{operation}: X={rows}")
            axis.set_xscale("log", base=2)
            axis.set_xticks(counts, [label_r(r) for r in counts], rotation=35)
            axis.set_yscale("log")
            axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            axis.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            axis.yaxis.set_minor_locator(NullLocator())
            axis.set_xlabel("R elements (K=1024)")
            axis.set_ylabel("Best correct kernel time (us, log scale)")
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=min(5, len(series)))
    block = records[0]["block"]
    figure.suptitle("fp32 (a+b) reduction: fastest measured correct RS per shape\n"
                    f"Fixed BLOCK={block}; barrier retained at RS=1; shading = sample min/max", fontsize=13)
    export(figure, out, "best-rs-latency")
    plt.close(figure)


def draw_splits(records, series, out):
    operations = [op for op in ("sum", "amax") if any(r["operation"] == op for r in records)]
    row_counts = sorted({r["X"] for r in records})
    counts = sorted({r["R"] for r in records})
    labels = [(scenario, label, rows) for scenario, label in series for rows in row_counts]
    figure, axes = plt.subplots(1, len(operations), figsize=(7 * len(operations), 1.6 + 0.48 * len(labels)),
                                squeeze=False, constrained_layout=True)
    normalization = Normalize(vmin=0, vmax=max(math.log2(int(r["best_RS"])) for r in records if r["best_RS"]))
    import numpy as np

    for axis, operation in zip(axes.flat, operations):
        values = np.full((len(labels), len(counts)), np.nan)
        for i, (scenario, _, rows) in enumerate(labels):
            for j, count in enumerate(counts):
                record = next((r for r in records if (
                    r["scenario"], r["operation"], r["X"], r["R"], r["status"]) == (
                        scenario, operation, rows, count, "PASS")), {})
                if record:
                    values[i, j] = math.log2(record["best_RS"])
                    text = str(record["best_RS"])
                else:
                    text = "--"
                axis.text(j, i, text, ha="center", va="center", color="#111111", fontsize=10)
        axis.imshow(values, cmap="YlGnBu", norm=normalization, aspect="auto", alpha=0.62)
        axis.set_title(operation)
        axis.set_xticks(range(len(counts)), [label_r(r) for r in counts])
        axis.set_yticks(range(len(labels)), [f"{label}, X={rows}" for _, label, rows in labels])
        axis.set_xlabel("R elements (K=1024)")
    figure.suptitle("Selected RS (correct configurations only)\n"
                    "Numbers are shard counts; color represents log2(RS)", fontsize=13)
    export(figure, out, "best-rs-choice")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=Path)
    parser.add_argument("--cuda", type=Path, action="append", default=[], help="may be repeated for other GPUs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.npu and not args.cuda:
        parser.error("provide --npu and/or --cuda candidates.json")
    records, series = [], []
    sources = ([(args.npu, "npu", 0)] if args.npu else []) + [
        (path, "cuda", i) for i, path in enumerate(args.cuda)]
    for path, backend, index in sources:
        loaded, labels = load(path, backend, index, len(args.cuda))
        records.extend(loaded)
        series.extend(labels)
    if len({r["block"] for r in records}) != 1:
        parser.error("comparison requires the same BLOCK across runs")
    order = {key: i for i, (key, _) in enumerate(series)}
    records.sort(key=lambda r: (("sum", "amax").index(r["operation"]), r["X"], r["R"], order[r["scenario"]]))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "best-rs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    draw_times(records, series, args.out)
    if any(r["best_RS"] for r in records):
        draw_splits(records, series, args.out)
    print(f"{len(records)} best-RS rows: {dict(Counter(r['status'] for r in records))}")


if __name__ == "__main__":
    main()
