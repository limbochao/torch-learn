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
        family = "npu" if backend == "npu" else ("cuda" if gpu_count == 1 else f"cuda-{index + 1}")
        scenario = f"npu-{kind}" if backend == "npu" else family + ("-plain" if kind == "plain" else "")
        label = f"A5 {kind}" if backend == "npu" else device.replace("NVIDIA GeForce ", "") + f" {kind}"
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
                    splits = run.get("kind_splits", {}).get(kind, [1] if kind == "plain" else run["splits"])
                    if {r["RS"] for r in selected} != set(splits):
                        status = "INCOMPLETE"
                    counts = Counter(r["status"] for r in selected)
                    result.append({"scenario": scenario, "device_name": device,
                                   "device_uuid": best.get("device_uuid", ""),
                                   "physical_device": best.get("physical_device", ""), "backend": backend,
                                   "kind": kind, "operation": operation, "dtype": "float32", "X": rows,
                                   "cooperative": kind != "plain", "device_run": family,
                                   "R": count, "best_RS": best.get("RS", ""), "median_us": best.get("median_us", ""),
                                   "min_us": best.get("min_us", ""), "max_us": best.get("max_us", ""),
                                   "status": status, "candidate_count": len(selected), "pass_count": len(valid),
                                   "wrong_count": counts["WRONG"], "error_count": counts["ERROR"] + counts[
                                       "PROCESS_ERROR"], "timeout_count": counts["TIMEOUT"],
                                   "over_core": best.get("over_core", ""), "block": run["block"],
                                   "RS_candidates": "|".join(map(str, splits)), "source": str(path)})
    return result, series


def attach_plain(records):
    baselines = {(r["device_run"], r["operation"], r["X"], r["R"]): r
                 for r in records if r["kind"] == "plain"}
    for record in records:
        baseline = baselines.get((record["device_run"], record["operation"], record["X"], record["R"]), {})
        compatible = bool(baseline) and baseline["status"] == record["status"] == "PASS"
        compatible = compatible and baseline["physical_device"] == record["physical_device"]
        compatible = compatible and baseline["device_uuid"] == record["device_uuid"]
        record["plain_us"] = baseline["median_us"] if compatible else ""
        record["speedup_vs_plain"] = baseline["median_us"] / record["median_us"] if compatible else ""


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
                          marker="o", markersize=4, color=COLORS[n % len(COLORS)], label=label,
                          linestyle="--" if any(r["kind"] == "plain" for r in selected) else "-")
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
    figure.suptitle("fp32 (a+b) reduction: plain and fastest measured correct cooperative RS\n"
                    f"Fixed BLOCK={block}; plain has no grid barrier; shading = sample min/max", fontsize=13)
    export(figure, out, "best-rs-latency")
    plt.close(figure)


def draw_splits(records, series, out):
    records = [r for r in records if r["cooperative"]]
    series = [(key, label) for key, label in series if any(r["scenario"] == key for r in records)]
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
                    mark = record["backend"] == "npu" and record["over_core"]
                    text = str(record["best_RS"]) + ("*" if mark else "")
                else:
                    text = "--"
                axis.text(j, i, text, ha="center", va="center", color="#111111", fontsize=10)
        axis.imshow(values, cmap="YlGnBu", norm=normalization, aspect="auto", alpha=0.62)
        axis.set_title(operation)
        axis.set_xticks(range(len(counts)), [label_r(r) for r in counts])
        axis.set_yticks(range(len(labels)), [f"{label}, X={rows}" for _, label, rows in labels])
        axis.set_xlabel("R elements (K=1024)")
    figure.suptitle("Selected RS (correct configurations only)\n"
                    "Color = log2(RS); * = A5 X*RS exceeds vector cores (measured PASS)", fontsize=13)
    export(figure, out, "best-rs-choice")
    plt.close(figure)


def draw_speedups(records, series, out):
    records = [r for r in records if r["cooperative"] and r["speedup_vs_plain"] != ""]
    if not records:
        return
    operations = [op for op in ("sum", "amax") if any(r["operation"] == op for r in records)]
    rows = sorted({r["X"] for r in records})
    counts = sorted({r["R"] for r in records})
    figure, axes = plt.subplots(len(operations), len(rows), figsize=(5.2 * len(rows), 4.1 * len(operations)),
                                squeeze=False, constrained_layout=True)
    for i, operation in enumerate(operations):
        for j, row in enumerate(rows):
            axis = axes[i, j]
            axis.axhline(1, color="#555555", linestyle="--", linewidth=1)
            for n, (scenario, label) in enumerate(series):
                selected = sorted((r for r in records if (r["scenario"], r["operation"], r["X"]) == (
                    scenario, operation, row)), key=lambda r: r["R"])
                if selected:
                    axis.plot([r["R"] for r in selected], [r["speedup_vs_plain"] for r in selected],
                              marker="o", markersize=4, color=COLORS[n % len(COLORS)], label=label)
            axis.set_title(f"{operation}: X={row}")
            axis.set_xscale("log", base=2)
            axis.set_xticks(counts, [label_r(r) for r in counts], rotation=35)
            axis.set_yscale("log")
            axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            axis.yaxis.set_major_formatter(StrMethodFormatter("{x:g}x"))
            axis.yaxis.set_minor_locator(NullLocator())
            axis.set_xlabel("R elements (K=1024)")
            axis.set_ylabel("Plain time / cooperative time (log scale)")
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=min(5, len(labels)))
    figure.suptitle("Cooperative speedup over same-device plain reduction\n"
                    "Above 1 = faster; below 1 = slower; best correct measured RS", fontsize=13)
    export(figure, out, "cooperative-speedup")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=Path)
    parser.add_argument("--cuda", type=Path, action="append", default=[], help="may be repeated for other GPUs")
    parser.add_argument("--npu-plain", type=Path, help="separately collected plain baseline for --npu")
    parser.add_argument("--cuda-plain", type=Path, action="append", default=[],
                        help="separate plain baselines, in the same order as --cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.npu and not args.cuda:
        parser.error("provide --npu and/or --cuda candidates.json")
    if args.npu_plain and not args.npu:
        parser.error("--npu-plain requires --npu")
    if args.cuda_plain and len(args.cuda_plain) != len(args.cuda):
        parser.error("provide one --cuda-plain for each --cuda, or use combined scans")
    records, series = [], []
    sources = ([(args.npu, "npu", 0)] if args.npu else []) + [
        (path, "cuda", i) for i, path in enumerate(args.cuda)]
    sources += ([(args.npu_plain, "npu", 0)] if args.npu_plain else []) + [
        (path, "cuda", i) for i, path in enumerate(args.cuda_plain)]
    for path, backend, index in sources:
        loaded, labels = load(path, backend, index, len(args.cuda))
        if any(key in {key for key, _ in series} for key, _ in labels):
            parser.error("duplicate series; do not add a separate baseline to a scan already containing plain")
        records.extend(loaded)
        series.extend(labels)
    if len({r["block"] for r in records}) != 1:
        parser.error("comparison requires the same BLOCK across runs")
    attach_plain(records)
    order = {key: i for i, (key, _) in enumerate(series)}
    records.sort(key=lambda r: (("sum", "amax").index(r["operation"]), r["X"], r["R"], order[r["scenario"]]))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "best-rs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    draw_times(records, series, args.out)
    if any(r["cooperative"] and r["best_RS"] for r in records):
        draw_splits(records, series, args.out)
    draw_speedups(records, series, args.out)
    print(f"{len(records)} best-RS rows: {dict(Counter(r['status'] for r in records))}")


if __name__ == "__main__":
    main()
