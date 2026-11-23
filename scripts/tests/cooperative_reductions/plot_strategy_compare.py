"""Rebuild strategy comparison figures and CSV from saved device measurements."""

import argparse
import csv
import json
import math
from pathlib import Path

from plot_barrier_suite import export
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator, StrMethodFormatter


def load(path, backend, suite):
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a nonempty results.json list")
    normalized, seen = [], set()
    for record in records:
        key = tuple(str(record.get(field)) for field in ("workload", "X", "R", "S", "strategy"))
        if key in seen:
            raise ValueError(f"{path}: duplicate configuration {key}")
        seen.add(key)
        timing = record.get("timing", {})
        if record["status"] == "PASS":
            values = [timing[field] for field in ("min_us", "median_us", "max_us")]
            if not all(math.isfinite(value) and value > 0 for value in values) or values != sorted(values):
                raise ValueError(f"{path}: invalid timing for {key}")
        normalized.append({"backend": backend, "suite": suite, "workload": record["workload"],
                           "X": int(record["X"]), "R": int(record["R"]), "S": record.get("S", ""),
                           "strategy": record["strategy"], "status": record["status"],
                           "actual_cooperative": bool(record.get("source_summary", {}).get(
                               "cooperative_decorators", 0)) if backend == "cuda" else (
                                   record["strategy"] == "cooperative"),
                           "run_calls": record.get("source_summary", {}).get("run_calls", ""),
                           "barrier_calls": record.get("source_summary", {}).get("barrier_calls", ""),
                           "min_us": timing.get("min_us", ""), "median_us": timing.get("median_us", ""),
                           "max_us": timing.get("max_us", ""), "source": str(path)})
    return normalized


def draw_cuda(records, out):
    selected = [r for r in records if r["backend"] == "cuda" and r["suite"] == "main"]
    if not selected:
        return
    workloads = [name for name in ("sum", "rmsnorm", "softmax", "chain")
                 if any(r["workload"] == name for r in selected)]
    figure, axes = plt.subplots(len(workloads), 1, figsize=(10, 2.7 * len(workloads)), squeeze=False,
                                constrained_layout=True)
    for axis, workload in zip(axes.flat, workloads):
        shapes = sorted({(r["X"], r["R"]) for r in selected if r["workload"] == workload})
        labels = []
        for index, (rows, count) in enumerate(shapes):
            candidates = {r["strategy"]: r for r in selected if (
                r["workload"], r["X"], r["R"]) == (workload, rows, count) and r["status"] == "PASS"}
            labels.append(f"X={rows}, R={count:,}")
            if not {"split", "cooperative"} <= candidates.keys():
                axis.text(index, 0.1, "missing / failed", ha="center")
                continue
            cooperative = candidates["cooperative"]
            ratio = candidates["split"]["median_us"] / cooperative["median_us"]
            actual = cooperative["actual_cooperative"]
            axis.bar(index, ratio, color="#167d8d" if actual else "#aaaaaa", width=0.65,
                     hatch=None if actual else "//")
            axis.text(index, ratio + 0.04, f"{ratio:.2f}x" + ("" if actual else " (split fallback)"),
                      ha="center", fontsize=9)
        axis.axhline(1, color="#444444", linestyle="--", linewidth=1)
        axis.set_xticks(range(len(shapes)), labels)
        axis.set_title("4 dependent softmax updates" if workload == "chain" else workload)
        axis.set_ylabel("Split / cooperative time")
        axis.set_ylim(0, max(1.45, max((p.get_height() for p in axis.patches), default=1) + 0.4))
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("RTX 4090: actual Inductor graphs, fp32\n"
                    "Above 1 = cooperative faster; gray = cooperative was not selected", fontsize=13)
    export(figure, out, "cuda-strategies")
    plt.close(figure)


def draw_npu(records, out):
    selected = [r for r in records if r["backend"] == "npu"]
    if not selected:
        return
    shapes = sorted({(r["workload"], r["X"], r["R"]) for r in selected})
    columns = min(2, len(shapes))
    rows = math.ceil(len(shapes) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(7 * columns, 3.5 * rows), squeeze=False,
                                constrained_layout=True)
    for axis, (workload, rows, count) in zip(axes.flat, shapes):
        candidates = [r for r in selected if (r["workload"], r["X"], r["R"]) == (workload, rows, count)]
        for strategy, color in (("split", "#3569b0"), ("cooperative", "#167d8d")):
            series = sorted((r for r in candidates if r["strategy"] == strategy and r["status"] == "PASS"),
                            key=lambda r: r["S"])
            axis.plot([r["S"] for r in series], [r["median_us"] for r in series], marker="o", color=color,
                      label="Two kernels" if strategy == "split" else "One kernel + all_vector")
            axis.fill_between([r["S"] for r in series], [r["min_us"] for r in series],
                              [r["max_us"] for r in series], alpha=0.1, color=color)
        plain = next((r for r in candidates if r["strategy"] == "plain" and r["status"] == "PASS"), None)
        if plain:
            axis.axhline(plain["median_us"], label="Unsplit prototype", color="#777777", linestyle="--")
        axis.set_title(f"{workload}: X={rows}, R={count:,}")
        axis.set_xlabel("R-axis shard count S")
        axis.set_ylabel("Full pipeline time (us)")
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        splits = sorted({r["S"] for r in candidates})
        axis.set_xticks(splits, [str(value) for value in splits])
        axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5, 7)))
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        axis.yaxis.set_minor_locator(NullLocator())
        axis.grid(alpha=0.2, which="both")
        axis.legend(fontsize=9)
    for axis in list(axes.flat)[len(shapes):]:
        axis.axis("off")
    figure.suptitle("A5 SIMD: matched handwritten prototypes, fp32, BLOCK=1024\n"
                    "Shading = sample min/max; these are not integrated Inductor results", fontsize=13)
    export(figure, out, "npu-strategies")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", type=Path)
    parser.add_argument("--cuda-mixed", type=Path)
    parser.add_argument("--cuda-hybrid", type=Path)
    parser.add_argument("--npu", type=Path)
    parser.add_argument("--npu-recheck", type=Path, action="append", default=[],
                        help="replace matching NPU points with this targeted repeat; may be specified again")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not any((args.cuda, args.cuda_mixed, args.cuda_hybrid, args.npu)):
        parser.error("provide at least one results.json input")
    if args.npu_recheck and not args.npu:
        parser.error("--npu-recheck requires --npu")
    records = []
    for name, backend, suite in (("cuda", "cuda", "main"), ("cuda_mixed", "cuda", "mixed"),
                                 ("cuda_hybrid", "cuda", "hybrid"), ("npu", "npu", "main")):
        if getattr(args, name):
            records.extend(load(getattr(args, name), backend, suite))
    for path in args.npu_recheck:
        for replacement in load(path, "npu", "main"):
            fields = ("backend", "suite", "workload", "X", "R", "S", "strategy")
            matching = [i for i, record in enumerate(records) if all(
                record[field] == replacement[field] for field in fields)]
            if len(matching) != 1:
                raise ValueError(f"{path}: targeted repeat must match exactly one original point")
            records[matching[0]] = replacement
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    draw_cuda(records, args.out)
    draw_npu(records, args.out)
    print(f"{len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
