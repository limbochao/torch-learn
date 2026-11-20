"""Plot device-event results collected by barrier_suite.py."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SERIES = (
    ("npu", "original", "A5: upstream semaphore", "#ba3d32", "o"),
    ("npu", "native", "A5: all_vector", "#087f8c", "s"),
    ("npu", "debug", "A5: debug + all_vector + debug", "#8860ac", "^"),
    ("cuda", "original", "RTX 4090: upstream semaphore", "#3569b0", "D"),
)


def export(figure, directory, stem):
    for suffix in ("svg", "png"):
        path = directory / f"{stem}.{suffix}"
        figure.savefig(path, dpi=170)
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def draw(axis, records, workload, rows, count, zoom=False):
    for backend, kind, label, color, marker in SERIES:
        if zoom and backend == "npu" and kind == "original":
            continue
        selected = sorted((r for r in records if r["backend"] == backend and r["kind"] == kind
                           and (r["workload"], r["X"], r["R"]) == (workload, rows, count)
                           and (not zoom or workload == "exchange" or r["S"] >= 8)),
                          key=lambda r: r["S"])
        if not selected:
            continue
        splits = [r["S"] for r in selected]
        axis.plot(splits, [r["median_us"] for r in selected], label=label, color=color,
                  marker=marker, linewidth=1.6, markersize=4)
        axis.fill_between(splits, [r["min_us"] for r in selected], [r["max_us"] for r in selected],
                          color=color, alpha=0.09)
    title = "Small integer exchange" if workload == "exchange" else f"Sum: X={rows}, R={count:,}"
    axis.set_title(title + (" (S >= 8)" if zoom and workload != "exchange" else ""))
    axis.set_xlabel("R-axis split count S")
    axis.set_ylabel("Device time per kernel (us)" + ("; log scale" if not zoom else ""))
    axis.set_xticks([8, 16, 32, 48, 56] if zoom and workload != "exchange" else [1, 8, 16, 32, 48, 56])
    if not zoom:
        axis.set_yscale("log")
    axis.grid(True, alpha=0.25, which="both")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for backend, source in (("npu", args.npu), ("cuda", args.cuda)):
        records.extend(dict(record, backend=backend) for record in json.loads(source.read_text())
                       if record["status"] == "PASS")
    fields = ("backend", "kind", "workload", "X", "R", "S", "median_us", "min_us", "max_us", "sample_count")
    with (args.out / "timings.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    workloads = [("exchange", 1, 0)] + [("sum", x, r) for x, r in
                  ((1, 4096), (1, 32768), (1, 1048576), (1, 1048593), (1, 16777216), (4, 1048576))]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)
    for axis, workload in zip(axes.flat, workloads):
        draw(axis, records, *workload)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", fontsize=11)
    axes.flat[-1].text(0.05, 0.15, "Median of device-event samples; shading = min/max.\n"
                        "Same all-read algorithm, fp32 accumulation, BLOCK=1024.\n"
                        "CUDA used shared physical GPU 6; A5 used physical NPU 2.\n"
                        "A full-kernel comparison, not isolated barrier latency.",
                        transform=axes.flat[-1].transAxes, fontsize=10)
    figure.suptitle("Cooperative reduction: barrier implementation and split count", fontsize=16)
    export(figure, args.out, "barrier-comparison")
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for axis, workload in zip(axes, (("exchange", 1, 0), ("sum", 1, 1048576), ("sum", 1, 16777216))):
        draw(axis, records, *workload, zoom=True)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3)
    figure.suptitle("Linear-scale detail: the two A5 rewrites and CUDA reference", fontsize=14)
    export(figure, args.out, "barrier-rewrites-detail")
    plt.close(figure)


if __name__ == "__main__":
    main()
