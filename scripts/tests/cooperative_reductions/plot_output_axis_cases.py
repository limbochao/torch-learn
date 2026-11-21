"""Plot output-axis occupancy cases, leaving wrong-result points untimed."""

import argparse
import csv
import json
from pathlib import Path

from plot_barrier_suite import export
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.data.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    rows = sorted({r["X"] for r in records})
    counts = sorted({r["R"] for r in records})
    splits = sorted({r["S"] for r in records})
    styles = (("original", "Upstream semaphore", "#ba3d32", "o"),
              ("native", "all_vector", "#087f8c", "s"),
              ("debug", "debug + all_vector + debug", "#8860ac", "^"))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(len(rows), len(counts), figsize=(14, 9), squeeze=False,
                                constrained_layout=True)
    for i, x in enumerate(rows):
        for j, count in enumerate(counts):
            axis = axes[i, j]
            selected = [r for r in records if r["X"] == x and r["R"] == count]
            cores = selected[0]["core_count"]
            for kind, label, color, marker in styles:
                series = {r["S"]: r for r in selected if r["kind"] == kind}
                values = [series[s]["median_us"] if series.get(s, {}).get("status") == "PASS"
                          else float("nan") for s in splits]
                axis.plot(splits, values, label=label, color=color, marker=marker, markersize=4)
                limits = [[series[s][key] if series.get(s, {}).get("status") == "PASS" else float("nan")
                           for s in splits] for key in ("min_us", "max_us")]
                axis.fill_between(splits, *limits, color=color, alpha=0.1)
                failed = [s for s, r in series.items() if r["status"] == "FAIL_WRONG_RESULT"]
                axis.plot(failed, [0.035 if kind == "native" else 0.07] * len(failed),
                          transform=axis.get_xaxis_transform(), linestyle="none", marker="x", color=color)
            baseline = next(r for r in selected if r["kind"] == "plain")
            axis.axhline(baseline["median_us"], label="Ordinary reduction (S=1)", color="#555555", linestyle=":")
            axis.axhspan(baseline["min_us"], baseline["max_us"], color="#555555", alpha=0.07)
            limit = cores / x
            if limit < max(splits):
                axis.axvspan(limit, max(splits), color="#777777", alpha=0.07)
                axis.axvline(limit, color="#777777", linewidth=0.8, linestyle="--")
            axis.set_title(f"X={x} ({'full' if x == cores else 'underfilled'} output axis), R={count:,}")
            axis.set_xlabel("R-axis split count S")
            axis.set_ylabel("Device time per kernel (us; log scale)")
            axis.set_xticks([1, 8, 16, 32, 48, 56])
            axis.set_yscale("log")
            axis.grid(True, alpha=0.2, which="both")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], marker="x", linestyle="none", color="black"))
    labels.append("Wrong result; no timing")
    figure.legend(handles, labels, loc="outside lower center", ncol=3)
    figure.suptitle("A5 output-axis occupancy and cooperative split count\n"
                    "Median with min/max bands; gray area: over-core mapping diagnostics", fontsize=14)
    export(figure, args.out, "output-axis-occupancy")
    plt.close(figure)
    fields = ("X", "R", "S", "kind", "status", "diagnostic_over_core", "median_us", "min_us", "max_us")
    with (args.out / "output-axis.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
