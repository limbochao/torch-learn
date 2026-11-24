"""Summarize verified device-duration results; classify emitted stages from saved code."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics

from common import write_json


def emitted_reductions(directory):
    kernels = []
    for path in sorted(directory.glob("output_code_*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute)
                    or node.func.attr != "triton" or len(node.args) < 2
                    or not isinstance(node.args[1], ast.Constant) or not isinstance(node.args[1].value, str)):
                continue
            source = node.args[1].value
            if "triton_heuristics.reduction(" not in source and "triton_heuristics.persistent_reduction(" not in source:
                continue
            for fn in ast.parse(source).body:
                if not isinstance(fn, ast.FunctionDef):
                    continue
                constants = {}
                for stmt in fn.body:
                    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant):
                        for target in stmt.targets:
                            if isinstance(target, ast.Name):
                                constants[target.id] = stmt.value.value
                kernels.append(dict(name=fn.name, x=constants.get("xnumel"), r=constants.get("r0_numel")))
    return kernels


def actual_path(record, case, directory):
    original = record.get("actual_path", "unknown")
    if record.get("entry") != "inductor" or original != "ordinary":
        return original, None
    kernels = emitted_reductions(directory)
    # The initial run instrumented Reduction but missed its online-softmax subclass.
    # Its saved DSL still provides an exact partial -> combine signature.
    for index, first in enumerate(kernels):
        x, r = first["x"], first["r"]
        if not isinstance(x, int) or not isinstance(r, int) or x <= case["X"] or x % case["X"]:
            continue
        splits = x // case["X"]
        for second in kernels[index + 1:]:
            if second["x"] == case["X"] and second["r"] == splits and r < case["R"] and r * splits >= case["R"]:
                return "split", dict(source="saved DSL partial/combine dimensions", first=first,
                                     second=second, splits=splits, original_annotation=original)
    return original, dict(source="saved DSL", reduction_dimensions=kernels)


def collect(root):
    records = []
    manifest = json.loads((root / "run.json").read_text())
    for path in sorted((root / "cases").glob("*/result.json")):
        value = json.loads(path.read_text())
        value.setdefault("submission", value["backend"].upper() + " Graph replay"
                         if manifest.get("profile_graph") == "1" else "direct callable")
        value["artifact_root"] = root.name
        value["artifact_case"] = str(path.parent.relative_to(root))
        if manifest.get("tuning_source"):
            value["tuning_source_run"] = Path(manifest["tuning_source"]).name
        for result in value.get("selected", []):
            result["verified_path"], result["classification_evidence"] = actual_path(
                result, value["case"], path.parent / result["strategy"])
        records.append(value)
    return records


def apply_confirmations(indexed, selections, confirmed_roots, retry_roots=()):
    confirmed = {}
    for root in confirmed_roots:
        for record in collect(root):
            source = record.get("tuning_source_run")
            if not source:
                raise ValueError(f"Confirmation has no fixed tuning source: {root}")
            key = source, record["case"]["id"]
            if key in confirmed:
                raise ValueError(f"Duplicate confirmation for {key}")
            confirmed[key] = record
    for root in retry_roots:
        for record in collect(root):
            key = record.get("tuning_source_run"), record["case"]["id"]
            old = confirmed.get(key)
            if old is None or old["status"] == "PASS" or record["status"] != "PASS":
                raise ValueError(f"Confirmation retry must replace an existing failed case: {key}")
            if (record["case"] != old["case"] or record["backend"] != old["backend"] or
                    record["submission"] != old["submission"]):
                raise ValueError(f"Confirmation retry case/submission differs: {key}")
            record["confirmation_retry"] = dict(artifact_root=old["artifact_root"],
                                               artifact_case=old["artifact_case"], status=old["status"])
            confirmed[key] = record
    requested = set()
    for path in selections:
        selection = json.loads(path.read_text())
        for run in selection["runs"]:
            for case_id in run["cases"]:
                key = run["source"], case_id
                if key in requested:
                    raise ValueError(f"Duplicate predeclared confirmation: {key}")
                requested.add(key)
                record = confirmed.get(key)
                if record is None or record["status"] != "PASS":
                    raise ValueError(f"Predeclared confirmation is missing or failed: {key}")
                original_run = record.get("confirmation_retry", {}).get("artifact_root", record["artifact_root"])
                if run.get("confirmation") and original_run != run["confirmation"]:
                    raise ValueError(f"Confirmation run differs from the predeclared selection: {key}")
                case_key = record["backend"], case_id
                old = indexed[case_key]
                source_roots = {old["artifact_root"], old.get("replaces", {}).get("artifact_root")}
                if run["source"] not in source_roots or record["case"] != old["case"]:
                    raise ValueError(f"Confirmation source/case differs: {key}")
                if record["submission"] != old["submission"]:
                    raise ValueError(f"Confirmation submission differs: {key}")
                history = list(old.get("confirmation_history", []))
                history.append(dict(artifact_root=old["artifact_root"], artifact_case=old["artifact_case"],
                                    status=old["status"], selection_file=path.name, reason=selection["reason"]))
                record["confirmation_history"] = history
                if old.get("replaces"):
                    record["replaces"] = old["replaces"]
                indexed[case_key] = record
    if requested != set(confirmed):
        raise ValueError("Confirmation input differs from the complete predeclared selection")


def summarize(records):
    rows = []
    for record in records:
        case = record["case"]
        selected = {item["strategy"]: item for item in record.get("selected", [])}
        row = dict(backend=record["backend"], **case, status=record["status"],
                   submission=record.get("submission", "direct callable"),
                   device=record.get("environment", {}).get("requested_device"),
                   artifact_root=record["artifact_root"], artifact_case=record["artifact_case"])
        if record.get("execution_policy"):
            row["execution_policy"] = record["execution_policy"]
        if record.get("replaces"):
            row["replaces"] = record["replaces"]
        if record.get("confirmation_history"):
            row["confirmation_history"] = record["confirmation_history"]
        if record.get("confirmation_retry"):
            row["confirmation_retry"] = record["confirmation_retry"]
        for name in ("plain", "split", "cooperative", "natural"):
            item = selected.get(name, {})
            timing = item.get("timing", {})
            row[name + "_status"] = item.get("status")
            row[name + "_path"] = item.get("verified_path")
            row[name + "_us"] = timing.get("median_us")
            row[name + "_min_us"] = timing.get("min_us")
            row[name + "_max_us"] = timing.get("max_us")
            row[name + "_events"] = timing.get("events_per_call")
            row[name + "_config"] = item.get("config") if item.get("entry") == "dsl_prototype" else item.get("selected_configs")
            row[name + "_config_phase"] = ("dsl_launch" if item.get("entry") == "dsl_prototype" else
                                            "initial_compile" if item.get("entry") == "inductor" else None)
            medians = timing.get("round_medians_us", [])
            row[name + "_session_variation"] = max(medians) / min(medians) if medians else None
        valid = all(selected.get(name, {}).get("status") == "PASS" for name in ("split", "cooperative"))
        strict = valid and row["split_path"] == "split" and row["cooperative_path"] == "cooperative"
        row["strict_split_cooperative_pair"] = strict
        if valid and row["split_us"] and row["cooperative_us"]:
            row["speedup_cooperative_vs_requested_split"] = row["split_us"] / row["cooperative_us"]
        if strict:
            speedup = row["split_us"] / row["cooperative_us"]
            row["speedup_cooperative_vs_split"] = speedup
            rounds = [a / b for a, b in zip(selected["split"]["timing"]["round_medians_us"],
                                           selected["cooperative"]["timing"]["round_medians_us"], strict=True)]
            row["paired_round_ratios"] = rounds
            row["outcome"] = ("cooperative" if speedup > 1.05 and min(rounds) > 1 else
                              "split" if speedup < 0.95 and max(rounds) < 1 else "close_or_variable")
            row["round_ratio_min"] = min(rounds)
            row["round_ratio_max"] = max(rounds)
            variation = max(max(selected[s]["timing"]["round_medians_us"]) /
                            min(selected[s]["timing"]["round_medians_us"]) for s in ("split", "cooperative"))
            row["session_variation_ratio"] = variation
            row["noisy"] = variation > 1.15
            if row["noisy"]:
                row["raw_outcome"] = row["outcome"]
                row["outcome"] = "noisy"
        else:
            row["outcome"] = "not_a_strict_pair"
        if row["plain_us"] and row["cooperative_us"]:
            row["speedup_cooperative_vs_plain"] = row["plain_us"] / row["cooperative_us"]
        if row["plain_us"] and row["split_us"]:
            row["speedup_split_vs_plain"] = row["plain_us"] / row["split_us"]
        if row["plain_us"] and row["natural_us"]:
            row["speedup_natural_vs_plain"] = row["plain_us"] / row["natural_us"]
        baseline = selected.get("plain", {})
        if baseline.get("status") == "PASS" and row.get("plain_us"):
            for strategy in ("split", "cooperative", "natural"):
                contender = selected.get(strategy, {})
                if contender.get("status") != "PASS" or not row.get(strategy + "_us"):
                    continue
                ratio = row["plain_us"] / row[strategy + "_us"]
                directions = [a / b for a, b in zip(baseline["timing"]["round_medians_us"],
                                                   contender["timing"]["round_medians_us"], strict=True)]
                noisy = max(row["plain_session_variation"], row[strategy + "_session_variation"]) > 1.15
                row[strategy + "_vs_plain_outcome"] = ("noisy" if noisy else strategy if ratio > 1.05 and min(directions) > 1
                                                       else "plain" if ratio < .95 and max(directions) < 1 else "close_or_variable")
        if (row["natural_path"] == "cooperative" and row["split_path"] == "split" and
                row["natural_status"] == "PASS" and row["split_status"] == "PASS"):
            ratio = row["split_us"] / row["natural_us"]
            paired = [a / b for a, b in zip(selected["split"]["timing"]["round_medians_us"],
                                          selected["natural"]["timing"]["round_medians_us"], strict=True)]
            row["natural_speedup_vs_split"] = ratio
            row["natural_outcome"] = ("cooperative" if ratio > 1.05 and min(paired) > 1 else
                                      "split" if ratio < 0.95 and max(paired) < 1 else "close_or_variable")
            row["natural_noisy"] = any(max(selected[s]["timing"]["round_medians_us"]) /
                                       min(selected[s]["timing"]["round_medians_us"]) > 1.15
                                       for s in ("split", "natural"))
            if row["natural_noisy"]:
                row["natural_outcome"] = "noisy"
        rows.append(row)
    return rows


def export(rows, out):
    out.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    write_json(out / "results.json", rows)
    stats = {}
    for backend in sorted({row["backend"] for row in rows}):
        selected = [row for row in rows if row["backend"] == backend]
        paired = [row for row in selected if row["strict_split_cooperative_pair"]]
        stable = [row for row in paired if not row.get("noisy")]
        natural = [row for row in paired if row["natural_path"] == "cooperative"]
        stats[backend] = dict(cases=len(selected), status=dict(Counter(row["status"] for row in selected)),
                             strict_pairs=len(paired), outcomes=dict(Counter(row["outcome"] for row in paired)),
                             noisy=sum(row.get("noisy", False) for row in paired),
                             natural_cooperative_pairs=len(natural),
                             natural_outcomes=dict(Counter(row.get("natural_outcome") for row in natural)),
                             split_vs_plain=dict(Counter(row["split_vs_plain_outcome"] for row in selected
                                                         if "split_vs_plain_outcome" in row)),
                             cooperative_vs_plain=dict(Counter(row["cooperative_vs_plain_outcome"] for row in selected
                                                               if "cooperative_vs_plain_outcome" in row)),
                             natural_vs_plain=dict(Counter(row["natural_vs_plain_outcome"] for row in selected
                                                           if "natural_vs_plain_outcome" in row)),
                             predeclared_confirmed_cases=sum(bool(row.get("confirmation_history")) for row in selected),
                             failed_confirmation_retries=sum(bool(row.get("confirmation_retry")) for row in selected),
                             geomean_speedup=math.exp(statistics.mean(math.log(row["speedup_cooperative_vs_split"])
                                                                     for row in paired)) if paired else None,
                             stable_geomean_speedup=math.exp(statistics.mean(math.log(row["speedup_cooperative_vs_split"])
                                                                            for row in stable)) if stable else None)
    write_json(out / "summary.json", stats)
    return stats


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    backends = sorted({row["backend"] for row in rows})
    figure, axes = plt.subplots(2, len(backends), figsize=(7 * len(backends), 8), squeeze=False)
    colors = {"sum": "#0072B2", "amax": "#D55E00", "rmsnorm": "#009E73", "softmax": "#CC79A7", "sum_amax": "#E69F00"}
    for column, backend in enumerate(backends):
        pairs = [row for row in rows if row["backend"] == backend and row["strict_split_cooperative_pair"]]
        ax = axes[0, column]
        for op, color in colors.items():
            points = [row for row in pairs if row["op"] == op and not row.get("noisy")]
            if points:
                ax.scatter([row["split_us"] for row in points], [row["cooperative_us"] for row in points],
                           label=op, color=color, alpha=0.8, s=35)
        noisy = [row for row in pairs if row.get("noisy")]
        if noisy:
            ax.scatter([row["split_us"] for row in noisy], [row["cooperative_us"] for row in noisy],
                       label="session variation >15%", color="#777777", marker="x", s=35)
        if pairs:
            limits = [min(min(row["split_us"], row["cooperative_us"]) for row in pairs) * .8,
                      max(max(row["split_us"], row["cooperative_us"]) for row in pairs) * 1.2]
            ax.plot(limits, limits, "--", color="#555555", linewidth=1)
            ax.set_xlim(limits)
            ax.set_ylim(limits)
        ax.set(xscale="log", yscale="log", xlabel="Split device duration (us)",
               ylabel="Cooperative device duration (us)", title=f"{backend.upper()} | strictly verified strategy pairs")
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, "No strictly paired samples", transform=ax.transAxes, ha="center", color="#666666")
        ax = axes[1, column]
        groups = ["core", "dtype", "layout", "tail", "output_control", "fusion"]
        for index, group in enumerate(groups):
            values = [row["speedup_cooperative_vs_split"] for row in pairs if row["group"] == group and not row.get("noisy")]
            if values:
                ax.scatter(index + np.linspace(-.15, .15, len(values)), values, color="#0072B2", alpha=.65, s=22)
                ax.plot([index - .2, index + .2], [statistics.median(values)] * 2, color="#D55E00", linewidth=2)
        ax.axhline(1, linestyle="--", color="#555555", linewidth=1)
        ax.set_xticks(range(len(groups)), groups, rotation=20)
        ax.set(yscale="log", ylabel="Split / cooperative (>1 favors cooperative)", title="Stable-session pairs; red line = group median")
        ax.grid(axis="y", alpha=.2)
    figure.suptitle("Profiler device operation duration only | same eager forward | shared devices", fontsize=12)
    figure.tight_layout()
    figure.savefig(out / "comparison.svg")
    figure.savefig(out / "comparison.png", dpi=160)
    plt.close(figure)

    operations = ["sum", "amax", "rmsnorm", "softmax"]
    xvalues = [1, 7, 14]
    rvalues = sorted({row["R"] for row in rows if row["group"] == "core"})
    if not rvalues:
        return
    figure, axes = plt.subplots(len(backends), 4, figsize=(max(15, len(rvalues) * 1.6), 3.2 * len(backends)), squeeze=False,
                                layout="constrained")
    for rowindex, backend in enumerate(backends):
        for column, op in enumerate(operations):
            ax = axes[rowindex, column]
            matrix = np.full((3, len(rvalues)), np.nan)
            annotations = {}
            for value in rows:
                if value["backend"] != backend or value["op"] != op or value["group"] != "core":
                    continue
                i, j = xvalues.index(value["X"]), rvalues.index(value["R"])
                if value.get("strict_split_cooperative_pair"):
                    ratio = value["speedup_cooperative_vs_split"]
                    matrix[i, j] = np.log2(ratio)
                    annotations[i, j] = f"{ratio:.2f}" + ("~" if value.get("noisy") else "")
                else:
                    annotations[i, j] = "ord" if value.get("split_path") == "ordinary" else "n/a"
            image = ax.imshow(matrix, cmap="RdBu", vmin=-2, vmax=2, aspect="auto")
            ax.set_facecolor("#eeeeee")
            for (i, j), text in annotations.items():
                color = "white" if np.isfinite(matrix[i, j]) and abs(matrix[i, j]) > 1.2 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
            labels = [f"{r // 1048576}M" if r >= 1048576 else f"{r // 1024}K" if r >= 1024 else str(r) for r in rvalues]
            ax.set_xticks(range(len(rvalues)), labels, rotation=45 if len(rvalues) > 6 else 0)
            ax.set_yticks(range(3), [str(x) for x in xvalues])
            ax.set(xlabel="R", ylabel=f"{backend.upper()} | X", title=op)
    bar = figure.colorbar(image, ax=axes.ravel().tolist(), shrink=.85, pad=.02)
    bar.set_ticks([-2, -1, 0, 1, 2], labels=["0.25", "0.5", "1", "2", "4"])
    bar.set_label("Split / cooperative")
    figure.suptitle("FP32 contiguous | device duration | blue >1 favors cooperative\n"
                   "ord = ordinary reduction | ~ = session variation >15%")
    figure.savefig(out / "core_matrix.svg")
    figure.savefig(out / "core_matrix.png", dpi=160)
    plt.close(figure)

    for backend in backends:
        present_xvalues = sorted({row["X"] for row in rows if row["backend"] == backend and row["group"] == "core"})
        if not present_xvalues:
            continue
        figure, axes = plt.subplots(len(present_xvalues), 4, figsize=(16, 10 if len(present_xvalues) == 3 else 3.5 * len(present_xvalues)),
                                   sharex=True, squeeze=False, layout="constrained")
        legend_axis, legend_entries = None, {}
        for i, x in enumerate(present_xvalues):
            for j, op in enumerate(operations):
                ax = axes[i, j]
                points = sorted([r for r in rows if r["backend"] == backend and r["X"] == x
                                 and r["op"] == op and r["group"] == "core"], key=lambda r: r["R"])
                for strategy, color in (("plain", "#777777"), ("split", "#0072B2"),
                                        ("cooperative", "#D55E00"), ("natural", "#CC79A7")):
                    valid = [r for r in points if r.get(strategy + "_status") == "PASS" and r.get(strategy + "_us")]
                    if valid:
                        ax.plot([r["R"] for r in valid], [r[strategy + "_us"] for r in valid],
                                ".--" if strategy == "natural" else ".-", label=strategy, color=color, markersize=5)
                        uncertain = [r for r in valid if (r.get(strategy + "_session_variation") or 0) > 1.15]
                        if uncertain:
                            ax.scatter([r["R"] for r in uncertain], [r[strategy + "_us"] for r in uncertain],
                                       marker="x", color="#555555", s=35, zorder=5)
                ax.set(xscale="log", yscale="log", title=f"{op} | X={x}", xlabel="R", ylabel="device duration (us)")
                ax.set_xticks([256, 4096, 65536, 1048576, 4194304], ["256", "4K", "64K", "1M", "4M"])
                ax.grid(alpha=.2)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    if legend_axis is None:
                        legend_axis = ax
                    legend_entries.update(zip(labels, handles))
                else:
                    ax.text(.5, .5, "No PASS samples", transform=ax.transAxes, ha="center", color="#666666")
        if legend_axis is not None:
            legend_axis.legend(legend_entries.values(), legend_entries.keys(), fontsize=8)
        choices = "plain / split / cooperative" + (" / natural" if backend == "cuda" else "")
        note = " | x = session variation >15%" if any(
            (row.get(strategy + "_session_variation") or 0) > 1.15 for row in rows if row["backend"] == backend
            for strategy in ("plain", "split", "cooperative", "natural")) else ""
        figure.suptitle(f"{backend.upper()} | same eager | {choices}{note}")
        figure.savefig(out / f"{backend}_r_sweep.svg")
        figure.savefig(out / f"{backend}_r_sweep.png", dpi=160)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--replace-failed", action="append", type=Path, default=[],
                        help="Use a complete rerun only for non-PASS primary cases; preserve provenance")
    parser.add_argument("--confirmation-selection", action="append", type=Path, default=[],
                        help="Apply complete predeclared fixed-finalist selections in the supplied order")
    parser.add_argument("--confirmed", action="append", type=Path, default=[],
                        help="Fixed-finalist confirmation runs; every selected case is used regardless of speed")
    parser.add_argument("--replace-failed-confirmed", action="append", type=Path, default=[],
                        help="Complete frozen-finalist reruns for failed confirmation cases; preserve error provenance")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    records = [record for root in args.input for record in collect(root)]
    indexed = {}
    for record in records:
        key = record["backend"], record["case"]["id"]
        if key in indexed:
            raise ValueError(f"Duplicate primary case: {key}; do not mix submission modes")
        indexed[key] = record
    for root in args.replace_failed:
        for record in collect(root):
            key = record["backend"], record["case"]["id"]
            old = indexed.get(key)
            if old is None or old["status"] == "PASS" or record["status"] != "PASS":
                continue
            # Error records may not contain submission; their run manifest still does.
            if old.get("submission") and old["submission"] != record.get("submission"):
                raise ValueError(f"Replacement submission mode differs: {key}")
            record["replaces"] = dict(artifact_root=old["artifact_root"], artifact_case=old["artifact_case"],
                                      status=old["status"])
            indexed[key] = record
    if bool(args.confirmation_selection) != bool(args.confirmed):
        raise ValueError("Confirmation inputs and predeclared selections must be supplied together")
    if args.replace_failed_confirmed and not args.confirmation_selection:
        raise ValueError("Failed confirmation retries require predeclared confirmation inputs")
    if args.confirmation_selection:
        apply_confirmations(indexed, args.confirmation_selection, args.confirmed, args.replace_failed_confirmed)
    records = list(indexed.values())
    rows = summarize(records)
    print(json.dumps(export(rows, args.out), indent=2))
    if not args.no_plot:
        plot(rows, args.out)


if __name__ == "__main__":
    main()
