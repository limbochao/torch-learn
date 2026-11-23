"""Merge compiled/DSL matrix runs into CSV and standalone SVG; no accelerator is required."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


KINDS = {"plain": "Inductor plain", "original": "DSL origin", "debug": "DSL debug",
         "cooperative": "Inductor forced cooperative", "natural": "Inductor natural selection"}
FIELDS = ("source_run", "backend", "device_name", "device_uuid", "physical_device", "entry", "operation",
          "dtype", "X", "R", "kind", "force_cooperative", "actual_path", "status", "RS", "block",
          "median_us", "min_us", "max_us", "plain_us", "speedup_vs_plain", "mismatched_values",
          "exchange_failed", "integration_candidate", "integration_note", "best_integration_config",
          "block_search_boundary", "selected_configs", "configs_by_round", "selection_varied", "round_medians_us",
          "artifact_dir")


def flatten(record, source):
    row = dict(record)
    device = row.get("device", {})
    row.update(source_run=source, device_name=device.get("name"), device_uuid=device.get("uuid"),
               physical_device=device.get("physical_device"), dtype=row.get("dtype", "float32"))
    checks = [*record.get("accuracy", []), record.get("post_accuracy", {})]
    row["mismatched_values"] = sum(item.get("mismatched_values", 0) for item in checks)
    row["exchange_failed"] = "error" in record.get("exchange", {})
    return row


def save_csv(path, records, fields=FIELDS):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for item in records:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list))
                             else value for key, value in item.items() if key in fields})


def same_config(records):
    pairs = defaultdict(dict)
    for row in records:
        if row.get("entry") != "dsl" or "/search/" not in row.get("artifact_dir", ""):
            continue
        key = tuple(row.get(name) for name in ("source_run", "device_uuid", "operation", "X", "R", "RS", "block"))
        pairs[key][row["kind"]] = row
    result = []
    for key, values in sorted(pairs.items()):
        original, debug = values.get("original", {}), values.get("debug", {})
        result.append(dict(zip(("source_run", "device_uuid", "operation", "X", "R", "RS", "block"), key),
                           original_status=original.get("status"), debug_status=debug.get("status"),
                           original_us=original.get("median_us"), debug_us=debug.get("median_us")))
    return result


def figures(records, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    operations = sorted({item["operation"] for item in records}, key=lambda op: (op != "sum", op))
    rows = sorted({item["X"] for item in records})
    groups = defaultdict(list)
    for item in records:
        if item["status"] != "PASS":
            continue
        key = (item["source_run"], item["device_name"], item["kind"])
        groups[key].append(item)
    metrics = [("latency", "median_us", "Latency (us)"), ("speedup", "speedup_vs_plain", "Plain / candidate"),
               ("selected-rs", "RS", "RS"), ("selected-block", "block", "BLOCK (NPU DSL)")]
    for filename, field, ylabel in metrics:
        if field == "block" and not any(item.get("entry") == "dsl" for item in records):
            continue
        fig, axes = plt.subplots(len(operations), len(rows), squeeze=False,
                                 figsize=(5 * len(rows), 3.8 * len(operations)), constrained_layout=True)
        for oi, operation in enumerate(operations):
            for xi, xnumel in enumerate(rows):
                ax = axes[oi][xi]
                for (run, name, kind), items in sorted(groups.items()):
                    if field in ("RS", "block", "speedup_vs_plain") and kind == "plain":
                        continue
                    subset = sorted((item for item in items if item["operation"] == operation and item["X"] == xnumel
                                     and item.get(field) is not None), key=lambda item: item["R"])
                    if not subset:
                        continue
                    label = f"{run}: {name} / {KINDS[kind]}"
                    xvalues, yvalues = [item["R"] for item in subset], [item[field] for item in subset]
                    line, = ax.plot(xvalues, yvalues, "o--" if kind == "plain" else "o-", label=label, markersize=4)
                    if field == "median_us":
                        ax.fill_between(xvalues, [item.get("min_us", item[field]) for item in subset],
                                        [item.get("max_us", item[field]) for item in subset],
                                        color=line.get_color(), alpha=0.12)
                ax.set_title(f"{operation}, X={xnumel}")
                ax.set_xscale("log", base=2)
                axis_r = sorted({item["R"] for item in records
                                 if item["operation"] == operation and item["X"] == xnumel})
                ax.set_xticks(axis_r, [f"{r // 1048576}M" if r >= 1048576 else f"{r // 1024}K" for r in axis_r],
                              rotation=30)
                if field != "RS":
                    ax.set_yscale("log", base=2 if field != "median_us" else 10)
                if field == "speedup_vs_plain":
                    ax.axhline(1, color="black", linewidth=0.8, linestyle="--")
                ax.set_xlabel("R")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.25, which="both")
        handles, labels = [], []
        for ax in axes.flat:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        if handles:
            fig.legend(handles, labels, loc="outside lower center", fontsize=8, ncol=1)
        fig.suptitle("fp32 (a+b) reduction / Graph + device Events\n"
                     "Correct results only; shaded range: observed sample min/max", fontsize=12)
        fig.savefig(out / f"{filename}.svg", bbox_inches="tight")
        fig.savefig(out / f"{filename}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Run directory; repeat to merge")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    selected, candidates, natural = [], [], []
    for source in args.input:
        for name, target in (("results", selected), ("candidates", candidates), ("natural", natural)):
            path = source / (name + ".json")
            if path.exists():
                target.extend(flatten(item, source.name) for item in json.loads(path.read_text()))
            elif name == "results":
                raise FileNotFoundError(path)
    save_csv(args.out / "results.csv", selected)
    save_csv(args.out / "candidates.csv", candidates)
    save_csv(args.out / "natural-selection.csv", natural)
    paired = same_config(candidates)
    save_csv(args.out / "same-config-sync.csv", paired,
             ("source_run", "device_uuid", "operation", "X", "R", "RS", "block", "original_status", "debug_status",
              "original_us", "debug_us"))
    if not args.csv_only:
        figures(selected, args.out)
    print(f"Saved {len(selected)} selected results, {len(candidates)} candidates, {len(natural)} natural selections "
          f"to {args.out}")


if __name__ == "__main__":
    main()
