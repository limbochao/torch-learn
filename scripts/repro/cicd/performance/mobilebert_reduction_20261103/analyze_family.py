"""Conservative cross-device compiler-gain checks for concrete MobileBERT reductions."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    case = "MobileBertForMaskedLM_Train"
    files = {
        tag: args.input_root / f"{case}_{tag}.csv" for tag in ["Eager_NPU", "Inductor_NPU", "Eager_GPU", "Inductor_GPU"]
    }
    rows = {tag: list(csv.DictReader(p.open())) for tag, p in files.items()}
    gpu_eager = {r["OP Type"]: r for r in rows["Eager_GPU"]}
    gpu_inductor = {r["OP Type"]: r for r in rows["Inductor_GPU"]}

    def label(name):
        for key, value in [("ReduceSum", "S"), ("aclnnAdd_Add", "A"), ("aclnnMul_", "M"), ("MatMul", "G")]:
            if key in name:
                return value
        return "O"

    windows = []
    eager_upper = {7: 0.0, 9: 0.0, 13: 0.0}
    eager_20_upper = 0.0
    for step in range(10, 20):
        entries = [(i + 2, r) for i, r in enumerate(rows["Eager_NPU"]) if r["Step Id"] == str(step)]
        labels = [label(r["Name"]) for _, r in entries]
        pairs = [i for i in range(len(entries) - 3) if labels[i : i + 4] == list("SMMS")]
        # One 512-channel head pair, then eight NoNorm pairs per layer, for 24 layers.
        # The branch order is corroborated by the original backward FX and wrapper excerpts.
        assert len(pairs) == 193, (step, len(pairs))
        for layer in range(24):
            base = 1 + layer * 8
            first_merge = [i for i in range(pairs[base + 4] + 4, pairs[base + 5]) if labels[i] == "A"]
            assert len(first_merge) == 2
            selected_indices = [first_merge[0], pairs[base + 6] - 1, pairs[base + 7] - 1]
            assert [labels[i] for i in selected_indices] == list("AAA")
            selected_indices += list(range(pairs[base + 7], pairs[base + 7] + 4))
            eager_20_upper += sum(float(entries[i][1]["Duration(us)"]) for i in selected_indices) / 10000
        for ordinal, start in enumerate(pairs[1:]):
            kind = [7, 11, 11, 11, 13, 11, 7, 20][ordinal % 8]
            if kind not in [7, 13]:
                continue
            lo = start - 1 if kind == 13 else start
            if kind == 13:
                assert labels[lo] == "A"
            selected = entries[lo : start + 4]
            duration = sum(float(r["Duration(us)"]) for _, r in selected)
            eager_upper[kind] += duration / 10000
            windows.append(
                {
                    "kind": kind,
                    "step": step,
                    "layer_ordinal": ordinal // 8,
                    "pair_ordinal": ordinal % 8,
                    "first_csv_row": selected[0][0],
                    "last_csv_row": selected[-1][0],
                    "duration_us": duration,
                    "includes_extra_input_gradient_mul": True,
                }
            )
        # Every target _9 is a linear bias reduction. Charge ALL bias sums after two GEMMs,
        # including other widths and additional bias gradients: this is deliberately an upper bound.
        for start in range(len(entries) - 2):
            if labels[start : start + 3] == list("GGS"):
                line, row = entries[start + 2]
                duration = float(row["Duration(us)"])
                eager_upper[9] += duration / 10000
                windows.append(
                    {
                        "kind": 9,
                        "step": step,
                        "layer_ordinal": "superset",
                        "pair_ordinal": "superset",
                        "first_csv_row": line,
                        "last_csv_row": line,
                        "duration_us": duration,
                        "includes_extra_input_gradient_mul": False,
                    }
                )

    # Full fused GPU kernels are charged, even when they also produce other gradients.
    names = {
        7: ["triton_red_fused_mul_sum_view_8", "triton_red_fused_add_mul_sum_view_12", "triton_red_fused_sum_view_9"],
        9: [
            "triton_red_fused_mul_sum_view_8",
            "triton_red_fused_add_mul_sum_view_11",
            "triton_red_fused_add_mul_sum_view_12",
            "triton_red_fused_sum_view_9",
        ],
        13: ["triton_red_fused_add_mul_sum_view_12", "triton_red_fused_sum_view_9"],
    }
    counts = {7: 48, 9: 168, 13: 24}
    minima = {op: float(gpu_eager[op]["Min Time(us)"]) for op in ["add", "mul", "sum"]}
    required = {7: {"mul": 1, "sum": 2}, 9: {"sum": 1}, 13: {"add": 1, "mul": 1, "sum": 2}}
    bounds = {}
    for kind in [7, 9, 13]:
        suffix = {7: "mul_sum_view_7", 9: "mul_sum_view_9", 13: "add_mul_sum_view_13"}[kind]
        selected = [r for r in rows["Inductor_NPU"] if r["Name"] == "triton_red_fused_" + suffix]
        assert len(selected) == counts[kind] * 10
        n_i = sum(float(r["Duration(us)"]) for r in selected) / 10000
        n_e = eager_upper[kind]
        g_e = counts[kind] * sum(minima[op] * count for op, count in required[kind].items()) / 1000
        g_i = sum(float(gpu_inductor[name]["Total Time(us)"]) for name in names[kind]) / 10000
        bounds[suffix] = {
            "calls_per_step": counts[kind],
            "npu_eager_upper_ms": n_e,
            "npu_inductor_ms": n_i,
            "gpu_eager_lower_ms": g_e,
            "gpu_inductor_upper_ms": g_i,
            "npu_gain_upper": 1 - n_i / n_e,
            "gpu_gain_lower": 1 - g_i / g_e,
            "budget_lower_ms": n_i - n_e * g_i / g_e,
            "gpu_charged_full_kernels": names[kind],
        }

    source = args.input_root / "NPU_g01_output_code.py"
    text = source.read_text()
    calls = []
    for match in re.finditer(r"(triton_red_fused_\w+)\.run\(([^\n]+)\)", text):
        calls.append((match[1], [s.strip() for s in match[2].split(",")], text[: match.start()].count("\n") + 1))
    duplicates = []
    for name, inputs, line in calls:
        if name != "triton_red_fused_mul_sum_view_9":
            continue
        for other, args2, line2 in calls:
            if other == "triton_red_fused_mul_sum_view_7" and args2[0] == inputs[0]:
                between = text.splitlines()[line : line2 - 1]
                # The live buffer name is not redefined or passed to another call in this interval.
                assert not any(re.search(r"\b" + re.escape(inputs[0]) + r"\b", s) for s in between)
                duplicates.append(
                    {
                        "input_buffer": inputs[0],
                        "bias_line": line,
                        "pair_line": line2,
                        "bias_output": inputs[1],
                        "pair_first_output": args2[2],
                    }
                )
    assert len(duplicates) == 24
    result = {
        "profile_steps": 10,
        "bounds": bounds,
        "gpu_minima_us": minima,
        "eager_window_counts_per_step": {
            str(kind): Counter(str(w["step"]) for w in windows if w["kind"] == kind) for kind in [7, 9, 13]
        },
        "npu_duplicate_reduction_pairs": duplicates,
        "unproven_individual_gain_deficit": ["add_mul_sum_view_20", "add_sum_view_0"],
        "add_mul_sum_view_20_schedule_control": {
            "npu_eager_upper_ms_including_extra_dh_mul": eager_20_upper,
            "gpu_eager_lower_ms": 24 * (3 * minima["add"] + minima["mul"] + 2 * minima["sum"]) / 1000,
            "gpu_inductor_upper_ms": sum(
                float(gpu_inductor[n]["Total Time(us)"])
                for n in [
                    "triton_red_fused_add_mul_sum_view_19",
                    "triton_red_fused_add_embedding_dense_backward_mul_sum_view_21",
                    "triton_red_fused_native_layer_norm_native_layer_norm_backward_nll_loss_forward_relu_sum_threshold_backward_view_6",
                ]
            )
            / 10000,
            "status": "Bounds do not prove NPU compiler-gain deficit versus GPU; no budget assigned.",
        },
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [*files.values(), source]},
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "profile_bound.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output_root / "eager_windows.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(windows[0]))
        writer.writeheader()
        writer.writerows(windows)
    print(json.dumps(bounds, indent=2))


if __name__ == "__main__":
    main()
