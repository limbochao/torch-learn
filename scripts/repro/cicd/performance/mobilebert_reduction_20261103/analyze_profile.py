"""Compute conservative four-way bounds for the concrete MobileBERT _11 reduction."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    case = "MobileBertForMaskedLM_Train"
    files = {
        key: args.input_root / f"{case}_{key}.csv" for key in ["Eager_NPU", "Inductor_NPU", "Eager_GPU", "Inductor_GPU"]
    }
    rows = {key: list(csv.DictReader(path.open())) for key, path in files.items()}
    pattern = [
        "aclnnAdd_AddAiCore_Add",
        "aclnnReduceSum_ReduceSumOpAiCore_ReduceSum",
        "aclnnMul_MulAiCore_Mul",
        "aclnnMul_MulAiCore_Mul",
        "aclnnReduceSum_ReduceSumOpAiCore_ReduceSum",
    ]
    windows = []
    for index in range(len(rows["Eager_NPU"]) - len(pattern) + 1):
        selected = rows["Eager_NPU"][index : index + len(pattern)]
        if len({row["Step Id"] for row in selected}) != 1:
            continue
        if [row["Name"] for row in selected] == pattern:
            windows.append(
                {
                    "step": selected[0]["Step Id"],
                    "first_csv_row": index + 2,
                    "duration_us": sum(float(row["Duration(us)"]) for row in selected),
                }
            )
    kernel = "triton_red_fused_add_mul_sum_view_11"
    npu = [row for row in rows["Inductor_NPU"] if row["Name"] == kernel]
    steps = len({row["Step Id"] for row in rows["Eager_NPU"]})
    assert steps == 10
    assert len(npu) == 960
    n_e = sum(row["duration_us"] for row in windows) / steps / 1000
    n_i = sum(float(row["Duration(us)"]) for row in npu) / steps / 1000
    eager_gpu = {row["OP Type"]: row for row in rows["Eager_GPU"]}
    gpu = {row["OP Type"]: row for row in rows["Inductor_GPU"]}
    charged_gpu = [kernel, "triton_red_fused_add_mul_sum_view_12", "triton_red_fused_sum_view_9"]
    gpu_minima = {op: float(eager_gpu[op]["Min Time(us)"]) for op in ["add", "mul", "sum"]}
    g_e = 96 * (gpu_minima["add"] + gpu_minima["mul"] + 2 * gpu_minima["sum"]) / 1000
    g_i = sum(float(gpu[op]["Total Time(us)"]) for op in charged_gpu) / steps / 1000
    budget = n_i - n_e * g_i / g_e
    result = {
        "kernel": kernel,
        "calls_per_step": 96,
        "profile_steps": steps,
        "npu_eager_window_pattern": pattern,
        "npu_eager_window_counts": dict(Counter(row["step"] for row in windows)),
        "npu_eager_upper_ms": n_e,
        "npu_inductor_ms": n_i,
        "gpu_eager_minima_us": gpu_minima,
        "gpu_eager_lower_ms": g_e,
        "gpu_inductor_charged_ops": charged_gpu,
        "gpu_inductor_upper_ms": g_i,
        "npu_gain_upper": 1 - n_i / n_e,
        "gpu_gain_lower": 1 - g_i / g_e,
        "budget_lower_ms": budget,
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files.values()},
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "profile_bound.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output_root / "eager_windows.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "first_csv_row", "duration_us"])
        writer.writeheader()
        writer.writerows(windows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
