"""Condense elementwise performance summary rows into CSV and XLSX reports."""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

from elementwise_op_cost_xlsx import write_xlsx_report


BASE_OUTPUT_COLUMNS = (
    "bound",
    "scalar_ops",
    "dtype",
    "first_shape",
    "runtime_shape",
)
EXECUTIONS = ("eager", "static", "dynamic")
SHAPE_DEPENDENT_EXECUTIONS = ("dynamic", "custom", "group")
DEVICES = ("cuda", "npu")
OUTPUT_NAME = "elementwise_op_cost_comparison.csv"
XLSX_OUTPUT_NAME = "elementwise_op_cost_comparison.xlsx"
LOGGER = logging.getLogger(__name__)


def read_rows(path: Path) -> list[dict[str, str]]:
    content = path.read_text(encoding="utf-8-sig")
    if not content.strip():
        return []
    dialect = csv.Sniffer().sniff(content[:8192], delimiters=",\t")
    reader = csv.DictReader(content.splitlines(), dialect=dialect)
    if reader.fieldnames is None:
        return []
    return [{key: value or "" for key, value in row.items()} for row in reader]


def row_value(row: dict[str, str], column: str) -> str:
    return row.get(column, "").strip()


def shape_key(shape: str) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in shape.replace("x", ",").split(",") if dim)
    except ValueError:
        return ()


def group_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row_value(row, "case"),
        row_value(row, "scalar_ops"),
        row_value(row, "dtype").removeprefix("torch."),
        row_value(row, "runtime_shape"),
    )


def first_shapes(rows: list[dict[str, str]]) -> list[str]:
    shapes = {
        row_value(row, "compile_shape")
        for row in rows
        if row_value(row, "execution") in SHAPE_DEPENDENT_EXECUTIONS
    }
    return sorted(shapes, key=shape_key) if shapes else [""]


def has_execution(
    present: set[tuple[str, str]],
    device: str,
    execution: str,
) -> bool:
    return (device, execution) in present


def output_columns(rows: list[dict[str, str]]) -> tuple[str, ...]:
    present = {
        (row_value(row, "device"), row_value(row, "execution")) for row in rows
    }
    columns = list(BASE_OUTPUT_COLUMNS)
    for device in DEVICES:
        for execution in EXECUTIONS:
            if has_execution(present, device, execution):
                columns.append(f"{device}_{execution}_us")
        if (
            has_execution(present, device, "dynamic")
            and has_execution(present, device, "static")
        ):
            columns.append(f"{device}_dynamic_static_ratio")
            if device == "npu" and (
                has_execution(present, "cuda", "dynamic")
                and has_execution(present, "cuda", "static")
            ):
                columns.append("npu_cuda_ratio_of_lift")

    for execution in ("custom", "group"):
        if has_execution(present, "npu", execution):
            columns.append(f"npu_{execution}_us")
        if (
            has_execution(present, "npu", execution)
            and has_execution(present, "npu", "static")
        ):
            columns.append(f"npu_{execution}_static_ratio")
        if (
            has_execution(present, "npu", execution)
            and has_execution(present, "npu", "static")
            and has_execution(present, "cuda", "dynamic")
            and has_execution(present, "cuda", "static")
        ):
            columns.append(f"npu_{execution}_cuda_ratio_of_lift")
    return tuple(columns)


def select_execution_row(
    rows: list[dict[str, str]],
    device: str,
    execution: str,
    first_shape: str,
) -> dict[str, str] | None:
    matches = []
    for row in rows:
        if row_value(row, "device") != device or row_value(row, "execution") != execution:
            continue
        if (
            execution in SHAPE_DEPENDENT_EXECUTIONS
            and row_value(row, "compile_shape") != first_shape
        ):
            continue
        matches.append(row)
    return matches[-1] if matches else None


def format_number(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def timing_us(row: dict[str, str] | None) -> float | None:
    if row is None:
        return None
    value = row_value(row, "device_call_mean_us")
    try:
        return float(value) if value else None
    except ValueError:
        return None


def cost_ratio(numerator_us: float | None, denominator_us: float | None) -> float | None:
    if numerator_us is None or denominator_us in (None, 0.0):
        return None
    return numerator_us / denominator_us


def comparison_row(
    rows: list[dict[str, str]],
    scalar_ops: str,
    dtype: str,
    runtime_shape: str,
    first_shape: str,
    columns: tuple[str, ...],
) -> dict[str, str]:
    result = {column: "" for column in columns}
    result.update(
        {
            "bound": row_value(rows[-1], "bound") if rows else "",
            "scalar_ops": scalar_ops,
            "dtype": dtype,
            "first_shape": first_shape,
            "runtime_shape": runtime_shape,
        }
    )

    ratios: dict[str, float | None] = {}
    for device in DEVICES:
        timings = {
            execution: timing_us(select_execution_row(rows, device, execution, first_shape))
            for execution in EXECUTIONS
        }
        for execution, timing in timings.items():
            column = f"{device}_{execution}_us"
            if column in result:
                result[column] = format_number(timing)
        ratio = cost_ratio(timings["dynamic"], timings["static"])
        ratios[device] = ratio
        column = f"{device}_dynamic_static_ratio"
        if column in result:
            result[column] = format_number(ratio)

    cuda_ratio = ratios["cuda"]
    npu_ratio = ratios["npu"]
    npu_static_us = timing_us(select_execution_row(rows, "npu", "static", first_shape))
    for execution in ("custom", "group"):
        execution_us = timing_us(
            select_execution_row(rows, "npu", execution, first_shape)
        )
        column = f"npu_{execution}_us"
        if column in result:
            result[column] = format_number(execution_us)
        column = f"npu_{execution}_static_ratio"
        if column in result:
            result[column] = format_number(cost_ratio(execution_us, npu_static_us))
        column = f"npu_{execution}_cuda_ratio_of_lift"
        if column in result:
            execution_static_ratio = cost_ratio(execution_us, npu_static_us)
            lift = None
            if execution_static_ratio is not None and cuda_ratio not in (None, 0.0):
                lift = execution_static_ratio / cuda_ratio
            result[column] = format_number(lift)
    ratio_of_lift = None
    if npu_ratio is not None and cuda_ratio not in (None, 0.0):
        ratio_of_lift = npu_ratio / cuda_ratio
    if "npu_cuda_ratio_of_lift" in result:
        result["npu_cuda_ratio_of_lift"] = format_number(ratio_of_lift)
    return result


def condensed_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["scalar_ops"],
        row["dtype"],
        row["first_shape"],
        row["runtime_shape"],
    )


def build_comparison_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    columns = output_columns(rows)
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)

    comparison_by_key: dict[tuple[str, ...], tuple[str, dict[str, str]]] = {}
    for (case, scalar_ops, dtype, runtime_shape), case_rows in sorted(grouped.items()):
        for first_shape in first_shapes(case_rows):
            comparison = comparison_row(
                case_rows,
                scalar_ops,
                dtype,
                runtime_shape,
                first_shape,
                columns,
            )
            key = condensed_key(comparison)
            if key in comparison_by_key:
                previous_case = comparison_by_key[key][0]
                LOGGER.warning(
                    "comparison key conflict for %s: cases %r and %r; keeping %r",
                    key,
                    previous_case,
                    case,
                    previous_case,
                )
                continue
            comparison_by_key[key] = (case, comparison)

    return sorted(
        (item[1] for item in comparison_by_key.values()),
        key=lambda row: (
            row["bound"],
            row["scalar_ops"],
            row["dtype"],
            shape_key(row["first_shape"]),
            shape_key(row["runtime_shape"]),
        ),
    )


def write_comparison(
    rows: list[dict[str, str]],
    output_path: Path,
    columns: tuple[str, ...],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="summary CSV or TSV file")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = parse_args()
    output_path = args.summary.parent / OUTPUT_NAME
    xlsx_output_path = args.summary.parent / XLSX_OUTPUT_NAME
    summary_rows = read_rows(args.summary)
    rows = build_comparison_rows(summary_rows)
    columns = output_columns(summary_rows)
    write_comparison(rows, output_path, columns)
    write_xlsx_report(rows, xlsx_output_path, columns)
    print(f"wrote {len(rows)} comparison rows to {output_path}")
    print(f"wrote {len(rows)} comparison rows to {xlsx_output_path}")


if __name__ == "__main__":
    main()
