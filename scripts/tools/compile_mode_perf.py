#!/usr/bin/env python3
"""Compare static, dynamic, and symbolic-group compilation performance."""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


GROUP_AUTOTUNE_ENV = "INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE"
ARTIFACT_MODE_COMPONENTS = {
    "static": "s",
    "dynamic": "d",
    "group": "g",
}
SUMMARY_COLUMNS = (
    "case",
    "mode",
    "first_shape",
    "shape",
    "mean_us",
    "tiling",
    "samples",
    "kernel_count",
    "kernels",
    "result_dir",
)
COMPARISON_COLUMNS = (
    "case",
    "first_shape",
    "shape",
    "static_us",
    "static_tiling",
    "dynamic_us",
    "dynamic_static_ratio",
    "dynamic_tiling",
    "group_us",
    "group_static_ratio",
    "group_tiling",
)
CASE_KEYS = (
    "name",
    "forward",
    "make_inputs",
    "sample_bindings",
    "compile_bindings",
    "dynamic_dims",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", type=Path, help="extracted eager case")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prof_log/compile_mode_perf"),
        help="output root",
    )
    parser.add_argument("--run-id", help="run directory name (default: timestamp)")
    parser.add_argument("--device", default="npu:0", help="NPU device, for example npu:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--active", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--_action", choices=("discover", "worker"), help=argparse.SUPPRESS)
    parser.add_argument("--_config", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as output:
        output.write(content)
        temporary_path = Path(output.name)
    os.replace(temporary_path, path)


def write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_case(path: Path) -> dict[str, object]:
    spec = importlib.util.spec_from_file_location(
        f"compile_mode_case_{os.getpid()}", path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load case module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case = getattr(module, "CASE", None)
    if not isinstance(case, dict):
        raise ValueError(f"{path} must define a CASE dictionary")
    missing = [name for name in CASE_KEYS if name not in case]
    if missing:
        raise ValueError(f"CASE is missing required keys: {missing}")
    if not callable(case["forward"]) or not callable(case["make_inputs"]):
        raise ValueError("CASE forward and make_inputs must be callable")
    if not isinstance(case["sample_bindings"], list):
        raise ValueError("CASE sample_bindings must be a list")
    if not isinstance(case["compile_bindings"], list):
        raise ValueError("CASE compile_bindings must be a list")
    if not case["sample_bindings"] or not case["compile_bindings"]:
        raise ValueError("CASE binding lists must not be empty")
    if not isinstance(case["dynamic_dims"], dict):
        raise ValueError("CASE dynamic_dims must be a dictionary")
    return case


def discover(config: dict[str, object]) -> None:
    case = load_case(Path(str(config["case_path"])))
    payload = {
        "name": str(case["name"]),
        "sample_bindings": case["sample_bindings"],
        "compile_bindings": case["compile_bindings"],
        "dynamic_dims": case["dynamic_dims"],
    }
    json.dumps(payload)
    write_json(Path(str(config["result_path"])), payload)


def device_index(device: str) -> int:
    match = re.fullmatch(r"npu(?::(\d+))?", device)
    if match is None:
        raise ValueError("--device must be 'npu' or 'npu:<index>'")
    return int(match.group(1) or 0)


def tensor_shape_label(signature: list[dict[str, object]]) -> str:
    tensors = [item for item in signature if item["kind"] == "tensor"]
    if not tensors:
        return "no_tensor_inputs"

    def dimensions(item: dict[str, object]) -> str:
        shape = item["shape"]
        return "x".join(str(dim) for dim in shape) if shape else "scalar"

    if len(tensors) == 1:
        return dimensions(tensors[0])
    return ";".join(f"{item['path']}={dimensions(item)}" for item in tensors)


def input_signature(torch: Any, args: tuple[object, ...], kwargs: dict[str, object]):
    result: list[dict[str, object]] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, torch.Tensor):
            result.append(
                {
                    "kind": "tensor",
                    "path": path,
                    "shape": list(value.shape),
                    "stride": list(value.stride()),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                }
            )
            return
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        if isinstance(value, dict):
            for key in sorted(value, key=repr):
                suffix = f".{key}" if isinstance(key, str) and key.isidentifier() else f"[{key!r}]"
                visit(value[key], path + suffix)
            return
        result.append(
            {
                "kind": "value",
                "path": path,
                "type": type(value).__name__,
                "value": repr(value),
            }
        )

    visit(args, "args")
    visit(kwargs, "kwargs")
    return result


def resolve_input_path(args: tuple[object, ...], kwargs: dict[str, object], path: str):
    try:
        expression = ast.parse(path, mode="eval").body
    except SyntaxError as error:
        raise ValueError(f"invalid dynamic input path: {path!r}") from error

    def resolve(node):
        if isinstance(node, ast.Name):
            if node.id == "args":
                return args
            if node.id == "kwargs":
                return kwargs
        if isinstance(node, ast.Attribute):
            value = resolve(node.value)
            if isinstance(value, dict) and node.attr in value:
                return value[node.attr]
        if isinstance(node, ast.Subscript):
            value = resolve(node.value)
            key = ast.literal_eval(node.slice)
            return value[key]
        raise ValueError(f"unsupported dynamic input path: {path!r}")

    return resolve(expression)


def mark_dynamic_inputs(torch: Any, case: dict[str, object], args, kwargs) -> None:
    for path, dims in case["dynamic_dims"].items():
        tensor = resolve_input_path(args, kwargs, str(path))
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"dynamic input path {path!r} does not resolve to a Tensor")
        for dim in dims:
            torch._dynamo.mark_dynamic(tensor, int(dim))


def make_case_inputs(torch: Any, case: dict[str, object], binding, device: str):
    torch.manual_seed(0)
    torch.npu.manual_seed_all(0)
    result = case["make_inputs"](dict(binding), device)
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("make_inputs must return (args, kwargs)")
    args, kwargs = result
    if not isinstance(args, tuple) or not isinstance(kwargs, dict):
        raise ValueError("make_inputs must return a tuple of args and a dict of kwargs")
    return args, kwargs


def latest_compile_debug_dir(debug_root: Path) -> Path:
    outputs = [path for path in debug_root.rglob("output_code.py") if path.is_file()]
    if not outputs:
        raise RuntimeError(f"no output_code.py found below {debug_root}")
    latest = max(
        outputs,
        key=lambda path: (path.stat().st_mtime_ns, path.stat().st_ctime_ns, str(path)),
    )
    return latest.parent


def replace_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def prepare_profile_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def profile_npu(torch: Any, fn, args, kwargs, profile_dir: Path, config):
    from npu_profiler import ProfileResultParser, TorchNpuProfiler

    prepare_profile_dir(profile_dir)
    profiler = TorchNpuProfiler(
        profile_dir,
        wait=0,
        warmup=int(config["warmup"]),
        active=int(config["active"]),
        repeat=int(config["repeat"]),
        with_stack=False,
    )
    profiler.run_steps(lambda: fn(*args, **kwargs))
    summaries = ProfileResultParser(profile_dir).kernel_time_by_name()
    call_count = int(config["active"]) * int(config["repeat"])
    kernel_count = sum(summary.count for summary in summaries)
    if kernel_count == 0:
        raise RuntimeError(f"no NPU device kernels found in {profile_dir}")
    return {
        "mean_us": sum(summary.total_us for summary in summaries) / call_count,
        "samples": call_count,
        "kernel_count": kernel_count,
        "kernels": [summary.key for summary in summaries],
    }


def compile_forward(torch: Any, case, args, kwargs, dynamic: bool):
    torch._dynamo.reset()
    if dynamic:
        mark_dynamic_inputs(torch, case, args, kwargs)
    compiled = torch.compile(
        case["forward"],
        backend="inductor",
        dynamic=None if dynamic else False,
    )
    compiled(*args, **kwargs)
    torch.npu.synchronize()
    return compiled


def artifact_mode_root(run_root: Path, mode: str, first_index=None) -> Path:
    try:
        component = ARTIFACT_MODE_COMPONENTS[mode]
    except KeyError as error:
        raise ValueError(f"unsupported artifact mode: {mode}") from error
    root = run_root / "artifacts" / component
    if first_index is not None:
        root = root / f"f{first_index:03d}"
    return root


def artifact_path(
    run_root: Path,
    mode: str,
    sample_index: int,
    _shape: str,
    first_index=None,
) -> Path:
    return artifact_mode_root(run_root, mode, first_index) / f"s{sample_index:03d}"


def result_record(
    case,
    mode,
    sample_index,
    binding,
    signature,
    timing,
    tiling,
    profile_dir,
    first_index=None,
    first_binding=None,
    first_signature=None,
):
    return {
        "case": str(case["name"]),
        "mode": mode,
        "first_index": first_index,
        "sample_index": sample_index,
        "first_binding": first_binding,
        "binding": binding,
        "first_input_signature": first_signature or [],
        "input_signature": signature,
        "first_shape": tensor_shape_label(first_signature or []) if first_index is not None else "",
        "shape": tensor_shape_label(signature),
        "mean_us": timing["mean_us"],
        "samples": timing["samples"],
        "kernel_count": timing["kernel_count"],
        "kernels": timing["kernels"],
        "tiling": tiling,
        "result_dir": str(profile_dir),
    }


def run_static(torch: Any, case, config, recorder):
    run_root = Path(str(config["run_root"]))
    device = str(config["device"])
    records = []
    for sample_index, binding in enumerate(case["sample_bindings"]):
        args, kwargs = make_case_inputs(torch, case, binding, device)
        signature = input_signature(torch, args, kwargs)
        shape = tensor_shape_label(signature)
        output_root = artifact_path(run_root, "static", sample_index, shape)
        recorder.start_capture()
        try:
            compiled = compile_forward(torch, case, args, kwargs, dynamic=False)
        finally:
            tiling = recorder.stop_capture()
        replace_tree(
            latest_compile_debug_dir(Path(str(config["debug_root"]))),
            output_root / "torch_compile_debug",
        )
        timing = profile_npu(
            torch, compiled, args, kwargs, output_root / "profiles", config
        )
        records.append(
            result_record(
                case,
                "static",
                sample_index,
                binding,
                signature,
                timing,
                tiling,
                output_root / "profiles",
            )
        )
    return records


def run_dynamic(torch: Any, case, config, recorder, group: bool):
    run_root = Path(str(config["run_root"]))
    device = str(config["device"])
    first_index = int(config["compile_index"])
    first_binding = case["compile_bindings"][first_index]
    compile_args, compile_kwargs = make_case_inputs(
        torch, case, first_binding, device
    )
    first_signature = input_signature(torch, compile_args, compile_kwargs)
    first_shape = tensor_shape_label(first_signature)
    mode = "group" if group else "dynamic"

    recorder.start_capture()
    try:
        compiled = compile_forward(
            torch, case, compile_args, compile_kwargs, dynamic=True
        )
    finally:
        compile_tiling = recorder.stop_capture()
    compile_debug_root = artifact_mode_root(
        run_root,
        mode,
        first_index=None if group else first_index,
    )
    replace_tree(
        latest_compile_debug_dir(Path(str(config["debug_root"]))),
        compile_debug_root / "torch_compile_debug",
    )

    records = []
    for sample_index, binding in enumerate(case["sample_bindings"]):
        args, kwargs = make_case_inputs(torch, case, binding, device)
        signature = input_signature(torch, args, kwargs)
        shape = tensor_shape_label(signature)
        output_root = artifact_path(
            run_root,
            mode,
            sample_index,
            shape,
            first_index=None if group else first_index,
        )
        if group:
            recorder.start_capture()
            try:
                timing = profile_npu(
                    torch, compiled, args, kwargs, output_root / "profiles", config
                )
            finally:
                tiling = recorder.stop_capture()
        else:
            timing = profile_npu(
                torch, compiled, args, kwargs, output_root / "profiles", config
            )
            tiling = compile_tiling
        records.append(
            result_record(
                case,
                mode,
                sample_index,
                binding,
                signature,
                timing,
                tiling,
                output_root / "profiles",
                first_index=None if group else first_index,
                first_binding=first_binding,
                first_signature=first_signature,
            )
        )
    return records


def run_worker(config: dict[str, object]) -> None:
    debug_root = Path(str(config["debug_root"]))
    debug_root.mkdir(parents=True, exist_ok=True)

    import torch
    import torch_npu  # noqa: F401

    from autotune_tiling import BestTilingRecorder

    torch._dynamo.config.debug_dir_root = str(debug_root)
    torch.npu.set_device(device_index(str(config["device"])))
    case = load_case(Path(str(config["case_path"])))
    recorder = BestTilingRecorder("npu")
    recorder.install()
    try:
        execution = str(config["execution"])
        if execution == "static":
            records = run_static(torch, case, config, recorder)
        elif execution == "dynamic":
            records = run_dynamic(torch, case, config, recorder, group=False)
        elif execution == "group":
            records = run_dynamic(torch, case, config, recorder, group=True)
        else:
            raise ValueError(f"unsupported execution: {execution}")
    finally:
        recorder.uninstall()
        torch.npu.synchronize()
    write_json(Path(str(config["result_path"])), records)


def run_internal(action: str, config_path: Path, env: dict[str, str]) -> None:
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_action",
            action,
            "--_config",
            str(config_path),
        ],
        check=True,
        env=env,
    )


def compact_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_raw_results(path: Path, records: list[dict[str, object]]) -> None:
    content = "".join(compact_json(record) + "\n" for record in records)
    atomic_write_text(path, content)


def write_csv(path: Path, columns, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(output.name)
    os.replace(temporary_path, path)


def summary_rows(records: list[dict[str, object]]):
    rows = []
    for record in records:
        rows.append(
            {
                "case": record["case"],
                "mode": record["mode"],
                "first_shape": record["first_shape"],
                "shape": record["shape"],
                "mean_us": f"{float(record['mean_us']):.6f}",
                "tiling": compact_json(record["tiling"]) if record["tiling"] else "",
                "samples": record["samples"],
                "kernel_count": record["kernel_count"],
                "kernels": "|".join(record["kernels"]),
                "result_dir": record["result_dir"],
            }
        )
    return rows


def ratio(numerator: float, denominator: float) -> str:
    return "" if denominator == 0 else f"{numerator / denominator:.3f}"


def comparison_rows(records: list[dict[str, object]]):
    static = {
        int(record["sample_index"]): record
        for record in records
        if record["mode"] == "static"
    }
    dynamic = {
        (int(record["first_index"]), int(record["sample_index"])): record
        for record in records
        if record["mode"] == "dynamic"
    }
    group = {
        int(record["sample_index"]): record
        for record in records
        if record["mode"] == "group"
    }
    first_indices = sorted({key[0] for key in dynamic})
    sample_indices = sorted(static)
    rows = []
    for first_index in first_indices:
        for sample_index in sample_indices:
            try:
                static_record = static[sample_index]
                dynamic_record = dynamic[first_index, sample_index]
                group_record = group[sample_index]
            except KeyError as error:
                raise ValueError(
                    f"incomplete result matrix at first={first_index}, "
                    f"sample={sample_index}"
                ) from error
            shapes = {
                static_record["shape"],
                dynamic_record["shape"],
                group_record["shape"],
            }
            if len(shapes) != 1:
                raise ValueError(
                    f"input shape mismatch at sample {sample_index}: {sorted(shapes)}"
                )
            signatures = {
                compact_json(static_record["input_signature"]),
                compact_json(dynamic_record["input_signature"]),
                compact_json(group_record["input_signature"]),
            }
            if len(signatures) != 1:
                raise ValueError(
                    f"input signature mismatch at sample {sample_index}"
                )
            static_us = float(static_record["mean_us"])
            dynamic_us = float(dynamic_record["mean_us"])
            group_us = float(group_record["mean_us"])
            rows.append(
                {
                    "case": dynamic_record["case"],
                    "first_shape": dynamic_record["first_shape"],
                    "shape": dynamic_record["shape"],
                    "static_us": f"{static_us:.3f}",
                    "static_tiling": compact_json(static_record["tiling"])
                    if static_record["tiling"]
                    else "",
                    "dynamic_us": f"{dynamic_us:.3f}",
                    "dynamic_static_ratio": ratio(dynamic_us, static_us),
                    "dynamic_tiling": compact_json(dynamic_record["tiling"])
                    if dynamic_record["tiling"]
                    else "",
                    "group_us": f"{group_us:.3f}",
                    "group_static_ratio": ratio(group_us, static_us),
                    "group_tiling": compact_json(group_record["tiling"])
                    if group_record["tiling"]
                    else "",
                }
            )
    return rows


def worker_environment(cache_dir: Path, group: bool) -> dict[str, str]:
    env = os.environ.copy()
    env[GROUP_AUTOTUNE_ENV] = "1" if group else "0"
    env["TORCH_COMPILE_DEBUG"] = "1"
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    return env


def run_one_worker(run_root: Path, control_root: Path, base_config, execution, index=None):
    suffix = (
        execution
        if index is None or execution == "group"
        else f"{execution}_{index:03d}"
    )
    cache_dir = run_root / ".cache" / suffix
    debug_root = run_root / ".debug" / suffix
    result_path = control_root / f"{suffix}.json"
    config = {
        **base_config,
        "execution": execution,
        "compile_index": index,
        "debug_root": str(debug_root),
        "result_path": str(result_path),
    }
    config_path = control_root / f"{suffix}_config.json"
    write_json(config_path, config)
    print(f"running {suffix}", flush=True)
    try:
        run_internal(
            "worker",
            config_path,
            worker_environment(cache_dir, group=execution == "group"),
        )
        return read_json(result_path)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.rmtree(debug_root, ignore_errors=True)


def validate_controller_args(args: argparse.Namespace) -> None:
    if args.case is None:
        raise ValueError("case path is required")
    if not args.case.is_file():
        raise ValueError(f"case file does not exist: {args.case}")
    device_index(args.device)
    if args.warmup < 0 or args.active <= 0 or args.repeat <= 0:
        raise ValueError("warmup must be non-negative; active and repeat must be positive")
    if args.run_id and re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id) is None:
        raise ValueError("run-id may only contain letters, digits, '.', '_', and '-'")


def controller(args: argparse.Namespace) -> None:
    validate_controller_args(args)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (args.output / run_id).resolve()
    if run_root.exists():
        raise ValueError(f"run directory already exists: {run_root}")
    control_root = run_root / ".control"
    control_root.mkdir(parents=True)
    case_path = args.case.resolve()

    discovery_result = control_root / "discovery.json"
    discovery_config = control_root / "discovery_config.json"
    write_json(
        discovery_config,
        {"case_path": str(case_path), "result_path": str(discovery_result)},
    )
    discovery_env = os.environ.copy()
    discovery_env[GROUP_AUTOTUNE_ENV] = "0"
    discovery_env["TORCH_COMPILE_DEBUG"] = "0"
    run_internal("discover", discovery_config, discovery_env)
    discovered = read_json(discovery_result)

    manifest = {
        "run_id": run_id,
        "case": discovered["name"],
        "case_path": str(case_path),
        "device": args.device,
        "warmup": args.warmup,
        "active": args.active,
        "repeat": args.repeat,
        "sample_bindings": discovered["sample_bindings"],
        "compile_bindings": discovered["compile_bindings"],
        "status": "running",
    }
    write_json(run_root / "run.json", manifest)
    base_config = {
        "case_path": str(case_path),
        "run_root": str(run_root),
        "device": args.device,
        "warmup": args.warmup,
        "active": args.active,
        "repeat": args.repeat,
    }

    records = []
    try:
        records.extend(run_one_worker(run_root, control_root, base_config, "static"))
        write_raw_results(run_root / "raw_results.jsonl", records)
        for index in range(len(discovered["compile_bindings"])):
            records.extend(
                run_one_worker(
                    run_root,
                    control_root,
                    base_config,
                    "dynamic",
                    index,
                )
            )
            write_raw_results(run_root / "raw_results.jsonl", records)
        records.extend(
            run_one_worker(
                run_root,
                control_root,
                base_config,
                "group",
                0,
            )
        )
        write_raw_results(run_root / "raw_results.jsonl", records)

        write_csv(run_root / "summary.csv", SUMMARY_COLUMNS, summary_rows(records))
        comparisons = comparison_rows(records)
        write_csv(
            run_root / "comparison.csv",
            COMPARISON_COLUMNS,
            comparisons,
        )
        from compile_mode_perf_xlsx import write_xlsx_report

        write_xlsx_report(comparisons, run_root / "comparison.xlsx")
        manifest["status"] = "completed"
        manifest["result_count"] = len(records)
        write_json(run_root / "run.json", manifest)
        shutil.rmtree(control_root, ignore_errors=True)
        print(f"wrote results to {run_root}")
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        write_json(run_root / "run.json", manifest)
        raise
    finally:
        shutil.rmtree(run_root / ".cache", ignore_errors=True)
        shutil.rmtree(run_root / ".debug", ignore_errors=True)


def main() -> None:
    args = parse_args()
    try:
        if args._action:
            if args._config is None:
                raise ValueError("internal action requires --_config")
            config = read_json(args._config)
            if args._action == "discover":
                discover(config)
            else:
                run_worker(config)
        else:
            controller(args)
    except (
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
