#!/usr/bin/env python3
"""Compare static, dynamic, and symbolic-group compilation performance."""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import threading
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
    "manual_tiling_dir",
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
    "group_buckets",
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
    parser.add_argument(
        "case",
        nargs="*",
        type=Path,
        help="one or more eager case files, directories, or glob patterns",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prof_log/compile_mode_perf"),
        help="output root",
    )
    parser.add_argument("--run-id", help="run directory name (default: timestamp)")
    parser.add_argument(
        "--device",
        default="npu:0",
        help="device to test, for example npu:0 or cuda:0",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--active", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--group-compile-time",
        action="store_true",
        help="only measure NPU group compile plus first-call autotune time",
    )
    parser.add_argument(
        "--retries",
        "--retry",
        type=int,
        default=3,
        help="number of deferred retries for failed cases (default: 3)",
    )
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


def parse_device(device: str) -> tuple[str, int]:
    match = re.fullmatch(r"(npu|cuda)(?::(\d+))?", device)
    if match is None:
        raise ValueError(
            "--device must be 'npu', 'npu:<index>', 'cuda', or 'cuda:<index>'"
        )
    return match.group(1), int(match.group(2) or 0)


def device_type(device: str) -> str:
    return parse_device(device)[0]


def initialize_device(torch: Any, device: str) -> str:
    kind, index = parse_device(device)
    if kind == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(index)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is not available for --device {device}")
        torch.cuda.set_device(index)
    return kind


def synchronize(torch: Any, device: str) -> None:
    getattr(torch, device_type(device)).synchronize()


def bind_case_device(case: dict[str, object], device: str) -> None:
    # Generated factory calls reference this module global so case edits stay device-only.
    case["forward"].__globals__["device"] = device


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
    getattr(torch, device_type(device)).manual_seed_all(0)
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


def find_kernel_output_code(debug_dir: Path, kernel_name: str) -> Path:
    marker = f"{kernel_name} = async_compile.triton("
    matches = [
        path
        for path in debug_dir.rglob("output_code.py")
        if marker in path.read_text(encoding="utf-8")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one output_code.py for {kernel_name!r}, found {len(matches)}"
        )
    return matches[0]


def archive_manual_tiling(
    debug_dir: Path,
    output_root: Path,
    tiling,
    first_binding=None,
    binding=None,
) -> str:
    if not tiling:
        return ""
    manual_root = output_root / "manual_tiling"
    manual_root.mkdir(parents=True, exist_ok=True)
    extractor = Path(__file__).with_name("extract_triton_kernel.py")
    used_names: set[str] = set()
    for index, record in enumerate(tiling):
        kernel_name = str(record.get("kernel_name", ""))
        if not kernel_name.isidentifier():
            raise ValueError(f"invalid recorded kernel name: {kernel_name!r}")
        suffix = "" if kernel_name not in used_names else f"_{index}"
        used_names.add(kernel_name)
        record_dir = manual_root / f"{kernel_name}{suffix}"
        tiling_path = record_dir / "tiling.json"
        write_json(tiling_path, record)
        source = find_kernel_output_code(debug_dir, kernel_name)
        output = record_dir / f"{kernel_name}_manual.py"
        write_json(
            record_dir / "case.json",
            {
                "compile_binding": first_binding or {},
                "sample_binding": binding or first_binding or {},
            },
        )
        subprocess.run(
            [
                sys.executable,
                str(extractor),
                str(source),
                kernel_name,
                "--manual-tiling",
                "--tiling-json",
                str(tiling_path),
                "-o",
                str(output),
            ],
            check=True,
        )
    return str(manual_root)


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


def profile_cuda(torch: Any, fn, args, kwargs, profile_dir: Path, config):
    from cuda_profiler import CudaProfileParser, TorchCudaProfiler, cuda_kernel_label

    prepare_profile_dir(profile_dir)
    profiler = TorchCudaProfiler(
        profile_dir,
        wait=0,
        warmup=int(config["warmup"]),
        active=int(config["active"]),
        repeat=int(config["repeat"]),
        with_stack=False,
    )
    profiler.run_steps(lambda: fn(*args, **kwargs))
    records = [
        record
        for trace_path in profiler.trace_paths
        for record in CudaProfileParser(trace_path).kernel_records()
    ]
    call_count = int(config["active"]) * int(config["repeat"])
    if not records:
        raise RuntimeError(f"no CUDA device kernels found in {profile_dir}")
    return {
        "mean_us": sum(record.duration for record in records) / call_count,
        "samples": call_count,
        "kernel_count": len(records),
        "kernels": sorted({cuda_kernel_label(record.kernel_name) for record in records}),
    }


def profile_device(torch: Any, fn, args, kwargs, profile_dir: Path, config):
    if device_type(str(config["device"])) == "npu":
        return profile_npu(torch, fn, args, kwargs, profile_dir, config)
    return profile_cuda(torch, fn, args, kwargs, profile_dir, config)


class GroupCompileProfiler:
    """Measure grouped binary compilation and serial group autotuning phases."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread_state = threading.local()
        self._patches: list[tuple[object, str, object]] = []
        self._group_plan_ids: set[int] = set()
        self.grouping_end_ns: int | None = None
        self.first_binary_start_ns: int | None = None
        self.last_binary_end_ns: int | None = None
        self.benchmark_start_ns: int | None = None
        self.autotune_end_ns: int | None = None
        self.grouped_kernel_count = 0
        self.compiled_kernel_count = 0
        self.candidate_count = 0
        self.reachable_group_count = 0
        self.variant_count = 0
        self.group_tiers: list[dict[str, object]] = []

    def install(self) -> None:
        import torch_npu._inductor.runtime.triton_heuristics as heuristics

        self._patch_function(heuristics, "_triton_config_npu_index_grouped")
        self._patch_function(heuristics.triton, "compile", binary=True)
        self._patch_method(heuristics.NPUCachingAutotuner, "_precompile_config")
        self._patch_method(
            heuristics.NPUSymbolicGroupedAutotuner,
            "_autotune_all_groups",
        )

    def uninstall(self) -> None:
        while self._patches:
            owner, name, original = self._patches.pop()
            setattr(owner, name, original)

    def _patch_function(self, owner: object, name: str, binary: bool = False) -> None:
        original = getattr(owner, name)
        profiler = self

        def wrapped(*args, **kwargs):
            active_autotuner = getattr(self._thread_state, "group_autotuner", None)
            if binary and active_autotuner is not None:
                self._record_binary_start(active_autotuner)
            result = original(*args, **kwargs)
            if binary and active_autotuner is not None:
                self._record_binary_end(active_autotuner)
            if not binary:
                profiler._record_grouped_plan(args, kwargs, result)
            return result

        setattr(owner, name, wrapped)
        self._patches.append((owner, name, original))

    def _patch_method(self, cls: type, name: str) -> None:
        original = getattr(cls, name)
        profiler = self

        def wrapped(instance, *args, **kwargs):
            is_group_autotune = cls.__name__ == "NPUSymbolicGroupedAutotuner"
            is_binary_task = name == "_precompile_config" and profiler._is_grouped(instance)
            if is_binary_task:
                profiler._thread_state.group_autotuner = instance
            if is_group_autotune:
                profiler._record_autotune(time.perf_counter_ns(), entering=True)
            try:
                result = original(instance, *args, **kwargs)
            except BaseException:
                raise
            else:
                if is_group_autotune:
                    profiler._record_autotune(time.perf_counter_ns(), entering=False)
                return result
            finally:
                if is_binary_task:
                    profiler._thread_state.group_autotuner = None

        setattr(cls, name, wrapped)
        self._patches.append((cls, name, original))

    def _record_grouped_plan(self, args, kwargs, result) -> None:
        inductor_meta = args[1] if len(args) > 1 else kwargs.get("inductor_meta")
        if not isinstance(inductor_meta, dict):
            return
        plan = inductor_meta.get("grouped_candidate_plan")
        if not isinstance(plan, dict):
            return
        plan_id = id(plan)
        with self._lock:
            if plan_id in self._group_plan_ids:
                return
            self._group_plan_ids.add(plan_id)
            self.grouped_kernel_count += 1
            self.grouping_end_ns = time.perf_counter_ns()
            self.candidate_count += sum(
                len(candidates)
                for candidates in plan.get("group_to_candidates", ())
            )
            self.reachable_group_count += len(plan.get("reachable_group_ids", ()))
            self.variant_count += len(plan.get("variant_order", ()))
            group_to_candidates = plan.get("group_to_candidates", ())
            feature_specs = tuple(plan.get("group_features", ()))
            benchmark_inputs = tuple(
                plan.get("benchmark_feature_inputs_by_group", ())
            )
            for group_id in plan.get("reachable_group_ids", ()):
                remaining = int(group_id)
                features = []
                for feature_index, feature_spec in enumerate(feature_specs):
                    buckets = tuple(feature_spec.get("buckets", ()))
                    radix = len(buckets) + 1
                    bucket_index = remaining % radix
                    remaining //= radix
                    features.append(
                        {
                            "name": feature_spec.get("name", ""),
                            "source": feature_spec.get("source", ""),
                            "axis_names": list(feature_spec.get("axis_names", ())),
                            "buckets": list(buckets),
                            "bucket_index": bucket_index,
                            "representative": (
                                benchmark_inputs[group_id][feature_index]
                                if group_id < len(benchmark_inputs)
                                and feature_index < len(benchmark_inputs[group_id])
                                else None
                            ),
                        }
                    )
                self.group_tiers.append(
                    {
                        "kernel_name": inductor_meta.get("kernel_name", ""),
                        "group_id": group_id,
                        "candidate_count": (
                            len(group_to_candidates[group_id])
                            if group_id < len(group_to_candidates)
                            else 0
                        ),
                        "features": features,
                    }
                )

    def _record_binary_start(self, autotuner: object) -> None:
        with self._lock:
            timestamp = time.perf_counter_ns()
            if self.first_binary_start_ns is None:
                self.first_binary_start_ns = timestamp

    def _record_binary_end(self, autotuner: object) -> None:
        with self._lock:
            self.compiled_kernel_count += 1
            self.last_binary_end_ns = time.perf_counter_ns()

    @staticmethod
    def _is_grouped(autotuner: object) -> bool:
        plan = getattr(autotuner, "candidate_plan", None)
        return isinstance(plan, dict) and bool(plan.get("group_to_candidates"))

    def _record_autotune(self, timestamp: int, entering: bool) -> None:
        if entering:
            with self._lock:
                if self.benchmark_start_ns is None:
                    self.benchmark_start_ns = timestamp
        else:
            with self._lock:
                self.autotune_end_ns = timestamp

    def summary(self) -> dict[str, object]:
        binary_ms = None
        if self.first_binary_start_ns is not None and self.last_binary_end_ns is not None:
            binary_ms = (self.last_binary_end_ns - self.first_binary_start_ns) / 1e6
        benchmark_ms = None
        if self.last_binary_end_ns is not None and self.autotune_end_ns is not None:
            benchmark_ms = (self.autotune_end_ns - self.last_binary_end_ns) / 1e6
        result = {
            "grouped_kernel_count": self.grouped_kernel_count,
            "compiled_kernel_count": self.compiled_kernel_count,
            "candidate_count": self.candidate_count,
            "reachable_group_count": self.reachable_group_count,
            "variant_count": self.variant_count,
            "group_tiers": self.group_tiers,
            "binary_compile_ms": binary_ms,
            "group_benchmark_ms": benchmark_ms,
        }
        if binary_ms is not None and benchmark_ms is not None:
            result["group_compile_total_ms"] = binary_ms + benchmark_ms
        return result

    def print_summary(self) -> None:
        for name, value in self.summary().items():
            if name == "group_tiers":
                print(f"{name}={json.dumps(value, sort_keys=True)}", flush=True)
                continue
            if isinstance(value, float):
                print(f"{name}={value:.3f}", flush=True)
            else:
                print(f"{name}={value}", flush=True)


def compile_forward(torch: Any, case, args, kwargs, dynamic: bool, device: str):
    torch._dynamo.reset()
    if dynamic:
        mark_dynamic_inputs(torch, case, args, kwargs)
    compiled = torch.compile(
        case["forward"],
        backend="inductor",
        dynamic=None if dynamic else False,
    )
    compiled(*args, **kwargs)
    synchronize(torch, device)
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
    manual_tiling_dir="",
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
        "manual_tiling_dir": manual_tiling_dir,
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
            compiled = compile_forward(
                torch, case, args, kwargs, dynamic=False, device=device
            )
        finally:
            tiling = recorder.stop_capture()
        debug_dir = output_root / "torch_compile_debug"
        replace_tree(latest_compile_debug_dir(Path(str(config["debug_root"]))), debug_dir)
        manual_tiling_dir = archive_manual_tiling(
            debug_dir, output_root, tiling, binding, binding
        )
        timing = profile_device(
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
                manual_tiling_dir=manual_tiling_dir,
            )
        )
    return records


def run_dynamic(torch: Any, case, config, recorder, group: bool, group_profiler=None):
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
            torch,
            case,
            compile_args,
            compile_kwargs,
            dynamic=True,
            device=device,
        )
    finally:
        compile_tiling = recorder.stop_capture()
    compile_debug_root = artifact_mode_root(
        run_root,
        mode,
        first_index=None if group else first_index,
    )
    if group_profiler is not None:
        replace_tree(
            latest_compile_debug_dir(Path(str(config["debug_root"]))),
            compile_debug_root / "torch_compile_debug",
        )
        group_profiler.print_summary()
        return []
    replace_tree(
        latest_compile_debug_dir(Path(str(config["debug_root"]))),
        compile_debug_root / "torch_compile_debug",
    )
    compile_debug_dir = compile_debug_root / "torch_compile_debug"

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
                timing = profile_device(
                    torch, compiled, args, kwargs, output_root / "profiles", config
                )
            finally:
                tiling = recorder.stop_capture()
        else:
            timing = profile_device(
                torch, compiled, args, kwargs, output_root / "profiles", config
            )
            tiling = compile_tiling
        manual_tiling_dir = archive_manual_tiling(
            compile_debug_dir, output_root, tiling, first_binding, binding
        )
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
                manual_tiling_dir=manual_tiling_dir,
            )
        )
    return records


def run_worker(config: dict[str, object]) -> None:
    debug_root = Path(str(config["debug_root"]))
    debug_root.mkdir(parents=True, exist_ok=True)

    import torch

    from autotune_tiling import BestTilingRecorder

    torch._dynamo.config.debug_dir_root = str(debug_root)
    device = str(config["device"])
    kind = initialize_device(torch, device)
    case = load_case(Path(str(config["case_path"])))
    bind_case_device(case, device)
    recorder = BestTilingRecorder(kind)
    recorder.install()
    group_profiler = None
    if config.get("measure_group_compile_time", False):
        group_profiler = GroupCompileProfiler()
        group_profiler.install()
    try:
        execution = str(config["execution"])
        if execution == "static":
            records = run_static(torch, case, config, recorder)
        elif execution == "dynamic":
            records = run_dynamic(torch, case, config, recorder, group=False)
        elif execution == "group":
            records = run_dynamic(
                torch,
                case,
                config,
                recorder,
                group=True,
                group_profiler=group_profiler,
            )
        else:
            raise ValueError(f"unsupported execution: {execution}")
    finally:
        if group_profiler is not None:
            group_profiler.uninstall()
        recorder.uninstall()
        synchronize(torch, device)
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
                "manual_tiling_dir": record.get("manual_tiling_dir", ""),
                "result_dir": record["result_dir"],
            }
        )
    return rows


def ratio(numerator: float, denominator: float) -> str:
    return "" if denominator == 0 else f"{numerator / denominator:.3f}"


def group_buckets(tiling: object) -> list[dict[str, object]]:
    if not isinstance(tiling, list):
        return []
    buckets = []
    for record in tiling:
        if not isinstance(record, dict):
            continue
        features = record.get("group_features")
        if not features:
            continue
        buckets.append(
            {
                "kernel_name": record.get("kernel_name", ""),
                "group_features": features,
            }
        )
    return buckets


def comparison_rows(
    records: list[dict[str, object]],
    require_group: bool = True,
):
    records_by_case: dict[str, list[dict[str, object]]] = {}
    for record in records:
        records_by_case.setdefault(str(record["case"]), []).append(record)

    rows = []
    for case_name, case_records in records_by_case.items():
        static = {
            int(record["sample_index"]): record
            for record in case_records
            if record["mode"] == "static"
        }
        dynamic = {
            (int(record["first_index"]), int(record["sample_index"])): record
            for record in case_records
            if record["mode"] == "dynamic"
        }
        group = {
            int(record["sample_index"]): record
            for record in case_records
            if record["mode"] == "group"
        }
        first_indices = sorted({key[0] for key in dynamic})
        sample_indices = sorted(static)
        for first_index in first_indices:
            for sample_index in sample_indices:
                try:
                    static_record = static[sample_index]
                    dynamic_record = dynamic[first_index, sample_index]
                except KeyError as error:
                    raise ValueError(
                        f"incomplete result matrix for case={case_name}, "
                        f"first={first_index}, sample={sample_index}"
                    ) from error
                group_record = group.get(sample_index)
                if require_group and group_record is None:
                    raise ValueError(
                        f"incomplete group result matrix for case={case_name}, "
                        f"sample={sample_index}"
                    )
                compared_records = [static_record, dynamic_record]
                if group_record is not None:
                    compared_records.append(group_record)
                shapes = {record["shape"] for record in compared_records}
                if len(shapes) != 1:
                    raise ValueError(
                        f"input shape mismatch for case={case_name}, "
                        f"sample={sample_index}: {sorted(shapes)}"
                    )
                signatures = {
                    compact_json(record["input_signature"])
                    for record in compared_records
                }
                if len(signatures) != 1:
                    raise ValueError(
                        f"input signature mismatch for case={case_name}, "
                        f"sample={sample_index}"
                    )
                static_us = float(static_record["mean_us"])
                dynamic_us = float(dynamic_record["mean_us"])
                group_us = (
                    float(group_record["mean_us"])
                    if group_record is not None
                    else None
                )
                bucket_records = (
                    group_buckets(group_record["tiling"])
                    if group_record is not None
                    else []
                )
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
                        "group_us": f"{group_us:.3f}" if group_us is not None else "",
                        "group_static_ratio": (
                            ratio(group_us, static_us) if group_us is not None else ""
                        ),
                        "group_buckets": compact_json(bucket_records)
                        if bucket_records
                        else "",
                        "group_tiling": compact_json(group_record["tiling"])
                        if group_record is not None and group_record["tiling"]
                        else "",
                    }
                )
    return rows


def runtime_binding_label(binding: object) -> str:
    if not isinstance(binding, dict):
        return str(binding)
    return ",".join(f"{key}={binding[key]}" for key in sorted(binding))


def static_group_summary_rows(
    records: list[dict[str, object]],
) -> list[tuple[str, float, float, str]]:
    static = {
        int(record["sample_index"]): record
        for record in records
        if record["mode"] == "static"
    }
    group = {
        int(record["sample_index"]): record
        for record in records
        if record["mode"] == "group"
    }
    rows = []
    for sample_index in sorted(static):
        group_record = group.get(sample_index)
        if group_record is None:
            continue
        static_us = float(static[sample_index]["mean_us"])
        group_us = float(group_record["mean_us"])
        rows.append(
            (
                runtime_binding_label(static[sample_index].get("binding", {})),
                static_us,
                group_us,
                ratio(group_us, static_us),
            )
        )
    return rows


def print_static_group_summary(case_name: str, records: list[dict[str, object]]) -> None:
    rows = static_group_summary_rows(records)

    print()
    print("=== static vs group summary ===")
    print(f"case={case_name}")
    print()
    print("runtime".ljust(16) + "static_us".rjust(12) + "group_us".rjust(12) + "g/static".rjust(12))
    for runtime, static_us, group_us, group_ratio in rows:
        print(
            runtime.ljust(16)
            + f"{static_us:12.3f}{group_us:12.3f}{group_ratio:>12}"
        )


def print_batch_static_group_summary(records: list[dict[str, object]]) -> None:
    records_by_case: dict[str, list[dict[str, object]]] = {}
    for record in records:
        records_by_case.setdefault(str(record["case"]), []).append(record)

    print()
    print("=== batch static vs group summary ===")
    print(
        "case".ljust(64)
        + "runtime".ljust(16)
        + "static_us".rjust(12)
        + "group_us".rjust(12)
        + "g/static".rjust(12)
    )
    for case_name, case_records in records_by_case.items():
        for runtime, static_us, group_us, group_ratio in static_group_summary_rows(
            case_records
        ):
            print(
                case_name.ljust(64)
                + runtime.ljust(16)
                + f"{static_us:12.3f}{group_us:12.3f}{group_ratio:>12}"
            )


def static_dynamic_summary_rows(
    records: list[dict[str, object]],
) -> list[tuple[str, str, float, float, str]]:
    static = {
        int(record["sample_index"]): record
        for record in records
        if record["mode"] == "static"
    }
    rows = []
    for record in records:
        if record["mode"] != "dynamic":
            continue
        static_record = static.get(int(record["sample_index"]))
        if static_record is None:
            continue
        static_us = float(static_record["mean_us"])
        dynamic_us = float(record["mean_us"])
        rows.append(
            (
                str(record["first_shape"]),
                runtime_binding_label(record.get("binding", {})),
                static_us,
                dynamic_us,
                ratio(dynamic_us, static_us),
            )
        )
    return rows


def print_static_dynamic_summary(case_name: str, records: list[dict[str, object]]) -> None:
    print()
    print("=== static vs dynamic summary ===")
    print(f"case={case_name}")
    print()
    print(
        "first_shape".ljust(24)
        + "runtime".ljust(16)
        + "static_us".rjust(12)
        + "dynamic_us".rjust(12)
        + "d/static".rjust(12)
    )
    for first_shape, runtime, static_us, dynamic_us, dynamic_ratio in (
        static_dynamic_summary_rows(records)
    ):
        print(
            first_shape.ljust(24)
            + runtime.ljust(16)
            + f"{static_us:12.3f}{dynamic_us:12.3f}{dynamic_ratio:>12}"
        )


def print_batch_static_dynamic_summary(records: list[dict[str, object]]) -> None:
    records_by_case: dict[str, list[dict[str, object]]] = {}
    for record in records:
        records_by_case.setdefault(str(record["case"]), []).append(record)

    print()
    print("=== batch static vs dynamic summary ===")
    print(
        "case".ljust(64)
        + "first_shape".ljust(24)
        + "runtime".ljust(16)
        + "static_us".rjust(12)
        + "dynamic_us".rjust(12)
        + "d/static".rjust(12)
    )
    for case_name, case_records in records_by_case.items():
        for first_shape, runtime, static_us, dynamic_us, dynamic_ratio in (
            static_dynamic_summary_rows(case_records)
        ):
            print(
                case_name.ljust(64)
                + first_shape.ljust(24)
                + runtime.ljust(16)
                + f"{static_us:12.3f}{dynamic_us:12.3f}{dynamic_ratio:>12}"
            )


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


def expand_case_paths(case_inputs: list[Path]) -> list[Path]:
    case_paths = []
    seen = set()
    list_stack = set()

    def expand_inputs(inputs: list[Path]) -> None:
        for case_input in inputs:
            pattern = str(case_input)
            if glob.has_magic(pattern):
                matched_paths = [
                    Path(path) for path in sorted(glob.glob(pattern, recursive=True))
                ]
                if not matched_paths:
                    raise ValueError(f"case pattern matched no paths: {case_input}")
            else:
                matched_paths = [case_input]
            for matched_path in matched_paths:
                if matched_path.is_file() and matched_path.suffix != ".py":
                    list_path = matched_path.resolve()
                    if list_path in list_stack:
                        raise ValueError(f"case list includes itself or has a cycle: {matched_path}")
                    try:
                        lines = matched_path.read_text(encoding="utf-8-sig").splitlines()
                    except OSError as error:
                        raise ValueError(f"cannot read case list {matched_path}: {error}") from error
                    listed_paths = [
                        Path(line.strip())
                        for line in lines
                        if line.strip()
                    ]
                    if not listed_paths:
                        raise ValueError(f"case list is empty: {matched_path}")
                    list_stack.add(list_path)
                    try:
                        expand_inputs(listed_paths)
                    finally:
                        list_stack.remove(list_path)
                    continue
                if matched_path.is_file():
                    candidates = [matched_path]
                elif matched_path.is_dir():
                    candidates = sorted(
                        path
                        for path in matched_path.iterdir()
                        if path.is_file() and path.name.endswith("_case.py")
                    )
                    if not candidates:
                        raise ValueError(
                            f"case directory contains no *_case.py files: {matched_path}"
                        )
                else:
                    raise ValueError(f"case path does not exist: {matched_path}")
                for candidate in candidates:
                    resolved = candidate.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        case_paths.append(resolved)

    expand_inputs(case_inputs)
    if not case_paths:
        raise ValueError("at least one case file or directory is required")
    return case_paths


def validate_controller_args(args: argparse.Namespace) -> list[Path]:
    if not args.case:
        raise ValueError("at least one case file or directory is required")
    case_paths = expand_case_paths(args.case)
    parse_device(args.device)
    if args.warmup < 0 or args.active <= 0 or args.repeat <= 0:
        raise ValueError("warmup must be non-negative; active and repeat must be positive")
    if args.retries < 0:
        raise ValueError("retries must be non-negative")
    if args.run_id and re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id) is None:
        raise ValueError("run-id may only contain letters, digits, '.', '_', and '-'")
    return case_paths


def safe_case_directory_name(index: int, case_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", case_path.stem).strip("._-")
    return f"{index:03d}_{stem or 'case'}"


def run_case(
    case_path: Path,
    case_root: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    control_root = case_root / ".control"
    shutil.rmtree(control_root, ignore_errors=True)
    control_root.mkdir(parents=True, exist_ok=True)
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

    base_config = {
        "case_path": str(case_path),
        "run_root": str(case_root),
        "device": args.device,
        "warmup": args.warmup,
        "active": args.active,
        "repeat": args.repeat,
        "measure_group_compile_time": args.group_compile_time,
    }
    records = []
    completed = False
    try:
        if args.group_compile_time:
            if device_type(args.device) != "npu":
                raise ValueError("--group-compile-time requires an NPU device")
            records.extend(
                run_one_worker(
                    case_root,
                    control_root,
                    base_config,
                    "group",
                    0,
                )
            )
        else:
            records.extend(run_one_worker(case_root, control_root, base_config, "static"))
            for index in range(len(discovered["compile_bindings"])):
                records.extend(
                    run_one_worker(
                        case_root,
                        control_root,
                        base_config,
                        "dynamic",
                        index,
                    )
                )
            if device_type(args.device) == "npu":
                records.extend(
                    run_one_worker(
                        case_root,
                        control_root,
                        base_config,
                        "group",
                        0,
                    )
                )
        completed = True
        return discovered, records
    finally:
        shutil.rmtree(case_root / ".cache", ignore_errors=True)
        shutil.rmtree(case_root / ".debug", ignore_errors=True)
        if completed:
            shutil.rmtree(control_root, ignore_errors=True)


def controller(args: argparse.Namespace) -> None:
    case_paths = validate_controller_args(args)
    kind = device_type(args.device)
    batch = len(case_paths) > 1
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (args.output / run_id).resolve()
    if run_root.exists():
        raise ValueError(f"run directory already exists: {run_root}")
    run_root.mkdir(parents=True)

    manifest = {
        "run_id": run_id,
        "case_paths": [str(case_path) for case_path in case_paths],
        "device": args.device,
        "warmup": args.warmup,
        "active": args.active,
        "repeat": args.repeat,
        "retries": args.retries,
        "retry_round": 0,
        "batch": batch,
        "total_cases": len(case_paths),
        "completed_cases": 0,
        "cases": [],
        "failed_cases": [],
        "status": "running",
    }
    write_json(run_root / "run.json", manifest)

    records = []
    discovered_cases = []
    pending_cases = list(enumerate(case_paths))
    failed_by_index = {}
    try:
        for retry_round in range(args.retries + 1):
            if not pending_cases:
                break
            next_pending = []
            manifest["retry_round"] = retry_round
            retry_label = f"retry {retry_round}/{args.retries} " if retry_round else ""
            for case_index, case_path in pending_cases:
                progress = f"[{case_index + 1}/{len(case_paths)}]"
                print(
                    f"{progress} {retry_label}starting case={case_path.name}".strip(),
                    flush=True,
                )
                case_root = (
                    run_root / "cases" / safe_case_directory_name(case_index, case_path)
                    if batch
                    else run_root
                )
                case_root.mkdir(parents=True, exist_ok=True)
                try:
                    discovered, case_records = run_case(case_path, case_root, args)
                    case_name = str(discovered["name"])
                    if any(item["name"] == case_name for item in discovered_cases):
                        raise ValueError(f"duplicate CASE name in batch: {case_name}")
                except Exception as error:
                    failure = {
                        "name": case_path.stem,
                        "case_path": str(case_path),
                        "attempts": retry_round + 1,
                        "retry_count": retry_round,
                        "error": f"{type(error).__name__}: {error}",
                    }
                    failed_by_index[case_index] = failure
                    next_pending.append((case_index, case_path))
                    print(
                        f"{progress} {retry_label}failed case={case_path.name}; "
                        f"deferred: {failure['error']}",
                        flush=True,
                    )
                    continue

                discovered_cases.append(
                    {
                        "name": case_name,
                        "case_path": str(case_path),
                        "sample_bindings": discovered["sample_bindings"],
                        "compile_bindings": discovered["compile_bindings"],
                    }
                )
                failed_by_index.pop(case_index, None)
                records.extend(case_records)
                write_raw_results(run_root / "raw_results.jsonl", records)
                manifest["cases"] = discovered_cases
                manifest["completed_cases"] = len(discovered_cases)
                manifest["failed_cases"] = [
                    failed_by_index[index] for index in sorted(failed_by_index)
                ]
                write_json(run_root / "run.json", manifest)
                if args.group_compile_time:
                    print(
                        f"{progress} {retry_label}completed case={case_name} "
                        f"(group compile timing only)",
                        flush=True,
                    )
                elif kind == "npu":
                    print_static_group_summary(case_name, case_records)
                else:
                    print_static_dynamic_summary(case_name, case_records)
                print(
                    f"{progress} {retry_label}completed case={case_name} "
                    f"(success={len(discovered_cases)}, "
                    f"pending={len(next_pending)})",
                    flush=True,
                )

            pending_cases = next_pending
            manifest["completed_cases"] = len(discovered_cases)
            manifest["failed_cases"] = [
                failed_by_index[index] for index in sorted(failed_by_index)
            ]
            write_json(run_root / "run.json", manifest)
            if pending_cases and retry_round < args.retries:
                print(
                    f"deferred retry queue: {len(pending_cases)} case(s); "
                    f"retrying after round {retry_round + 1} completes",
                    flush=True,
                )

        failed_cases = [failed_by_index[index] for index in sorted(failed_by_index)]
        if failed_cases and not batch:
            raise RuntimeError(failed_cases[0]["error"])

        if args.group_compile_time:
            if failed_cases:
                print()
                print(
                    f"=== batch failures ({len(failed_cases)}/{len(case_paths)}) ==="
                )
                for failure in failed_cases:
                    print(f"case={failure['name']} error={failure['error']}")
            manifest["status"] = (
                "completed_with_failures" if failed_cases else "completed"
            )
            manifest["result_count"] = 0
            manifest["failed_cases"] = failed_cases
            if not batch and discovered_cases:
                manifest.update(discovered_cases[0])
            write_json(run_root / "run.json", manifest)
            return

        write_csv(run_root / "summary.csv", SUMMARY_COLUMNS, summary_rows(records))
        comparisons = comparison_rows(records, require_group=kind == "npu")
        write_csv(
            run_root / "comparison.csv",
            COMPARISON_COLUMNS,
            comparisons,
        )
        from compile_mode_perf_xlsx import write_xlsx_report

        write_xlsx_report(comparisons, run_root / "comparison.xlsx")
        if batch:
            if records:
                if kind == "npu":
                    print_batch_static_group_summary(records)
                else:
                    print_batch_static_dynamic_summary(records)
            else:
                print()
                if kind == "npu":
                    print("=== batch static vs group summary ===")
                else:
                    print("=== batch static vs dynamic summary ===")
                print("no successful cases")
        if failed_cases:
            print()
            print(
                f"=== batch failures ({len(failed_cases)}/{len(case_paths)}) ==="
            )
            for failure in failed_cases:
                print(f"case={failure['name']} error={failure['error']}")
        if not records:
            raise RuntimeError("all batch cases failed; no comparison results generated")
        manifest["status"] = "completed_with_failures" if failed_cases else "completed"
        manifest["result_count"] = len(records)
        manifest["failed_cases"] = failed_cases
        if not batch:
            case_info = discovered_cases[0]
            manifest.update(case_info)
        write_json(run_root / "run.json", manifest)
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
