"""Compare static and dynamic elementwise kernel performance.

The cases are split into memory-bound and compute-bound workloads. P0 is the
small first-pass suite; P1 adds scalar operation classes for follow-up analysis.
Each run profiles only the selected execution. Device-side kernel durations are
parsed from profiler artifacts; correctness is not checked.

Run static kernels, compiling one specialized kernel for each runtime shape:

    RUN_ID=baseline_001 DEVICE=npu EXECUTION=static PRIORITY=P0 \
        python scripts/tests/elementwise_dynamic_perf/elementwise_op_cost_cases.py

Run one dynamic kernel compiled/autotuned with COMPILE_SHAPE and reuse it for
all SHAPES. SYMBOLIC_DIMS identifies which dimensions are allowed to change:

    RUN_ID=baseline_001 DEVICE=npu EXECUTION=dynamic SYMBOLIC_DIMS=0 COMPILE_SHAPE=8192 \
        PRIORITY=P0 python scripts/tests/elementwise_dynamic_perf/elementwise_op_cost_cases.py

Use EXECUTION=custom on NPU to benchmark a locally modified torch_npu dynamic
path. Use EXECUTION=group to enable symbolic group autotune.

Useful environment variables:

    DEVICE=npu|cuda
    EXECUTION=eager|static|dynamic|custom|group
    PRIORITY=P0 or PRIORITY=P0,P1
    CASES=memory_add,exp_log
    COMPILE_SHAPE=8192
    SHAPES=127;128;129;2047;2048;2049;8191;8192;8193;1048575;1048576;1048577
    SYMBOLIC_DIMS=0
    DTYPE=float32
    WARMUP=5 ACTIVE=20 REPEAT=1
    RUN_ID=elementwise_baseline_001
    RECORD_RESULTS=0|1
    PROFILE_ROOT=prof_log/elementwise_dynamic_perf

The default SHAPES bracket 128, 2048, 8192, and 2^20 boundaries. Use separate
dynamic runs with COMPILE_SHAPE=128, 8192, and 1048576 to quantify how the
first shape affects reuse performance.

Runs sharing RUN_ID are merged into <PROFILE_ROOT>/<RUN_ID>/summary.csv. Raw
profiles are stored below <PROFILE_ROOT>/<RUN_ID>/profiles/. Cross-device
automatic merging requires both processes to access the same PROFILE_ROOT;
file locking and atomic replacement protect concurrent summary updates.
"""

from __future__ import annotations

import csv
import fcntl
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch


SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from tools.autotune_tiling import BestTilingRecorder
from tools.cuda_profiler import CudaProfileParser, TorchCudaProfiler, cuda_kernel_label
from tools.npu_profiler import ProfileResultParser, TorchNpuProfiler


TensorArgs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
CaseFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
DEFAULT_SHAPES = (
    "127;128;129;2047;2048;2049;8191;8192;8193;"
    "1048575;1048576;1048577"
)
GROUP_AUTOTUNE_ENV = "INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE"
DYNAMIC_EXECUTIONS = ("dynamic", "custom", "group")
RESULT_COLUMNS = (
    "run_id",
    "device",
    "execution",
    "priority",
    "bound",
    "case",
    "group",
    "input_kind",
    "dtype",
    "compile_shape",
    "runtime_shape",
    "symbolic_dims",
    "scalar_ops",
    "samples",
    "device_kernel_count",
    "device_call_mean_us",
    "kernels",
    "autotune_tiling_configs",
    "result_dir",
)
SUMMARY_KEY_COLUMNS = (
    "device",
    "execution",
    "case",
    "input_kind",
    "dtype",
    "compile_shape",
    "runtime_shape",
    "symbolic_dims",
)


@dataclass(frozen=True)
class Case:
    fn: CaseFn
    priority: str
    bound: str
    group: str
    scalar_ops: tuple[str, ...]
    input_kind: str = "float"


@dataclass(frozen=True)
class TimingResult:
    mean_us: float
    sample_count: int
    kernel_count: int
    kernels: tuple[str, ...]


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be one of 0, 1, false, true, no, yes, off, or on")


def env_run_id() -> str:
    value = os.environ.get("RUN_ID", "").strip()
    if not value:
        raise ValueError("RUN_ID is required to isolate and aggregate experiment results")
    if re.fullmatch(r"[A-Za-z0-9._-]+", value) is None:
        raise ValueError("RUN_ID may only contain letters, digits, '.', '_', and '-'")
    return value


def env_dtype(name: str, default: torch.dtype) -> torch.dtype:
    value = os.environ.get(name)
    if value is None:
        return default
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = value.strip().lower()
    if key not in mapping:
        raise ValueError(
            f"Unsupported DTYPE={value!r}, expected one of {sorted(mapping)}"
        )
    return mapping[key]


def parse_shape(value: str) -> tuple[int, ...]:
    shape = tuple(
        int(dim.strip())
        for dim in value.replace("x", ",").split(",")
        if dim.strip()
    )
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"Invalid shape {value!r}; dimensions must be positive")
    return shape


def parse_shapes() -> list[tuple[int, ...]]:
    value = os.environ.get("SHAPES")
    if value is None:
        legacy_shape = os.environ.get("SHAPE")
        value = legacy_shape if legacy_shape is not None else DEFAULT_SHAPES
    shapes = [parse_shape(item) for item in value.split(";") if item.strip()]
    if not shapes:
        raise ValueError("SHAPES must contain at least one shape")
    return shapes


def parse_symbolic_dims(rank: int) -> tuple[int, ...]:
    value = os.environ.get("SYMBOLIC_DIMS", "0")
    dims = []
    for item in value.split(","):
        if not item.strip():
            continue
        dim = int(item)
        dim = dim + rank if dim < 0 else dim
        if dim < 0 or dim >= rank:
            raise ValueError(f"SYMBOLIC_DIMS contains {item!r}, but rank is {rank}")
        dims.append(dim)
    if not dims:
        raise ValueError("SYMBOLIC_DIMS must contain at least one dimension")
    return tuple(sorted(set(dims)))


def validate_symbolic_shapes(
    compile_shape: tuple[int, ...],
    runtime_shapes: list[tuple[int, ...]],
    symbolic_dims: tuple[int, ...],
) -> None:
    for runtime_shape in runtime_shapes:
        if len(runtime_shape) != len(compile_shape):
            raise ValueError(
                f"Runtime shape {runtime_shape} and compile shape {compile_shape} "
                "must have the same rank"
            )
        for dim, (compile_size, runtime_size) in enumerate(
            zip(compile_shape, runtime_shape)
        ):
            if dim not in symbolic_dims and compile_size != runtime_size:
                raise ValueError(
                    f"Dimension {dim} is static but changes from {compile_size} "
                    f"to {runtime_size}; add it to SYMBOLIC_DIMS"
                )


def shape_label(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape)


def sync(device: str) -> None:
    getattr(torch, device).synchronize()


def make_float_inputs(
    shape: tuple[int, ...], device: str, dtype: torch.dtype
) -> TensorArgs:
    torch.manual_seed(0)
    x = torch.rand(shape, device=device, dtype=dtype) + 0.5
    y = torch.rand(shape, device=device, dtype=dtype) + 1.0
    z = torch.rand(shape, device=device, dtype=dtype) + 1.5
    return x, y, z


def make_int_inputs(shape: tuple[int, ...], device: str) -> TensorArgs:
    torch.manual_seed(0)
    x = torch.randint(1, 1024, shape, device=device, dtype=torch.int32)
    y = torch.randint(1, 1024, shape, device=device, dtype=torch.int32)
    z = torch.randint(1, 1024, shape, device=device, dtype=torch.int32)
    return x, y, z


def make_inputs(
    shape: tuple[int, ...],
    device: str,
    dtype: torch.dtype,
    input_kind: str,
) -> TensorArgs:
    if input_kind == "float":
        return make_float_inputs(shape, device, dtype)
    if input_kind == "int":
        return make_int_inputs(shape, device)
    raise ValueError(f"Unsupported input_kind={input_kind!r}")


def memory_add(x, y, z):
    return x + y


def exp_log_ops(x, y, z):
    return (
        torch.exp(x * 0.01)
        + torch.expm1(y * 0.01)
        + torch.log1p(z)
        + torch.log2(y + 1.0)
    )


def bitwise_ops(x, y, z):
    return ((x & y) ^ (z | 17)) + ((x << 1) & 255) - ((y >> 1) & 127)


def division_ops(x, y, z):
    return x / (y + 0.25) + torch.div(z, x + 0.5)


def pow_ops(x, y, z):
    return torch.pow(x + 0.1, 2.0) + torch.pow(y + 0.1, z * 0.1)


def trig_hyperbolic_ops(x, y, z):
    return torch.sin(x) * torch.cos(y) + torch.tanh(z) + torch.atan(x * 0.1)


CASES: dict[str, Case] = {
    "memory_add": Case(
        memory_add,
        priority="P0",
        bound="memory_bound",
        group="single float add",
        scalar_ops=("add",),
    ),
    "exp_log": Case(
        exp_log_ops,
        priority="P0",
        bound="compute_bound",
        group="exp and log",
        scalar_ops=("exp", "expm1", "log1p", "log2"),
    ),
    "bitwise": Case(
        bitwise_ops,
        priority="P1",
        bound="memory_bound",
        group="integer bitwise",
        scalar_ops=("and", "or", "xor", "left_shift", "right_shift"),
        input_kind="int",
    ),
    "division": Case(
        division_ops,
        priority="P1",
        bound="compute_bound",
        group="division",
        scalar_ops=("div",),
    ),
    "pow": Case(
        pow_ops,
        priority="P1",
        bound="compute_bound",
        group="constant and tensor power",
        scalar_ops=("pow",),
    ),
    "trig_hyperbolic": Case(
        trig_hyperbolic_ops,
        priority="P1",
        bound="compute_bound",
        group="trig and hyperbolic",
        scalar_ops=("sin", "cos", "tanh", "atan"),
    ),
}


def selected_case_names() -> list[str]:
    selected = os.environ.get("CASES")
    if selected:
        names = [item.strip() for item in selected.split(",") if item.strip()]
        unknown = sorted(set(names) - set(CASES))
        if unknown:
            raise ValueError(
                f"Unknown CASES entries: {unknown}; valid cases: {sorted(CASES)}"
            )
        return names

    priorities = {
        item.strip().upper()
        for item in os.environ.get("PRIORITY", "P0").split(",")
        if item.strip()
    }
    unknown = sorted(priorities - {"P0", "P1"})
    if unknown:
        raise ValueError(f"Unknown PRIORITY entries: {unknown}; expected P0 or P1")
    return [name for name, case in CASES.items() if case.priority in priorities]


def mark_symbolic(args: TensorArgs, symbolic_dims: tuple[int, ...]) -> None:
    for tensor in args:
        for dim in symbolic_dims:
            torch._dynamo.mark_dynamic(tensor, dim)


def prepare_profile_dir(profile_dir: Path) -> None:
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
    profile_dir.mkdir(parents=True)


def serialize_best_tiling_configs(records: list[dict[str, object]]) -> str:
    if not records:
        return ""
    return json.dumps(records, sort_keys=True, separators=(",", ":"))


def profile_npu_case(
    case_fn: CaseFn,
    args: TensorArgs,
    profile_dir: Path,
    warmup: int,
    active: int,
    repeat: int,
) -> TimingResult:
    prepare_profile_dir(profile_dir)
    profiler = TorchNpuProfiler(
        profile_dir,
        wait=0,
        warmup=warmup,
        active=active,
        repeat=repeat,
        with_stack=False,
    )
    profiler.run_steps(lambda: case_fn(*args))
    summaries = ProfileResultParser(profile_dir).kernel_time_by_name()
    call_count = active * repeat
    total_us = sum(summary.total_us for summary in summaries)
    kernel_count = sum(summary.count for summary in summaries)
    if kernel_count == 0:
        raise RuntimeError(f"No NPU device kernels found in {profile_dir}")
    return TimingResult(
        mean_us=total_us / call_count,
        sample_count=call_count,
        kernel_count=kernel_count,
        kernels=tuple(summary.key for summary in summaries),
    )


def profile_cuda_case(
    case_fn: CaseFn,
    args: TensorArgs,
    profile_dir: Path,
    warmup: int,
    active: int,
    repeat: int,
) -> TimingResult:
    prepare_profile_dir(profile_dir)
    profiler = TorchCudaProfiler(
        profile_dir,
        wait=0,
        warmup=warmup,
        active=active,
        repeat=repeat,
        with_stack=False,
    )
    profiler.run_steps(lambda: case_fn(*args))
    records = [
        record
        for trace_path in profiler.trace_paths
        for record in CudaProfileParser(trace_path).kernel_records()
    ]
    call_count = active * repeat
    if not records:
        raise RuntimeError(f"No CUDA device kernels found in {profile_dir}")
    return TimingResult(
        mean_us=sum(record.duration for record in records) / call_count,
        sample_count=call_count,
        kernel_count=len(records),
        kernels=tuple(sorted({cuda_kernel_label(record.kernel_name) for record in records})),
    )


def profile_case(
    case_fn: CaseFn,
    args: TensorArgs,
    device: str,
    profile_dir: Path,
    warmup: int,
    active: int,
    repeat: int,
) -> TimingResult:
    if device == "npu":
        return profile_npu_case(
            case_fn, args, profile_dir, warmup, active, repeat
        )
    return profile_cuda_case(case_fn, args, profile_dir, warmup, active, repeat)


def result_dir(
    run_root: Path,
    device: str,
    case_name: str,
    runtime_shape: tuple[int, ...],
    compile_shape: tuple[int, ...],
    symbolic_dims: tuple[int, ...],
    execution: str,
) -> Path:
    root = run_root / "profiles" / device / execution
    if execution in DYNAMIC_EXECUTIONS:
        dims_label = "-".join(str(dim) for dim in symbolic_dims)
        root = root / f"compile_{shape_label(compile_shape)}" / f"dims_{dims_label}"
    return root / case_name / f"runtime_{shape_label(runtime_shape)}"


def summary_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[column] for column in SUMMARY_KEY_COLUMNS)


def merge_summary_rows(summary_path: Path, new_rows: list[dict[str, str]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = summary_path.with_suffix(f"{summary_path.suffix}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        rows_by_key: dict[tuple[str, ...], dict[str, str]] = {}
        if summary_path.exists():
            with summary_path.open(newline="", encoding="utf-8") as summary_file:
                reader = csv.DictReader(summary_file)
                if tuple(reader.fieldnames or ()) != RESULT_COLUMNS:
                    raise ValueError(
                        f"Existing summary schema does not match current script: {summary_path}"
                    )
                for row in reader:
                    rows_by_key[summary_key(row)] = row
        for row in new_rows:
            rows_by_key[summary_key(row)] = row

        temporary_path = summary_path.with_name(
            f".{summary_path.name}.tmp.{os.getpid()}"
        )
        try:
            with temporary_path.open("w", newline="", encoding="utf-8") as summary_file:
                writer = csv.DictWriter(summary_file, fieldnames=RESULT_COLUMNS)
                writer.writeheader()
                writer.writerows(rows_by_key.values())
            os.replace(temporary_path, summary_path)
        finally:
            temporary_path.unlink(missing_ok=True)


def compile_static(case: Case, args: TensorArgs) -> CaseFn:
    torch._dynamo.reset()
    compiled = torch.compile(case.fn, backend="inductor", dynamic=False)
    compiled(*args)
    return compiled


def compile_dynamic(
    case: Case,
    args: TensorArgs,
    symbolic_dims: tuple[int, ...],
) -> CaseFn:
    torch._dynamo.reset()
    mark_symbolic(args, symbolic_dims)
    compiled = torch.compile(case.fn, backend="inductor", dynamic=None)
    compiled(*args)
    return compiled


def initialize_device(device: str) -> None:
    if device == "npu":
        import torch_npu  # noqa: F401
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda requested, but CUDA is not available")
    else:
        raise ValueError("DEVICE must be 'npu' or 'cuda'")


def main() -> None:
    device = os.environ.get("DEVICE", "npu").strip().lower()
    selected_execution = os.environ.get("EXECUTION", "static").strip().lower()
    if selected_execution not in ("eager", "static", "dynamic", "custom", "group"):
        raise ValueError(
            "EXECUTION must be 'eager', 'static', 'dynamic', 'custom', or 'group'"
        )
    if selected_execution in ("custom", "group") and device != "npu":
        raise ValueError(f"EXECUTION={selected_execution} is only supported with DEVICE=npu")
    os.environ[GROUP_AUTOTUNE_ENV] = "1" if selected_execution == "group" else "0"

    record_results = env_bool("RECORD_RESULTS", True)
    run_id = (
        env_run_id()
        if record_results
        else os.environ.get("RUN_ID", "temporary").strip() or "temporary"
    )
    initialize_device(device)
    tiling_recorder = BestTilingRecorder(device)
    if selected_execution != "eager":
        tiling_recorder.install()
    dtype = env_dtype("DTYPE", torch.float32)
    runtime_shapes = parse_shapes()
    if selected_execution in DYNAMIC_EXECUTIONS:
        compile_shape = parse_shape(os.environ.get("COMPILE_SHAPE", "8192"))
        symbolic_dims = parse_symbolic_dims(len(compile_shape))
        validate_symbolic_shapes(compile_shape, runtime_shapes, symbolic_dims)
    else:
        compile_shape = runtime_shapes[0]
        symbolic_dims = ()

    warmup = env_int("WARMUP", 5)
    active = env_int("ACTIVE", 20)
    repeat = env_int("REPEAT", 1)
    if warmup < 0 or active <= 0 or repeat <= 0:
        raise ValueError("WARMUP must be non-negative; ACTIVE and REPEAT must be positive")

    profile_root = Path(
        os.environ.get("PROFILE_ROOT", "prof_log/elementwise_dynamic_perf")
    )
    run_root = profile_root / run_id
    summary_path = run_root / "summary.csv"
    temporary_profile_root = None
    if record_results:
        result_root = run_root
    else:
        temporary_profile_root = tempfile.TemporaryDirectory(
            prefix="elementwise_dynamic_perf_"
        )
        result_root = Path(temporary_profile_root.name)
    names = selected_case_names()

    print("elementwise_op_cost_cases")
    print(
        f"run_id={run_id} device={device} execution={selected_execution} dtype={dtype}"
    )
    if selected_execution in DYNAMIC_EXECUTIONS:
        print(f"compile_shape={compile_shape} symbolic_dims={symbolic_dims}")
        print(f"{GROUP_AUTOTUNE_ENV}={os.environ[GROUP_AUTOTUNE_ENV]}")
    elif selected_execution == "static":
        print("compile_shape=per-runtime-shape")
    else:
        print("compile_shape=not-applicable")
    print(f"runtime_shapes={runtime_shapes}")
    print(f"warmup={warmup} active={active} repeat={repeat}")
    print(f"record_results={record_results}")
    print(f"profile_root={profile_root}")
    print(f"summary_csv={summary_path if record_results else 'disabled'}")
    print(f"cases={','.join(names)}")
    print()

    writer = csv.DictWriter(sys.stdout, fieldnames=RESULT_COLUMNS)
    writer.writeheader()

    for name in names:
        case = CASES[name]
        dynamic_compiled = None
        dynamic_tiling_records: list[dict[str, object]] = []
        if selected_execution in DYNAMIC_EXECUTIONS:
            compile_args = make_inputs(
                compile_shape, device, dtype, case.input_kind
            )
            tiling_recorder.start_capture()
            try:
                dynamic_compiled = compile_dynamic(
                    case, compile_args, symbolic_dims
                )
            finally:
                dynamic_tiling_records = tiling_recorder.stop_capture()
            sync(device)

        for runtime_index, runtime_shape in enumerate(runtime_shapes):
            args = make_inputs(runtime_shape, device, dtype, case.input_kind)
            if selected_execution == "static":
                tiling_recorder.start_capture()
                try:
                    compiled = compile_static(case, args)
                finally:
                    tiling_records = tiling_recorder.stop_capture()
                row_compile_shape = runtime_shape
            elif selected_execution in DYNAMIC_EXECUTIONS:
                compiled = dynamic_compiled
                tiling_records = dynamic_tiling_records if runtime_index == 0 else []
                row_compile_shape = compile_shape
            else:
                compiled = None
                tiling_records = []
                row_compile_shape = runtime_shape
            if selected_execution != "eager" and compiled is None:
                raise AssertionError("compiled function was not initialized")
            sync(device)

            execution_fn = case.fn if compiled is None else compiled
            output_dir = result_dir(
                result_root,
                device,
                name,
                runtime_shape,
                row_compile_shape,
                symbolic_dims,
                selected_execution,
            )
            timing = profile_case(
                execution_fn,
                args,
                device,
                output_dir,
                warmup,
                active,
                repeat,
            )
            tiling_configs = serialize_best_tiling_configs(tiling_records)
            result_row = {
                "run_id": run_id,
                "device": device,
                "execution": selected_execution,
                "priority": case.priority,
                "bound": case.bound,
                "case": name,
                "group": case.group,
                "input_kind": case.input_kind,
                "dtype": str(dtype),
                "compile_shape": shape_label(row_compile_shape)
                if selected_execution != "eager"
                else "",
                "runtime_shape": shape_label(runtime_shape),
                "symbolic_dims": "|".join(str(dim) for dim in symbolic_dims)
                if selected_execution in DYNAMIC_EXECUTIONS
                else "",
                "scalar_ops": "|".join(case.scalar_ops),
                "samples": str(timing.sample_count),
                "device_kernel_count": str(timing.kernel_count),
                "device_call_mean_us": f"{timing.mean_us:.6f}",
                "kernels": "|".join(timing.kernels),
                "autotune_tiling_configs": tiling_configs,
                "result_dir": str(output_dir) if record_results else "",
            }
            writer.writerow(result_row)
            if record_results:
                merge_summary_rows(summary_path, [result_row])

    if temporary_profile_root is not None:
        temporary_profile_root.cleanup()


if __name__ == "__main__":
    main()
