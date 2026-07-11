"""
Microbench cases for elementwise aten op cost classes.

This script intentionally avoids broadcast/view/reindex cases. Each case focuses on
one class of elementwise aten ops so the generated pointwise kernel can be compared
by the scalar work inside each output element.

The timing result is parsed from torch_npu profiler `kernel_details.csv`, so the
reported numbers are device-side Triton kernel duration in microseconds.

Run on NPU after sourcing the target environment:

    python scripts/tests/pointwise_op_cost_cases.py

Useful environment variables:

    DEVICE=npu DYNAMIC=0 CHECK=1 PROFILE_ROOT=prof_log/pointwise_op_cost_cases
    PROF_WAIT=1 PROF_WARMUP=1 PROF_ACTIVE=5 PROF_REPEAT=1
    SHAPE=1048576 DTYPE=float32 CASES=arithmetic_light,exp_log
    TORCHINDUCTOR_CACHE_DIR=/path/to/inductor_cache
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from tools.npu_profiler import ProfileResultParser, TorchNpuProfiler


TensorArgs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
CaseFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Case:
    fn: CaseFn
    group: str
    aten_ops: tuple[str, ...]
    input_desc: str
    output_desc: str
    input_kind: str = "float"


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


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
        expected = sorted(mapping)
        raise ValueError(f"Unsupported DTYPE={value!r}, expected one of {expected}")
    return mapping[key]


def parse_shape() -> tuple[int, ...]:
    value = os.environ.get("SHAPE", "1048576")
    return tuple(
        int(x.strip()) for x in value.replace("x", ",").split(",") if x.strip()
    )


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "npu":
        torch.npu.synchronize()


def make_float_inputs(shape: tuple[int, ...], device: str, dtype: torch.dtype) -> TensorArgs:
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


def make_bool_inputs(shape: tuple[int, ...], device: str) -> TensorArgs:
    torch.manual_seed(0)
    x = torch.rand(shape, device=device) > 0.2
    y = torch.rand(shape, device=device) > 0.4
    z = torch.rand(shape, device=device) > 0.6
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
    if input_kind == "bool":
        return make_bool_inputs(shape, device)
    raise ValueError(f"Unsupported input_kind={input_kind!r}")


def arithmetic_light(x, y, z):
    return ((x + y) * 1.125 - z * 0.25) + x * y


def minmax_clamp(x, y, z):
    return torch.clamp_max(torch.maximum(x, y), 2.0) + torch.clamp_min(
        torch.minimum(y, z), 1.0
    )


def compare_ops(x, y, z):
    return (x < y) | (y <= z) | (x != z)


def logical_ops(x, y, z):
    return torch.logical_xor(
        torch.logical_and(x, y),
        torch.logical_or(torch.logical_not(y), z),
    )


def bitwise_ops(x, y, z):
    return ((x & y) ^ (z | 17)) + ((x << 1) & 255) - ((y >> 1) & 127)


def division_ops(x, y, z):
    return x / (y + 0.25) + torch.div(z, x + 0.5)


def root_reciprocal_ops(x, y, z):
    return torch.sqrt(x * y + 0.01) + torch.rsqrt(z + 1.0) + torch.reciprocal(y + 0.5)


def pow_ops(x, y, z):
    return torch.pow(x + 0.1, 2.0) + torch.pow(y + 0.1, z * 0.1)


def exp_log_ops(x, y, z):
    return (
        torch.exp(x * 0.01)
        + torch.expm1(y * 0.01)
        + torch.log1p(z)
        + torch.log2(y + 1.0)
    )


def trig_hyperbolic_ops(x, y, z):
    return torch.sin(x) * torch.cos(y) + torch.tanh(z) + torch.atan(x * 0.1)


def special_math_ops(x, y, z):
    positive = x * 0.1 + 1.0
    return torch.erf(x * 0.1) + torch.erfc(y * 0.1) + torch.lgamma(positive) + z * 0.01


def round_sign_ops(x, y, z):
    return torch.ceil(x) + torch.sign(y - 1.5) + torch.signbit(z - 1.5).to(x.dtype)


CASES: dict[str, Case] = {
    "arithmetic_light": Case(
        arithmetic_light,
        "light arithmetic",
        ("aten.add", "aten.sub", "aten.mul"),
        input_desc="x,y,z: same-shape float tensors; x in [0.5,1.5), y in [1,2), z in [1.5,2.5)",
        output_desc="out = ((x + y) * 1.125 - z * 0.25) + x * y",
    ),
    "minmax_clamp": Case(
        minmax_clamp,
        "min/max clamp",
        ("aten.maximum", "aten.minimum", "aten.clamp_max", "aten.clamp_min"),
        input_desc="x,y,z: same-shape float tensors; positive values around [0.5,2.5)",
        output_desc="out = clamp_max(maximum(x, y), 2.0) + clamp_min(minimum(y, z), 1.0)",
    ),
    "compare": Case(
        compare_ops,
        "comparison",
        ("aten.lt", "aten.le", "aten.ne"),
        input_desc="x,y,z: same-shape float tensors; positive values around [0.5,2.5)",
        output_desc="out = (x < y) | (y <= z) | (x != z)",
    ),
    "logical": Case(
        logical_ops,
        "logical bool",
        ("aten.logical_and", "aten.logical_or", "aten.logical_xor", "aten.logical_not"),
        input_desc="x,y,z: same-shape bool tensors generated from random thresholds",
        output_desc="out = logical_xor(logical_and(x, y), logical_or(logical_not(y), z))",
        input_kind="bool",
    ),
    "bitwise": Case(
        bitwise_ops,
        "integer bitwise",
        (
            "aten.bitwise_and",
            "aten.bitwise_or",
            "aten.bitwise_xor",
            "aten.bitwise_left_shift",
            "aten.bitwise_right_shift",
        ),
        input_desc="x,y,z: same-shape int32 tensors with values in [1,1024)",
        output_desc="out = ((x & y) ^ (z | 17)) + ((x << 1) & 255) - ((y >> 1) & 127)",
        input_kind="int",
    ),
    "division": Case(
        division_ops,
        "division",
        ("aten.div", "aten.div.Tensor"),
        input_desc="x,y,z: same-shape float tensors with positive denominators",
        output_desc="out = x / (y + 0.25) + div(z, x + 0.5)",
    ),
    "root_reciprocal": Case(
        root_reciprocal_ops,
        "sqrt rsqrt reciprocal",
        ("aten.sqrt", "aten.rsqrt", "aten.reciprocal"),
        input_desc="x,y,z: same-shape positive float tensors; expressions keep sqrt inputs positive",
        output_desc="out = sqrt(x * y + 0.01) + rsqrt(z + 1.0) + reciprocal(y + 0.5)",
    ),
    "pow": Case(
        pow_ops,
        "power",
        ("aten.pow",),
        input_desc="x,y,z: same-shape positive float tensors; one constant exponent and one tensor exponent",
        output_desc="out = pow(x + 0.1, 2.0) + pow(y + 0.1, z * 0.1)",
    ),
    "exp_log": Case(
        exp_log_ops,
        "exp/log",
        ("aten.exp", "aten.expm1", "aten.log1p", "aten.log2"),
        input_desc="x,y,z: same-shape positive float tensors; log inputs stay positive",
        output_desc="out = exp(x * 0.01) + expm1(y * 0.01) + log1p(z) + log2(y + 1.0)",
    ),
    "trig_hyperbolic": Case(
        trig_hyperbolic_ops,
        "trig/hyperbolic",
        ("aten.sin", "aten.cos", "aten.tanh", "aten.atan"),
        input_desc="x,y,z: same-shape float tensors; values around [0.5,2.5)",
        output_desc="out = sin(x) * cos(y) + tanh(z) + atan(x * 0.1)",
    ),
    "special_math": Case(
        special_math_ops,
        "special math",
        ("aten.erf", "aten.erfc", "aten.lgamma"),
        input_desc="x,y,z: same-shape positive float tensors; lgamma input is x * 0.1 + 1.0",
        output_desc="out = erf(x * 0.1) + erfc(y * 0.1) + lgamma(x * 0.1 + 1.0) + z * 0.01",
    ),
    "round_sign": Case(
        round_sign_ops,
        "round/sign",
        ("aten.ceil", "aten.sign", "aten.signbit"),
        input_desc="x,y,z: same-shape float tensors; sign/signbit use values shifted by 1.5",
        output_desc="out = ceil(x) + sign(y - 1.5) + signbit(z - 1.5).to(x.dtype)",
    ),
}


def output_spec(tensor: torch.Tensor) -> str:
    return f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"


def assert_same_output(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype == torch.bool or not actual.dtype.is_floating_point:
        if not torch.equal(actual, expected):
            raise AssertionError("compiled output differs from eager output")
        return
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def profile_compiled_case(
    case_name: str,
    compiled_fn: CaseFn,
    args: TensorArgs,
    profile_root: Path,
    wait: int,
    warmup: int,
    active: int,
    repeat: int,
) -> tuple[float, float, int, str]:
    case_profile_dir = profile_root / case_name
    if case_profile_dir.exists():
        shutil.rmtree(case_profile_dir)

    profiler = TorchNpuProfiler(
        case_profile_dir,
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat,
        with_stack=False,
    )
    profiler.run_steps(lambda: compiled_fn(*args))

    summaries = ProfileResultParser(case_profile_dir).kernel_time_by_name(
        name_prefix="triton"
    )
    total_us = sum(item.total_us for item in summaries)
    count = sum(item.count for item in summaries)
    mean_us = total_us / count if count else 0.0
    kernels = "|".join(item.key for item in summaries)
    return mean_us, total_us, count, kernels


def case_names() -> list[str]:
    selected = os.environ.get("CASES")
    if not selected:
        return list(CASES)
    names = [x.strip() for x in selected.split(",") if x.strip()]
    unknown = sorted(set(names) - set(CASES))
    if unknown:
        raise ValueError(f"Unknown CASES entries: {unknown}; valid cases: {sorted(CASES)}")
    return names


def main() -> None:
    device = os.environ.get("DEVICE", "npu")
    shape = parse_shape()
    dtype = env_dtype("DTYPE", torch.float32)
    dynamic = env_bool("DYNAMIC", False)
    check = env_bool("CHECK", True)
    profile_root = Path(os.environ.get("PROFILE_ROOT", "prof_log/pointwise_op_cost_cases"))
    prof_wait = env_int("PROF_WAIT", 1)
    prof_warmup = env_int("PROF_WARMUP", 1)
    prof_active = env_int("PROF_ACTIVE", 5)
    prof_repeat = env_int("PROF_REPEAT", 1)

    if device == "npu":
        import torch_npu  # noqa: F401
    else:
        raise RuntimeError("This script uses torch_npu profiler and expects DEVICE=npu.")

    names = case_names()

    print("pointwise_op_cost_cases")
    print(f"device={device} dtype={dtype} shape={shape} dynamic={int(dynamic)}")
    print(
        f"prof_wait={prof_wait} prof_warmup={prof_warmup} "
        f"prof_active={prof_active} prof_repeat={prof_repeat}"
    )
    print(f"profile_root={profile_root}")
    print(f"check={int(check)}")
    print(f"cases={','.join(names)}")
    print()
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "case",
            "group",
            "input_kind",
            "input_desc",
            "output_desc",
            "actual_output",
            "aten_ops",
            "triton_kernel_mean_us",
            "triton_kernel_total_us",
            "triton_kernel_count",
            "kernels",
        ]
    )

    for name in names:
        case = CASES[name]
        args = make_inputs(shape, device, dtype, case.input_kind)
        compiled = torch.compile(case.fn, backend="inductor", dynamic=dynamic)

        # Compile before profiling so profiler records steady-state kernel launches.
        compiled_out = compiled(*args)
        sync(device)

        if check:
            eager_out = case.fn(*args)
            sync(device)
            assert_same_output(compiled_out, eager_out)

        mean_us, total_us, count, kernels = profile_compiled_case(
            name,
            compiled,
            args,
            profile_root,
            prof_wait,
            prof_warmup,
            prof_active,
            prof_repeat,
        )
        aten_ops = "|".join(case.aten_ops)
        writer.writerow(
            [
                name,
                case.group,
                case.input_kind,
                case.input_desc,
                case.output_desc,
                output_spec(compiled_out),
                aten_ops,
                f"{mean_us:.6f}",
                f"{total_us:.6f}",
                count,
                kernels,
            ]
        )


if __name__ == "__main__":
    main()
