#!/usr/bin/env python3
"""Run all 26 extracted DLRM embedding-dense-backward full cases.

The 0914 NPU Inductor graph contains one zero-initialization Triton kernel for
each embedding table. This script keeps the corresponding ATen ``full`` call,
with the exact ``[rows, 16]`` shape, and compares Eager with Inductor on CUDA
or NPU. It does not import or embed generated Triton source.

Use ``--case all`` (the default) to run every table, or ``--case N`` for one
zero-based case. Generated artifacts can be inspected with
``--compile-debug``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path


EMBEDDING_DIM = 16

# Mapping extracted from model__0_backward_3.1/output_code.py. The suffix is
# the generated kernel name in the supplied 0914 NPU artifact; rows is its
# full output's first dimension.
FULL_CASES = (
    ("triton_poi_fused_embedding_dense_backward_1", 3),
    ("triton_poi_fused_embedding_dense_backward_2", 4),
    ("triton_poi_fused_embedding_dense_backward_3", 10),
    ("triton_poi_fused_embedding_dense_backward_4", 15),
    ("triton_poi_fused_embedding_dense_backward_5", 18),
    ("triton_poi_fused_embedding_dense_backward_6", 24),
    ("triton_poi_fused_embedding_dense_backward_7", 27),
    ("triton_poi_fused_embedding_dense_backward_8", 105),
    ("triton_poi_fused_embedding_dense_backward_9", 305),
    ("triton_poi_fused_embedding_dense_backward_10", 583),
    ("triton_poi_fused_embedding_dense_backward_11", 633),
    ("triton_poi_fused_embedding_dense_backward_12", 1460),
    ("triton_poi_fused_embedding_dense_backward_14", 2173),
    ("triton_poi_fused_embedding_dense_backward_15", 3194),
    ("triton_poi_fused_embedding_dense_backward_17", 5652),
    ("triton_poi_fused_embedding_dense_backward_18", 5683),
    ("triton_poi_fused_embedding_dense_backward_39", 12517),
    ("triton_poi_fused_embedding_dense_backward_41", 14992),
    ("triton_poi_fused_embedding_dense_backward_43", 93145),
    ("triton_poi_fused_embedding_dense_backward_45", 142572),
    ("triton_poi_fused_embedding_dense_backward_47", 286181),
    ("triton_poi_fused_embedding_dense_backward_49", 2202608),
    ("triton_poi_fused_embedding_dense_backward_51", 5461306),
    ("triton_poi_fused_embedding_dense_backward_53", 7046547),
    ("triton_poi_fused_embedding_dense_backward_55", 8351593),
    ("triton_poi_fused_embedding_dense_backward_57", 10131227),
)


def make_eager_forward(rows: int):
    def eager_forward(device: torch.device):
        return torch.ops.aten.full.default(
            [rows, EMBEDDING_DIM],
            0,
            dtype=torch.float32,
            layout=torch.strided,
            device=device,
            pin_memory=False,
        )

    return eager_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "npu"), default="npu")
    parser.add_argument("--device-id", type=int, default=0, help="physical device id exposed to the process")
    parser.add_argument(
        "--case",
        default="all",
        help="zero-based case index or exact generated kernel name (default: all)",
    )
    parser.add_argument("--warmup", type=int, default=int(os.getenv("WARMUP", "10")))
    parser.add_argument("--repeat", type=int, default=int(os.getenv("REPEAT", "50")))
    parser.add_argument("--compile-debug", action="store_true", help="enable TORCH_COMPILE_DEBUG")
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="profile output root (default: prof_log/dlrm_embedding_dense_full_26/<timestamp>)",
    )
    return parser.parse_args()


def setup_device(device_name: str, device_id: int):
    global torch
    if device_id < 0:
        raise ValueError("--device-id must be non-negative")
    visible_devices = "ASCEND_RT_VISIBLE_DEVICES" if device_name == "npu" else "CUDA_VISIBLE_DEVICES"
    os.environ[visible_devices] = str(device_id)
    import torch

    if device_name == "npu":
        try:
            import torch_npu
            from torch_npu.utils._dynamo import register_inductor_npu
        except ImportError as error:
            raise RuntimeError("--device npu requires torch_npu") from error
        register_inductor_npu()
        torch.npu.set_device(0)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.set_device(0)
    return torch.device(f"{device_name}:0")


def synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize(device)


def warmup(function, device: torch.device, count: int) -> None:
    for _ in range(count):
        function(device)
    synchronize(device)


def cuda_device_time_us(function, device: torch.device, repeat: int, profile_dir: Path) -> float:
    profile_dir.mkdir(parents=True, exist_ok=True)
    trace_path = profile_dir / "trace.json"
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=repeat, repeat=1),
    ) as profiler:
        for _ in range(repeat):
            result = function(device)
            del result
            torch.cuda.synchronize(device)
            profiler.step()
    profiler.export_chrome_trace(str(trace_path))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    total_us = sum(
        event["dur"]
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "kernel"
        and isinstance(event.get("dur"), (int, float))
    )
    if total_us == 0:
        raise RuntimeError("CUDA profiler did not record device kernel events")
    return total_us / repeat


def npu_device_time_us(function, device: torch.device, repeat: int, profile_dir: Path) -> float:
    import torch_npu

    profile_dir.mkdir(parents=True, exist_ok=True)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=repeat, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        ),
    ) as profiler:
        for _ in range(repeat):
            result = function(device)
            del result
            torch.npu.synchronize()
            profiler.step()

    total_us = 0.0
    for csv_path in profile_dir.rglob("kernel_details.csv"):
        with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
            for row in csv.DictReader(csv_file):
                duration = row.get("Duration(us)") or row.get("Task Duration(us)")
                if duration:
                    total_us += float(duration)
    if total_us == 0:
        raise RuntimeError("NPU profiler did not record device kernel events")
    return total_us / repeat


def device_time_us(function, device: torch.device, warmup_count: int, repeat: int, profile_dir: Path) -> float:
    warmup(function, device, warmup_count)
    if device.type == "cuda":
        return cuda_device_time_us(function, device, repeat, profile_dir)
    return npu_device_time_us(function, device, repeat, profile_dir)


def select_cases(case_arg: str):
    if case_arg == "all":
        return list(enumerate(FULL_CASES))
    if case_arg.isdigit():
        index = int(case_arg)
        if not 0 <= index < len(FULL_CASES):
            raise ValueError(f"--case index must be in [0, {len(FULL_CASES) - 1}]")
        return [(index, FULL_CASES[index])]
    for index, case in enumerate(FULL_CASES):
        if case[0] == case_arg:
            return [(index, case)]
    raise ValueError(f"unknown --case {case_arg!r}")


def run_case(index, kernel_name, rows, device, args, profile_root):
    eager_forward = make_eager_forward(rows)
    compiled_forward = torch.compile(eager_forward, backend="inductor")

    # Compile before profiling and verify the same all-zero result.
    compiled_result = compiled_forward(device)
    synchronize(device)
    eager_result = eager_forward(device)
    synchronize(device)
    if eager_result.shape != (rows, EMBEDDING_DIM) or not torch.equal(eager_result, compiled_result):
        raise AssertionError(f"case {index} ({kernel_name}) eager/Inductor outputs differ")
    del eager_result, compiled_result

    case_root = profile_root / f"case_{index:02d}_{rows}"
    eager_us = device_time_us(
        eager_forward, device, args.warmup, args.repeat, case_root / "eager"
    )
    inductor_us = device_time_us(
        compiled_forward, device, args.warmup, args.repeat, case_root / "inductor"
    )
    improvement = 1.0 - inductor_us / eager_us
    print(
        f"case={index:02d} kernel={kernel_name} shape=({rows}, {EMBEDDING_DIM}) "
        f"eager_device_us={eager_us:.3f} inductor_device_us={inductor_us:.3f} "
        f"improvement={improvement:.2%}"
    )
    return eager_us, inductor_us


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("warmup must be non-negative and repeat must be positive")
    if args.compile_debug:
        os.environ["TORCH_COMPILE_DEBUG"] = "1"

    selected_cases = select_cases(args.case)
    device = setup_device(args.device, args.device_id)
    profile_root = args.profile_dir or (
        Path("prof_log/dlrm_embedding_dense_full_26") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    profile_root = profile_root.resolve()
    print(
        f"device={device} cases={len(selected_cases)} warmup={args.warmup} "
        f"repeat={args.repeat} profile_dir={profile_root}"
    )

    eager_total = 0.0
    inductor_total = 0.0
    for index, (kernel_name, rows) in selected_cases:
        eager_us, inductor_us = run_case(index, kernel_name, rows, device, args, profile_root)
        eager_total += eager_us
        inductor_total += inductor_us

    print(f"cases_total={len(selected_cases)}")
    print(f"eager_total_us={eager_total:.3f}")
    print(f"inductor_total_us={inductor_total:.3f}")
    print(f"improvement={1.0 - inductor_total / eager_total:.2%}")


if __name__ == "__main__":
    main()
