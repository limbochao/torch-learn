#!/usr/bin/env python3
"""Run one DLRM embedding-select graph fragment in eager and Inductor.

The ATen graph and input metadata come from this DLRM 0914 artifact:

    NPU_Inductor/torchinductor/model__0_forward_1.0/output_code.py

The script lets ``torch.compile`` generate the device kernel and has no direct
Triton dependency.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path


BATCH = 128
NUM_TABLES = 26
TABLE_ROWS = 1460
EMBEDDING_DIM = 16
TABLE_INDEX = 0


def eager_forward(all_indices, weight):
    """The full-network ATen graph fragment for one embedding lookup."""

    indices = torch.ops.aten.select.int(all_indices, 0, TABLE_INDEX)
    return torch.ops.aten.embedding.default(weight, indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "npu"), default="npu")
    parser.add_argument("--device-id", type=int, default=0, help="physical device id exposed to the process")
    parser.add_argument("--warmup", type=int, default=int(os.getenv("WARMUP", "10")))
    parser.add_argument("--repeat", type=int, default=int(os.getenv("REPEAT", "50")))
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="profile output root (default: prof_log/dlrm_embedding_select_perf/<timestamp>)",
    )
    return parser.parse_args()


def setup_device(device_name: str, device_id: int):
    global torch
    if device_id < 0:
        raise ValueError("--device-id must be non-negative")
    variable = "ASCEND_RT_VISIBLE_DEVICES" if device_name == "npu" else "CUDA_VISIBLE_DEVICES"
    os.environ[variable] = str(device_id)
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


def make_inputs(device: torch.device):
    torch.manual_seed(0)
    all_indices = torch.randint(
        TABLE_ROWS,
        (NUM_TABLES, BATCH),
        dtype=torch.int64,
        device=device,
    )
    weight = torch.randn(
        (TABLE_ROWS, EMBEDDING_DIM),
        dtype=torch.float32,
        device=device,
    )
    return all_indices, weight


def warmup(function, inputs, device: torch.device, count: int) -> None:
    for _ in range(count):
        function(*inputs)
    synchronize(device)


def cuda_device_time_us(function, inputs, device: torch.device, repeat: int, profile_dir: Path) -> float:
    profile_dir.mkdir(parents=True, exist_ok=True)
    trace_path = profile_dir / "trace.json"
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=repeat, repeat=1),
    ) as profiler:
        for _ in range(repeat):
            function(*inputs)
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


def npu_device_time_us(function, inputs, repeat: int, profile_dir: Path) -> float:
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
            function(*inputs)
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


def device_time_us(function, inputs, device: torch.device, warmup_count: int, repeat: int, profile_dir: Path) -> float:
    warmup(function, inputs, device, warmup_count)
    if device.type == "cuda":
        return cuda_device_time_us(function, inputs, device, repeat, profile_dir)
    return npu_device_time_us(function, inputs, repeat, profile_dir)


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("warmup must be non-negative and repeat must be positive")

    global torch
    device = setup_device(args.device, args.device_id)
    inputs = make_inputs(device)
    compiled_forward = torch.compile(
        eager_forward,
        backend="inductor",
        fullgraph=True,
        dynamic=False,
    )

    eager_output = eager_forward(*inputs)
    compiled_output = compiled_forward(*inputs)
    synchronize(device)
    torch.testing.assert_close(compiled_output, eager_output)

    profile_root = args.profile_dir or (
        Path("prof_log/dlrm_embedding_select_perf")
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    profile_root = profile_root.resolve()
    eager_us = device_time_us(
        eager_forward,
        inputs,
        device,
        args.warmup,
        args.repeat,
        profile_root / "eager",
    )
    inductor_us = device_time_us(
        compiled_forward,
        inputs,
        device,
        args.warmup,
        args.repeat,
        profile_root / "inductor",
    )
    improvement = 1.0 - inductor_us / eager_us

    print(f"device={device} device_id={args.device_id}")
    print(
        f"indices_shape=({NUM_TABLES}, {BATCH}) table_index={TABLE_INDEX} "
        f"weight_shape=({TABLE_ROWS}, {EMBEDDING_DIM})"
    )
    print(f"warmup={args.warmup} repeat={args.repeat} correctness=PASS")
    print(f"eager_device_us={eager_us:.3f}")
    print(f"inductor_device_us={inductor_us:.3f}")
    print(f"improvement={improvement:.4%}")
    print(f"profile_dir={profile_root}")


if __name__ == "__main__":
    main()
