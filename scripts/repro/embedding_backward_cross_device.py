#!/usr/bin/env python3
"""Run the AOT embedding-backward fragment in eager and Inductor modes.

The input shapes and ATen operation sequence are kept from the supplied
``eager_forward`` graph. The script has no Triton dependency and supports only
CUDA and NPU devices.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path


BATCH = 4096
SLOTS = 38
SOURCE_WIDTH = 2280
EMBEDDING_DIM = 128
SLICE_START = 1064
NUM_EMBEDDINGS = 98166


def eager_forward(where_4, ge_4, getitem_46, getitem_58, getitem_106, index_put_1):
    """The supplied eager graph, with device selected from its tensor inputs."""

    full_default_18 = torch.ops.aten.full.default(
        [], 0.0, dtype=torch.float32, layout=torch.strided, device=getitem_46.device, pin_memory=False
    )
    view_205 = torch.ops.aten.reshape.default(getitem_46, [BATCH, 1, SOURCE_WIDTH])
    slice_34 = torch.ops.aten.slice.Tensor(view_205, 2, SLICE_START, SLICE_START + EMBEDDING_DIM)
    add_1 = torch.ops.aten.add.Tensor(slice_34, getitem_58)
    add_21 = torch.ops.aten.add.Tensor(add_1, getitem_106)
    squeeze_5 = torch.ops.aten.squeeze.dim(add_21, 1)
    unsqueeze_22 = torch.ops.aten.unsqueeze.default(squeeze_5, 1)
    expand_61 = torch.ops.aten.expand.default(unsqueeze_22, [BATCH, SLOTS, EMBEDDING_DIM])
    unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(ge_4, 2)
    full_default_92 = torch.ops.aten.full.default([], 0, dtype=torch.float32, device=getitem_46.device)
    where_self_1 = torch.ops.aten.where.self(unsqueeze_default_1, expand_61, full_default_92)
    eq_7 = torch.ops.aten.eq.Scalar(where_4, -1)
    unsqueeze_23 = torch.ops.aten.unsqueeze.default(eq_7, -1)
    where_47 = torch.ops.aten.where.self(unsqueeze_23, full_default_18, where_self_1)
    full_default_49 = torch.ops.aten.full.default(
        [NUM_EMBEDDINGS, EMBEDDING_DIM], 0, dtype=torch.float32,
        layout=torch.strided, device=getitem_46.device, pin_memory=False
    )
    return torch.ops.aten.index_put_.default(full_default_49, [where_4], where_47, True)


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
        help="profile output root (default: prof_log/embedding_backward_cross_device/<timestamp>)",
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
    # The selected physical device is remapped to runtime index 0.
    device = torch.device(f"{device_name}:0")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize(device)


def make_inputs(device: torch.device):
    torch.manual_seed(0)
    where_4 = torch.randint(-1, NUM_EMBEDDINGS, (BATCH, SLOTS), device=device, dtype=torch.int64)
    ge_4 = torch.rand((BATCH, SLOTS), device=device) >= 0.25
    getitem_46 = torch.randn((BATCH, SOURCE_WIDTH), device=device, dtype=torch.float32)
    getitem_58 = torch.randn((BATCH, 1, EMBEDDING_DIM), device=device, dtype=torch.float32)
    getitem_106 = torch.randn((BATCH, 1, EMBEDDING_DIM), device=device, dtype=torch.float32)
    index_put_1 = torch.empty((NUM_EMBEDDINGS, EMBEDDING_DIM), device=device, dtype=torch.float32)
    return where_4, ge_4, getitem_46, getitem_58, getitem_106, index_put_1


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
    compiled_forward = torch.compile(eager_forward, backend="inductor")

    compiled_forward(*inputs)
    synchronize(device)

    profile_root = args.profile_dir or (
        Path("prof_log/embedding_backward_cross_device")
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    profile_root = profile_root.resolve()
    eager_us = device_time_us(eager_forward, inputs, device, args.warmup, args.repeat, profile_root / "eager")
    inductor_us = device_time_us(compiled_forward, inputs, device, args.warmup, args.repeat, profile_root / "inductor")
    print(f"device={device} device_id={args.device_id} shape=({BATCH}, {SLOTS}, {EMBEDDING_DIM})")
    print(f"warmup={args.warmup} repeat={args.repeat}")
    print(f"eager_device_us={eager_us:.3f}")
    print(f"inductor_device_us={inductor_us:.3f}")
    print(f"profile_dir={profile_root}")


if __name__ == "__main__":
    main()
