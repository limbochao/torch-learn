#!/usr/bin/env python3
"""Compare the DLRM embedding-dense-backward full and atomic phases.

The default case is table 2 from the 0914 DLRM graph. It keeps the original
``[10131227, 16]`` output and the exact ATen fragment surrounding the fused
atomic scatter. No generated Triton source is embedded in this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path


BATCH = 128
NUM_SLOTS = 27
EMBEDDING_DIM = 16
TABLE_ROWS = (
    1460,
    583,
    10131227,
    2202608,
    305,
    24,
    12517,
    633,
    3,
    93145,
    5683,
    8351593,
    3194,
    27,
    14992,
    5461306,
    10,
    5652,
    2173,
    4,
    7046547,
    18,
    15,
    286181,
    105,
    142572,
)


def make_full_forward(num_embeddings: int):
    def full_forward(anchor):
        return torch.ops.aten.full.default(
            [num_embeddings, EMBEDDING_DIM],
            0,
            dtype=torch.float32,
            layout=torch.strided,
            device=anchor.device,
            pin_memory=False,
        )

    return full_forward


def make_atomic_forward(table_index: int):
    slice_start = (table_index + 1) * EMBEDDING_DIM

    def atomic_forward(indices, bmm_2, bmm_1, output):
        zero = torch.ops.aten.full.default(
            [], 0.0, dtype=torch.float32, layout=torch.strided,
            device=output.device, pin_memory=False
        )
        permute = torch.ops.aten.permute.default(bmm_1, [0, 2, 1])
        add = torch.ops.aten.add.Tensor(bmm_2, permute)
        view = torch.ops.aten.reshape.default(add, [BATCH, NUM_SLOTS * EMBEDDING_DIM])
        values = torch.ops.aten.slice.Tensor(view, 1, slice_start, slice_start + EMBEDDING_DIM)
        is_padding = torch.ops.aten.eq.Scalar(indices, -1)
        is_padding = torch.ops.aten.unsqueeze.default(is_padding, -1)
        values = torch.ops.aten.where.self(is_padding, zero, values)
        return torch.ops.aten.index_put_.default(output, [indices], values, True)

    return atomic_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "npu"), default="npu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--table-index", type=int, default=2)
    parser.add_argument("--phase", choices=("full", "atomic", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=int(os.getenv("WARMUP", "10")))
    parser.add_argument("--repeat", type=int, default=int(os.getenv("REPEAT", "50")))
    parser.add_argument("--compile-debug", action="store_true", help="enable TORCH_COMPILE_DEBUG")
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="profile output root (default: prof_log/dlrm_embedding_dense_phases/<timestamp>)",
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


def make_atomic_inputs(device: torch.device, num_embeddings: int):
    torch.manual_seed(0)
    # Duplicate rows exercise accumulate=True; -1 entries reproduce DLRM padding.
    indices = ((torch.arange(BATCH, dtype=torch.int64) // 2) * 7919) % num_embeddings
    indices[::17] = -1
    indices = indices.to(device)
    bmm_2 = torch.randn(
        (BATCH, NUM_SLOTS, EMBEDDING_DIM), dtype=torch.float32, device=device
    )
    bmm_1 = torch.randn(
        (BATCH, EMBEDDING_DIM, NUM_SLOTS), dtype=torch.float32, device=device
    )
    output = torch.zeros((num_embeddings, EMBEDDING_DIM), dtype=torch.float32, device=device)
    return indices, bmm_2, bmm_1, output


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


def device_time_us(function, inputs, device: torch.device, warmup_count: int, repeat: int, profile_dir: Path):
    warmup(function, inputs, device, warmup_count)
    if device.type == "cuda":
        return cuda_device_time_us(function, inputs, device, repeat, profile_dir)
    return npu_device_time_us(function, inputs, repeat, profile_dir)


def validate_full(eager_forward, compiled_forward, anchor, device: torch.device) -> None:
    eager_output = eager_forward(anchor)
    compiled_output = compiled_forward(anchor)
    synchronize(device)
    if eager_output.shape != compiled_output.shape or not torch.equal(eager_output, compiled_output):
        raise AssertionError("full eager and Inductor outputs differ")


def validate_atomic(eager_forward, compiled_forward, device: torch.device, num_embeddings: int) -> None:
    eager_inputs = make_atomic_inputs(device, num_embeddings)
    compiled_inputs = (*eager_inputs[:-1], torch.zeros_like(eager_inputs[-1]))
    eager_output = eager_forward(*eager_inputs)
    compiled_output = compiled_forward(*compiled_inputs)
    synchronize(device)
    valid_rows = torch.unique(eager_inputs[0][eager_inputs[0] >= 0])
    torch.testing.assert_close(eager_output[valid_rows], compiled_output[valid_rows], rtol=1e-5, atol=1e-5)


def run_phase(name, eager_forward, eager_inputs, compiled_inputs, device, args, profile_root):
    compiled_forward = torch.compile(eager_forward, backend="inductor")
    compiled_forward(*compiled_inputs)
    synchronize(device)
    if name == "full":
        validate_full(eager_forward, compiled_forward, eager_inputs[0], device)
    else:
        validate_atomic(eager_forward, compiled_forward, device, TABLE_ROWS[args.table_index])

    eager_us = device_time_us(
        eager_forward, eager_inputs, device, args.warmup, args.repeat, profile_root / name / "eager"
    )
    inductor_us = device_time_us(
        compiled_forward, compiled_inputs, device, args.warmup, args.repeat,
        profile_root / name / "inductor"
    )
    speedup = 1.0 - inductor_us / eager_us
    print(f"phase={name} eager_device_us={eager_us:.3f} inductor_device_us={inductor_us:.3f} ")
    print(f"phase={name} improvement={speedup:.2%}")


def main() -> None:
    args = parse_args()
    if not 0 <= args.table_index < len(TABLE_ROWS):
        raise ValueError(f"--table-index must be in [0, {len(TABLE_ROWS) - 1}]")
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("warmup must be non-negative and repeat must be positive")
    if args.compile_debug:
        os.environ["TORCH_COMPILE_DEBUG"] = "1"

    device = setup_device(args.device, args.device_id)
    num_embeddings = TABLE_ROWS[args.table_index]
    profile_root = args.profile_dir or (
        Path("prof_log/dlrm_embedding_dense_phases") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    profile_root = profile_root.resolve()
    memory_mib = num_embeddings * EMBEDDING_DIM * 4 / 1024**2
    print(
        f"device={device} table_index={args.table_index} output_shape=({num_embeddings}, {EMBEDDING_DIM}) "
        f"output_mib={memory_mib:.2f} warmup={args.warmup} repeat={args.repeat}"
    )

    if args.phase in ("full", "both"):
        full_forward = make_full_forward(num_embeddings)
        anchor = torch.empty((), dtype=torch.float32, device=device)
        run_phase("full", full_forward, (anchor,), (anchor,), device, args, profile_root)

    if args.phase in ("atomic", "both"):
        atomic_forward = make_atomic_forward(args.table_index)
        eager_inputs = make_atomic_inputs(device, num_embeddings)
        compiled_inputs = (*eager_inputs[:-1], torch.zeros_like(eager_inputs[-1]))
        run_phase("atomic", atomic_forward, eager_inputs, compiled_inputs, device, args, profile_root)

    print(f"profile_dir={profile_root}")


if __name__ == "__main__":
    main()
