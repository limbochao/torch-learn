#!/usr/bin/env python3
"""Reproduce the DLRM embedding_dense_backward performance case.

The default graph has 26 tables, ``bmm_2=[128,27,16]`` and
``bmm_1=[128,16,27]``.  It compares the original full-add eager granularity,
the split eager granularity, and Inductor's split ``add +
embedding_dense_backward`` path.  The fallback is intentionally kept as the
ATen operation; this repro measures whether the fused preprocessing around it
is slower.

Example on NPU::

    ASCEND_RT_VISIBLE_DEVICES=2 python scripts/repro/dlrm_embedding_dense_backward_perf.py
"""

from __future__ import annotations

import argparse
import os
import statistics
import time


BATCH = 128
SLOTS = 27
EMBEDDING_DIM = 16
DEFAULT_TABLES = 26
DEFAULT_ROWS = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("npu", "cuda"), default="npu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--tables", type=int, default=DEFAULT_TABLES)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=20)
    return parser.parse_args()


def prepare_device(device_name: str, device_id: int):
    os.environ["CUDA_VISIBLE_DEVICES" if device_name == "cuda" else "ASCEND_RT_VISIBLE_DEVICES"] = str(
        device_id
    )
    import torch

    if device_name == "npu":
        import torch_npu
        from torch_npu.utils._dynamo import register_inductor_npu

        register_inductor_npu()
        torch.npu.set_device(0)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.set_device(0)
    return torch, torch.device(f"{device_name}:0")


def synchronize(torch, device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize(device)


def fallback(grad, indices, rows):
    return torch.ops.aten.embedding_dense_backward.default(grad, indices, rows, -1, False)


def eager_full(bmm_2, bmm_1, indices, rows):
    full = (bmm_2 + bmm_1.permute(0, 2, 1)).reshape(BATCH, SLOTS * EMBEDDING_DIM)
    return tuple(
        fallback(full[:, i * EMBEDDING_DIM : (i + 1) * EMBEDDING_DIM], indices[i], rows[i])
        for i in range(len(rows))
    )


def eager_split(bmm_2, bmm_1, indices, rows):
    return tuple(
        fallback(bmm_2[:, i, :] + bmm_1[:, :, i], indices[i], rows[i])
        for i in range(len(rows))
    )


def timed(torch, device, function, inputs, warmup: int, repeat: int) -> list[float]:
    for _ in range(warmup):
        function(*inputs)
    synchronize(torch, device)
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(repeat):
            function(*inputs)
        synchronize(torch, device)
        samples.append((time.perf_counter() - start) * 1e6 / repeat)
    return samples


def check_outputs(torch, expected, actual) -> None:
    for expected_output, actual_output in zip(expected, actual):
        torch.testing.assert_close(expected_output, actual_output)


def main() -> None:
    global torch
    args = parse_args()
    if args.tables <= 0 or args.tables >= SLOTS or args.rows <= 0:
        raise ValueError(f"tables must be in [1, {SLOTS - 1}]")

    torch, device = prepare_device(args.device, args.device_id)
    torch.manual_seed(0)
    bmm_2 = torch.randn((BATCH, SLOTS, EMBEDDING_DIM), dtype=torch.float32, device=device)
    bmm_1 = torch.randn((BATCH, EMBEDDING_DIM, SLOTS), dtype=torch.float32, device=device)
    indices = torch.randint(args.rows, (args.tables, BATCH), dtype=torch.int64, device=device)
    rows = tuple(args.rows for _ in range(args.tables))
    inputs = (bmm_2, bmm_1, indices, rows)

    expected = eager_full(*inputs)
    eager_split_outputs = eager_split(*inputs)
    check_outputs(torch, expected, eager_split_outputs)

    compiled = torch.compile(eager_split, backend="inductor", fullgraph=True, dynamic=False)
    compiled_outputs = compiled(*inputs)
    synchronize(torch, device)
    check_outputs(torch, expected, compiled_outputs)

    eager_full_samples = timed(torch, device, eager_full, inputs, args.warmup, args.repeat)
    eager_split_samples = timed(torch, device, eager_split, inputs, args.warmup, args.repeat)
    inductor_samples = timed(torch, device, compiled, inputs, args.warmup, args.repeat)
    medians = {
        "eager_full": statistics.median(eager_full_samples),
        "eager_split": statistics.median(eager_split_samples),
        "inductor_split": statistics.median(inductor_samples),
    }
    print(f"device={device} tables={args.tables} rows={args.rows} grad_shape=({BATCH}, {EMBEDDING_DIM})")
    print("correctness=PASS")
    for name, value in medians.items():
        print(f"{name}_us={value:.3f}")
    print(f"inductor_vs_eager_full={1.0 - medians['inductor_split'] / medians['eager_full']:.4%}")
    print(f"inductor_vs_eager_split={1.0 - medians['inductor_split'] / medians['eager_split']:.4%}")
    print(f"eager_full_samples_us={[round(value, 3) for value in eager_full_samples]}")
    print(f"eager_split_samples_us={[round(value, 3) for value in eager_split_samples]}")
    print(f"inductor_samples_us={[round(value, 3) for value in inductor_samples]}")


if __name__ == "__main__":
    main()
