#!/usr/bin/env python3
"""Reproduce the DLRM embedding_select Inductor performance case.

The default graph performs 26 independent lookups with indices ``[26, 128]``
and embedding width 16.  It compares synchronized eager and Inductor time and
checks every output.  Run on the target device, for example:

    ASCEND_RT_VISIBLE_DEVICES=2 python scripts/repro/dlrm_embedding_select_perf.py
"""

from __future__ import annotations

import argparse
import os
import statistics
import time


BATCH = 128
EMBEDDING_DIM = 16
DEFAULT_TABLES = 26
DEFAULT_ROWS = 1460


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("npu", "cuda"), default="npu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--tables", type=int, default=DEFAULT_TABLES)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
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


def forward(indices, weights):
    return tuple(torch.ops.aten.embedding.default(weight, indices[i]) for i, weight in enumerate(weights))


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


def main() -> None:
    global torch
    args = parse_args()
    if args.tables <= 0 or args.rows <= 0 or args.warmup < 0 or args.repeat <= 0:
        raise ValueError("tables, rows and repeat must be positive")

    torch, device = prepare_device(args.device, args.device_id)
    torch.manual_seed(0)
    indices = torch.randint(args.rows, (args.tables, BATCH), dtype=torch.int64, device=device)
    weights = tuple(
        torch.randn((args.rows, EMBEDDING_DIM), dtype=torch.float32, device=device)
        for _ in range(args.tables)
    )
    inputs = (indices, weights)
    eager_outputs = forward(*inputs)
    compiled = torch.compile(forward, backend="inductor", fullgraph=True, dynamic=False)
    compiled_outputs = compiled(*inputs)
    synchronize(torch, device)
    for eager_output, compiled_output in zip(eager_outputs, compiled_outputs):
        torch.testing.assert_close(eager_output, compiled_output)

    eager_samples = timed(torch, device, forward, inputs, args.warmup, args.repeat)
    inductor_samples = timed(torch, device, compiled, inputs, args.warmup, args.repeat)
    eager_us = statistics.median(eager_samples)
    inductor_us = statistics.median(inductor_samples)
    print(f"device={device} tables={args.tables} rows={args.rows} shape=({BATCH}, {EMBEDDING_DIM})")
    print(f"correctness=PASS eager_us={eager_us:.3f} inductor_us={inductor_us:.3f}")
    print(f"improvement={1.0 - inductor_us / eager_us:.4%}")
    print(f"eager_samples_us={[round(value, 3) for value in eager_samples]}")
    print(f"inductor_samples_us={[round(value, 3) for value in inductor_samples]}")


if __name__ == "__main__":
    main()
