#!/usr/bin/env python3
"""Profile a Triton kernel for every num_warps/superblock_factor combination."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import torch
import torch_npu
import triton
import triton.language as tl
from torch._dynamo.testing import rand_strided

from tools.npu_profiler import ProfileResultParser, TorchNpuProfiler


VALUES = (1, 2, 4, 8, 16, 32, 64)
BATCH = 4096
SOURCE_WIDTH = 2280
SLOTS = 38
EMBEDDING_DIM = 128
NUM_EMBEDDINGS = 98166


@triton.jit
def triton_poi_fused_add_eq_expand_full_index_put__1(
    in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, z0_numel, y1_numel, x2_numel,
    Z0BLOCK, Z0BLOCK_SUB: tl.constexpr, Y1BLOCK_SUB: tl.constexpr,
):
    x2_numel = 128
    X2BLOCK_SUB: tl.constexpr = 128
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0 = tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    base_y1 = tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (y1_numel + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2 = tl.arange(0, X2BLOCK_SUB)
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:, None, None]
        z0_mask = z0 < min(Z0BLOCK + z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = (loop_y1 * Y1BLOCK_SUB) + base_y1[None, :, None]
            y1_mask = y1 < y1_numel
            x2 = base_x2[None, None, :]
            x2_mask = x2 < x2_numel
            tmp0 = tl.load(in_ptr0 + (y1 + 38 * z0), y1_mask & z0_mask)
            tmp8 = tl.load(in_ptr1 + (y1 + 38 * z0), y1_mask & z0_mask)
            tmp9 = tl.load(in_ptr2 + (1064 + x2 + 2280 * z0), x2_mask & z0_mask)
            tmp10 = tl.load(in_ptr3 + (x2 + 128 * z0), x2_mask & z0_mask)
            tmp12 = tl.load(in_ptr4 + (x2 + 128 * z0), x2_mask & z0_mask)
            tmp1 = tl.full([Z0BLOCK_SUB, Y1BLOCK_SUB, X2BLOCK_SUB], 98166, tl.int32)
            tmp2 = tmp0 + tmp1
            tmp3 = tmp0 < 0
            tmp4 = tl.where(tmp3, tmp2, tmp0)
            tl.device_assert(
                ((0 <= tmp4) & (tmp4 < 98166)) | ~(y1_mask & z0_mask),
                "index out of bounds: 0 <= tmp4 < 98166",
            )
            tmp6 = tl.full([1, 1, 1], -1, tl.int64)
            tmp7 = tmp0 == tmp6
            tmp11 = tmp9 + tmp10
            tmp13 = tmp11 + tmp12
            tmp14 = 0.0
            tmp15 = tl.where(tmp8, tmp13, tmp14)
            tmp16 = tl.where(tmp7, tmp14, tmp15)
            tl.atomic_add(out_ptr0 + (x2 + 128 * tmp4), tmp16, x2_mask & y1_mask & z0_mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=int(os.getenv("WARMUP", "10")))
    parser.add_argument("--repeat", type=int, default=int(os.getenv("REPEAT", "20")))
    parser.add_argument("--profile-root", type=Path, default=Path("prof_log/num_warps_superblock_factor"))
    parser.add_argument("--discard-profile", action="store_true", help="remove raw profile files after parsing")
    return parser.parse_args()


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    return (
        rand_strided((BATCH, SOURCE_WIDTH), (SOURCE_WIDTH, 1), device="npu", dtype=torch.float32),
        rand_strided((BATCH, 1, EMBEDDING_DIM), (EMBEDDING_DIM, EMBEDDING_DIM, 1), device="npu", dtype=torch.float32),
        rand_strided((BATCH, 1, EMBEDDING_DIM), (EMBEDDING_DIM, EMBEDDING_DIM, 1), device="npu", dtype=torch.float32),
        rand_strided((BATCH, SLOTS), (SLOTS, 1), device="npu", dtype=torch.bool),
        rand_strided((BATCH, SLOTS), (SLOTS, 1), device="npu", dtype=torch.int64),
    )


def launch(inputs: tuple[torch.Tensor, ...], output: torch.Tensor, num_warps: int, superblock_factor: int) -> None:
    triton_poi_fused_add_eq_expand_full_index_put__1[(triton.cdiv(128, 1), 1, 1)](
        inputs[4], inputs[3], inputs[0], inputs[1], inputs[2], output, BATCH, SLOTS, EMBEDDING_DIM,
        Z0BLOCK=128, Z0BLOCK_SUB=1, Y1BLOCK_SUB=16, compile_mode="simt_only",
        multibuffer=False, num_ctas=1, num_stages=2, num_warps=num_warps,
        superblock_factor=superblock_factor,
    )


def profile_config(inputs, num_warps, superblock_factor, warmup, repeat, profile_dir):
    output = torch.zeros((NUM_EMBEDDINGS, EMBEDDING_DIM), device="npu", dtype=torch.float32)
    for _ in range(warmup):
        launch(inputs, output, num_warps, superblock_factor)
    torch.npu.synchronize()
    TorchNpuProfiler(
        profile_dir, wait=0, warmup=0, active=repeat, repeat=1,
        record_shapes=False, profile_memory=False, with_stack=False,
    ).run_steps(lambda: launch(inputs, output, num_warps, superblock_factor), steps=repeat)
    durations = []
    for row in ProfileResultParser(profile_dir).kernel_rows():
        value = row.get("Duration(us)") or row.get("Task Duration(us)")
        if value:
            durations.append(float(value))
    if not durations:
        raise RuntimeError(f"no device kernel records found under {profile_dir}")
    return sum(durations) / repeat, len(durations)


def main() -> None:
    args = parse_args()
    if args.device_id < 0 or args.warmup < 0 or args.repeat <= 0:
        raise ValueError("device-id must be non-negative, warmup >= 0, and repeat > 0")
    torch_npu.npu._initialized = torch_npu.npu.is_initialized()
    torch.npu.set_device(args.device_id)
    inputs = make_inputs()
    args.profile_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for num_warps in VALUES:
        for superblock_factor in VALUES:
            label = f"num_warps_{num_warps}_superblock_factor_{superblock_factor}"
            profile_dir = args.profile_root / label
            if profile_dir.exists():
                shutil.rmtree(profile_dir)
            try:
                mean_us, kernel_count = profile_config(
                    inputs, num_warps, superblock_factor, args.warmup, args.repeat, profile_dir
                )
                status, error = "ok", ""
                print(f"num_warps={num_warps:>2} superblock_factor={superblock_factor:>2} device_us={mean_us:.3f}")
            except Exception as exc:
                mean_us, kernel_count = float("nan"), 0
                status, error = "error", f"{type(exc).__name__}: {exc}"
                print(f"num_warps={num_warps:>2} superblock_factor={superblock_factor:>2} ERROR {error}")
            rows.append({
                "num_warps": num_warps, "superblock_factor": superblock_factor,
                "warmup": args.warmup, "repeat": args.repeat,
                "device_us": mean_us, "kernel_count": kernel_count,
                "status": status, "error": error, "profile_dir": str(profile_dir),
            })
            if args.discard_profile and status == "ok":
                shutil.rmtree(profile_dir, ignore_errors=True)
    summary_path = args.profile_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["status"] != "ok", row["device_us"])))
    print(f"summary={summary_path.resolve()}")


if __name__ == "__main__":
    main()
