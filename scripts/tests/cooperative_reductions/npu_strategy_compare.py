"""Matched A5 SIMD prototypes: split kernels versus an all-Vector cooperative kernel."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

from strategy_compare import compare, make_function, write_json


def build_kernels():
    import triton
    import triton.language as tl
    from triton.language.extra.cann import extension

    globals().update(triton=triton, tl=tl, extension=extension)

    @triton.jit
    def local_stats(A, B, row, shard, R: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr, OP: tl.constexpr):
        chunk = tl.cdiv(R, S * BLOCK) * BLOCK
        acc = tl.full((BLOCK,), 0, tl.float32)
        maximum = tl.full((), float("-inf"), tl.float32)
        exp_sum = tl.full((), 0, tl.float32)
        for begin in range(shard * chunk, tl.minimum((shard + 1) * chunk, R), BLOCK):
            offsets = begin + tl.arange(0, BLOCK)
            a = tl.load(A + row * R + offsets, offsets < R, other=0)
            b = tl.load(B + row * R + offsets, offsets < R, other=0)
            z = a + b
            if OP == 2:
                z = tl.where(offsets < R, z, float("-inf"))
                new_maximum = tl.maximum(maximum, tl.max(z, 0))
                exp_sum = exp_sum * tl.exp(maximum - new_maximum) + tl.sum(tl.exp(z - new_maximum), 0)
                maximum = new_maximum
            elif OP == 1:
                acc += z * z
            else:
                acc += z
        if OP == 2:
            return maximum, exp_sum
        else:
            return maximum, tl.sum(acc, 0)

    @triton.jit
    def read_stats(W, row, R: tl.constexpr, S: tl.constexpr, P: tl.constexpr, X: tl.constexpr, OP: tl.constexpr):
        peers = tl.arange(0, P)
        sums = tl.load(W + row * S + peers, peers < S, other=0)
        if OP == 2:
            maxima = tl.load(W + X * S + row * S + peers, peers < S, other=float("-inf"))
            maximum = tl.max(maxima, 0)
            total = tl.sum(sums * tl.exp(maxima - maximum), 0)
        else:
            maximum = tl.full((), 0, tl.float32)
            total = tl.sum(sums, 0)
        return maximum, total

    @triton.jit
    def emit(A, B, Out, row, shard, maximum, total, R: tl.constexpr, S: tl.constexpr,
             BLOCK: tl.constexpr, OP: tl.constexpr):
        if OP == 0:
            tl.store(Out + row, total, shard == 0)
        else:
            chunk = tl.cdiv(R, S * BLOCK) * BLOCK
            if OP == 1:
                scale = tl.rsqrt(total / R + 1e-5)
            for begin in range(shard * chunk, tl.minimum((shard + 1) * chunk, R), BLOCK):
                offsets = begin + tl.arange(0, BLOCK)
                a = tl.load(A + row * R + offsets, offsets < R, other=0)
                b = tl.load(B + row * R + offsets, offsets < R, other=0)
                z = a + b
                if OP == 2:
                    value = tl.exp(z - maximum) / total
                else:
                    value = z * scale
                tl.store(Out + row * R + offsets, value, offsets < R)

    globals().update(local_stats=local_stats, read_stats=read_stats, emit=emit)

    @triton.jit
    def partial(A, B, W, R: tl.constexpr, S: tl.constexpr, P: tl.constexpr,
                X: tl.constexpr, BLOCK: tl.constexpr, OP: tl.constexpr):
        shard, row = tl.program_id(0), tl.program_id(1)
        maximum, total = local_stats(A, B, row, shard, R, S, BLOCK, OP)
        tl.store(W + row * S + shard, total)
        if OP == 2:
            tl.store(W + X * S + row * S + shard, maximum)

    @triton.jit
    def finish(A, B, W, Out, R: tl.constexpr, S: tl.constexpr, P: tl.constexpr,
               X: tl.constexpr, BLOCK: tl.constexpr, OP: tl.constexpr):
        shard, row = tl.program_id(0), tl.program_id(1)
        maximum, total = read_stats(W, row, R, S, P, X, OP)
        emit(A, B, Out, row, shard, maximum, total, R, S, BLOCK, OP)

    @triton.jit
    def cooperative(A, B, W, Out, R: tl.constexpr, S: tl.constexpr, P: tl.constexpr,
                    X: tl.constexpr, BLOCK: tl.constexpr, OP: tl.constexpr):
        shard, row = tl.program_id(0), tl.program_id(1)
        maximum, total = local_stats(A, B, row, shard, R, S, BLOCK, OP)
        if S > 1:
            tl.store(W + row * S + shard, total)
            if OP == 2:
                tl.store(W + X * S + row * S + shard, maximum)
            extension.sync_block_all("all_vector", 0)
            maximum, total = read_stats(W, row, R, S, P, X, OP)
        emit(A, B, Out, row, shard, maximum, total, R, S, BLOCK, OP)

    return partial, finish, cooperative


def time_kernel(torch, invoke, args):
    for _ in range(5):
        invoke()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(args.batch):
            invoke()
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    samples = []
    for _ in range(args.samples):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(args.replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (args.batch * args.replays))
    return {"median_us": statistics.median(samples), "min_us": min(samples), "max_us": max(samples),
            "samples_us": samples, "batch": args.batch, "replays": args.replays,
            "method": "NPU Graph + device Events"}


def worker(args):
    import torch
    import torch_npu
    import triton

    torch.set_num_threads(4)
    torch.manual_seed(1731)
    torch.npu.set_device(0)
    torch.npu.set_op_timeout_ms(10000)
    physical = ctypes.c_int32(-1)
    query = ctypes.CDLL("libascendcl.so").aclrtGetPhyDevIdByUserDevId
    if query(0, ctypes.byref(physical)) or physical.value != args.physical_device:
        raise RuntimeError(f"wrong physical device: {physical.value}")
    props = torch.npu.get_device_properties(0)
    rows, rnumel, splits = int(args.rows), int(args.rnumels), int(args.splits)
    assert rows * splits <= props.vector_core_num
    a = torch.randn((rows, rnumel), device="npu")
    b = torch.randn_like(a)
    workspace = torch.full((2 * rows * splits,), float("nan"), device="npu")
    output = torch.empty((rows,) if args.workloads == "sum" else (rows, rnumel), device="npu")
    kernels = build_kernels()
    constants = {"R": rnumel, "S": splits, "P": triton.next_power_of_2(splits),
                 "X": rows, "BLOCK": args.block, "OP": ("sum", "rmsnorm", "softmax").index(args.workloads)}
    options = {"compile_mode": "simd", "num_warps": 4, "launch_cooperative_grid": True}
    grid = (splits, rows)
    fn = make_function(args.workloads)
    records = []
    for strategy in (["plain"] if splits == 1 else ["split", "cooperative"]):
        record = {"workload": args.workloads, "strategy": strategy, "X": rows, "R": rnumel,
                  "S": splits, "block": args.block, "physical_device": physical.value,
                  "effective_vector_cores": props.vector_core_num,
                  "torch": torch.__version__, "torch_npu": torch_npu.__version__, "triton": triton.__version__}
        try:
            workspace.fill_(float("nan"))
            output.fill_(float("nan"))

            def invoke():
                if strategy == "split":
                    first = kernels[0][grid](a, b, workspace, **constants, **options)
                    second = kernels[1][grid](a, b, workspace, output, **constants, **options)
                    return first, second
                return (kernels[2][grid](a, b, workspace, output, **constants, **options),)

            compiled = invoke()
            torch.npu.synchronize()
            record["accuracy"] = [compare(output, fn(a, b), args.workloads)]
            for epoch in range(2):
                a.normal_()
                b.normal_()
                output.fill_(float("nan"))
                invoke()
                torch.npu.synchronize()
                record["accuracy"].append(compare(output, fn(a, b), args.workloads))
            for index, kernel in enumerate(compiled):
                meta = kernel.metadata
                record.setdefault("metadata", []).append(meta._asdict() if hasattr(meta, "_asdict") else vars(meta))
                for stage in ("ttir", "ttadapter"):
                    if isinstance(kernel.asm.get(stage), str):
                        (args.out / f"{strategy}-{index}.{stage}").write_text(kernel.asm[stage])
            record["timing"] = time_kernel(torch, invoke, args)
            record["post_accuracy"] = compare(output, fn(a, b), args.workloads)
            record["status"] = "PASS" if not any(item["mismatched_values"] for item in (
                *record["accuracy"], record["post_accuracy"])) else "WRONG"
        except Exception:
            record.update(status="ERROR", error=traceback.format_exc())
        records.append(record)
        write_json(args.out / "result.json", records)
        print(json.dumps({k: record.get(k) for k in ("workload", "strategy", "X", "R", "S", "status", "timing")}),
              flush=True)
    return 0 if all(r["status"] == "PASS" for r in records) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workloads", default="sum,rmsnorm,softmax")
    parser.add_argument("--rows", default="1")
    parser.add_argument("--rnumels", default="32768,1048576")
    parser.add_argument("--splits", default="1,2,4,8,16,32,56")
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--case-timeout", type=int, default=180)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker(args)
    write_json(args.out / "manifest.json", {"argv": sys.argv, "time_utc": datetime.now(timezone.utc).isoformat(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "visible_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES")})
    results = []
    for workload in args.workloads.split(","):
        for rows in args.rows.split(","):
            for rnumel in args.rnumels.split(","):
                for splits in args.splits.split(","):
                    if int(rows) * int(splits) > 56:
                        continue
                    case = args.out / f"{workload}-x{rows}-r{rnumel}-s{splits}"
                    case.mkdir(exist_ok=True)
                    if args.resume and (case / "result.json").exists():
                        records = json.loads((case / "result.json").read_text())
                    else:
                        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--out", str(case),
                                   "--workloads", workload, "--rows", rows, "--rnumels", rnumel, "--splits", splits,
                                   "--physical-device", str(args.physical_device), "--block", str(args.block),
                                   "--batch", str(args.batch), "--replays", str(args.replays), "--samples", str(args.samples)]
                        with (case / "run.log").open("w") as stream:
                            try:
                                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=args.case_timeout)
                                records = json.loads((case / "result.json").read_text()) if (
                                    case / "result.json").exists() else [{"status": "PROCESS_ERROR"}]
                            except subprocess.TimeoutExpired:
                                records = [{"status": "TIMEOUT"}]
                    for record in records:
                        record.update(workload=workload, X=int(rows), R=int(rnumel), S=int(splits))
                        results.append(record)
                    write_json(args.out / "results.json", results)
                    print(workload, rows, rnumel, splits, [(r.get("strategy"), r["status"]) for r in records], flush=True)
    return 0 if all(r["status"] == "PASS" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
