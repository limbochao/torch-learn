"""Device-timed barrier comparisons and correctness checks on CUDA and Ascend."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import traceback


KINDS = {"original": 0, "native": 1, "debug": 2}


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)


def load_stack(args):
    os.environ.setdefault("TRITON_CACHE_DIR", str(args.out / "triton_cache"))
    import torch
    import triton
    import triton.language as tl

    if args.backend == "cuda" and getattr(args, "cuda_barrier", "installed") == "snapshot":
        from cuda_triton_helpers import x_grid_barrier as original_barrier
    else:
        from torch._inductor.runtime.triton_helpers import x_grid_barrier as original_barrier

    if args.backend == "npu":
        import torch_npu
        from npu_triton_helpers import x_grid_barrier as native_barrier, x_grid_barrier_debug as debug_barrier

        device_api = torch.npu
        device_api.set_device(0)
        device_api.set_op_timeout_ms(args.op_timeout_ms)
        physical = ctypes.c_int32(-1)
        query = ctypes.CDLL("libascendcl.so").aclrtGetPhyDevIdByUserDevId
        query.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
        query.restype = ctypes.c_int32
        status = query(0, ctypes.byref(physical))
        if status or physical.value != args.physical_device:
            raise RuntimeError(f"NPU mapping mismatch: status={status}, physical={physical.value}")
        if args.mapping == "off":
            import triton.backends.ascend.compiler as compiler

            # Diagnostic process-local switch, matching the existing component probe.
            compiler._is_auto_map_parallel_blocks_enabled = lambda: False
        options = {"compile_mode": "simd", "num_warps": 4, "launch_cooperative_grid": True}
    else:
        device_api = torch.cuda
        if not device_api.is_available():
            raise RuntimeError("CUDA is unavailable; check the container and CUDA_VISIBLE_DEVICES")
        device_api.set_device(0)
        native_barrier = original_barrier
        debug_barrier = original_barrier
        options = {"num_warps": 4, "launch_cooperative_grid": True}
    globals().update(torch=torch, triton=triton, tl=tl, original_barrier=original_barrier,
                     native_barrier=native_barrier, debug_barrier=debug_barrier)
    globals()["device_api"] = device_api
    torch.manual_seed(args.seed)
    return torch, triton, device_api, options


def build_kernels():
    @triton.jit
    def barrier(Sem, KIND: tl.constexpr):
        if KIND == 0:
            original_barrier(Sem)
        elif KIND == 1:
            native_barrier(Sem)
        else:
            debug_barrier(Sem)

    globals()["barrier"] = barrier

    @triton.jit
    def exchange(W, Out, Sem, epoch, S: tl.constexpr, P: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr,
                 PHASES: tl.constexpr, KIND: tl.constexpr, SKEW: tl.constexpr):
        shard = tl.program_id(0)
        row = tl.program_id(1) + Y * tl.program_id(2)
        sem = Sem.to(tl.pointer_type(tl.uint32)) + row
        peers = tl.arange(0, P)
        for phase in range(PHASES):
            extra = 0
            if SKEW:
                for step in range((shard % 4) * 8):
                    extra += step
            value = shard + 1 + row * 1000 + phase * 10000 + epoch * 1000 + extra
            tl.store(W + row * S + shard, value)
            barrier(sem, KIND)
            values = tl.load(W + row * S + peers, peers < S, other=0)
            tl.store(Out + (phase * Y * Z + row) * S + shard, tl.sum(values, 0))
            # Protect readers before the next phase overwrites their workspace.
            if PHASES > 1:
                barrier(sem, KIND)

    @triton.jit
    def reduce(A, Bptr, W, Out, Sem, R: tl.constexpr, S: tl.constexpr, P: tl.constexpr,
               Y: tl.constexpr, BLOCK: tl.constexpr, KIND: tl.constexpr, OP: tl.constexpr):
        shard = tl.program_id(0)
        row = tl.program_id(1) + Y * tl.program_id(2)
        chunk = tl.cdiv(R, S * BLOCK) * BLOCK
        if OP == 2:
            acc = tl.full((BLOCK,), float("-inf"), tl.float32)
        else:
            acc = tl.full((BLOCK,), 0, tl.float32)
        for begin in range(shard * chunk, tl.minimum((shard + 1) * chunk, R), BLOCK):
            offsets = begin + tl.arange(0, BLOCK)
            a = tl.load(A + row * R + offsets, offsets < R, other=0).to(tl.float32)
            b = tl.load(Bptr + row * R + offsets, offsets < R, other=0).to(tl.float32)
            value = a + b
            if OP == 2:
                acc = tl.maximum(acc, tl.where(offsets < R, value, float("-inf")))
            else:
                acc += value
        if OP == 2:
            partial = tl.max(acc, 0)
        else:
            partial = tl.sum(acc, 0)
        tl.store(W + row * S + shard, partial)
        barrier(Sem.to(tl.pointer_type(tl.uint32)) + row, KIND)
        peers = tl.arange(0, P)
        if OP == 2:
            values = tl.load(W + row * S + peers, peers < S, other=float("-inf"))
            result = tl.max(values, 0)
        else:
            values = tl.load(W + row * S + peers, peers < S, other=0)
            result = tl.sum(values, 0)
            if OP == 1:
                result /= R
        # Keep the same all-read combine algorithm for all three implementations.
        tl.store(Out + row, result, shard == 0)

    return exchange, reduce


def metadata_dict(kernel):
    value = kernel.metadata
    return value._asdict() if hasattr(value, "_asdict") else vars(value)


def save_kernel(args, kernel, name):
    directory = args.out / "ir"
    directory.mkdir(exist_ok=True)
    write_json(directory / (name + ".metadata.json"), metadata_dict(kernel))
    for stage in ("ttir", "ttgir", "ttadapter", "llir", "ptx"):
        source = kernel.asm.get(stage)
        if isinstance(source, str):
            (directory / (name + "." + stage)).write_text(source)


def manifest(args, api):
    properties = api.get_device_properties(0)
    packages = {}
    for name in ("torch", "torch-npu", "triton", "triton-ascend"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    record = {"argv": sys.argv, "time_utc": datetime.now(timezone.utc).isoformat(),
              "cwd": os.getcwd(), "pid": os.getpid(), "backend": args.backend,
              "physical_device": args.physical_device, "packages": packages,
              "properties": str(properties), "mapping": args.mapping,
              "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES") if args.backend == "cuda"
              else os.getenv("ASCEND_RT_VISIBLE_DEVICES"), "options": args.options,
              "op_timeout_ms": args.op_timeout_ms if args.backend == "npu" else None,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    repository = next((parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()), None)
    try:
        record["git_commit"] = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip() if repository else None
    except subprocess.CalledProcessError:
        record["git_commit"] = None
    record["helper_sha256"] = {}
    for name in args.kinds:
        helper = {"original": original_barrier, "native": native_barrier, "debug": debug_barrier}[name]
        source = inspect.getsource(helper.fn)
        (args.out / f"helper-{name}.py").write_text(source)
        record["helper_sha256"][name] = hashlib.sha256(source.encode()).hexdigest()
    if args.backend == "npu":
        from triton.backends.ascend.utils import _get_npucompiler_path

        compiler, environment = _get_npucompiler_path()
        version = subprocess.run([compiler, "--version"], env=environment, capture_output=True,
                                 text=True, timeout=30)
        record["compiler_version"] = version.stdout + version.stderr
        record["core_count"] = properties.vector_core_num
    else:
        record["core_count"] = properties.multi_processor_count
        record["device_name"] = properties.name
        record["cuda_version"] = torch.version.cuda
        record["device_uuid"] = str(getattr(properties, "uuid", "unknown"))
        record["cuda_barrier"] = getattr(args, "cuda_barrier", "installed")
        if record["cuda_barrier"] == "snapshot":
            record["barrier_source_commit"] = "cf30153c4c131c8164ee7798e5022d810682e2cb"
    write_json(args.out / "manifest.json", record)
    return record


def device_time(args, api, invoke):
    for _ in range(5):
        invoke()
    api.synchronize()
    batch, replays = args.batch, args.replays
    sample_limit = getattr(args, "max_sample_ms", None)
    pilot_us = None
    if sample_limit:
        start, end = api.Event(enable_timing=True), api.Event(enable_timing=True)
        start.record()
        invoke()
        end.record()
        end.synchronize()
        pilot_us = start.elapsed_time(end) * 1000
        call_budget = max(1, int(sample_limit * 1000 / max(pilot_us, 1)))
        batch = min(batch, call_budget)
        replays = min(replays, max(1, call_budget // batch))
    graph = api.CUDAGraph() if args.backend == "cuda" else api.NPUGraph()
    with api.graph(graph):
        for _ in range(batch):
            invoke()
    for _ in range(3):
        graph.replay()
    api.synchronize()
    samples = []
    for _ in range(args.samples):
        start, end = api.Event(enable_timing=True), api.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (batch * replays))
    return {"median_us": statistics.median(samples), "min_us": min(samples), "max_us": max(samples),
            "samples_us": samples, "method": f"{args.backend} graph + device events",
            "batch": batch, "replays": replays, "sample_count": args.samples,
            "pilot_us": pilot_us, "max_sample_ms": sample_limit}


def exchange_expected(splits, rows, phases, epoch, skew):
    extra = sum(sum(range((shard % 4) * 8)) for shard in range(splits)) if skew else 0
    expected = torch.empty((phases, rows, splits), dtype=torch.int32)
    for phase in range(phases):
        for row in range(rows):
            expected[phase, row].fill_(splits * (splits + 1) // 2
                                       + splits * (row * 1000 + phase * 10000 + epoch * 1000) + extra)
    return expected


def make_exchange(args, kernel, splits, y, z, kind, phases=1, epoch=0, skew=False, prepared=None):
    rows = y * z
    if prepared is None:
        workspace = torch.full((rows, splits), -12345, dtype=torch.int32, device=args.backend)
        output = torch.full((phases, rows, splits), -54321, dtype=torch.int32, device=args.backend)
        initial = 0 if kind == "original" else 123
        sem = torch.full((rows,), initial, dtype=torch.int32, device=args.backend)
    else:
        output, sem, buffers = prepared
        workspace = buffers[0]

    def invoke():
        return kernel[(splits, y, z)](workspace, output, sem, epoch,
                                      S=splits, P=triton.next_power_of_2(splits), Y=y, Z=z,
                                      PHASES=phases, KIND=KINDS[kind], SKEW=skew, **args.options)

    return invoke, output, sem, (workspace,)


def verify_exchange(args, api, kernel, splits, y, z, kind, phases=1, epoch=0, skew=False):
    invoke, output, sem, keepalive = make_exchange(args, kernel, splits, y, z, kind, phases, epoch, skew)
    compiled = invoke()
    api.synchronize()
    actual = output.cpu()
    expected = exchange_expected(splits, y * z, phases, epoch, skew)
    if args.phase == "mapping":
        # Keep the launched IR and observed values even when the numerical assertion fails.
        save_kernel(args, compiled, "mapping")
        write_json(args.out / "mapping-observations.json", {
            "grid": [splits, y, z], "metadata": metadata_dict(compiled),
            "mismatched_values": int((actual != expected).sum()), "checked_values": actual.numel(),
            "first_phase_observed": actual[0].tolist(), "first_phase_expected": expected[0].tolist(),
            "last_phase_observed": actual[-1].tolist(), "last_phase_expected": expected[-1].tolist()})
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if kind != "original":
        assert bool((sem.cpu() == 123).all()), "Native helper unexpectedly modified the semaphore"
    return compiled, {"max_abs_error": 0, "checked_values": actual.numel(),
                      "final_sem": sem.cpu().tolist()}


def make_reduction(args, kernel, rows, count, splits, kind, dtype="float32", operation="sum", seed=0, prepared=None):
    if prepared is None:
        generator = torch.Generator().manual_seed(args.seed + seed)
        data_dtype = getattr(torch, dtype)
        left = torch.randn((rows, count), generator=generator, dtype=torch.float32).to(data_dtype)
        right = torch.randn((rows, count), generator=generator, dtype=torch.float32).to(data_dtype)
        # Define the prologue in fp32, then use an fp64 CPU reference for the reduction.
        reference_input = (left.float() + right.float()).double()
        if operation == "amax":
            expected = reference_input.amax(-1)
        elif operation == "mean":
            expected = reference_input.mean(-1)
        else:
            expected = reference_input.sum(-1)
        left, right = left.to(args.backend), right.to(args.backend)
        workspace = torch.full((rows, splits), float("nan"), device=args.backend)
        output = torch.full((rows,), float("nan"), device=args.backend)
        sem = torch.zeros(rows, dtype=torch.int32, device=args.backend)
    else:
        output, expected, buffers = prepared
        left, right, workspace, sem = buffers

    def invoke():
        return kernel[(splits, rows)](left, right, workspace, output, sem,
                                      R=count, S=splits, P=triton.next_power_of_2(splits),
                                      Y=rows, BLOCK=args.block, KIND=KINDS[kind],
                                      OP={"sum": 0, "mean": 1, "amax": 2}[operation], **args.options)

    return invoke, output, expected, (left, right, workspace, sem)


def check_reduction(output, expected, operation):
    actual = output.cpu().double()
    atol = 2e-2 if operation == "sum" else 2e-5
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=atol)
    difference = (actual - expected).abs()
    return {"max_abs_error": difference.max().item(),
            "max_relative_error": (difference / expected.abs().clamp_min(1e-12)).max().item(),
            "atol": atol, "rtol": 2e-4, "observed": actual.tolist(), "reference": expected.tolist()}


def benchmark(args, api, kernels, info):
    exchange, reduce = kernels
    records = []
    shapes = [tuple(map(int, shape.split("x"))) for shape in args.shapes.split(",")]
    for workload, rows, count in [("exchange", 1, 0)] + [("sum", x, r) for x, r in shapes]:
        for splits in args.splits:
            if rows * splits > info["core_count"]:
                continue
            # Rotate the order across S to reduce a fixed ordering bias.
            shift = args.splits.index(splits) % len(args.kinds)
            order = args.kinds[shift:] + args.kinds[:shift]
            prepared = None
            for kind in order:
                record = {"workload": workload, "X": rows, "R": count, "S": splits,
                          "kind": kind, "dtype": "float32", "block": args.block}
                try:
                    if workload == "exchange":
                        invoke, output, sem, keepalive = make_exchange(
                            args, exchange, splits, rows, 1, kind, prepared=prepared)
                        prepared = (output, sem, keepalive)
                        keepalive[0].fill_(-12345)
                        output.fill_(-54321)
                        sem.zero_()
                        api.synchronize()
                        compiled = invoke()
                        api.synchronize()
                        torch.testing.assert_close(output.cpu(), exchange_expected(splits, rows, 1, 0, False),
                                                   rtol=0, atol=0)
                        accuracy = {"max_abs_error": 0}
                    else:
                        invoke, output, expected, keepalive = make_reduction(
                            args, reduce, rows, count, splits, kind, prepared=prepared)
                        prepared = (output, expected, keepalive)
                        keepalive[2].fill_(float("nan"))
                        output.fill_(float("nan"))
                        keepalive[3].zero_()
                        api.synchronize()
                        compiled = invoke()
                        api.synchronize()
                        accuracy = check_reduction(output, expected, "sum")
                    timing = device_time(args, api, invoke)
                    if workload == "exchange":
                        torch.testing.assert_close(output.cpu(), exchange_expected(splits, rows, 1, 0, False),
                                                   rtol=0, atol=0)
                    else:
                        check_reduction(output, expected, "sum")
                    record.update(status="PASS", accuracy=accuracy, **timing, metadata=metadata_dict(compiled))
                    record["buffer_addresses"] = [tensor.data_ptr() for tensor in keepalive] + [output.data_ptr()]
                    record["same_buffers_across_helpers"] = True
                    if count == 1048576 and rows == 1 and splits in (1, 8, 32, 56):
                        save_kernel(args, compiled, f"{workload}-{kind}-x{rows}-r{count}-s{splits}")
                except Exception:
                    record.update(status="FAIL", error=traceback.format_exc())
                    records.append(record)
                    write_json(args.out / "bench.json", records)
                    raise
                records.append(record)
                write_json(args.out / "bench.json", records)
                print(json.dumps({k: record[k] for k in ("workload", "X", "R", "S", "kind", "status", "median_us")}),
                      flush=True)
    return records


def correctness(args, api, kernels, info):
    exchange, reduce = kernels
    records = []
    grids = [(s, 1, 1) for s in args.splits]
    grids += [(4, 2, 1), (8, 4, 1), (7, 4, 2), (14, 2, 2), (28, 2, 1)]
    for kind in args.kinds:
        for splits, y, z in grids:
            if splits * y * z > info["core_count"]:
                continue
            record = {"test": "exchange", "kind": kind, "grid": [splits, y, z],
                      "phases": args.phases, "epochs": [0, 11, 97], "skew": True}
            try:
                errors = []
                for epoch in record["epochs"]:
                    compiled, accuracy = verify_exchange(args, api, exchange, splits, y, z, kind,
                                                         args.phases, epoch, skew=True)
                    errors.append(accuracy)
                record.update(status="PASS", accuracy=errors, metadata=metadata_dict(compiled))
                if (splits, y, z) == (7, 4, 2):
                    save_kernel(args, compiled, f"exchange-{kind}-s7-y4-z2")
            except Exception:
                record.update(status="FAIL", error=traceback.format_exc())
                records.append(record)
                write_json(args.out / "correctness.json", records)
                raise
            records.append(record)
            write_json(args.out / "correctness.json", records)
            print("check", kind, record["grid"], record["status"], flush=True)
        for dtype in ("float32", "float16", "bfloat16"):
            for operation in ("sum", "mean", "amax"):
                for rows, count, splits in ((1, 17, 32), (1, 1048593, 32), (4, 32771, 8)):
                    record = {"test": "reduction", "kind": kind, "dtype": dtype, "operation": operation,
                              "X": rows, "R": count, "S": splits}
                    invoke, output, expected, keepalive = make_reduction(
                        args, reduce, rows, count, splits, kind, dtype, operation)
                    try:
                        for _ in range(3):
                            compiled = invoke()
                        api.synchronize()
                        record.update(status="PASS", accuracy=check_reduction(output, expected, operation),
                                      metadata=metadata_dict(compiled))
                    except Exception:
                        record.update(status="FAIL", error=traceback.format_exc())
                        records.append(record)
                        write_json(args.out / "correctness.json", records)
                        raise
                    records.append(record)
                    write_json(args.out / "correctness.json", records)
                    print("check", kind, dtype, operation, count, record["status"], flush=True)
    return records


def concurrency_worker(config, rank, ready):
    args = argparse.Namespace(**config)
    args.out = Path(args.out) / f"worker-{rank}"
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        _, _, api, args.options = load_stack(args)
        kernels = build_kernels()
        manifest(args, api)
        kind = args.kinds[0]
        streams = [api.Stream() for _ in range(args.streams)]
        tasks = []
        for index, stream in enumerate(streams):
            invoke, output, expected, keepalive = make_reduction(
                args, kernels[1], args.concurrent_rows, args.concurrent_r, args.concurrent_s, kind,
                seed=100 * rank + index)
            epoch_base = 10000 + rank * 1000 + index * 100
            exchange_invoke, exchange_output, exchange_sem, exchange_buffers = make_exchange(
                args, kernels[0], args.concurrent_s, args.concurrent_rows, 1, kind,
                phases=2, epoch=epoch_base + 3, skew=True)
            # Initialization uses the default stream; make it visible before another stream launches.
            api.synchronize()
            with api.stream(stream):
                compiled = invoke()
                exchange_invoke()
            tasks.append((stream, output, expected, keepalive, exchange_output,
                          exchange_sem, exchange_buffers, epoch_base))
        api.synchronize()
        for task in tasks:
            check_reduction(task[1], task[2], "sum")
        ready.wait(timeout=240)
        # Separate output tensors preserve every result; each stream owns its workspace and semaphore.
        results = []
        launch_gate = threading.Barrier(args.streams)
        failures = []

        def run_stream(index):
            (stream, output, expected, keepalive, exchange_output,
             exchange_sem, exchange_buffers, epoch_base) = tasks[index]
            left, right, workspace, sem = keepalive
            observations = []
            exchanges = []
            try:
                with api.stream(stream):
                    launch_gate.wait(timeout=60)
                    for iteration in range(args.iterations):
                        target = torch.empty_like(output)
                        kernels[1][(args.concurrent_s, args.concurrent_rows)](
                            left, right, workspace, target, sem,
                            R=args.concurrent_r, S=args.concurrent_s, P=triton.next_power_of_2(args.concurrent_s),
                            Y=args.concurrent_rows,
                            BLOCK=args.block, KIND=KINDS[kind], OP=0, **args.options)
                        observations.append(target)
                        # Every launch has different integer data, exposing stale workspace reads.
                        epoch = epoch_base + 2 * iteration + 3
                        exchanged = torch.empty_like(exchange_output)
                        kernels[0][(args.concurrent_s, args.concurrent_rows, 1)](
                            exchange_buffers[0], exchanged, exchange_sem, epoch,
                            S=args.concurrent_s, P=triton.next_power_of_2(args.concurrent_s),
                            Y=args.concurrent_rows, Z=1, PHASES=2, KIND=KINDS[kind], SKEW=True, **args.options)
                        exchanges.append((exchanged, epoch))
                stream.synchronize()
                accuracy = [check_reduction(value, expected, "sum") for value in observations]
                for value, epoch in exchanges:
                    reference = exchange_expected(args.concurrent_s, args.concurrent_rows, 2, epoch, True)
                    torch.testing.assert_close(value.cpu(), reference, rtol=0, atol=0)
                results.append({"stream_index": index, "stream_id": stream.npu_stream if args.backend == "npu"
                                else stream.cuda_stream, "iterations": len(accuracy), "accuracy": accuracy,
                                "integer_exchange_launches": len(exchanges), "integer_exchange_error": 0,
                                "first_epoch": exchanges[0][1], "last_epoch": exchanges[-1][1]})
            except Exception:
                failures.append(traceback.format_exc())

        threads = [threading.Thread(target=run_stream, args=(index,)) for index in range(args.streams)]
        started = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        api.synchronize()
        record = {"rank": rank, "pid": os.getpid(), "status": "FAIL" if failures else "PASS",
                  "kind": kind, "workers": args.workers, "streams": args.streams,
                  "X": args.concurrent_rows, "R": args.concurrent_r, "S": args.concurrent_s,
                  "wall_seconds": time.monotonic() - started, "results": results, "errors": failures,
                  "metadata": metadata_dict(compiled),
                  "concurrency_scope": "simultaneous host submission; device overlap not assumed"}
    except Exception:
        record = {"rank": rank, "pid": os.getpid(), "status": "FAIL", "error": traceback.format_exc()}
    write_json(args.out / "result.json", record)


def concurrency(args):
    context = mp.get_context("spawn")
    ready = context.Barrier(args.workers)
    config = vars(args).copy()
    processes = [context.Process(target=concurrency_worker, args=(config, rank, ready))
                 for rank in range(args.workers)]
    for process in processes:
        process.start()
    deadline = time.monotonic() + args.process_timeout
    while any(process.is_alive() for process in processes) and time.monotonic() < deadline:
        for process in processes:
            process.join(timeout=0.2)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
    records = []
    for rank in range(args.workers):
        result_file = args.out / f"worker-{rank}" / "result.json"
        if result_file.exists():
            records.append(json.loads(result_file.read_text()))
    summary = {"status": "PASS" if len(records) == args.workers and
               all(r["status"] == "PASS" for r in records) and
               all(p.exitcode == 0 for p in processes) else "FAIL",
               "kind": args.kinds[0], "workers": args.workers, "streams": args.streams,
               "S": args.concurrent_s, "X": args.concurrent_rows, "R": args.concurrent_r,
               "iterations": args.iterations, "exitcodes": [p.exitcode for p in processes], "results": records}
    write_json(args.out / "concurrency.json", summary)
    print(json.dumps(summary, default=str), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "npu"), required=True)
    parser.add_argument("--phase", choices=("inspect", "bench", "check", "mapping", "concurrency"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--kinds", default="original,native,debug")
    parser.add_argument("--splits", default="1,2,4,8,16,32,48,56")
    parser.add_argument("--shapes", default="1x4096,1x32768,1x1048576,1x1048593,1x16777216,4x1048576")
    parser.add_argument("--mapping", choices=("auto", "off"), default="auto")
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=213)
    parser.add_argument("--phases", type=int, default=16)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--op-timeout-ms", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--streams", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--concurrent-s", type=int, default=16)
    parser.add_argument("--concurrent-rows", type=int, default=1)
    parser.add_argument("--concurrent-r", type=int, default=1048593)
    parser.add_argument("--process-timeout", type=int, default=360)
    parser.add_argument("--grid", default="8,1,1", help="One isolated mapping diagnostic; may exceed core count")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    args.kinds = args.kinds.split(",")
    args.splits = list(map(int, args.splits.split(",")))
    if args.backend == "cuda" and args.kinds != ["original"]:
        parser.error("CUDA supports the original helper; the two other helpers require CANN")
    if any(kind not in KINDS for kind in args.kinds):
        parser.error("Unknown helper kind")
    if args.phase == "concurrency":
        if len(args.kinds) != 1:
            parser.error("Each concurrency run isolates one helper")
        result = concurrency(args)
        raise SystemExit(0 if result["status"] == "PASS" else 1)
    _, _, api, args.options = load_stack(args)
    info = manifest(args, api)
    kernels = build_kernels()
    if args.phase == "bench":
        benchmark(args, api, kernels, info)
    elif args.phase == "check":
        correctness(args, api, kernels, info)
    elif args.phase == "mapping":
        splits, y, z = map(int, args.grid.split(","))
        record = {"grid": [splits, y, z], "mapping": args.mapping, "kind": args.kinds[0]}
        try:
            compiled, accuracy = verify_exchange(args, api, kernels[0], splits, y, z, args.kinds[0],
                                                 args.phases, 11, skew=True)
            record.update(status="PASS", accuracy=accuracy, metadata=metadata_dict(compiled))
            save_kernel(args, compiled, "mapping")
        except Exception:
            record.update(status="FAIL", error=traceback.format_exc())
        write_json(args.out / "mapping.json", record)
        print(json.dumps(record, default=str), flush=True)
        raise SystemExit(0 if record["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
