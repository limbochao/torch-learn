"""Validate x_grid_barrier before comparing it with TA native synchronization."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

from probe import event_time, parse_shapes


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--shapes", default="1x4096,1x32768,1x1048576,8x1048576,1x1048593")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(args.out.resolve() / "triton_cache"))

    import torch
    import torch_npu
    import triton
    import triton.language as tl
    from triton.backends.ascend.utils import _get_npucompiler_path
    from triton.language.extra.cann import extension as ext
    from torch._inductor.runtime.triton_helpers import x_grid_barrier

    globals().update(triton=triton, tl=tl, ext=ext, x_grid_barrier=x_grid_barrier)

    @triton.jit
    def checked_barrier(Sem, MAX_POLLS: tl.constexpr):
        # Diagnostic copy of x_grid_barrier, with a finite polling budget.
        tl.debug_barrier()
        one = tl.full((), 1, tl.uint32)
        expected = tl.num_programs(0).to(tl.uint32)
        nb = tl.where(tl.program_id(0) == 0, 0x80000000 - (expected - one), one)
        old = tl.atomic_add(Sem, nb, sem="release")
        flipped = False
        polls = 0
        while (not flipped) & (polls < MAX_POLLS):
            current = tl.atomic_add(Sem, 0, sem="acquire")
            flipped = ((old ^ current) & 0x80000000) != 0
            polls += 1
        tl.debug_barrier()
        return flipped

    globals()["checked_barrier"] = checked_barrier

    @triton.jit
    def visibility(W, Y, Sem, OK, S: tl.constexpr, X: tl.constexpr,
                   PHASES: tl.constexpr, KIND: tl.constexpr):
        pid = tl.program_id(0)
        row = tl.program_id(1)
        sem = Sem.to(tl.pointer_type(tl.uint32)) + row
        for phase in range(PHASES):
            tl.store(W + row * S + pid, pid + 1 + phase)
            if KIND == 0:
                ready = checked_barrier(sem, 4096)
            elif KIND == 1:
                x_grid_barrier(sem)
                ready = True
            else:
                ext.sync_block_all("all_vector", 0)
                ready = True
            peers = tl.arange(0, S)
            values = tl.load(W + row * S + peers)
            offset = (phase * X + row) * S + pid
            tl.store(Y + offset, tl.sum(values, 0))
            tl.store(OK + offset, ready)
            # All readers finish before the next phase overwrites the workspace.
            if KIND == 0:
                done = checked_barrier(sem, 4096)
                tl.store(OK + offset, ready & done)
            elif KIND == 1:
                x_grid_barrier(sem)
            else:
                ext.sync_block_all("all_vector", 1)

    @triton.jit
    def reduction(A, W, Y, Sem, R: tl.constexpr, S: tl.constexpr,
                  B: tl.constexpr, KIND: tl.constexpr):
        shard = tl.program_id(0)
        row = tl.program_id(1)
        chunk = tl.cdiv(R, S * B) * B
        acc = tl.full((B,), 0, tl.float32)
        for start in range(shard * chunk, tl.minimum((shard + 1) * chunk, R), B):
            offsets = start + tl.arange(0, B)
            acc += tl.load(A + row * R + offsets, offsets < R, other=0)
        partial = tl.sum(acc, 0)
        if KIND == 0:
            tl.store(Y + row, partial)
        else:
            tl.store(W + row * S + shard, partial)
            if KIND == 1:
                x_grid_barrier(Sem.to(tl.pointer_type(tl.uint32)) + row)
            else:
                ext.sync_block_all("all_vector", 0)
            peers = tl.arange(0, S)
            values = tl.load(W + row * S + peers)
            tl.store(Y + row, tl.sum(values, 0), shard == 0)

    torch.npu.set_device(args.device)
    torch.manual_seed(1234)
    physical = ctypes.c_int32(-1)
    library = ctypes.CDLL("libascendcl.so")
    get_physical = library.aclrtGetPhyDevIdByUserDevId
    get_physical.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
    get_physical.restype = ctypes.c_int32
    status = get_physical(args.device, ctypes.byref(physical))
    if status != 0 or physical.value != args.physical_device:
        raise RuntimeError(f"Unexpected device mapping: status={status}, physical={physical.value}")
    properties = torch.npu.get_device_properties(args.device)
    compiler, compiler_env = _get_npucompiler_path()
    version = subprocess.run([compiler, "--version"], env=compiler_env,
                             capture_output=True, text=True, timeout=30)
    helper_source = inspect.getsource(x_grid_barrier.fn)
    (args.out / "x_grid_barrier.py").write_text(helper_source)
    write_json(args.out / "manifest.json", {
        "argv": sys.argv, "cwd": str(Path.cwd()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"], text=True
        ).strip(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "torch-npu", "triton", "triton-ascend")},
        "torch_npu_version": torch_npu.__version__,
        "visible_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
        "logical_device": args.device, "physical_device": physical.value, "properties": str(properties),
        "compiler": compiler, "compiler_version": version.stdout + version.stderr,
        "helper_source_sha256": hashlib.sha256(helper_source.encode()).hexdigest(),
    })
    options = {"compile_mode": "simd", "num_warps": 4, "launch_cooperative_grid": True}

    def save_ir(kernel, name):
        for stage in ("ttir", "ttadapter", "llir"):
            value = kernel.asm.get(stage)
            if isinstance(value, str):
                (args.out / f"{name}.{stage}").write_text(value)
        write_json(args.out / f"{name}.metadata.json", kernel.metadata._asdict())

    checks = []
    # The exact helper is called only after all bounded visibility checks pass.
    for kind in (0, 1, 2):
        for rows, programs in ((1, 1), (1, 2), (1, 8), (1, 32), (4, 8)):
            if rows * programs > properties.vector_core_num:
                raise ValueError("All test programs must fit within the vector core count")
            workspace = torch.empty((rows, programs), device="npu", dtype=torch.float32)
            output = torch.empty((8, rows, programs), device="npu", dtype=torch.float32)
            ok = torch.empty((8, rows, programs), device="npu", dtype=torch.int32)
            semaphores = torch.zeros(rows, device="npu", dtype=torch.int32)
            kernel = visibility[(programs, rows)](
                workspace, output, semaphores, ok, S=programs, X=rows, PHASES=8, KIND=kind, **options)
            torch.npu.synchronize()
            expected = torch.tensor([programs * (programs + 1) / 2 + p * programs for p in range(8)])
            expected = expected[:, None, None].expand_as(output.cpu())
            passed = bool(ok.cpu().all()) and torch.equal(output.cpu(), expected)
            name = f"visibility-k{kind}-x{rows}-s{programs}"
            save_ir(kernel, name)
            record = {"kind": kind, "X": rows, "S": programs, "phases": 8, "correct": passed,
                      "ready": ok.cpu().tolist(), "observed": output.cpu().tolist()}
            checks.append(record)
            write_json(args.out / "visibility.json", checks)
            print(name, "PASS" if passed else "FAIL", flush=True)
            if not passed:
                raise SystemExit("Functional validation failed; no performance samples collected")

    measurements = []
    for rows, count in parse_shapes(args.shapes):
        ratio = max(1, properties.vector_core_num // rows)
        splits = min(64, 1 << (ratio.bit_length() - 1))
        if rows * splits > properties.vector_core_num:
            raise ValueError("All cooperative programs must fit within the vector core count")
        data = torch.randn((rows, count), device="npu", dtype=torch.float32)
        workspace = torch.empty((rows, splits), device="npu", dtype=torch.float32)
        output = torch.empty(rows, device="npu", dtype=torch.float32)
        semaphores = torch.zeros(rows, device="npu", dtype=torch.int32)
        expected = data.sum(-1).cpu()
        for kind, strategy in ((0, "single"), (1, "x_grid_barrier"), (2, "native_all_vector")):
            actual_splits = 1 if kind == 0 else splits

            def invoke():
                return reduction[(actual_splits, rows)](
                    data, workspace, output, semaphores, R=count, S=actual_splits, B=1024, KIND=kind, **options)

            kernel = invoke()
            torch.npu.synchronize()
            torch.testing.assert_close(output.cpu(), expected, atol=2e-2, rtol=2e-4)
            save_ir(kernel, f"sum-{strategy}-x{rows}-r{count}")
            record = {"strategy": strategy, "X": rows, "R": count, "S": actual_splits, "B": 1024,
                      "options": options, "correct": True, **event_time(torch, invoke, graph=True)}
            measurements.append(record)
            write_json(args.out / "bench.json", measurements)
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
