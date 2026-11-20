"""Measure the atomic and synchronization components of the A5 x-grid slow case."""

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

from probe import event_time


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--programs", default="1,2,8,32")
    parser.add_argument("--debug-ir", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(args.out.resolve() / "triton_cache"))
    os.environ.setdefault("TRITON_DUMP_DIR", str(args.out.resolve() / "compiler_dump"))

    import torch
    import torch_npu
    import triton
    import triton.language as tl
    import triton.backends.ascend.compiler as backend_compiler
    from triton.backends.ascend.utils import _get_npucompiler_path
    from triton.language.extra.cann import extension as ext
    from torch._inductor.runtime.triton_helpers import x_grid_barrier

    globals().update(tl=tl, triton=triton, ext=ext, x_grid_barrier=x_grid_barrier)

    @triton.jit
    def atomic_component(A, Y, KIND: tl.constexpr, SHARED: tl.constexpr,
                         DELTA: tl.constexpr, ORDER: tl.constexpr):
        pid = tl.program_id(0)
        address = A + (0 if SHARED else pid * 32)
        if KIND == 0:
            old = tl.load(address, volatile=True)
        else:
            old = tl.atomic_add(address, DELTA, sem=ORDER)
        if KIND == 2 or KIND == 0:
            tl.store(Y + pid, old)
        else:
            tl.store(Y + pid, pid)

    @triton.jit
    def barrier_component(W, Y, Sem, KIND: tl.constexpr, PHASES: tl.constexpr,
                          POLICY: tl.constexpr):
        pid = tl.program_id(0)
        for phase in range(PHASES):
            tl.store(W + pid, phase + pid)
            if KIND == 1:
                x_grid_barrier(Sem.to(tl.pointer_type(tl.uint32)))
            elif KIND == 2:
                ext.sync_block_all("all_vector", 0)
        tl.store(Y + pid, tl.load(W + pid))

    @triton.jit
    def count_polls(Sem, Counts, OK):
        pid = tl.program_id(0)
        sem = Sem.to(tl.pointer_type(tl.uint32))
        tl.debug_barrier()
        expected = tl.num_programs(0).to(tl.uint32)
        one = tl.full((), 1, tl.uint32)
        increment = tl.where(pid == 0, 0x80000000 - (expected - one), one)
        old = tl.atomic_add(sem, increment, sem="release")
        polls = 0
        flipped = False
        while (not flipped) & (polls < 4096):
            current = tl.atomic_add(sem, 0, sem="acquire")
            flipped = ((old ^ current) & 0x80000000) != 0
            polls += 1
        tl.debug_barrier()
        tl.store(Counts + pid, polls)
        tl.store(OK + pid, flipped)

    torch.npu.set_device(0)
    physical = ctypes.c_int32(-1)
    get_physical = ctypes.CDLL("libascendcl.so").aclrtGetPhyDevIdByUserDevId
    get_physical.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
    get_physical.restype = ctypes.c_int32
    status = get_physical(0, ctypes.byref(physical))
    if status != 0 or physical.value != args.physical_device:
        raise RuntimeError(f"Unexpected device mapping: {status=}, physical={physical.value}")
    properties = torch.npu.get_device_properties(0)
    programs = [int(value) for value in args.programs.split(",")]
    if not all(0 < p <= properties.vector_core_num for p in programs):
        raise ValueError("All participants must fit within the vector core count")
    compiler, env = _get_npucompiler_path()
    version = subprocess.run([compiler, "--version"], env=env, capture_output=True, text=True, timeout=30)
    helper = inspect.getsource(x_grid_barrier.fn)
    (args.out / "x_grid_barrier.py").write_text(helper)
    save_json(args.out / "manifest.json", {
        "git_commit": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "packages": {n: importlib.metadata.version(n) for n in ("torch", "torch-npu", "triton", "triton-ascend")},
        "torch_npu_version": torch_npu.__version__, "visible_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
        "physical_device": physical.value, "properties": str(properties),
        "compiler": compiler, "compiler_version": version.stdout + version.stderr,
        "helper_source_sha256": hashlib.sha256(helper.encode()).hexdigest(),
        "argv": vars(args), "timing": "32 captured calls x 10 replays, 5 samples, median",
    })
    options = {"compile_mode": "simd", "num_warps": 4,
               "launch_cooperative_grid": True, "debug": args.debug_ir}
    measurements = []

    def save_kernel(kernel, name):
        for stage in ("ttir", "ttadapter"):
            if isinstance(kernel.asm.get(stage), str):
                (args.out / f"{name}.{stage}").write_text(kernel.asm[stage])
        save_json(args.out / f"{name}.metadata.json", kernel.metadata._asdict())

    def measure(name, invoke, check, details):
        kernel = invoke()
        torch.npu.synchronize()
        check()
        save_kernel(kernel, name)
        timing = event_time(torch, invoke, graph=True)
        record = {"name": name, **details, "correct": True, **timing}
        measurements.append(record)
        save_json(args.out / "components.json", measurements)
        print(json.dumps(record), flush=True)

    poll_records = []
    for p in programs:
        for kind, shared, delta, order, label in (
            (0, True, 0, "acquire", "load"),
            (1, True, 1, "release", "atomic_dead_shared"),
            (2, True, 1, "release", "atomic_old_shared"),
            (2, False, 1, "release", "atomic_old_private"),
            (2, True, 0, "acquire", "atomic_poll_shared"),
        ):
            data = torch.zeros(p * 32, device="npu", dtype=torch.int32)
            output = torch.empty(p, device="npu", dtype=torch.int32)

            def invoke_atomic():
                return atomic_component[(p,)](
                    data, output, KIND=kind, SHARED=shared, DELTA=delta, ORDER=order, **options)

            def check_atomic():
                expected = torch.zeros(p * 32, dtype=torch.int32)
                if delta:
                    if shared:
                        expected[0] = p
                    else:
                        expected[::32] = 1
                torch.testing.assert_close(data.cpu(), expected, rtol=0, atol=0)
                if kind == 2:
                    expected_old = (torch.arange(p, dtype=torch.int32) if shared and delta
                                    else torch.zeros(p, dtype=torch.int32))
                    torch.testing.assert_close(output.cpu().sort().values, expected_old, rtol=0, atol=0)

            measure(f"{label}-p{p}", invoke_atomic, check_atomic,
                    {"programs": p, "kind": label, "shared": shared, "delta": delta, "order": order})

        for kind, phases, policy in ((0, 1, 0), (1, 1, 0), (1, 4, 0), (2, 1, 0), (2, 4, 0), (2, 1, 1)):
            workspace = torch.empty(p, device="npu", dtype=torch.int32)
            output = torch.empty_like(workspace)
            sem = torch.zeros(1, device="npu", dtype=torch.int32)

            def invoke_barrier():
                return barrier_component[(p,)](
                    workspace, output, sem, KIND=kind, PHASES=phases, POLICY=policy, **options)

            def check_barrier():
                expected = torch.arange(p, dtype=torch.int32) + phases - 1
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

            # Diagnostic control only: disable the compiler policy in this Python process.
            # POLICY changes the JIT key, and the output cache is exclusive to this experiment.
            saved_policy = backend_compiler._is_auto_map_parallel_blocks_enabled
            try:
                if policy:
                    backend_compiler._is_auto_map_parallel_blocks_enabled = lambda: False
                measure(f"barrier-k{kind}-p{p}-n{phases}-policy{policy}", invoke_barrier, check_barrier,
                        {"programs": p, "kind": kind, "phases": phases, "disable_blockify": bool(policy)})
            finally:
                backend_compiler._is_auto_map_parallel_blocks_enabled = saved_policy

        sem = torch.zeros(1, device="npu", dtype=torch.int32)
        counts = torch.empty(p, device="npu", dtype=torch.int32)
        ok = torch.empty_like(counts)
        kernel = count_polls[(p,)](sem, counts, ok, **options)
        torch.npu.synchronize()
        if not bool(ok.cpu().all()):
            raise RuntimeError("Bounded poll-count diagnostic did not complete")
        save_kernel(kernel, f"poll-counts-p{p}")
        poll_records.append({"programs": p, "counts": counts.cpu().tolist(), "correct": True})
        save_json(args.out / "poll-counts.json", poll_records)


if __name__ == "__main__":
    main()
