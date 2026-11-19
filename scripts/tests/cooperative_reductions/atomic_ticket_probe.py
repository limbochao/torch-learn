"""Check the contended atomic return values required by x_grid_barrier; no spinning."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--programs", default="1,2,8,32")
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(args.out.resolve() / "triton_cache"))

    import torch
    import torch_npu
    import triton
    import triton.language as tl
    from torch._inductor.runtime.triton_helpers import x_grid_barrier

    globals().update(triton=triton, tl=tl)

    @triton.jit
    def tickets(Counter, Old, ORDER: tl.constexpr):
        pid = tl.program_id(0)
        previous = tl.atomic_add(Counter, 1, sem=ORDER, scope="gpu")
        tl.store(Old + pid, previous)

    torch.npu.set_device(args.device)
    properties = torch.npu.get_device_properties(args.device)
    helper_source = inspect.getsource(x_grid_barrier.fn)
    manifest = {
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "triton": triton.__version__,
        "visible_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
        "logical_device": args.device,
        "properties": str(properties),
        "compile_mode": "simd",
        "num_warps": 4,
        "launch_cooperative_grid": True,
        "helper_source_sha256": hashlib.sha256(helper_source.encode()).hexdigest(),
        "git_commit": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "x_grid_barrier.py").write_text(helper_source)
    records = []
    for programs in map(int, args.programs.split(",")):
        if not 1 <= programs <= properties.vector_core_num:
            raise ValueError("program count must be within the vector core count")
        for order in ("release", "acquire", "acq_rel"):
            counter = torch.zeros(1, device="npu", dtype=torch.int32)
            previous = torch.empty(programs, device="npu", dtype=torch.int32)
            trials = []
            for repeat in range(args.repeats):
                counter.zero_()
                compiled = tickets[(programs,)](
                    counter, previous, ORDER=order, compile_mode="simd", num_warps=4,
                    launch_cooperative_grid=True,
                )
                torch.npu.synchronize()
                observed = previous.cpu().tolist()
                total = counter.item()
                passed = total == programs and sorted(observed) == list(range(programs))
                trials.append({"repeat": repeat, "counter": total, "old_values": observed, "correct": passed})
                if repeat == 0:
                    for stage in ("ttir", "ttadapter", "llir"):
                        value = compiled.asm.get(stage)
                        if isinstance(value, str):
                            (args.out / f"tickets-p{programs}-{order}.{stage}").write_text(value)
            record = {"programs": programs, "order": order,
                      "correct": all(trial["correct"] for trial in trials), "trials": trials}
            records.append(record)
            (args.out / "results.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record), flush=True)
    raise SystemExit(0 if all(record["correct"] for record in records) else 1)


if __name__ == "__main__":
    main()
