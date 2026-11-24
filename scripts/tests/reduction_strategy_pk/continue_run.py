"""Continue a frozen experiment after its coordinator stops, recording card migration."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--adopt", type=int, nargs="*", default=[])
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--max-hbm-percent", type=int, default=95)
    args = parser.parse_args()
    args.source, args.out = args.source.resolve(), args.out.resolve()
    sys.path.insert(0, str(args.source))
    run = importlib.import_module("run")
    manifest = json.loads((args.out / "run.json").read_text())
    for name, digest in manifest["source_sha256"].items():
        if run.sha256(args.source / name) != digest:
            raise RuntimeError(f"Frozen source changed: {name}")
    args.backend, args.smoke, args.resume = manifest["backend"], manifest["smoke"], True
    args.tuning_source = Path(manifest["tuning_source"]) if manifest.get("tuning_source") else None
    if manifest.get("execution_policy"):
        args.execution_policy = manifest["execution_policy"]
        args.resource_wait_timeout = args.execution_policy.get("resource_wait_timeout", 0)
    if args.backend != "npu":
        raise RuntimeError("This continuation supports NPU only; CUDA has separate memory limits")
    if os.getenv("PK_PROFILE_GRAPH", "0") != manifest["profile_graph"]:
        raise RuntimeError("Submission mode differs from original run")
    adopted = []
    for pid in args.adopt:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        tokens = [part.decode() for part in command if part]
        if str(args.source / "run.py") not in tokens or "--job" not in tokens:
            raise RuntimeError(f"Not a worker from the frozen experiment: {pid}")
        job_path = Path(tokens[tokens.index("--job") + 1]).resolve()
        if job_path.parent.parent != args.out / "cases":
            raise RuntimeError(f"Unexpected job path: {job_path}")
        job = json.loads(job_path.read_text())
        adopted.append(dict(pid=pid, job=job, directory=job_path.parent))
    run.write_json(args.out / ("continuation-" + str(int(time.time())) + ".json"),
                   dict(date=datetime.now(timezone.utc).isoformat(), devices=args.devices.split(","),
                        adopted=adopted, source=str(args.source), reason="Shared card resource changes"))

    def finish_existing(item):
        pid, directory = item["pid"], item["directory"]
        while Path(f"/proc/{pid}/cmdline").exists() and not (directory / "result.json").exists():
            try:
                command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            except FileNotFoundError:
                break
            tokens = [part.decode() for part in command if part]
            if not tokens:
                break
            if (str(args.source / "run.py") not in tokens or "--job" not in tokens or
                    Path(tokens[tokens.index("--job") + 1]).resolve() != directory / "job.json"):
                raise RuntimeError(f"Adopted PID no longer belongs to the experiment: {pid}")
            progress = directory / "active.json"
            if time.time() - progress.stat().st_mtime > args.timeout:
                if os.getpgid(pid) != pid:
                    raise RuntimeError("Adopted worker is not its own process group")
                os.killpg(pid, signal.SIGKILL)
                run.write_json(directory / "result.json", dict(case=item["job"]["case"],
                               backend=args.backend, status="TIMEOUT", returncode=-9))
                break
            time.sleep(1)
        if not (directory / "result.json").exists():
            run.write_json(directory / "result.json", dict(case=item["job"]["case"], backend=args.backend,
                           status="PROCESS_ERROR", error="Adopted worker exited without a terminal result"))
        print("adopted", item["job"]["case"]["id"], "finished", flush=True)

    pending = Queue()
    reserved = {item["job"]["case"]["id"] for item in adopted}
    for case in manifest["cases"]:
        if case["id"] not in reserved and not (args.out / "cases" / case["id"] / "result.json").exists():
            pending.put(case)

    def queue(device):
        for item in adopted:
            if item["job"]["device"] == device:
                finish_existing(item)
        while True:
            if pending.empty():
                return
            try:
                usage = subprocess.run(["npu-smi", "info", "-i", device, "-t", "usages"],
                                       capture_output=True, text=True, timeout=20)
                rates = [int(line.split(":", 1)[1].strip()) for line in usage.stdout.splitlines()
                         if "HBM Usage Rate(%)" in line]
                if not rates or rates[0] > args.max_hbm_percent:
                    print("waiting for free HBM", device, rates, flush=True)
                    time.sleep(30)
                    continue
            except subprocess.TimeoutExpired:
                time.sleep(30)
                continue
            try:
                case = pending.get_nowait()
            except Empty:
                return
            run.launch_case(args, case, device, False)

    devices = args.devices.split(",")
    with ThreadPoolExecutor(max_workers=len(devices) + len(adopted)) as executor:
        futures = [executor.submit(queue, device) for device in devices]
        futures += [executor.submit(finish_existing, item) for item in adopted if item["job"]["device"] not in devices]
        for future in futures:
            future.result()
    results = [json.loads((args.out / "cases" / case["id"] / "result.json").read_text())
               for case in manifest["cases"]]
    run.write_json(args.out / "results.json", results)


if __name__ == "__main__":
    main()
