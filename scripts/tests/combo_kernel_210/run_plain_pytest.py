"""Collect selected source methods and run ordinary pytest without a custom plugin."""

import argparse
import ast
from collections import Counter
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def fingerprint(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def execute(command, cwd, env, log, timeout):
    started = time.time()
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            returncode = 124
    return returncode, round(time.time() - started, 2)


def read_reports(path, nodes):
    reports = {}
    if not path.exists():
        return reports
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return reports
    for case in root.iter("testcase"):
        matches = [node for node in nodes if node.split("::", 2)[-1] == case.get("name")
                   and case.get("classname", "").endswith("." + node.split("::")[1])]
        if len(matches) != 1:
            raise RuntimeError(f"Unexpected JUnit testcase: {case.attrib}")
        problems = [child for child in case if child.tag in ("failure", "error", "skipped")]
        status = "FAIL" if any(c.tag == "failure" for c in problems) else (
            "ERROR" if any(c.tag == "error" for c in problems) else "SKIP" if problems else "PASS")
        node = matches[0]
        previous = reports.get(node)
        if previous and previous["status"] != "PASS":
            continue
        reports[node] = {"status": status, "seconds": case.get("time"),
                         "detail": "\n".join((c.get("message", "") + "\n" + (c.text or ""))
                                               for c in problems)}
    return reports


def write_table(output, jobs):
    with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["测试文件", "函数名", "最新结果", "实例结果", "当前分析"])
        for job in jobs:
            reports = job.get("reports", {})
            counts = Counter(row["status"] for row in reports.values())
            detail = next((row["detail"] for row in reports.values() if row["status"] != "PASS"), "")
            # Preserve the actual exception; classification/root cause belongs to later analysis.
            analysis = "—" if job.get("status") == "PASS" else detail.split("\n\n")[0][:600]
            if not analysis:
                analysis = job.get("error", "待执行")
            writer.writerow([job["file"], job["name"], job.get("status", "待执行"),
                             "; ".join(f"{key} {value}" for key, value in sorted(counts.items())), analysis])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--test-commit", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--representatives", action="store_true")
    args = parser.parse_args()
    base, output = args.test_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    devices = [int(value) for value in args.devices.split(",")]
    assert devices and len(set(devices)) == len(devices)
    selected = json.loads(args.manifest.read_text())
    if args.representatives:
        preferred = {"test_combo_kernels.py": "test_activation_functions",
                     "test_foreach.py": "test_single_list",
                     "test_torchinductor.py": "test_combo_kernel_filter_cpu"}
        selected = [next((r for r in selected if r["file"] == name and
                          r["name"] == preferred.get(name)), next(r for r in selected if r["file"] == name))
                    for name in sorted({r["file"] for r in selected})]
    save(output / "manifest.json", selected)
    state = {"stage": "collecting", "started": time.time(), "devices": devices,
             "test_root": str(base), "test_commit": args.test_commit, "python": sys.executable,
             "custom_pytest_plugin": False, "methods": len(selected)}
    save(output / "status.json", state)
    try:
        import torch
        import torch_npu
        import triton

        sources = list(Path(torch_npu.__file__).parent.joinpath("_inductor").rglob("*.py"))
        sources += list(Path(torch.__file__).parent.joinpath("_inductor").rglob("*.py"))
        before = fingerprint(sources)
        save(output / "installed_sha256.json", before)
        state.update(torch=torch.__version__, torch_git=torch.version.git_version,
                     torch_npu=torch_npu.__version__, torch_npu_git=torch_npu.version.git_version,
                     triton=triton.__version__)
        save(output / "status.json", state)
        catalog, classes = {}, {}
        for filename in sorted({row["file"] for row in selected}):
            tree = ast.parse((base / filename).read_text(encoding="utf-8-sig"))
            for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
                classes[filename, cls.name] = [node.name for node in cls.body
                                               if isinstance(node, ast.FunctionDef)]
            log = output / (filename + ".collection.log")
            env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(devices[0]))
            command = [sys.executable, "-m", "pytest", "--collect-only", "-q", "--color=no", filename]
            code, _ = execute(command, base, env, log, args.timeout)
            if code:
                raise RuntimeError(f"Collection failed: {filename}, exit={code}; see {log.name}")
            catalog[filename] = [line.strip() for line in log.read_text(errors="replace").splitlines()
                                 if line.startswith(filename + "::")]
        jobs = []
        for index, row in enumerate(selected):
            allowed_classes = {row["class"], row["class"] + "NPU", row["class"] + "PRIVATEUSE1"}
            if row["class"] == "CommonTemplate":
                allowed_classes.add("GPUTests")
            source_names = classes[row["file"], row["class"]]
            nodes = []
            for node in catalog[row["file"]]:
                _, cls, method = node.split("::", 2)
                candidates = [name for name in source_names if method == name or
                              method.startswith(name + "_") or method.startswith(name + "[")]
                if cls in allowed_classes and candidates and max(candidates, key=len) == row["name"]:
                    nodes.append(node)
            if not nodes:
                raise RuntimeError(f"No collected instances for {row['file']}::{row['class']}::{row['name']}")
            if args.representatives:
                nodes = nodes[:1]
            jobs.append({**row, "index": index, "nodes": sorted(nodes), "status": "PENDING"})
        nodes = [node for job in jobs for node in job["nodes"]]
        assert len(nodes) == len(set(nodes)), "Overlapping source method selection"
        save(output / "collection.json", jobs)
        (output / "nodeids.txt").write_text("\n".join(sorted(nodes)) + "\n")
        state.update(stage="collected" if args.collect_only else "testing", instances=len(nodes))
        save(output / "status.json", state)
        if args.collect_only:
            return
        lock = threading.Lock()
        lanes = {device: [] for device in devices}
        loads = dict.fromkeys(devices, 0)
        for job in sorted(jobs, key=lambda item: len(item["nodes"]), reverse=True):
            device = min(loads, key=loads.get)
            lanes[device].append(job)
            loads[device] += len(job["nodes"]) + 3

        def worker(device, assigned):
            for job in assigned:
                stem = f"{job['index']:02d}-{job['name']}"
                env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(device))
                xml = output / (stem + ".xml")
                command = [sys.executable, "-m", "pytest", "-vs", "-ra", "--color=no", "--tb=short",
                           "--junitxml=" + str(xml), *job["nodes"]]
                job.update(device=device, command=command, status="RUNNING", started=time.time())
                with lock:
                    save(output / "jobs.json", jobs)
                code, elapsed = execute(command, base, env, output / (stem + ".log"), args.timeout)
                reports = read_reports(xml, job["nodes"])
                counts = Counter(row["status"] for row in reports.values())
                status = "ERROR" if code not in (0, 1) or len(reports) != len(job["nodes"]) else (
                    "FAIL" if counts["FAIL"] or counts["ERROR"] else "SKIP" if counts["SKIP"] else "PASS")
                job.update(status=status, returncode=code, seconds=elapsed, reports=reports, finished=time.time())
                with lock:
                    save(output / "jobs.json", jobs)
                    write_table(output, jobs)
                    state.update(completed=sum(j["status"] not in ("PENDING", "RUNNING") for j in jobs))
                    save(output / "status.json", state)
                print(f"{device}: {job['name']}: {status} ({len(reports)}/{len(job['nodes'])})", flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [pool.submit(worker, device, assigned) for device, assigned in lanes.items()]
            for future in futures:
                future.result()
        after = fingerprint(sources)
        if before != after:
            raise RuntimeError("Installed Inductor Python sources changed during the run")
        state.update(stage="complete", finished=time.time(), installed_source_unchanged=True,
                     method_results=dict(Counter(j["status"] for j in jobs)),
                     instance_results=dict(Counter(r["status"] for j in jobs for r in j["reports"].values())))
        save(output / "status.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
    except Exception as error:
        state.update(stage="error", error=repr(error), finished=time.time())
        save(output / "status.json", state)
        raise


if __name__ == "__main__":
    main()
