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
import shutil
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
        writer = csv.writer(stream, lineterminator="\n")
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


def select_instances(nodes, limit):
    """Select a deterministic spread, including both ends when limit is two."""
    nodes = sorted(nodes)
    if not limit or len(nodes) <= limit:
        return nodes
    if limit == 1:
        return nodes[:1]
    return [nodes[index * (len(nodes) - 1) // (limit - 1)] for index in range(limit)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--test-commit", required=True)
    parser.add_argument("--npu-fix-commit", help="Python overlay commit applied to the installed NPU package")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--cache-root", type=Path, help="Fresh, isolated Inductor cache outside the evidence directory")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--representatives", action="store_true")
    parser.add_argument("--max-instances-per-method", type=int, default=0,
                        help="Sample up to this many instances from every selected source method; zero runs all")
    parser.add_argument("--reuse", type=Path, help="Reuse verified, completed jobs from this output directory")
    parser.add_argument("--rerun-files", nargs="*", default=[], help="Files explicitly excluded from reuse")
    parser.add_argument("--foreach-functional", action="store_true", help="Record the NPU functional test policy")
    parser.add_argument("--layout", type=Path, help="Original table row mapping, including nested functions")
    args = parser.parse_args()
    if args.max_instances_per_method < 0:
        parser.error("--max-instances-per-method must be nonnegative")
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
    if args.layout:
        shutil.copy2(args.layout, output / "table_rows.json")
    state = {"stage": "collecting", "started": time.time(), "devices": devices,
             "test_root": str(base), "test_commit": args.test_commit, "python": sys.executable,
             "custom_pytest_plugin": False, "methods": len(selected)}
    state.update(rerun_files=args.rerun_files, foreach_functional=args.foreach_functional,
                 npu_fix_commit=args.npu_fix_commit, max_instances_per_method=args.max_instances_per_method)
    if args.cache_root:
        cache = args.cache_root.resolve()
        assert cache != output and output not in cache.parents, "Keep the compilation cache outside evidence"
        cache.mkdir(parents=True, exist_ok=False)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
        state["cache_root"] = str(cache)
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
            full_nodes = sorted(nodes)
            if "only_nodes" in row:
                requested = row["only_nodes"]
                if not requested or len(requested) != len(set(requested)):
                    raise RuntimeError(f"Empty or duplicated instance selection: {row['name']}")
                missing = set(requested) - set(full_nodes)
                if missing:
                    raise RuntimeError(f"Selected instances were not collected: {sorted(missing)}")
                nodes = sorted(requested)
            else:
                nodes = select_instances(full_nodes, args.max_instances_per_method)
            job = {**row, "index": index, "nodes": nodes, "status": "PENDING"}
            if nodes != full_nodes:
                job.update(full_nodes=full_nodes, sample_limit=args.max_instances_per_method)
            jobs.append(job)
        nodes = [node for job in jobs for node in job["nodes"]]
        assert len(nodes) == len(set(nodes)), "Overlapping source method selection"
        save(output / "collection.json", jobs)
        (output / "nodeids.txt").write_text("\n".join(sorted(nodes)) + "\n")
        state.update(stage="collected" if args.collect_only else "testing", instances=len(nodes))
        state["full_instances"] = sum(len(job.get("full_nodes", job["nodes"])) for job in jobs)
        save(output / "status.json", state)
        if args.collect_only:
            return
        if args.reuse:
            previous = args.reuse.resolve()
            old_state = json.loads((previous / "status.json").read_text())
            assert old_state["stage"] == "complete" and old_state["installed_source_unchanged"]
            assert json.loads((previous / "installed_sha256.json").read_text()) == before
            for key in ("torch_git", "torch_npu_git", "triton"):
                assert old_state[key] == state[key], f"Changed environment: {key}"
            old_jobs = {(j["file"], j["class"], j["name"]): j
                        for j in json.loads((previous / "jobs.json").read_text())}
            for job in jobs:
                if job["file"] in args.rerun_files:
                    continue
                old = old_jobs.get((job["file"], job["class"], job["name"]))
                if old is None:
                    continue
                assert (base / job["file"]).read_bytes() == (
                    Path(old_state["test_root"]) / job["file"]).read_bytes(), "Changed test source"
                assert set(old["nodes"]) == set(job["nodes"]) == set(old["reports"])
                assert old["status"] in ("PASS", "FAIL", "SKIP") and old["returncode"] in (0, 1)
                old_stem = f"{old['index']:02d}-{old['name']}"
                new_stem = f"{job['index']:02d}-{job['name']}"
                for extension in (".log", ".xml"):
                    shutil.copy2(previous / (old_stem + extension), output / (new_stem + extension))
                index = job["index"]
                job.update(old, index=index, reused_from=str(previous))
            state.update(reused_methods=sum("reused_from" in job for job in jobs),
                         completed=sum("reused_from" in job for job in jobs))
            save(output / "jobs.json", jobs)
            save(output / "status.json", state)
            write_table(output, jobs)
        lock = threading.Lock()
        lanes = {device: [] for device in devices}
        loads = dict.fromkeys(devices, 0)
        for job in sorted(jobs, key=lambda item: len(item["nodes"]), reverse=True):
            if "reused_from" in job:
                continue
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
