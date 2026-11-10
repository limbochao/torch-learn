"""Resume a phase after a card migration, adopting its still-running pytest processes."""

import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from run_validation import finalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "evidence"
    plan = json.loads((root / "migration_plan.json").read_text())
    state = json.loads((root / "validation_status.json").read_text())
    devices = plan["physical_devices"]
    assert devices and 4 not in devices
    state.update(physical_devices=devices, stage="testing",
                 migration=plan.get("description", "physical card 4 released"))

    def save(**fields):
        state.update(fields)
        (root / "validation_status.json").write_text(json.dumps(state, indent=2) + "\n")

    def alive(job):
        try:
            fields = (Path("/proc") / str(job["pid"]) / "stat").read_text().rsplit(")", 1)[1].split()
            return fields[0] != "Z" and fields[19] == job["proc_starttime"]
        except FileNotFoundError:
            return False

    def adopt(job):
        while alive(job):
            time.sleep(30)
        record = job["record"]
        stem = record["file"].removesuffix(".py") + "." + record["name"]
        reports = {}
        for line in (out / (stem + ".jsonl")).read_text().splitlines():
            row = json.loads(line)
            reports.setdefault(row["nodeid"], {})[row["phase"]] = row["outcome"]
        assert set(reports) == set(record["nodes"]), "Adopted process has missing reports"
        assert all({"setup", "teardown"} <= phases.keys() for phases in reports.values())
        assert all("call" in phases or phases["setup"] != "passed" for phases in reports.values())
        xml = ET.parse(out / (stem + ".xml")).getroot()
        suites = list(xml.iter("testsuite"))
        assert sum(int(suite.attrib["tests"]) for suite in suites) == len(record["nodes"])
        failed = any("failed" in phases.values() for phases in reports.values())
        xml_failed = any(int(suite.attrib.get("failures", 0)) or int(suite.attrib.get("errors", 0))
                         for suite in suites)
        assert failed == xml_failed
        record.update(event="finish", returncode=int(failed), finished=datetime.datetime.now().isoformat(),
                      completion_evidence="adopted pytest: complete phase reports and JUnit XML",
                      adopted_pid=job["pid"])
        with (out / "jobs.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("ADOPTED_FINISHED", record["physical_device"], record["name"], record["returncode"], flush=True)

    def worker(device):
        job = next((job for job in plan["adopted_jobs"] if job["physical_device"] == device), None)
        if job:
            adopt(job)
        names = plan["queues"][str(device)]
        if not names:
            return
        env = dict(os.environ, COMBO_PHYSICAL_DEVICES=str(device), ASCEND_RT_VISIBLE_DEVICES=str(device),
                   PYTHONPATH=str(root), COMBO_SELECTED_MANIFEST=str(root / "combo_npu_a5_selected.json"),
                   COMBO_TEST_ROOT=state["test_root"], COMBO_SUMMARY_NAME=f"resumed_card{device}_summary.json",
                   TORCHINDUCTOR_COMPILE_THREADS="1", OMP_NUM_THREADS="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        env.pop("COMBO_EXCLUDE_METHODS", None)
        with (out / f"resumed_card{device}_driver.log").open("w") as log:
            subprocess.run([sys.executable, "-u", "run_combo_selected.py", *names], cwd=root, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        save()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [pool.submit(worker, device) for device in devices]
            for future in futures:
                future.result()
        # A later migration can retain whole lane drivers, which already write
        # their own finish records, while this coordinator runs an interrupted job.
        for job in plan.get("wait_drivers", []):
            while alive(job):
                time.sleep(30)
        save(stage="finalizing")
        env = dict(os.environ, COMBO_TEST_ROOT=state["test_root"])
        for script in ["export_combo_results.py", "capture_combo_evidence.py"]:
            with (out / (script + ".log")).open("w") as log:
                subprocess.run([sys.executable, script], cwd=root, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        save(stage="complete", finished=time.time(), result=finalize(root, state["expected_instances"]))
    except Exception as error:
        save(stage="error", error=repr(error))
        raise


if __name__ == "__main__":
    main()
