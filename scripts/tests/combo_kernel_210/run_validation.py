"""Collect, execute and finalize one prepared Combo validation directory."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def finalize(root, expected):
    out = root / "evidence"
    result = json.loads((out / "functional_results.json").read_text())
    reports = defaultdict(dict)
    for path in out.glob("test_*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            reports[row["nodeid"]][row["phase"]] = row["outcome"]
    nodes = json.loads((out / "collection.json").read_text())
    assert set(reports) == {row["nodeid"] for row in nodes}
    assert expected is None or len(reports) == expected
    assert all({"setup", "teardown"} <= set(row) for row in reports.values())
    assert all("call" in row or row["setup"] != "passed" for row in reports.values())
    counts = Counter(
        "failed" if "failed" in row.values() else "skipped" if "skipped" in row.values() else "passed"
        for row in reports.values()
    )
    assert dict(counts) == result["instance_counts"]
    assert all(m["job_finished"] and m["completed_instances"] == m["expected_instances"]
               for m in result["methods"])
    assert all(m["returncode"] in (0, 1) for m in result["methods"])
    check = {"all_expected_instances_reported": True, "all_pytest_phases_checked": True,
             "method_count": len(result["methods"]), "instance_count": len(reports),
             "instance_counts": dict(counts)}
    hashes = root / "expected_source_sha256.json"
    if hashes.exists():
        import sysconfig
        site = Path(sysconfig.get_paths()["purelib"])
        before = json.loads(hashes.read_text())
        after = {name: hashlib.sha256((site / name).read_bytes()).hexdigest() for name in before}
        assert before == after, "Installed source changed during validation"
        check.update(installed_source_unchanged=True, post_run_source_sha256=after)
    (out / "integrity_checks.json").write_text(json.dumps(check, indent=2) + "\n")
    return check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Prepared directory with adapted tests and runner")
    parser.add_argument("--devices", default="1,2,3", help="Explicitly authorized physical cards")
    parser.add_argument("--with-f3", action="store_true", help="Enable the prepared F3 overlay plugin")
    parser.add_argument("--expected-instances", type=int)
    parser.add_argument("--detach", action="store_true", help="Run in background; write validation.log/status JSON")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.detach:
        command = [sys.executable, str(Path(__file__).resolve()), *[x for x in sys.argv[1:] if x != "--detach"]]
        with (root / "validation.log").open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (root / "validation.pid").write_text(str(child.pid) + "\n")
        print(json.dumps({"pid": child.pid, "log": str(root / "validation.log")}))
        return
    out = root / "evidence"
    out.mkdir(exist_ok=True)
    assert not (out / "jobs.jsonl").exists(), "Use a fresh directory to preserve earlier evidence"
    devices = [int(x) for x in args.devices.split(",")]
    assert devices and len(devices) == len(set(devices)) and set(devices) <= {1, 2, 3}
    env = dict(os.environ, COMBO_PHYSICAL_DEVICES=args.devices, COMBO_WITH_F3=str(int(args.with_f3)),
               ASCEND_RT_VISIBLE_DEVICES=str(devices[0]), PYTHONPATH=str(root),
               COMBO_SELECTED_MANIFEST=str(root / "combo_npu_a5_selected.json"),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", TORCHINDUCTOR_COMPILE_THREADS="1", OMP_NUM_THREADS="1")
    if args.with_f3:
        assert (root / "combo_baseline_audit.py").exists() and (root / "f3_horizontal_partition.py").exists()
        assert "combo_baseline_audit.install()" in (root / "combo_validation_plugin.py").read_text()
    selected = json.loads((root / "combo_npu_a5_selected.json").read_text())
    state = {"started": time.time(), "physical_devices": devices, "with_f3": args.with_f3}

    def save(stage, **kwargs):
        state.update(stage=stage, **kwargs)
        (root / "validation_status.json").write_text(json.dumps(state, indent=2) + "\n")

    def execute(command, cwd, name):
        with (out / name).open("w") as log:
            subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        save("collecting")
        env["COMBO_COLLECTION_JSON"] = str(out / "collection.json")
        execute([sys.executable, "-m", "pytest", "-p", "combo_validation_plugin", "--collect-only", "-q",
                 *sorted({row["file"] for row in selected})],
                root / "Community-Use-Case-Testing/v2.13.0/inductor_npu", "collection.log")
        nodes = json.loads((out / "collection.json").read_text())
        assert args.expected_instances is None or len(nodes) == args.expected_instances
        env.pop("COMBO_COLLECTION_JSON")
        save("testing", expected_methods=len(selected), expected_instances=len(nodes))
        execute([sys.executable, "-u", "run_combo_selected.py"], root, "driver.log")
        save("finalizing")
        for script in ["export_combo_results.py", "capture_combo_evidence.py"]:
            execute([sys.executable, script], root, script + ".log")
        save("complete", finished=time.time(), result=finalize(root, args.expected_instances))
    except Exception as error:
        save("error", error=repr(error))
        raise


if __name__ == "__main__":
    main()
