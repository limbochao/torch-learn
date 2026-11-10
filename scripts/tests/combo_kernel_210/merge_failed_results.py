"""Update an existing result snapshot with failed-only reruns, preserving validation provenance."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil

from run_plain_pytest import save


def provenance(directory, state):
    return {"artifact": directory.name, "test_commit": state["test_commit"],
            "npu_fix_commit": state.get("npu_fix_commit")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads((args.reference / "status.json").read_text())
    assert state["stage"] == "complete" and state["installed_source_unchanged"]
    jobs = json.loads((args.reference / "jobs.json").read_text())
    previous = provenance(args.reference, state)
    for job in jobs:
        job.pop("rerun_nodes", None)
        for report in job["reports"].values():
            report.setdefault("validation", previous)
    by_method = {(j["file"], j["class"], j["name"]): j for j in jobs}
    seen = set()
    rerun_results = []
    fingerprints = []
    for directory in args.rerun:
        current = json.loads((directory / "status.json").read_text())
        assert current["stage"] == "complete" and current["installed_source_unchanged"]
        for key in ("torch_git", "torch_npu_git", "triton"):
            assert current[key] == state[key], f"Different base environment: {key}"
        fingerprints.append(json.loads((directory / "installed_sha256.json").read_text()))
        source = provenance(directory, current)
        for new in json.loads((directory / "jobs.json").read_text()):
            job = by_method[new["file"], new["class"], new["name"]]
            assert set(new["nodes"]) == set(new["reports"]), "Incomplete rerun reports"
            assert set(new["nodes"]) <= set(job["nodes"]), "Rerun changed the original selection"
            for node, report in new["reports"].items():
                assert node not in seen, f"Instance rerun twice in the same merge: {node}"
                assert job["reports"][node]["status"] != "PASS", f"Reran a historical PASS: {node}"
                seen.add(node)
                job["reports"][node] = {**copy.deepcopy(report), "validation": source}
            job["rerun_nodes"] = new["nodes"]
            job.setdefault("rerun_history", []).append({**source, "nodes": new["nodes"],
                                                       "command": new["command"], "device": new["device"]})
            statuses = {report["status"] for report in job["reports"].values()}
            job["status"] = ("ERROR" if "ERROR" in statuses else "FAIL" if "FAIL" in statuses
                             else "SKIP" if "SKIP" in statuses else "PASS")
        rerun_results.append({**source, "method_results": current["method_results"],
                              "instance_results": current["instance_results"]})
        state.update({key: current[key] for key in ("test_commit", "npu_fix_commit", "finished")})
    assert all(item == fingerprints[0] for item in fingerprints), "Rerun phases used different installed sources"
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("manifest.json", "table_rows.json", "full-collection.json", "foreach-sampling.json"):
        if (args.reference / name).exists():
            shutil.copy2(args.reference / name, args.output / name)
    state.update(failed_only_merge=True, reference_artifact=args.reference.name, rerun_results=rerun_results,
                 method_results=dict(Counter(j["status"] for j in jobs)),
                 instance_results=dict(Counter(r["status"] for j in jobs for r in j["reports"].values())))
    state.pop("cache_root", None)
    # These fields describe a historical live-queue handoff, not this failed-only update.
    for key in ("execution_order", "phase", "phase_state", "phase_runs", "reused_methods"):
        state.pop(key, None)
    save(args.output / "status.json", state)
    save(args.output / "jobs.json", jobs)
    save(args.output / "installed_sha256.json", fingerprints[-1])
    print(json.dumps({"methods": state["method_results"], "instances": state["instance_results"],
                      "rerun_instances": len(seen)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
