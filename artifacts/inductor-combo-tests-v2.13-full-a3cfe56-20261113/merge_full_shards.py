"""Merge a complete, disjoint plain-pytest run against its full collection."""

import argparse
from collections import Counter
import json
from pathlib import Path

from run_plain_pytest import save, write_table


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact.resolve()
    collection = root / "full-collection"
    expected_state = load(collection / "status.json")
    expected_hashes = load(collection / "installed_sha256.json")
    jobs = load(collection / "collection.json")
    by_method = {(job["file"], job["class"], job["name"]): job for job in jobs}
    assert len(by_method) == len(jobs), "Duplicate source methods"
    expected_nodes = {node for job in jobs for node in job["nodes"]}
    assert len(expected_nodes) == sum(len(job["nodes"]) for job in jobs), "Overlapping collection"
    assert expected_state["instances"] == expected_state["full_instances"] == len(expected_nodes)
    source_check = load(root / "test-source-validation.json")
    assert source_check["unchanged"], "Staged test sources changed during execution"
    reports, states = {}, []
    for shard in load(root / "shards.json"):
        directory = root / "results" / shard["name"]
        state = load(directory / "status.json")
        assert state["stage"] == "complete", f"Incomplete shard: {shard['name']}"
        assert state["installed_source_unchanged"], f"Changed installation: {shard['name']}"
        assert load(directory / "installed_sha256.json") == expected_hashes, "Changed installed sources"
        for key in ("torch", "torch_git", "torch_npu", "torch_npu_git", "triton", "test_commit", "npu_fix_commit"):
            assert state[key] == expected_state[key], f"Different {key}: {shard['name']}"
        states.append(state)
        for part in load(directory / "jobs.json"):
            key = part["file"], part["class"], part["name"]
            assert key in by_method, f"Unexpected source method: {key}"
            assert set(part["nodes"]) == set(part["reports"]), f"Missing reports: {key}"
            assert part["returncode"] in (0, 1), f"Execution error: {key}"
            full = by_method[key]
            full.setdefault("executions", []).append({
                "shard": shard["name"], "index": part["index"], "device": part["device"],
                "command": part["command"], "returncode": part["returncode"],
                "seconds": part["seconds"], "started": part["started"], "finished": part["finished"],
            })
            for node, report in part["reports"].items():
                assert node in full["nodes"] and node not in reports, f"Unexpected or duplicate node: {node}"
                assert report["status"] in ("PASS", "FAIL", "SKIP"), f"Execution error: {node}"
                report["validation"] = {
                    "artifact": root.name, "shard": shard["name"], "test_commit": state["test_commit"],
                    "torch_npu_git": state["torch_npu_git"], "npu_fix_commit": state["npu_fix_commit"],
                }
                reports[node] = report
    assert set(reports) == expected_nodes, "The shards do not cover the full collection"
    for job in jobs:
        job["reports"] = {node: reports[node] for node in job["nodes"]}
        counts = Counter(report["status"] for report in job["reports"].values())
        job["status"] = "FAIL" if counts["FAIL"] else "SKIP" if counts["SKIP"] else "PASS"
    state = dict(expected_state, stage="complete", started=min(s["started"] for s in states),
                 finished=max(s["finished"] for s in states), installed_source_unchanged=True,
                 devices=sorted({device for s in states for device in s["devices"]}),
                 test_source_unchanged=True, full_coverage=True, sampling=False, reusing_old_results=False,
                 method_results=dict(Counter(job["status"] for job in jobs)),
                 instance_results=dict(Counter(report["status"] for report in reports.values())),
                 shards=[shard["name"] for shard in load(root / "shards.json")])
    state["cache_root"] = str(Path(states[0]["cache_root"]).parent)
    state["cache_roots"] = {shard["name"]: item["cache_root"]
                            for shard, item in zip(load(root / "shards.json"), states)}
    save(root / "jobs.json", jobs)
    save(root / "status.json", state)
    save(root / "installed_sha256.json", expected_hashes)
    for name, status in (("nodeids.txt", None), ("passed-nodeids.txt", "PASS"),
                         ("failed-nodeids.txt", "FAIL"), ("skipped-nodeids.txt", "SKIP")):
        nodes = sorted(node for node, report in reports.items() if status is None or report["status"] == status)
        (root / name).write_text("\n".join(nodes) + ("\n" if nodes else ""))
    write_table(root, jobs)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
