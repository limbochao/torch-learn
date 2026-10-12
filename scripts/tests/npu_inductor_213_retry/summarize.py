"""Merge retry evidence with the original ordered case list."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re


STATUSES = ("PASS", "FAIL", "ERROR", "SKIP", "XFAIL", "XPASS")
PHASES = ("failures", "dynamic", "skips", "xfails", "skips_followup", "dynamic_followup")


def node_status(node):
    if any(node.get(phase, {}).get("outcome") == "failed" for phase in ("setup", "teardown")):
        return "ERROR"
    call = node.get("call", {})
    if call.get("wasxfail") is not None:
        return "XFAIL" if call["outcome"] == "skipped" else "XPASS"
    if call.get("outcome") == "failed":
        return "FAIL"
    if call.get("outcome") == "passed":
        return "PASS"
    if any(node.get(phase, {}).get("outcome") == "skipped" for phase in ("setup", "call")):
        return "SKIP"
    return "INCOMPLETE"


def normalized(nodes):
    return {node.removeprefix("test/inductor/") for node in nodes}


def read_attempt(root, phase, row):
    prefix = root / phase / "results" / f"case-{row['row']:03d}"
    invocation_path = prefix.with_suffix(".invocation.json")
    if not invocation_path.exists():
        return None
    invocation = json.loads(invocation_path.read_text())
    report_path = prefix.with_suffix(".json")
    report = json.loads(report_path.read_text()) if report_path.exists() else {"reports": {}}
    nodes = report["reports"]
    counts = Counter(node_status(node) for node in nodes.values())
    expected = normalized(row["nodes"])
    collected = normalized(report.get("collected", []))
    if invocation["timed_out"]:
        status = "TIMEOUT"
    elif expected != collected or report.get("collection_errors"):
        status = "COLLECTION_ERROR"
    elif invocation["returncode"] not in (0, 1):
        status = "PROCESS_ERROR"
    elif counts["ERROR"]:
        status = "ERROR"
    elif counts["FAIL"]:
        status = "FAIL"
    elif counts["INCOMPLETE"] or normalized(nodes) != expected or not counts:
        status = "INCOMPLETE"
    elif len(counts) == 1:
        status = next(iter(counts))
    else:
        status = "MIXED"
    reasons = []
    for nodeid, node in nodes.items():
        for phase_name in ("setup", "call", "teardown"):
            detail = node.get(phase_name, {}).get("longrepr", "")
            if detail:
                lines = [line.strip() for line in detail.splitlines()
                         if re.search(r"Error:|Exception:|Skipped|XFailed|AssertionError", line)]
                reasons.append(nodeid.split("::")[-1] + ": " + " | ".join((lines or [detail])[-3:]))
    reasons.extend(report.get("collection_errors", []))
    if invocation["timed_out"]:
        reasons.append("Reached the 900-second per-row limit; incomplete nodes are not counted as PASS.")
    return {"row": row["row"], "phase": phase, "status": status, "counts": dict(counts),
            "reason": "\n".join(reasons), "invocation": invocation, "report": report}


def sanitize(value):
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items() if key not in {"pid"}}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if not isinstance(value, str):
        return value
    value = re.sub(r"/home/[^/\s]+/PTA_213/Community-Use-Case-Testing/v2\.13\.0/pytorch", "<test-repo>", value)
    value = re.sub(r"/home/[^/\s]+/miniforge3/envs/pta_213_case", "<test-env>", value)
    value = re.sub(r"/home/[^/\s]+/PTA_213/retry_20261011", "<retry-workdir>", value)
    value = re.sub(r"/home/[^/\s]+", "<home>", value)
    value = re.sub(r"/tmp/tmp[\w-]+", "<temporary>", value)
    value = re.sub(r"PID[:= ]*\d+", "PID:<redacted>", value)
    value = re.sub(r"backend_hash['\"]?\s*[:=]\s*['\"][^'\"]+['\"]", "backend_hash=<redacted>", value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    baseline = list(csv.DictReader((args.baseline / "results.csv").open(encoding="utf-8-sig")))
    plan = json.loads((args.baseline / "plan.json").read_text())
    assert len(baseline) == len(plan) == 331
    assert [int(row["序号"]) for row in baseline] == list(range(1, 332))
    audit = {row["row"]: row for row in json.loads((args.run / "skip-audit.json").read_text())}
    attempts, latest, missing = [], {}, []
    for phase in PHASES:
        phase_plan = json.loads((args.run / f"{phase}-plan.json").read_text())
        for row in phase_plan:
            attempt = read_attempt(args.run, phase, row)
            if attempt is None:
                missing.append([phase, row["row"]])
                continue
            attempts.append(attempt)
            latest[row["row"]] = attempt
    for number, assessment in audit.items():
        result = latest.get(number)
        assessment["latest_status"] = result["status"] if result else "SKIP"
        assessment["latest_reason"] = result["reason"] if result else assessment["reason"]
        assessment["pytest_invocations"] = sum(item["row"] == number for item in attempts)
    rows = []
    for old, planned in zip(baseline, plan):
        number = int(old["序号"])
        assert (old["文件"], old["原始用例名"]) == (planned["file"], planned["test"])
        result = latest.get(number)
        if result:
            current = result["status"]
            counts = result["counts"]
            invocation = result["invocation"]
            reason = result["reason"]
            action = "RETRIED"
        else:
            current = old["结果"]
            counts = {key: int(old[key]) for key in STATUSES}
            invocation = {}
            reason = audit.get(number, {}).get("reason", old["说明"])
            action = "RETAIN_SKIP" if number in audit else "PREVIOUS_RESULT"
        rows.append({"序号": number, "文件": old["文件"], "原始用例名": old["原始用例名"],
                     "上轮结果": old["结果"], "本次处理": action, "最新结果": current,
                     "本次调用数": sum(item["row"] == number for item in attempts),
                     **{key: counts.get(key, 0) for key in STATUSES},
                     "说明": reason, "跳过评估": audit.get(number, {}).get("reason", ""),
                     "批次": result["phase"] if result else "baseline",
                     "耗时秒": round(invocation.get("duration", 0), 3),
                     "环境变量": json.dumps(invocation.get("environment", {}), ensure_ascii=False),
                     "命令参数": json.dumps(invocation.get("command", []), ensure_ascii=False),
                     "实际节点": "\n".join(result["report"].get("collected", [])) if result else old["实际节点"]})
    required = {int(row["序号"]) for row in baseline
                if row["结果"] in {"FAIL", "PROCESS_ERROR", "TIMEOUT", "XFAIL"}}
    if args.require_complete:
        assert not missing, missing
        assert required <= latest.keys(), sorted(required - latest.keys())
        assert len(audit) == 102
        assert all(row["action"] == "RETAIN_SKIP" or number in latest for number, row in audit.items())
        assert all(item["status"] != "INCOMPLETE" for item in attempts)
    transitions = Counter(row["上轮结果"] + " -> " + row["最新结果"] for row in rows)
    summary = {
        "input_rows": len(rows), "retried_rows": len(latest), "pytest_invocations": len(attempts),
        "missing_invocations": missing, "reviewed_skip_rows": len(audit),
        "initial_retained_skip_rows": sum(row["action"] == "RETAIN_SKIP" for row in audit.values()),
        "latest_row_statuses": dict(Counter(row["最新结果"] for row in rows)),
        "retried_row_statuses": dict(Counter(item["status"] for item in latest.values())),
        "latest_node_statuses": {key: sum(row[key] for row in rows) for key in STATUSES},
        "incomplete_nodes_in_timed_out_rows": sum(item["counts"].get("INCOMPLETE", 0)
                                                  for item in latest.values() if item["status"] == "TIMEOUT"),
        "transitions": dict(sorted(transitions.items())),
        "phase_rows": {phase: dict(Counter(item["status"] for item in attempts if item["phase"] == phase))
                       for phase in PHASES},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sanitize(rows))
    outputs = [("summary.json", summary), ("node-results.json", attempts),
               ("skip-audit.json", list(audit.values()))]
    for name, value in outputs:
        (args.output / name).write_text(json.dumps(sanitize(value), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
