"""Merge phase reports and reviewed diagnoses into the cumulative CSV."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re


def sanitize(value):
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if not isinstance(value, str):
        return value
    # Pytest may truncate the beginning of a home path in an exception repr.
    value = re.sub(r'''\.\.\.[^\s"'\\]*/PTA_213/pytorch/''', '...<pytorch>/', value)
    value = re.sub(
        r"(?:/home/[^/\s\"']+/miniforge3|(?:\.\./)+miniforge3)/envs/[^/\s]+/lib/python[\d.]+/site-packages/",
        "<site-packages>/", value,
    )
    value = re.sub(r"/home/[^/\s\"']+/PTA_213/pytorch/", "<pytorch>/", value)
    value = re.sub(r"/home/[^/\s\"']+/[^\s\"'\\:)]*", "<workspace-path>", value)
    value = re.sub(r"/tmp/[^\s\"'\\:)]*", "<temporary-artifact>", value)
    value = value.replace("/__w/pytorch/pytorch/", "<pytorch>/")
    value = re.sub(r"PID:\s*\d+", "PID:<redacted>", value)
    value = re.sub(r"(?<=at )0x[0-9a-fA-F]{8,}", "<address>", value)
    value = re.sub(r"(backend_hash[\\\"']*\s*[:=]\s*[\\\"']+)[0-9a-fA-F]+", r"\1<redacted>", value)
    value = re.sub(r"(\b[VDIWEF]\d{4} \d{2}:\d{2}:\d{2}\.\d+\s+)\d+", r"\1<redacted>", value)
    return value


def node_status(reports):
    if any(r["outcome"] == "failed" for r in reports):
        return "failed"
    if any(r.get("wasxfail") for r in reports):
        return "xfail" if any(r["outcome"] == "skipped" for r in reports) else "xpass"
    if any(r["outcome"] == "skipped" for r in reports):
        return "skipped"
    if len(reports) == 3 and all(r["outcome"] == "passed" for r in reports):
        return "passed"
    return "no_call"


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    rows = list(csv.DictReader((out / "results.csv").open(encoding="utf-8-sig")))
    annotations_path = out / "analysis.json"
    annotations = json.loads(annotations_path.read_text()) if annotations_path.exists() else {}
    coverage_path = out / "coverage.json"
    coverage = json.loads(coverage_path.read_text()) if coverage_path.exists() else {}
    fields = list(rows[0])
    for field in ("执行范围", "参数化统计", "分析状态"):
        if field not in fields:
            fields.append(field)
    for row in rows:
        row.setdefault("分析状态", "完成" if row["批次"] == "01" else "待分析")
    plan = json.loads((args.raw_dir / "plan.json").read_text())
    first = [row for row in rows if row["批次"] == "01"]
    summaries = [{"batch": "01", "logical": dict(Counter(r["状态"] for r in first)),
                  "instances": 25, "reviewed": sum(r["分析状态"] == "完成" for r in first)}] if first else []
    for batch in plan:
        bid = batch["batch"]
        filename = f"batch-{bid}-results.json"
        raw = args.raw_dir / filename
        if not raw.exists():
            continue
        data = json.loads(raw.read_text())
        if "summary" not in data:
            continue
        data = sanitize(data)
        data["validation_date"] = "2026-10-08"
        (out / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        group = []
        for index, requested in zip(batch["rows"], batch["cases"], strict=True):
            row = rows[index - 1]
            result = data["case_results"][requested]
            nodeids = result.get("nodeids", [result["nodeid"]] if result.get("nodeid") else [])
            reports = [r for r in data["reports"] if r["nodeid"] in nodeids]
            counts = Counter(node_status([r for r in reports if r["nodeid"] == n]) for n in nodeids)
            status = result["status"]
            state = {"passed": "通过", "failed": "失败", "skipped": "跳过",
                     "xfail_or_xpass": "预期失败/意外通过", "collection_error": "收集失败",
                     "no_call": "未执行"}.get(status, status)
            if status == "skipped" and counts["passed"]:
                state = "部分通过/部分跳过"
            if status == "xfail_or_xpass":
                state = "意外通过" if counts["xpass"] else "预期失败"
                if counts["passed"] and not counts["xpass"]:
                    state = "部分通过/含预期失败"
            nonpass = [r for r in reports if r["outcome"] != "passed"]
            errors = [r["longrepr"] for r in nonpass]
            if status == "collection_error":
                errors.extend(data["collection_errors"])
            row.update({
                "批次": bid, "测试类": "; ".join(sorted({n.split("::")[1] for n in nodeids})),
                "nodeid": "; ".join(nodeids), "状态": state,
                "失败阶段": "; ".join(sorted({r["when"] for r in nonpass})) or "无",
                "日志证据": filename + "#" + requested,
                "源码证据": f"batch-{bid}-source-evidence.json#" + requested,
                "验证日期": "2026-10-08", "参数化统计": json.dumps(dict(counts), ensure_ascii=False),
                "执行范围": data.get("selection_metadata", {}).get(requested, {}).get("scope", ""),
            })
            if status == "passed":
                row.update({
                    "归因分类": "原用例通过", "失败原因": "无", "NPU机制关系": "未观察到失败",
                    "能否修复": "不适用", "修复方向": "无", "未修复原因": "不适用",
                    "覆盖边界": coverage.get(
                        requested, "选中实例全部执行原测试断言；输入、dtype 和分支范围见对应测试源码。"
                    ),
                    "分析状态": "完成",
                })
                if row["执行范围"] in ("cpu_only", "explicit_cpu"):
                    row["覆盖边界"] = "CPU 专用实例通过，不计为 NPU 功能通过。" + coverage.get(requested, "")
            else:
                row["分析状态"] = "待分析"
                row["失败原因"] = " | ".join(
                    line.strip() for error in errors for line in error.splitlines()
                    if line.startswith("E ") and any(x in line for x in ("Error", "Assertion", "not supported"))
                )[-2200:] or " | ".join(errors)[-1000:]
            if requested in annotations:
                row.update(annotations[requested])
                row["分析状态"] = "完成"
            group.append(row)
        write_csv(out / f"batch-{bid}.csv", group, fields)
        summaries.append({"batch": bid, "logical": dict(Counter(r["状态"] for r in group)),
                          "instances": len(data["selected"]),
                          "reviewed": sum(r["分析状态"] == "完成" for r in group)})
    write_csv(out / "results.csv", rows, fields)
    (out / "progress.json").write_text(json.dumps({
        "total": len(rows), "status": dict(Counter(r["状态"] for r in rows)),
        "reviewed": sum(r["分析状态"] == "完成" for r in rows), "batches": summaries,
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": dict(Counter(r["状态"] for r in rows)),
                      "reviewed": sum(r["分析状态"] == "完成" for r in rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
