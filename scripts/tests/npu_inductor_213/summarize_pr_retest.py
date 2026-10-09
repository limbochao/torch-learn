"""Compare the PR 47384 rerun with the archived 247-case baseline."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from summarize_batches import node_status, sanitize, write_csv


LABELS = {
    "passed": "通过", "failed": "失败", "skipped": "跳过", "no_call": "未完成",
    "collection_error": "收集失败", "xfail_or_xpass": "预期失败",
}
BASELINE = "npu-inductor-213-batches-20261008"


def summarize_skips(raw, out, objectives):
    baseline = out.parent / BASELINE
    original = list(csv.DictReader((baseline / "skip-validation.csv").open(encoding="utf-8-sig")))
    review_path = out / "skip-review.json"
    reviews = json.loads(review_path.read_text()) if review_path.exists() else {}
    expected_hash = json.loads((raw / "installation.json").read_text())["new_sha256"]
    rows, total_instances = [], Counter()

    def read_result(name, requested):
        path = raw / "skip" / name
        if not path.exists():
            return "待测试", {}, ""
        data = json.loads(path.read_text())
        if "summary" not in data:
            return "待测试", {}, ""
        assert data["environment"]["transfer_sha256"] == expected_hash
        result = data["case_results"][requested]
        nodes = result.get("nodeids", [result["nodeid"]] if result.get("nodeid") else [])
        counts = Counter(node_status([r for r in data["reports"] if r["nodeid"] == n]) for n in nodes)
        assert nodes and not counts["no_call"], name
        target = out / "skip" / name
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(sanitize(data), ensure_ascii=False, indent=2) + "\n")
        errors = "\n".join(r["longrepr"] for r in data["reports"] if r["outcome"] != "passed")
        return LABELS.get(result["status"], result["status"]), dict(counts), errors

    for old in original:
        index = old["原序号"]
        requested = old["文件"] + "::" + old["用例"]
        first, counts, errors = read_result(f"case-{index}.json", requested)
        final = first
        supplement = ""
        evidence = f"skip/case-{index}.json"
        if index == "201":
            supplement, final_counts, final_errors = read_result("case-201-hf32off.json", requested)
            if supplement != "待测试":
                final, counts, errors = supplement, final_counts, final_errors
                evidence += "; skip/case-201-hf32off.json"
            else:
                final = "待测试"
                counts = {}
        old_final = old["临时验证结果"]
        change = "待测试" if final == "待测试" else (
            "状态不变" if final == old_final else old_final + "→" + final
        )
        row = {
            "原序号": index, "文件": old["文件"], "用例": old["用例"],
            "测试目的": objectives.get(index, {}).get("测试目的", "待补充"),
            "NPU覆盖建议": objectives.get(index, {}).get("NPU覆盖建议", "待补充"),
            "基线适配结果": old_final, "PR适配首轮": first, "PR适配补测": supplement,
            "PR最终配置结果": final, "与基线适配对比": change,
            "参数化统计": json.dumps(counts, ensure_ascii=False),
            "当前结论": "待核对本轮适配报告", "分析状态": "待分析",
            "修复建议": old["适配后建议"], "是否涉及其他模块": old["是否涉及其他模块"],
            "日志证据": evidence if first != "待测试" else "",
            "适配源码证据": "../" + BASELINE + "/skip-validation/manifest.json",
            "目的源码证据": objectives.get(index, {}).get("目的源码证据", ""),
            "验证日期": "2026-10-08",
        }
        if "libcuda.so cannot found!" in errors:
            row.update(
                当前结论="解除原设备/能力条件后，在生成模块导入或 worker 内误选 CudaDriver，"
                "因 libcuda.so 缺失失败；见 driver-probe.json。未完成原用例全部断言。",
                分析状态="完成",
            )
        row.update(reviews.get(index, {}))
        rows.append(row)
        total_instances.update(counts)
    write_csv(out / "skip-validation.csv", rows, list(rows[0]))
    (out / "skip-progress.json").write_text(json.dumps({
        "total": len(rows), "status": dict(Counter(r["PR最终配置结果"] for r in rows)),
        "instances": dict(total_instances), "reviewed": sum(r["分析状态"] == "完成" for r in rows),
        "selection": "第 201 项固定使用 HF32-off 配置，其余使用原临时适配配置；首轮和补测分别保留。",
    }, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw, out = args.raw_dir, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    baseline = out.parent / BASELINE
    old_rows = list(csv.DictReader((baseline / "results.csv").open(encoding="utf-8-sig")))
    plan = json.loads((raw / "plan.json").read_text())
    reviews = json.loads((out / "review.json").read_text()) if (out / "review.json").exists() else {}
    objectives = json.loads((out / "objectives.json").read_text()) if (out / "objectives.json").exists() else {}
    overrides = json.loads((raw / "overrides.json").read_text()) if (raw / "overrides.json").exists() else {}
    rows, batches = [], []
    for batch in plan:
        bid = batch["batch"]
        filename = f"batch-{bid}-results.json"
        source = raw / filename
        data = json.loads(source.read_text()) if source.exists() else {}
        group = []
        for index, requested in zip(batch["rows"], batch["cases"], strict=True):
            old = old_rows[index - 1]
            effective_file = overrides.get(str(index), filename)
            effective = json.loads((raw / effective_file).read_text()) if effective_file != filename else data
            ready = "summary" in effective
            result = effective.get("case_results", {}).get(requested, {}) if ready else {}
            nodes = result.get("nodeids", [result["nodeid"]] if result.get("nodeid") else [])
            reports = [r for r in effective.get("reports", []) if r["nodeid"] in nodes]
            counts = Counter(node_status([r for r in reports if r["nodeid"] == node]) for node in nodes)
            state = LABELS.get(result.get("status"), "待测试")
            if counts["passed"] and result.get("status") == "xfail_or_xpass":
                state = "部分通过/含预期失败"
            if counts["xpass"]:
                state = "意外通过"
            if counts["passed"] and state == "跳过":
                state = "部分通过/部分跳过"
            errors = "\n".join(r["longrepr"] for r in reports if r["outcome"] != "passed")
            if state == "待测试":
                change = "待测试"
            elif state == old["状态"]:
                change = "状态不变"
            elif old["状态"] == "通过":
                change = "回归"
            elif state == "通过":
                change = "转为通过"
            else:
                change = old["状态"] + "→" + state
            row = {
                "序号": str(index), "批次": bid, "文件": old["文件"], "用例": old["用例"],
                "测试目的": objectives.get(str(index), {}).get("测试目的", "待补充"),
                "NPU覆盖建议": objectives.get(str(index), {}).get("NPU覆盖建议", "待补充"),
                "目的源码证据": objectives.get(str(index), {}).get("目的源码证据", ""),
                "测试类": "; ".join(sorted({n.split("::")[1] for n in nodes})),
                "基线状态": old["状态"], "PR状态": state, "变化": change,
                "参数化统计": json.dumps(dict(counts), ensure_ascii=False),
                "失败阶段": "; ".join(sorted({r["when"] for r in reports if r["outcome"] != "passed"})) or "无",
                "当前结论": "待核对本轮报告", "修复建议": "", "是否涉及其他模块": "",
                "覆盖边界": old["覆盖边界"], "分析状态": "待分析",
                "日志证据": effective_file + "#" + requested if ready else "",
                "基线源码证据": "; ".join("../" + BASELINE + "/" + p.strip()
                                       for p in old["源码证据"].split(";") if p.strip()),
                "验证日期": "2026-10-08",
            }
            if state == "通过" and old["状态"] == "通过":
                row.update(当前结论="原测试全部选中实例的 setup/call/teardown 通过；覆盖范围沿用未变更的测试源码。",
                           分析状态="完成")
            if "libcuda.so cannot found!" in errors:
                failures = [r for r in reports if r["outcome"] == "failed"]
                same_failure = all("libcuda.so cannot found!" in r["longrepr"] for r in failures)
                row.update(
                    失败阶段="生成模块导入/编译 worker 的 Triton 驱动选择",
                    当前结论="本组失败实例出现：PR 已使 CudaDriver.is_active() 为 False，但 PyTorch _is_backend_active "
                    "又用 torch.cuda.is_available() 兜底；迁移别名返回 True，仍构造 CudaDriver 并因缺少 libcuda.so 失败。",
                    修复建议="transfer_to_npu 还需覆盖当前 PyTorch helper 的兜底判定，并在编译 worker 中生效；"
                    "保持实际 NPU driver。PR 原样方案不足。",
                    是否涉及其他模块="可在 torch_npu/contrib/transfer_to_npu.py 适配；"
                    "涉及当前 torch/_inductor/runtime/triton_helpers.py 的接口契约，超出测试/torch_npu._inductor 范围。",
                    覆盖边界="首错前的子步骤可能已执行；未完成原用例全部数值或代码断言。原有更深层问题可能被本次首错遮挡。",
                    分析状态="完成" if same_failure else "待分析",
                )
            row.update(reviews.get(str(index), {}))
            rows.append(row)
            group.append(row)
            if ready:
                assert nodes and len(nodes) == sum(counts.values())
                assert not counts["no_call"], (index, counts)
                assert effective["environment"]["transfer_sha256"] == json.loads(
                    (raw / "installation.json").read_text()
                )["new_sha256"]
                (out / effective_file).write_text(json.dumps(sanitize(effective), ensure_ascii=False, indent=2) + "\n")
        if "summary" in data:
            (out / filename).write_text(json.dumps(sanitize(data), ensure_ascii=False, indent=2) + "\n")
            write_csv(out / f"batch-{bid}.csv", group, list(rows[0]))
        batches.append({"batch": bid, "status": dict(Counter(r["PR状态"] for r in group)),
                        "reviewed": sum(r["分析状态"] == "完成" for r in group)})
    assert [int(r["序号"]) for r in rows] == list(range(1, 248))
    write_csv(out / "results.csv", rows, list(rows[0]))
    totals = Counter()
    for row in rows:
        totals.update(json.loads(row["参数化统计"]))
    progress = {
        "total": len(rows), "status": dict(Counter(r["PR状态"] for r in rows)),
        "changes": dict(Counter(r["变化"] for r in rows)), "instances": dict(totals),
        "reviewed": sum(r["分析状态"] == "完成" for r in rows), "batches": batches,
    }
    (out / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n")
    summarize_skips(raw, out, objectives)
    print(json.dumps({k: v for k, v in progress.items() if k != "batches"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
