"""Align liulm results with the original 116 rows and prefer the latest recorded execution."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re


FIELDS = ["测试文件", "函数名", "v2.10.0存在性/实现差异", "执行范围", "原 e32+F1/F3 结果", "最新结果", "当前分析"]
KNOWN = {
    "test_combo_kernel_per_config_subkernel_red": "F2 仍存在：Combo store 编译失败后回退普通 kernel，功能通过；暂不优先分析",
    "test_combo_kernel_per_config_subkernel_block_size": (
        "F6 仍存在：eager 不支持 channels_last contiguous，未进入编译；不列入 Combo 修复重点"),
    "test_adaptive_avg_pool2d1": "F4 仍存在：Combo 索引化简解包失败后回退；两个开关分支最终均被 output_shape 断言阻塞",
    "test_rand_like_deterministic": ("F5 仍存在：Combo 索引 int64 与 int32 检查不一致后回退；"
                                     "两个开关分支最终均被 output_shape 断言阻塞"),
    "test_combo_kernel_size_hint_failure": "F8 仍存在：wrapper 在 nonzero 绑定 u0 前使用 u0，优先分析",
    "test_triton_kernel_to_post_grad_tracing_combo_kernel": "F7 仍存在：provenance 映射为空，AOTI 未执行；后置",
    "test_dynamic_shapes_persistent_reduction_mixed_x_dim_cuda": "F10 仍存在：mean 输出被后续 sum store 覆盖，精度失败；最优先分析",
    "test_mean": "F12 全量新增：mean 用例第二个输出 8/8 元素不一致，相对误差约 7；与 F10 是否同源待确认",
}


def load_csv(path):
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def from_snapshot(snapshot):
    collection = snapshot["collection"]
    methods = []
    for source in snapshot["methods"]:
        phases = defaultdict(dict)
        for record in source["records"]:
            phases[record["nodeid"]][record["phase"]] = record
        outcomes = []
        failures = []
        for records in phases.values():
            # A passing call without teardown is still running, not a completed instance.
            if "teardown" not in records:
                continue
            outcome = ("failed" if any(r["outcome"] == "failed" for r in records.values()) else
                       "skipped" if any(r["outcome"] == "skipped" for r in records.values()) else "passed")
            outcomes.append(outcome)
            failures.extend(r for r in records.values() if r["outcome"] != "passed")
        expected = sum(n["file"] == source["file"] and n["name"] == source["name"] for n in collection)
        methods.append({**source, "expected_instances": expected, "completed_instances": len(outcomes),
                        "counts": dict(Counter(outcomes)), "failures": failures})
    return methods


def passed(method):
    return (method["job_finished"] and method["returncode"] == 0
            and method["counts"].get("passed", 0) == method["expected_instances"])


def format_result(method):
    count = method["counts"].get("passed", 0)
    total = method["expected_instances"]
    if not method["job_finished"]:
        return f"执行中（已完成 {method['completed_instances']}/{total}）" if method["job_started"] else "待执行"
    if method["returncode"] not in (0, 1) or method["completed_instances"] != total:
        return f"执行异常（有效结果 {method['completed_instances']}/{total}）"
    if method["counts"].get("skipped", 0):
        return f"含跳过（通过 {count}/{total}）"
    value = ("通过" if passed(method) else "失败") + f"（{count}/{total}）"
    commit = method.get("fix_commit")
    return value + f"；修复 {commit[:10]}" if commit and passed(method) else value


def analysis(method, fallback):
    if passed(method):
        return "—"
    name = method["name"]
    if not method["job_finished"]:
        return fallback if fallback != "—" else "全量尚未结束，当前无新增失败证据"
    log = method.get("execution_log", "") + "\n".join(x.get("longrepr", "") for x in method["failures"])
    if "Too many streams" in log or "207005" in log:
        return "环境执行异常：采集用例时 NPU stream 资源不足（207005），未进入用例；不能判为代码缺陷，需复测"
    if method.get("fresh_failure_analysis"):
        if method["returncode"] not in (0, 1) or method["completed_instances"] != method["expected_instances"]:
            return "本轮执行异常，实例未完整记录；见本轮执行日志，不能计为功能通过或失败"
        if method["counts"].get("skipped", 0):
            return "本轮有跳过实例，跳过不计为通过；原因见本轮报告"
        errors = [s[2:].strip() for s in log.splitlines() if s.startswith("E ") and "Error" in s]
        return "本轮全量失败，待按新日志分析：" + next(iter(errors), "见本轮报告")[:180]
    if "out_ptr1 is not defined" in log:
        return "F11 仍存在：Combo 生成 store 使用未声明的 out_ptr1，Triton 编译失败；是否与 F10 写回状态有关尚待确认"
    if name in KNOWN:
        return KNOWN[name]
    errors = [s[2:].strip() for s in log.splitlines() if s.startswith("E ") and "Error" in s]
    return "最新执行失败，待分析：" + next(iter(errors), "见报告")[:180]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact", default="combo-liulm-0fd5a10-20261106")
    parser.add_argument("--running", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    art = repo / "artifacts" / args.artifact
    table = repo / "docs/notes/compiler/codegen/inductor-combo-tests-liulm-v2.10.0-combokernel-dev.csv"
    previous = load_csv(table)
    focused = json.loads((art / "with_fixes/functional_results.json").read_text())["methods"]
    focused_map = {(m["file"], m["name"]): m for m in focused}
    full_path = art / "full/functional_results.json"
    snapshot_path = art / "execution/full_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else None
    if full_path.exists() and (art / "full/integrity_checks.json").exists():
        latest = json.loads(full_path.read_text())["methods"]
        stamp = "完整全量报告"
    else:
        latest = from_snapshot(snapshot) if snapshot else []
        stamp = snapshot["captured_at"] if snapshot else "定向复测报告"
    full_methods = latest
    latest_map = {(m["file"], m["name"]): m for m in latest}
    retests = []
    index = art / "retests/index.json"
    if index.exists():
        for phase in json.loads(index.read_text()):
            folder = art / "retests" / phase
            result = json.loads((folder / "functional_results.json").read_text())
            context_path = folder / "validation_context.json"
            context = json.loads(context_path.read_text()) if context_path.exists() else {}
            result["validation_context"] = context
            for method in result["methods"]:
                method["fix_commit"] = context.get("fix_commits", {}).get(method["name"])
                method["fresh_failure_analysis"] = context.get("fresh_failure_analysis", False)
                if not method["job_finished"]:
                    continue
                if method["returncode"] not in (0, 1):
                    name = method["file"].removesuffix(".py") + "." + method["name"] + ".log"
                    method["execution_log"] = (folder / name).read_text()
                latest_map[method["file"], method["name"]] = method
            retests.append((phase, result))
        latest = list(latest_map.values())
        stamp = "全量快照 + " + retests[-1][0]
    rows = []
    for old in previous:
        key = old["测试文件"], old["函数名"]
        baseline = old["原 e32+F1/F3 结果"]
        row = {"测试文件": key[0], "函数名": key[1], "v2.10.0存在性/实现差异": old["v2.10.0存在性/实现差异"],
               "执行范围": old.get("执行范围", "未运行"),
               "原 e32+F1/F3 结果": baseline,
               "最新结果": "未运行", "当前分析": old["当前分析"]}
        if old["执行范围"] == "随外层方法验证":
            row.update(执行范围="随外层方法验证", 最新结果=baseline)
        elif baseline == "未运行":
            row["执行范围"] = "未运行"
        else:
            method = latest_map.get(key)
            fallback = old["当前分析"]
            if method:
                row["最新结果"] = format_result(method)
                row["当前分析"] = analysis(method, fallback)
            elif key in focused_map:
                method = focused_map[key]
                row["最新结果"] = format_result(method) + "（定向；全量待完成）"
                row["当前分析"] = analysis(method, fallback)
            else:
                row["最新结果"] = "待执行"
                row["当前分析"] = "最新全量尚无结果"
            if method and not method["job_finished"] and key in focused_map:
                row["最新结果"] += "；定向：" + format_result(focused_map[key])
                row["当前分析"] = analysis(focused_map[key], fallback) + "（全量待完成）"
        rows.append(row)
    if retests:
        context = retests[-1][1].get("validation_context", {})
        note = context.get("untested_failure_note")
        tested_keys = {(m["file"], m["name"]) for m in retests[-1][1]["methods"]}
        for row in rows:
            key = row["测试文件"], row["函数名"]
            if note and row["最新结果"].startswith("失败") and key not in tested_keys:
                row["当前分析"] = row["当前分析"].replace("仍存在", "历史失败") + "；" + note
    assert [(r["测试文件"], r["函数名"]) for r in rows] == [(r["测试文件"], r["函数名"]) for r in previous]
    with table.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    done = [m for m in latest if m["job_finished"]]
    good = sum(passed(m) for m in done)
    invalid = sum(m["returncode"] not in (0, 1) or m["completed_instances"] != m["expected_instances"] for m in done)
    scope = "历史全量与定向复测的滚动汇总" if retests else "全量"
    text = (f"最新结果已刷新：{scope} {len(done)}/62 个方法有终态，其中 {good} 个通过、"
            f"{len(done)-good-invalid} 个失败、{invalid} 个执行异常；其余状态按实际记录展示。")
    # Legacy adapted-case results remain in CSV/evidence, not the current native 2.10 report.
    readme = art / "README.md"
    content = re.sub(r'(`[A-Fa-f0-9]{40}`。)[^\n]*', lambda m: m[1] + text, readme.read_text(), count=1)
    readme.write_text(content)
    summary = json.loads((art / "validation_summary.json").read_text())
    summary["latest_table_snapshot"] = {"captured_at": stamp, "finished_methods": len(done), "passed_methods": good,
                                        "failed_methods": len(done)-good-invalid, "execution_errors": invalid}
    if retests:
        phase, result = retests[-1]
        summary["latest_retest"] = {"phase": phase, "methods": len(result["methods"]),
                                   "expected_instances": result["expected_instances"],
                                   "completed_instances": result["completed_instances"],
                                   "instance_counts": result["instance_counts"]}
    (art / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(text)


if __name__ == "__main__":
    main()
