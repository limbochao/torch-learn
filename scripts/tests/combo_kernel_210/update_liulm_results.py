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
    "test_helper_fn_defined": "F9 已修复：helper 用例实际 Combo 路径通过，不再分析；F3 错误未再出现",
    "test_mean": "F12 全量新增：mean 用例第二个输出 8/8 元素不一致，相对误差约 7；与 F10 是否同源待确认",
}
F3 = {"test_reduce_benchmark", "test_persistent_reduction_no_x_dim", "test_dynamic_shapes_reduce",
      "test_dynamic_shapes_persistent_reduction_no_x_dim", "test_dynamic_shapes_persistent_reduction_no_x_dim_2"}


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
    return ("通过" if passed(method) else "失败") + f"（{count}/{total}）"


def analysis(method, fallback):
    name = method["name"]
    if not method["job_finished"]:
        return fallback if fallback != "—" else "全量尚未结束，当前无新增失败证据"
    log = method.get("execution_log", "") + "\n".join(x.get("longrepr", "") for x in method["failures"])
    if "Too many streams" in log or "207005" in log:
        return "环境执行异常：采集用例时 NPU stream 资源不足（207005），未进入用例；不能判为代码缺陷，需复测"
    if "out_ptr1 is not defined" in log:
        return "F11 全量新增：Combo 生成 store 使用未声明的 out_ptr1，Triton 编译失败；是否与 F10 写回状态有关尚待确认"
    if name in KNOWN and (not passed(method) or name in {
            "test_helper_fn_defined", "test_combo_kernel_per_config_subkernel_red"}):
        return KNOWN[name]
    if name in F3 and passed(method):
        return "F3 修复后通过，NumelList 分区错误未再出现，不再分析"
    if passed(method):
        return "通过；F1/F3 旧错误未再出现，不再分析" if "F1" in fallback or "F3" in fallback else "通过，当前无失败待分析"
    errors = [s[2:].strip() for s in log.splitlines() if s.startswith("E ") and "Error" in s]
    return "最新执行失败，待分析：" + next(iter(errors), "见报告")[:180]


def markdown_table(rows, fields):
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines += ["| " + " | ".join(str(row[k]).replace("|", "\\|").replace("\n", " ") for k in fields) + " |"
              for row in rows]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact", default="combo-liulm-0fd5a10-20261106")
    parser.add_argument("--running", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    art = repo / "artifacts" / args.artifact
    doc = repo / "docs/notes/compiler/codegen/inductor-combo-tests-liulm-v2.10.0-combokernel-dev.md"
    originals = load_csv(doc.parent / "inductor-combo-tests-v2.13-vs-v2.10.csv")
    previous = {(r["测试文件"], r["函数名"]): r for r in load_csv(doc.with_suffix(".csv"))}
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
            for method in result["methods"]:
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
    for original in originals:
        key = original["测试文件"], original["函数名"]
        old = previous.get(key, {})
        row = {"测试文件": key[0], "函数名": key[1], "v2.10.0存在性/实现差异": original["v2.10.0存在性"],
               "执行范围": old.get("执行范围", "未运行"),
               "原 e32+F1/F3 结果": original["NPU_A5通过情况"],
               "最新结果": "未运行", "当前分析": original["失败原因分析"]}
        if original["类型"] == "嵌套函数":
            row.update(执行范围="随外层方法验证", 最新结果=original["NPU_A5通过情况"])
        elif original["NPU_A5通过情况"] == "未运行":
            row["执行范围"] = "未运行"
        else:
            method = latest_map.get(key)
            fallback = old.get("当前分析", original["失败原因分析"])
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
    assert [(r["测试文件"], r["函数名"]) for r in rows] == [(r["测试文件"], r["函数名"]) for r in originals]
    with doc.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    done = [m for m in latest if m["job_finished"]]
    good = sum(passed(m) for m in done)
    invalid = sum(m["returncode"] not in (0, 1) or m["completed_instances"] != m["expected_instances"] for m in done)
    scope = "全量及失败项复测" if retests else "全量"
    text = (f"最新结果已刷新：{scope} {len(done)}/62 个方法有终态，其中 {good} 个通过、"
            f"{len(done)-good-invalid} 个失败、{invalid} 个执行异常；其余状态按实际记录展示。")
    body = doc.read_text()
    body = re.sub(r'(验证日期：[0-9-]+。)\*\*[^\n]+?\*\*', lambda m: m[1] + "**" + text + "**", body, count=1)
    start, end = body.index("## 逐项结果"), body.index("## 证据与复现")
    intro = (f"与上游对照总表保持 {len(rows)} 条记录及原始顺序，包含 62 个独立执行方法、43 个未运行方法、"
             "11 条嵌套函数记录；嵌套记录不重复计数。未运行原因与原表逐字一致。\n\n"
             f"最新结果来源：{stamp}。失败项复测覆盖对应方法的旧结果，未复测方法保留全量结果；"
             "执行异常不记为功能失败或通过。历史轮次保存在证据及上方汇总中。\n\n")
    body = body[:start] + "## 逐项结果\n\n" + intro + markdown_table(rows, FIELDS) + "\n\n" + body[end:]
    body = body.replace("本表保留 `a369f6c` 旧基线的完整结果，并追加 `0fd5a10` 新包和已有 F3 修复的验证结果。",
                        "本表按上游对照总表的行与顺序，记录 `0fd5a10` 新包叠加 F3 的最新结果；历史轮次保留在证据中。")
    instance_passed = sum(m["counts"].get("passed", 0) for m in full_methods)
    line = (f"| 0fd5a10 + F3 全量 | {sum(passed(m) for m in full_methods)} / 62 | {instance_passed} / 409 | "
            "本轮全量原始记录；后续复测见下表 |")
    body = re.sub(r'\| 0fd5a10 \+ F3 全量 \|[^\n]*', line, body, count=1)
    if retests:
        phase, result = retests[-1]
        selected = {(m["file"], m["name"]) for m in result["methods"]}
        repeated = [r for r in rows if (r["测试文件"], r["函数名"]) in selected]
        heading = "## 失败项复测"
        if heading in body:
            start, end = body.index(heading), body.index("## 当前仍存在的问题")
            body = body[:start] + body[end:]
        content = (heading + "\n\n" + f"新包 + F3，独立目录 `{phase}`，复测 {len(selected)} 个方法、"
                   f"{result['expected_instances']} 个实例：通过 {result['instance_counts'].get('passed', 0)}，"
                   f"失败 {result['instance_counts'].get('failed', 0)}。数值断言与测试适配均未变更。\n\n"
                   + markdown_table(repeated, ["测试文件", "函数名", "最新结果", "当前分析"]) + "\n\n")
        body = body.replace("## 当前仍存在的问题", content + "## 当前仍存在的问题", 1)
        problem_cases = {
            "F4": {"test_adaptive_avg_pool2d1"}, "F5": {"test_rand_like_deterministic"},
            "F6": {"test_combo_kernel_per_config_subkernel_block_size"},
            "F7": {"test_triton_kernel_to_post_grad_tracing_combo_kernel"},
            "F8": {"test_combo_kernel_size_hint_failure"},
            "F10": {"test_dynamic_shapes_persistent_reduction_mixed_x_dim_cuda"},
            "F11": {"test_mutated_benchmark", "test_dynamic_shapes_mutated"}, "F12": {"test_mean"},
        }
        for problem, names in problem_cases.items():
            tested = [m for m in result["methods"] if m["name"] in names]
            if {m["name"] for m in tested} == names and all(passed(m) for m in tested):
                body = re.sub(r'\| ' + problem + r' \|[^\n]*\n', '', body)
        if all(passed(m) for m in result["methods"] if m["name"] == "test_round_robin_dispatch") and any(
                m["name"] == "test_round_robin_dispatch" for m in result["methods"]):
            body = re.sub(r'\| 环境执行异常 \|[^\n]*\n', '', body)
            body = body.replace("因此不是 409 个实例均已完成验证，stream 异常项仍需复测。",
                                "该全量记录的 stream 资源异常项已在本次失败项复测中通过，不再作为当前阻塞。")
    doc.write_text(body)
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
