"""Refresh the existing liulm Markdown/CSV table from archived pytest reports."""
import argparse
import csv
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact", default="combo-liulm-0fd5a10-20261106")
    parser.add_argument("--running", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    art = repo / "artifacts" / args.artifact
    doc = repo / "docs/notes/compiler/codegen/inductor-combo-tests-liulm-v2.10.0-combokernel-dev.md"
    with doc.with_suffix(".csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    summary = json.loads((art / "validation_summary.json").read_text())
    short = summary["baseline"][:7]
    full_column = next(k for k in rows[0] if k.startswith(short) and "全量" in k)
    result_path = art / "full/functional_results.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    if result:
        assert len(result["methods"]) == len(rows)
        assert result["completed_instances"] == result["expected_instances"]
        methods = {(m["file"], m["name"]): m for m in result["methods"]}
        unknown = []
        for row in rows:
            m = methods[row["测试文件"], row["函数名"]]
            passed = m["returncode"] == 0 and m["counts"].get("passed", 0) == m["expected_instances"]
            row[full_column] = ("通过" if passed else "失败") + (
                f'（{m["counts"].get("passed", 0)}/{m["expected_instances"]}）'
            )
            if not passed and row["当前分析"] == "—":
                errors = [line[2:] for failure in m["failures"]
                          for line in failure.get("longrepr", "").splitlines() if line.startswith("E ")]
                row["当前分析"] = "全量新增失败，待分析：" + next((x for x in errors if "Error" in x), "见报告")[:160]
                unknown.append(m["name"])
        summary["unreviewed_full_failures"] = unknown
        text = (f'全量完成：{len(rows)} 个方法中 {sum(m["returncode"] == 0 for m in methods.values())} 个全部通过；'
                f'{result["expected_instances"]} 个实例中 {result["instance_counts"].get("passed", 0)} 个通过、'
                f'{result["instance_counts"].get("failed", 0)} 个失败。定向失败证据继续保留。')
    else:
        for row in rows:
            row[full_column] = "运行中" if args.running else "待执行"
        text = "两轮定向复测已完成；全量后台运行中，结束后由脚本自动刷新。" if args.running else "两轮定向复测已完成；全量待执行。"
    fields = list(rows[0])
    with doc.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    body = doc.read_text()
    body = re.sub(r'(验证日期：[0-9-]+。)\*\*[^\n]+?\*\*', lambda m: m[1] + "**" + text + "**", body, count=1)
    if result:
        line = (f'| {short} + F3 全量 | {sum(m["returncode"] == 0 for m in result["methods"])} / {len(rows)} | '
                f'{result["instance_counts"].get("passed", 0)} / {result["expected_instances"]} | 本轮完整执行 |')
    else:
        state = "运行中" if args.running else "待执行"
        line = f'| {short} + F3 全量 | {state} | {state} | 后台脚本完成后自动更新 |'
    body = re.sub(r'\| ' + re.escape(short) + r' \+ F3 全量 \|[^\n]*', line, body, count=1)
    table = "| " + " | ".join(fields) + " |\n| " + " | ".join("---" for _ in fields) + " |\n"
    table += "\n".join("| " + " | ".join(str(row[k]).replace("|", "\\|") for k in fields) + " |" for row in rows)
    start, end = body.index("## 逐项结果"), body.index("## 证据与复现")
    body = body[:start] + "## 逐项结果\n\n" + table + "\n\n" + body[end:]
    heading = "## 原失败方法的最新复测"
    if heading in body:
        start, end = body.index(heading), body.index("## 当前仍存在的问题")
        body = body[:start] + body[end:]
    focused = "| 测试文件 | 方法 | 仅新包 | 新包+F3 | 当前分析 |\n| --- | --- | --- | --- | --- |\n"
    for row in rows:
        if row[f"{short} 定向结果"] != "未选入定向":
            focused += "| " + " | ".join(row[k] for k in ["测试文件", "函数名", f"{short} 定向结果",
                                                        f"{short}+F3 定向结果", "当前分析"]) + " |\n"
    body = body.replace("## 当前仍存在的问题", heading + "\n\n" + focused + "\n## 当前仍存在的问题", 1)
    doc.write_text(body)
    readme = art / "README.md"
    content = readme.read_text()
    content = re.sub(r'(`[A-Fa-f0-9]{40}`。)[^\n]*',
                     lambda m: m[1] + text, content, count=1)
    readme.write_text(content)
    (art / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(text)


if __name__ == "__main__":
    main()
