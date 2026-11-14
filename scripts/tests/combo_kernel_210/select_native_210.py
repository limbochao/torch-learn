"""Select existing 2.10 source methods without importing or modifying test files."""

import argparse
import ast
import csv
import hashlib
import io
import json
import subprocess
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = REPO.parent / "Community-Use-Case-Testing/v2.10.0/test_inductor"
DEFAULT_TABLE = REPO / "docs/notes/compiler/codegen/inductor-combo-tests-v2.13.csv"
DEFAULT_OUTPUT = REPO / "docs/notes/compiler/codegen/inductor-combo-tests-native-v2.10.csv"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def definitions(path):
    found = {}
    if not path.is_file():
        return found

    def visit(node, parents=()):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                is_function = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                if is_function:
                    parent = parents[-1] if parents else None
                    method = isinstance(parent, ast.ClassDef) and not any(
                        isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)) for p in parents
                    )
                    found.setdefault(child.name, []).append({
                        "kind": "method" if method else "nested_or_helper",
                        "class": parent.name if method else None,
                    })
                visit(child, (*parents, child))
            else:
                visit(child, parents)

    visit(ast.parse(path.read_text(encoding="utf-8")))
    return found


def execution_note(file, name):
    if name == "test_mean":
        return "2.10 原版不主动开启 Combo；按原始配置执行"
    if name == "test_rand_like_deterministic":
        return "保留原版 False/True Combo 参数；不添加 2.13 的 random.seed"
    if name == "test_adaptive_avg_pool2d1":
        return "保留原版 False/True Combo 参数及全部输入、数值断言"
    if name == "test_combo_kernel_filter_cpu":
        return "原方法包含 CPU/CUDA 分支；NPU 收集及实际分支待确认"
    if file in ("test_combo_kernels.py", "test_foreach.py"):
        return "保留 2.10 原版配置、参数化和 kernel 数量/源码结构断言"
    return "保留 2.10 原版装饰器、输入和断言"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("combo_native_210_selected.json"))
    parser.add_argument("--audit", type=Path, default=Path(__file__).with_name("combo_native_210_selection.json"))
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    with args.table.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    cache = {file: definitions(source / file) for file in dict.fromkeys(row["测试文件"] for row in rows)}
    selected, audit, table, seen = [], [], [], set()
    for index, row in enumerate(rows, 1):
        file, name = row["测试文件"], row["函数名"]
        hits = cache[file].get(name, [])
        methods = [hit for hit in hits if hit["kind"] == "method"]
        if row["类型"] == "嵌套函数":
            status = "nested" if hits else "absent"
        elif not methods:
            status = "absent"
        else:
            if len(methods) != 1:
                raise ValueError(f"Ambiguous source method: {file}:{name}: {methods}")
            status = "selected"
            key = file, methods[0]["class"], name
            if key not in seen:
                seen.add(key)
                selected.append({
                    "file": file, "name": name, "class": key[1],
                    "support": row["v2.10.0存在性"],
                })
                table.append({
                    "测试文件": file,
                    "函数名": name,
                    "存在性/实现差异": row["v2.10.0存在性"],
                    "2.10 原生执行说明": execution_note(file, name),
                    "执行状态": "待收集、未执行",
                })
        audit.append({"row": index, "file": file, "name": name, "status": status})
    head = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "source": str(source), "source_commit": head,
        "input_table": str(args.table.resolve()), "input_table_sha256": digest(args.table),
        "source_sha256": {file: digest(source / file) for file in cache if (source / file).is_file()},
        "source_rows": len(rows), "row_status_counts": dict(Counter(row["status"] for row in audit)),
        "selected_methods": len(selected), "selected_by_file": dict(Counter(row["file"] for row in selected)),
        "collected_instances": None, "executed": False, "rows": audit,
    }
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, lineterminator="\n", fieldnames=[
        "测试文件", "函数名", "存在性/实现差异", "2.10 原生执行说明", "执行状态",
    ])
    writer.writeheader()
    writer.writerows(table)
    for path, content in (
        (args.output, stream.getvalue()),
        (args.manifest, json.dumps(selected, ensure_ascii=False, indent=2) + "\n"),
        (args.audit, json.dumps(report, ensure_ascii=False, indent=2) + "\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8-sig" if path.suffix == ".csv" else "utf-8")
    print(json.dumps({key: report[key] for key in (
        "source_rows", "row_status_counts", "selected_methods", "selected_by_file", "executed",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
