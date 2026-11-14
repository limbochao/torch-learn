"""Map the 2.13 progress table to source methods and select only current non-PASS instances."""

import argparse
import ast
from collections import Counter
import csv
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
DEFAULT_TABLE = REPO / "docs/notes/compiler/codegen/inductor-combo-tests-v2.13.csv"
RERUN_STATUSES = {"FAIL", "SKIP", "ERROR"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Latest complete result directory containing status.json and jobs.json")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    args = parser.parse_args()
    with args.table.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    state = json.loads((args.reference / "status.json").read_text())
    if state["stage"] != "complete":
        raise ValueError("Select reruns from a complete result snapshot")
    jobs = json.loads((args.reference / "jobs.json").read_text())
    by_method = {(job["file"], job["class"], job["name"]): job for job in jobs}
    trees, occurrences, layout, selected = {}, Counter(), [], []
    for index, row in enumerate(rows):
        filename, name = row["测试文件"], row["函数名"]
        if filename not in trees:
            tree = ast.parse((args.source / filename).read_text(encoding="utf-8-sig"))
            parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
            trees[filename] = tree, parents
        tree, parents = trees[filename]
        candidates = sorted((node for node in ast.walk(tree)
                             if isinstance(node, ast.FunctionDef) and node.name == name), key=lambda n: n.lineno)
        occurrence = occurrences[filename, name]
        occurrences[filename, name] += 1
        if occurrence >= len(candidates):
            raise RuntimeError(f"Cannot resolve table row {index}: {filename}::{name}")
        node = candidates[occurrence]
        methods = [node.name]
        parent = parents.get(node)
        while parent is not None and not isinstance(parent, ast.ClassDef):
            if isinstance(parent, ast.FunctionDef):
                methods.append(parent.name)
            parent = parents.get(parent)
        if parent is None:
            raise RuntimeError(f"No test class for {filename}::{name}")
        item = {"file": filename, "name": name, "class": parent.name,
                "source_method": methods[-1], "nested": len(methods) > 1, "table_index": index}
        layout.append(item)
        if not item["nested"]:
            job = by_method[filename, parent.name, name]
            if job["status"] != row["最新结果"]:
                raise ValueError(f"Table and reference disagree: {filename}::{name}")
            if set(job["nodes"]) != set(job["reports"]):
                raise ValueError(f"Incomplete instance reports: {filename}::{name}")
            nodes = sorted(node for node, report in job["reports"].items()
                           if report["status"] in RERUN_STATUSES)
            if nodes:
                selected.append({"file": filename, "name": name, "class": parent.name,
                                 "only_nodes": nodes})
    keys = {(r["file"], r["class"], r["source_method"]) for r in layout if not r["nested"]}
    assert len(keys) == sum(not r["nested"] for r in layout), "Repeated independent method"
    assert all((r["file"], r["class"], r["source_method"]) in keys for r in layout)
    for path, data in ((args.manifest, selected), (args.layout, layout)):
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"table_rows": len(layout), "independent_methods": len(keys),
                      "selected_methods": len(selected),
                      "selected_instances": sum(len(row["only_nodes"]) for row in selected),
                      "nested_rows": sum(r["nested"] for r in layout),
                      "files": dict(Counter(r["file"] for r in selected))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
