"""Map every row of the requested comparison table to its actual 2.13 source method."""

import argparse
import ast
from collections import Counter
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    args = parser.parse_args()
    with args.table.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
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
            selected.append({"file": filename, "name": name, "class": parent.name})
    keys = {(r["file"], r["class"], r["name"]) for r in selected}
    assert len(keys) == len(selected), "Repeated independent method"
    assert all((r["file"], r["class"], r["source_method"]) in keys for r in layout)
    for path, data in ((args.manifest, selected), (args.layout, layout)):
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"table_rows": len(layout), "independent_methods": len(selected),
                      "nested_rows": sum(r["nested"] for r in layout),
                      "files": dict(Counter(r["file"] for r in selected))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
