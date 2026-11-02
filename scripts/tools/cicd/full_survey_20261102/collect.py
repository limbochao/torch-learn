import argparse
import ast
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description="Collect four-way CICD profiles and generated kernel provenance.")
parser.add_argument("--input-root", type=Path, required=True)
parser.add_argument("--work-dir", type=Path, required=True)
args = parser.parse_args()
ROOT = args.input_root.resolve()
OUT = args.work_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path, device):
    rows = list(enumerate(csv.DictReader(path.open()), 2))
    steps = sorted({r["Step Id"] for _, r in rows}) if device == "NPU" else None
    count_steps = len(steps) if steps else 10
    groups = defaultdict(lambda: {"ms": 0.0, "count": 0.0, "min_us": float("inf"), "max_us": 0.0})
    for line, r in rows:
        key = r["Name"] if device == "NPU" else r["OP Type"]
        g = groups[key]
        value = float(r["Duration(us)"] if device == "NPU" else r["Total Time(us)"])
        g["ms"] += value / count_steps / 1000
        g["count"] += (1 if device == "NPU" else float(r["Count"])) / count_steps
        g["min_us"] = min(g["min_us"], float(r["Duration(us)"] if device == "NPU" else r["Min Time(us)"]))
        g["max_us"] = max(g["max_us"], float(r["Duration(us)"] if device == "NPU" else r["Max Time(us)"]))
        if "first_line" not in g:
            g["first_line"] = line
    return {
        "total_ms": sum(g["ms"] for g in groups.values()),
        "groups": dict(groups),
        "steps": steps,
        "profile_steps": count_steps,
    }


def index_code(path):
    source = path.read_text()
    lines = source.splitlines()
    tree = ast.parse(source)
    result = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "triton":
            continue
        if len(call.args) < 2 or not isinstance(call.args[1], ast.Constant):
            continue
        name = ast.literal_eval(call.args[0])
        preceding = "\n".join(lines[: node.lineno - 1])
        headers = list(
            re.finditer(r"# Topologically Sorted Source Nodes: \[(.*?)\], Original ATen: \[(.*?)\]", preceding)
        )
        h = headers[-1] if headers else None
        metadata = preceding[h.start() :] if h else ""
        aten = [x.strip() for x in h.group(2).split(",")] if h else []
        result.append(
            {
                "name": name,
                "file": path.name,
                "line": node.lineno,
                "end_line": node.end_lineno,
                "aten": aten,
                "nodes": h.group(1) if h else "",
                "graph_header": source.splitlines()[0],
                "metadata": metadata,
                "dsl_sha256": hashlib.sha256(str(call.args[1].value).encode()).hexdigest(),
            }
        )
    return result


summaries = {}
for device in ("NPU", "GPU"):
    summaries[device] = {
        r["model"] + ("_Infer" if r["mode"] == "inference" else "_Train"): r
        for r in csv.DictReader((ROOT / f"summary_{device}.csv").open())
    }
cases = {}
for directory in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    name = directory.name
    item = {"model": name.rsplit("_", 1)[0], "mode": name.rsplit("_", 1)[1], "profile": {}, "code": {}, "summary": {}}
    for device in ("NPU", "GPU"):
        item["summary"][device] = {
            k: v for k, v in summaries[device].get(name, {}).items() if k not in ("log_path", "profile_path")
        }
        for backend in ("Eager", "Inductor"):
            item["profile"][device + "_" + backend] = read_csv(directory / f"{name}_{backend}_{device}.csv", device)
        item["code"][device] = []
        for path in sorted(directory.glob(device + "_g*_output_code.py")):
            item["code"][device] += index_code(path)
    ne, ni, ge, gi = [
        item["profile"][k]["total_ms"] for k in ("NPU_Eager", "NPU_Inductor", "GPU_Eager", "GPU_Inductor")
    ]
    item["whole_model"] = {
        "NPU_gain": 1 - ni / ne,
        "GPU_gain": 1 - gi / ge,
        "NPU_target_ms": ne * gi / ge,
        "deficit_ms": ni - ne * gi / ge,
    }
    cases[name] = item
(OUT / "inventory.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
print("collected", len(cases), "scenarios;", len({v["model"] for v in cases.values()}), "models")
print("kernel definitions", sum(len(c) for v in cases.values() for c in v["code"].values()))
print(
    "batch mismatches",
    [
        (k, v["summary"]["NPU"].get("batch_size"), v["summary"]["GPU"].get("batch_size"))
        for k, v in cases.items()
        if v["summary"]["NPU"].get("batch_size") != v["summary"]["GPU"].get("batch_size")
    ],
)
