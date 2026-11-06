"""Prepare an isolated validation phase from an already adapted test tree and launch it."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-nodes", type=Path, required=True)
    parser.add_argument("--source-hashes", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    template = args.template.resolve()
    root = args.root.resolve()
    selected = json.loads(args.manifest.read_text())
    nodes = json.loads(args.expected_nodes.read_text())
    assert len(nodes) == len(set(nodes)), "Duplicate expected node IDs"
    assert len(selected) == len({(row["file"], row["name"]) for row in selected})
    expected = json.loads(args.source_hashes.read_text())
    site = Path(sysconfig.get_paths()["purelib"])
    actual = {name: hashlib.sha256((site / name).read_bytes()).hexdigest() for name in expected}
    assert actual == expected, "Installed source differs from the approved commit"
    context = json.loads(args.context.read_text())
    assert context.get("installed_commit") and not context.get("with_f3_overlay")
    for name in ["Community-Use-Case-Testing", "original_cases"]:
        assert (template / name).is_dir(), name
    assert "combo_baseline_audit.install()" in (template / "combo_validation_plugin.py").read_text()
    root.mkdir(parents=True, exist_ok=False)
    for name in ["Community-Use-Case-Testing", "original_cases"]:
        (root / name).symlink_to((template / name).resolve(), target_is_directory=True)
    for source in template.glob("*.py"):
        shutil.copy2(source, root / source.name)
    scripts = Path(__file__).resolve().parent
    for name in ["run_validation.py", "run_combo_selected.py", "export_combo_results.py",
                 "capture_combo_evidence.py"]:
        shutil.copy2(scripts / name, root / name)
    for name, value in [("combo_npu_a5_selected.json", selected), ("expected_nodes.json", nodes),
                        ("expected_source_sha256.json", expected), ("validation_context.json", context)]:
        (root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    command = [sys.executable, str(root / "run_validation.py"), "--root", str(root),
               "--devices", args.devices, "--expected-instances", str(len(nodes))]
    if args.detach:
        command.append("--detach")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
