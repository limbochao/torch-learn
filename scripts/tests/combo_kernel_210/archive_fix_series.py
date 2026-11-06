"""Archive downloaded Combo validation phases with sanitized logs and source receipts."""
import argparse
import hashlib
import json
from pathlib import Path

from archive_liulm_run import clean, clean_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phases", nargs="+", required=True)
    args = parser.parse_args()
    excluded = {"installed_source", "original_tests", "executed_tests", "scripts", "generated_python",
                "production_overlay"}
    for phase in args.phases:
        source = args.raw_root / phase
        state = json.loads((source / "validation_status.json").read_text())
        assert state["stage"] == "complete", (phase, state["stage"])
        check = json.loads((source / "evidence/integrity_checks.json").read_text())
        assert check["all_expected_instances_reported"] and check["installed_source_unchanged"]
        receipts = []
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".json", ".jsonl", ".log"}:
                continue
            relative = path.relative_to(source)
            if set(relative.parts) & excluded or path.name == "generated_python_manifest.json":
                continue
            original = path.read_bytes()
            content = original.decode()
            if path.suffix == ".json":
                content = json.dumps(clean_value(json.loads(content)), ensure_ascii=False, indent=2) + "\n"
            elif path.suffix == ".jsonl":
                content = "\n".join(json.dumps(clean_value(json.loads(line)), ensure_ascii=False)
                                    for line in content.splitlines()) + "\n"
            else:
                content = clean(content)
            target = args.output / phase / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            receipts.append({"path": str(relative), "original_sha256": hashlib.sha256(original).hexdigest(),
                             "archived_sha256": hashlib.sha256(content.encode()).hexdigest()})
        (args.output / phase / "archive_manifest.json").write_text(json.dumps(receipts, indent=2) + "\n")
        print(phase, check["instance_counts"])


if __name__ == "__main__":
    main()
