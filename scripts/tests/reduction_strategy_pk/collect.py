"""Collect completed experiment evidence, omitting rebuildable caches and private paths."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile


def sanitize(text, workspace):
    text = text.replace(str(workspace), "<workspace>")
    text = re.sub(r"/home/[^/\s\"']+/miniforge3/envs/[^/\s\"']+", "<python-env>", text)
    text = re.sub(r"/home/[^/\s\"']+", "<home>", text)
    text = re.sub(r"/mnt/raid1/[^/\s\"']+/data/dev", "<workspace-parent>", text)
    text = re.sub(r"(['\"]backend_hash['\"]\s*:\s*['\"])[^'\"]+", r"\1<backend-hash>", text)
    text = re.sub(r"\bPID\s*[:=]\s*\d+", "PID:<process>", text)
    text = re.sub(r'("pid"\s*:\s*)\d+', r'\1"<process>"', text)
    text = re.sub(r"\[[0-9]{5,}\](?=\s+profiler\.py)", "[<process>]", text)
    text = re.sub(r"\b[^/\s\"']+_\d+_\d{14,}_ascend_pt\b", "<profile>", text)
    text = re.sub(r"PROF_\d+_\d+_[A-Za-z0-9]+", "<device-profile>", text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--helper", action="append", type=Path, default=[],
                        help="Additional coordinator/monitor scripts used outside the frozen worker source")
    args = parser.parse_args()
    root = args.input.resolve()
    workspace = root.parent
    selected = set()
    terminal = set()
    statuses = {}
    for result in (root / "cases").glob("*/result.json"):
        value = json.loads(result.read_text())
        terminal.add(result.parent.name)
        statuses[result.parent.name] = value["status"]
        for record in value.get("selected", []):
            selected.add((result.parent.name, record.get("id", record["strategy"])))
    sources = []
    output_manifest = []
    total_bytes = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        def add(path, name, filtered_trace=False):
            nonlocal total_bytes
            original = path.read_bytes()
            if filtered_trace:
                trace = json.loads(original)
                events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
                chosen = []
                for event in events:
                    category = event.get("cat")
                    label = (event.get("name", "").startswith("pk:") or event.get("name") == "pk-primer") and category in ("cpu_op", "user_annotation")
                    device = category in ("kernel", "gpu_memset", "gpu_memcpy")
                    if event.get("ph") == "X" and (label or device):
                        copy = {key: event[key] for key in ("name", "cat", "ph", "ts", "dur") if key in event}
                        copy["args"] = {key: value for key, value in event.get("args", {}).items()
                                        if key in ("grid", "block", "stream", "device", "registers per thread", "shared memory")}
                        chosen.append(copy)
                text = json.dumps(dict(traceEvents=chosen, evidence_scope="Original device events and labels; host details removed"))
            else:
                text = original.decode("utf-8")
            payload = sanitize(text, workspace).encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
            total_bytes += len(payload)
            output_manifest.append(dict(path=name, original_sha256=hashlib.sha256(original).hexdigest(),
                                        archived_sha256=hashlib.sha256(payload).hexdigest(),
                                        bytes=len(payload), filtered_trace=filtered_trace))

        for folder, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [name for name in dirs if name not in ("cache", "logs", "FRAMEWORK", "__pycache__")
                       and not name.startswith("PROF_")]
            for name in files:
                path = Path(folder) / name
                rel = path.relative_to(root)
                parts = list(rel.parts)
                if parts[0] == "cases" and (len(parts) < 2 or parts[1] not in terminal):
                    continue
                if (name in ("active.json", "progress.json") and
                        (len(parts) < 2 or statuses.get(parts[1]) == "PASS")) or name.endswith(".tmp"):
                    continue
                if name.endswith(".log") and (len(parts) < 2 or statuses.get(parts[1]) == "PASS"):
                    continue
                allowed = (path.suffix in (".json", ".py", ".ttir", ".ttgir", ".ptx", ".log") or
                           name == "kernel_details.csv")
                if path.suffix == ".ttadapter":
                    allowed = len(parts) > 2 and ((parts[1], parts[2]) in selected or parts[2] == "exchange"
                                                 or statuses.get(parts[1]) != "PASS")
                if not allowed or name in ("profiler_metadata.json", "profiler_info.json", "sample.json"):
                    continue
                for index, part in enumerate(parts):
                    if part.endswith("_ascend_pt"):
                        parts[index] = "profile"
                add(path, root.name + "/" + "/".join(parts), name in ("trace.json", "trace_view.json"))
        for path in sorted(args.source.glob("*.py")):
            name = "source/" + path.name
            add(path, name)
            sources.append(name)
        for path in args.helper:
            name = "supporting-source/" + path.name
            add(path, name)
            sources.append(name)
        manifest = dict(experiment=root.name, completed_cases=len(terminal), statuses=statuses,
                        source_files=sources, files=output_manifest,
                        omitted="Rebuildable caches, CANN binary/database duplication, non-winning adapter IR, routine logs",
                        redaction="Private paths/backend hashes; traces retain device events and CPU labels with original timestamps/durations")
        payload = (json.dumps(manifest, indent=2) + "\n").encode()
        info = tarfile.TarInfo("evidence_manifest.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    print(json.dumps(dict(cases=len(terminal), files=len(output_manifest), uncompressed_bytes=total_bytes,
                          archive_bytes=args.output.stat().st_size)))


if __name__ == "__main__":
    main()
