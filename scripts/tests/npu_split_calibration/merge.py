#!/usr/bin/env python3
"""Merge independently collected ranges offline, then recompute paired reports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

from analyze import summarize
from common import (collection_complete, external_output, measurement_settings, read_json,
                    require_profiler_settings, source_hashes, validate_collection, write_json)
from profiling import validate_captures
from policy import digest
from run import budget_cases


def prepare(inputs, out):
    """Validate every manifest and result before creating the destination."""
    if len(set(inputs)) != len(inputs):
        raise ValueError("input directories must be distinct")
    manifests = []
    for root in inputs:
        if root == out or root in out.parents or out in root.parents:
            raise ValueError("merge output and input directories must not contain one another")
        active = root / "active.json"
        if active.exists() and read_json(active).get("status") in ("RUNNING", "MERGING"):
            raise ValueError(f"collection is still marked RUNNING: {root}; finish/resume it before merging")
        manifest = read_json(root / "manifest.json")
        require_profiler_settings(manifest["settings"])
        validate_collection(manifest)
        if manifest.get("schema") != "npu-split-manifest-v3":
            raise ValueError(f"unsupported manifest schema: {root}")
        manifests.append(manifest)
    first = manifests[0]
    if first["source_hashes"] != source_hashes():
        raise ValueError("collector source differs from this merger; use the same suite revision on every machine")
    universe = first["collection"]["universe"]
    criteria = first["settings"]["criteria"]
    owners, reports = {}, []
    for root, manifest in zip(inputs, manifests):
        for key in ("environment", "source_hashes", "settings", "table"):
            if manifest[key] != first[key]:
                raise ValueError(f"{key} mismatch in {root}; use separate collections for different experiments")
        if manifest["collection"]["universe"] != universe:
            raise ValueError(f"case universe/order differs in {root}; use identical filters or --cases-json")
        expected, excluded = budget_cases(universe, manifest["settings"])
        selected = set(manifest["collection"]["selected_case_ids"])
        expected_ids = {case["id"] for case in expected}
        actual_ids = {case["id"] for case in manifest["cases"]}
        declared_ids = expected_ids if manifest["collection"]["partition"]["kind"] == "merged" else expected_ids & selected
        excluded_by_id = {item["case"]["id"]: item for item in excluded}
        declared_excluded = excluded_by_id if manifest["collection"]["partition"]["kind"] == "merged" else {
            key: value for key, value in excluded_by_id.items() if key in selected}
        if actual_ids != declared_ids or {item["case"]["id"]: item for item in manifest["excluded_cases"]} != declared_excluded:
            raise ValueError(f"memory exclusions differ from the declared budget in {root}")
        overlap = selected & owners.keys()
        if overlap:
            raise ValueError(f"overlapping ranges: {len(overlap)} cases, e.g. {sorted(overlap)[0]}; no automatic winner selection")
        for case in manifest["cases"]:
            path = root / "cases" / case["id"] / "result.json"
            if not path.exists():
                continue  # Preserve missing/failed cases; analysis must report them.
            if case["id"] not in selected:
                raise ValueError(f"result outside the selected collection: {path}")
            result = read_json(path)
            if (result.get("environment") != manifest["environment"]
                    or result.get("source_hashes") != manifest["source_hashes"]
                    or measurement_settings(result.get("settings", {})) != measurement_settings(manifest["settings"])
                    or result.get("table_digest") != digest(manifest["table"])
                    or result.get("case") != case):
                raise ValueError(f"result identity mismatch: {path}")
            if result.get("status") == "RUNNING" and not (path.parent / "error.json").exists():
                raise ValueError(f"unfinished result: {path}; resume the collection before merging")
            if result.get("status") == "PASS":
                if result.get("schema") != "npu-split-result-v3":
                    raise ValueError(f"legacy timing result: {path}")
                for probe in result["probes"]:
                    validate_captures(probe, manifest["settings"], path.parent)
        owners.update({case_id: root for case_id in selected})
        reports.append(dict(directory=str(root), created_utc=manifest.get("created_utc"),
                            manifest_digest=digest(manifest), partition=manifest["collection"]["partition"],
                            visible_devices=manifest.get("visible_devices"), selected_cases=len(selected)))

    cases, excluded = budget_cases(universe, first["settings"])
    missing = [case["id"] for case in universe if case["id"] not in owners]
    collection = dict(first["collection"], partition=dict(kind="merged"),
                      selected_case_ids=[case["id"] for case in universe if case["id"] in owners])
    provenance = dict(inputs=reports, case_sources={key: str(value) for key, value in owners.items()})
    manifest = dict(schema="npu-split-manifest-v3", cases=cases, excluded_cases=excluded,
                    settings=first["settings"], table=first["table"],
                    initial_case_ids=[case["id"] for case in universe],
                    environment=first["environment"], source_hashes=first["source_hashes"],
                    visible_devices=None, schedule="offline_merge", collection=collection,
                    merge=provenance, created_utc=datetime.now(timezone.utc).isoformat())
    validate_collection(manifest)
    return manifest, owners, missing, criteria


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="finished run.py output directories copied to this machine")
    parser.add_argument("--out", type=Path, required=True, help="new merged directory outside the repository")
    args = parser.parse_args()
    inputs = [path.expanduser().resolve() for path in args.inputs]
    out = external_output(args.out)
    if out.exists() and any(out.iterdir()):
        parser.error("merge output must be empty; use a new directory")
    try:
        manifest, owners, missing, criteria = prepare(inputs, out)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))

    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "active.json", dict(status="MERGING"))
    write_json(out / "manifest.json", manifest)
    # The full definition can be used directly for a continuation with run.py
    # --cases-json ... --resume --extend, retaining already completed cases.
    write_json(out / "all_cases.json", dict(cases=manifest["collection"]["universe"]))
    for index, root in enumerate(inputs):
        evidence = out / "inputs" / f"{index:03d}"
        evidence.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "manifest.json", evidence / "manifest.json")
        if (root / "preflight").exists():
            shutil.copytree(root / "preflight", evidence / "preflight")
    for case in manifest["cases"]:
        root = owners.get(case["id"])
        if root is None:
            continue
        source = root / "cases" / case["id"]
        if not source.exists():
            continue
        destination = out / "cases" / case["id"]
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("result.json", "error.json", "worker.log"):
            if (source / name).exists():
                shutil.copy2(source / name, destination / name)
        if (source / "result.json").exists():
            result = read_json(source / "result.json")
            for probe in result.get("probes", []):
                for capture in probe.get("profiles", []):
                    relative = Path(capture["csv"])
                    csv_source = (source / relative).resolve()
                    if relative.is_absolute() or source.resolve() not in csv_source.parents:
                        raise ValueError("profiler CSV path escaped source case")
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if csv_source.exists():
                        shutil.copy2(csv_source, target)
    _, failures = summarize(out, criteria["relative_margin"], criteria["absolute_margin_us"], criteria["metric"])
    complete = collection_complete(manifest) and not failures
    holdout_passed = None
    if manifest["settings"]["policy"] == "table" and "policy" in manifest["settings"]["modes"]:
        holdout_passed = read_json(out / "holdout.audit.json")["passed"]
    status = "COMPLETE" if complete else "INCOMPLETE"
    write_json(out / "merge.audit.json", dict(status=status, inputs=manifest["merge"]["inputs"],
               expected_cases=len(manifest["collection"]["universe"]), selected_cases=len(owners),
               missing_case_ids=missing, failed_cases=failures, planned_exclusions=len(manifest["excluded_cases"]),
               holdout_passed=holdout_passed, profiler_artifacts="kernel_details.csv only"))
    write_json(out / "active.json", dict(status=status, failed_cases=len(failures), missing_ranges=len(missing)))
    print(f"Merged {len(inputs)} directories: {len(owners)}/{len(manifest['collection']['universe'])} cases assigned, "
          f"{len(failures)} failed/missing results. {out / 'merge.audit.json'}")
    if not complete:
        raise SystemExit(1)
    if holdout_passed is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
