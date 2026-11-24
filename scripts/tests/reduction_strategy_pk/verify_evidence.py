"""Audit archive hashes, actual graph outputs, and timings recomputed from profiler files."""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import tempfile

from common import parse_profile, write_json


def recompute_profile(backend, source, batch):
    with tempfile.TemporaryDirectory(prefix="reduction-evidence-") as temporary:
        target = Path(temporary)
        names = ("trace.json",) if backend == "cuda" else ("trace_view.json", "kernel_details.csv")
        for name in names:
            found = list(source.rglob(name))
            if len(found) != 1:
                raise RuntimeError(f"Expected one {name} under {source}")
            shutil.copyfile(found[0], target / name)
        return parse_profile(backend, target, batch=batch)


def check_measurement(parsed, recorded, label, errors):
    maximum = 0.0
    samples, originals = parsed["samples_us"], recorded["samples_us"]
    if len(samples) != len(originals):
        errors.append("Profiler sample count differs: " + label)
    else:
        maximum = max(abs(a - b) for a, b in zip(samples, originals))
        # Python versions differ in float sum compensation. This tolerance is
        # far below profiler timestamp resolution.
        if not all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(samples, originals)):
            errors.append("Profiler samples differ: " + label)
    for field in ("median_us", "min_us", "max_us"):
        if not math.isclose(parsed[field], recorded[field], rel_tol=1e-12, abs_tol=1e-12):
            errors.append(f"Profiler {field} differs: {label}")
    for field in ("events_per_call", "batch"):
        if parsed[field] != recorded[field]:
            errors.append(f"Profiler {field} differs: {label}")
    return maximum


def check_contract(run, result, errors):
    case = result["case"]
    prefix = case["id"]
    selected = result.get("selected", [])
    vector_cores = run.get("execution_policy", {}).get("npu_vector_cores", 56)
    if result["status"] == "PASS":
        expected = {"plain", "split", "cooperative"}
        if result["backend"] == "cuda":
            expected.add("natural")
        if {r["strategy"] for r in selected} != expected or len(selected) != len(expected):
            errors.append(f"Missing strategy: {prefix}")
        if not result.get("eager_check", {}).get("passed"):
            errors.append(f"Failed eager reference: {prefix}")
        stride = {"contiguous": [case["R"], 1], "stride2": [2 * case["R"], 2],
                  "transpose": [1, case["X"]]}[case["layout"]]
        if result.get("input_stride") != stride:
            errors.append(f"Input layout differs: {prefix}")
    for record in selected:
        strategy = record["strategy"]
        label = f"{prefix}/{strategy}"
        if record["status"] == "NOT_APPLICABLE":
            if result["backend"] != "npu" or strategy != "cooperative" or case["X"] * 2 <= vector_cores:
                errors.append(f"Unexpected applicability exclusion: {label}")
            continue
        if record["status"] != "PASS":
            continue
        if record.get("entry") == "inductor":
            config = record["config"]
            wanted = (strategy in ("split", "natural"), strategy in ("cooperative", "natural"),
                      strategy == "cooperative")
            actual = (config.get("split_reductions"), config.get("triton.cooperative_reductions"),
                      config.get("triton.force_cooperative_reductions"))
            if actual != wanted:
                errors.append(f"Feature settings differ: {label}")
        elif strategy == "cooperative":
            splits = record["config"]["S"]
            if splits > vector_cores or case["X"] * splits > vector_cores or record.get("grid") != [splits, case["X"]]:
                errors.append(f"Cooperative residency constraint violated: {label}")
            if not record.get("exchange", {}).get("passed"):
                errors.append(f"Missing strict integer exchange: {label}")
        timing = record.get("timing", {})
        medians = timing.get("round_medians_us", [])
        if len(medians) != 3 or any(not math.isfinite(v) or v <= 0 for v in medians):
            errors.append(f"Invalid device durations: {label}")
        elif not math.isclose(timing["median_us"], statistics.median(medians), rel_tol=1e-10):
            errors.append(f"Final median differs: {label}")
        if len(record.get("confirmation", [])) != 3:
            errors.append(f"Missing confirmation session: {label}")
        for check in record.get("accuracy", []):
            if any(output["mismatched"] or output["nonfinite"] for output in check["outputs"]):
                errors.append(f"Numerical errors in a passing record: {label}")
        if record.get("entry") == "dsl_prototype":
            finalists = [item for item in result["candidates"] if item.get("strategy") == strategy
                         and item.get("entry") == "dsl_prototype" and item["status"] == "PASS"
                         and item.get("confirmation") and item.get("timing")]
            if not finalists or not math.isclose(timing["median_us"], min(
                    item["timing"]["median_us"] for item in finalists), rel_tol=1e-12):
                errors.append(f"Selected candidate is not the fastest correct finalist: {label}")
    if result["status"] == "PASS" and result["backend"] == "npu":
        env = result["environment"]
        if str(env["physical_device"]) != env["requested_device"] or env["effective_vector_cores"] != vector_cores:
            errors.append(f"Physical NPU differs: {prefix}")


def audit(root):
    manifest = json.loads((root / "evidence_manifest.json").read_text())
    errors, sessions, checked = [], 0, 0
    max_sample_error = 0.0
    search_sessions, searched_candidates = 0, 0
    for item in manifest["files"]:
        path = root / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["archived_sha256"]:
            errors.append("Hash mismatch: " + item["path"])
    runroot = root / manifest["experiment"]
    run = json.loads((runroot / "run.json").read_text())
    expected_cases = {case["id"]: case for case in run["cases"]}
    actual_cases = {path.parent.name for path in runroot.glob("cases/*/result.json")}
    if actual_cases != set(expected_cases):
        errors.append("Run manifest/result coverage differs")
    installed_path = runroot / "installed-source.json"
    installed = json.loads(installed_path.read_text())["files"] if installed_path.exists() else []
    archived_sources = {Path(item["path"]).name: item["original_sha256"] for item in manifest["files"]
                        if item["path"].startswith("source/")}
    for name, digest in run["source_sha256"].items():
        if archived_sources.get(name) != digest:
            errors.append("Run/source mismatch: " + name)
    for path in sorted(runroot.glob("cases/*/result.json")):
        result = json.loads(path.read_text())
        if result["case"] != expected_cases.get(path.parent.name):
            errors.append("Case definition differs: " + path.parent.name)
        check_contract(run, result, errors)
        for item in installed:
            module = item["module_path"].removesuffix(".py").replace("/", ".")
            recorded = result.get("environment", {}).get("source_sha256", {}).get(module)
            if recorded and recorded != item["sha256"]:
                errors.append(f"Installed source changed: {path.parent.name}/{module}")
        if result["status"] not in ("PASS", "INCOMPLETE"):
            continue
        finalists = {record.get("id", record["strategy"]): record
                     for record in result.get("candidates", []) if record.get("confirmation")}
        for record in result.get("selected", []):
            if record["status"] == "PASS":
                label = record.get("id", record["strategy"])
                if label in finalists and finalists[label] != record:
                    errors.append(f"Selected/finalist record differs: {path.parent.name}/{label}")
                finalists[label] = record
        for record in finalists.values():
            if record["status"] != "PASS":
                continue
            if len(record.get("accuracy", [])) < 4 or not all(check["passed"] for check in record["accuracy"]):
                errors.append(f"Missing accuracy check: {path.parent.name}/{record['strategy']}")
            if run.get("profile_graph") == "1" and (len(record.get("graph_accuracy", [])) < 6 or
                                                     not all(check["passed"] for check in record.get("graph_accuracy", []))):
                errors.append(f"Failed graph check: {path.parent.name}/{record['strategy']}")
        for repeat in range(3):
            source = path.parent / f"confirm-{repeat}"
            parsed = recompute_profile(result["backend"], source, result.get("confirmation_batch", 5))
            graph_checks = json.loads((source / "graph_checks.json").read_text()) if run.get("profile_graph") == "1" else {}
            if set(parsed) != set(finalists):
                errors.append(f"Confirmation profiler candidate coverage differs: {path.parent.name}/{repeat}")
            for label, record in finalists.items():
                max_sample_error = max(max_sample_error, check_measurement(
                    parsed[label], record["confirmation"][repeat], f"{path.parent.name}/{label}/{repeat}", errors))
                if run.get("profile_graph") == "1":
                    checks = graph_checks.get(label, [])
                    if len(checks) != 2 or {c["phase"] for c in checks} != {"before_profile", "after_profile"} or not all(c["passed"] for c in checks):
                        errors.append(f"Missing graph output validation: {path.parent.name}/{label}/{repeat}")
            sessions += 1
        if result["backend"] == "npu" and not result.get("tuning_source"):
            candidates = {item["id"]: item for item in result["candidates"] if item.get("search_timing")}
            consumed = set()
            for phase in ("search", "refine"):
                source = path.parent / phase
                if not source.exists():
                    continue
                parsed = recompute_profile("npu", source, 3)
                checks = json.loads((source / "graph_checks.json").read_text()) if run.get("profile_graph") == "1" else {}
                for label, measurement in parsed.items():
                    candidate = candidates.get(label)
                    if candidate is None:
                        errors.append(f"Unrecorded search candidate: {path.parent.name}/{label}")
                        continue
                    max_sample_error = max(max_sample_error, check_measurement(
                        measurement, candidate["search_timing"], f"{path.parent.name}/{phase}/{label}", errors))
                    if run.get("profile_graph") == "1" and (len(checks.get(label, [])) != 2 or
                            not all(check["passed"] for check in checks.get(label, []))):
                        errors.append(f"Missing search graph validation: {path.parent.name}/{label}")
                    consumed.add(label)
                search_sessions += 1
            if consumed != set(candidates):
                errors.append("Search/refine candidate coverage differs: " + path.parent.name)
            searched_candidates += len(consumed)
            for strategy in ("split", "cooperative"):
                expected = sorted((item for item in candidates.values() if item["strategy"] == strategy),
                                  key=lambda item: item["search_timing"]["median_us"])[:2]
                actual = {item["id"] for item in result["candidates"] if item.get("confirmation")
                          and item.get("entry") == "dsl_prototype" and item["strategy"] == strategy}
                if actual != {item["id"] for item in expected}:
                    errors.append(f"Profiler shortlist selection differs: {path.parent.name}/{strategy}")
        checked += 1
    return dict(experiment=manifest["experiment"], hashed_files=len(manifest["files"]),
                checked_cases=checked, recomputed_profiler_sessions=sessions,
                recomputed_search_sessions=search_sessions, checked_search_candidates=searched_candidates,
                max_recomputed_sample_abs_error_us=max_sample_error, errors=errors)


def audit_shared_semantics(roots):
    fingerprints = {}
    originals = {}
    results = {}
    errors = []
    for root in roots:
        manifest = json.loads((root / "evidence_manifest.json").read_text())
        experiment = manifest["experiment"]
        for item in manifest["files"]:
            if item["path"].startswith(experiment + "/cases/") and item["path"].endswith("/result.json"):
                originals[experiment, item["path"]] = item["original_sha256"]
        for path in (root / experiment).glob("cases/*/result.json"):
            results[experiment, path.parent.name] = json.loads(path.read_text())
        source_text = (root / "source/common.py").read_text()
        tree = ast.parse(source_text)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in (
                    "eager_function", "cpu_inputs", "device_inputs", "reference", "compare"):
                # ast.dump changes between Python releases (type_params and
                # omitted empty fields). Compare the exact normalized function
                # source so independent machines produce the same fingerprint.
                digest = hashlib.sha256(ast.get_source_segment(source_text, node).encode()).hexdigest()
                old = fingerprints.setdefault(node.name, digest)
                if old != digest:
                    errors.append(f"Shared input/math/check semantics differ: {experiment}/{node.name}")
        backend = json.loads((root / experiment / "run.json").read_text())["backend"]
        if backend == "npu":
            tree = ast.parse((root / "source/npu_kernels.py").read_text())
            barrier = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                           and node.name == "x_grid_barrier_debug")
            expected = ast.parse('''def x_grid_barrier_debug(sem):
    tl.debug_barrier()
    extension.sync_block_all("all_vector", 0)
    tl.debug_barrier()
''').body[0]
            if ast.dump(ast.Module(body=barrier.body, type_ignores=[])) != ast.dump(
                    ast.Module(body=expected.body, type_ignores=[])):
                errors.append("Requested debug barrier differs: " + experiment)
    checked = 0
    for (experiment, case_id), result in results.items():
        source = result.get("tuning_source")
        if not source:
            continue
        source_name = Path(source["path"]).parts[-4]
        source_path = f"{source_name}/cases/{case_id}/result.json"
        if originals.get((source_name, source_path)) != source["sha256"]:
            errors.append(f"Fixed finalist source digest differs: {experiment}/{case_id}")
            continue
        previous = results[source_name, case_id]
        expected = {json.dumps(item["config"], sort_keys=True) for item in previous["candidates"]
                    if item.get("entry") == "dsl_prototype" and item["status"] == "PASS"
                    and item.get("confirmation")}
        actual = {json.dumps(item["config"], sort_keys=True) for item in result["candidates"]
                  if item.get("entry") == "dsl_prototype"}
        if actual != expected:
            errors.append(f"Fixed finalist configuration set differs: {experiment}/{case_id}")
        if result["status"] == "PASS" and any(
                item["status"] != "PASS" or len(item.get("confirmation", [])) != 3
                for item in result["candidates"] if item.get("entry") == "dsl_prototype"):
            errors.append(f"Fixed finalist confirmation is incomplete: {experiment}/{case_id}")
        checked += 1
    return dict(shared_semantics_sha256=fingerprints,
                shared_semantics_hash_method="SHA256 of newline-normalized exact function source segments",
                checked_fixed_finalist_cases=checked, errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    auditor_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    results = [audit(root) for root in args.input]
    contracts = audit_shared_semantics(args.input)
    report = dict(auditor_sha256=auditor_sha256, archives=results, cross_archive_contracts=contracts)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if any(result["errors"] for result in results) or contracts["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
