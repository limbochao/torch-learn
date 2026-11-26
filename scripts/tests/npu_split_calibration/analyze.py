"""Paired measurements and conservative, bounded admission proposals (no S search)."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
import statistics

from common import (METRIC, TIMING_METHOD, collection_complete, measurement_settings, read_json,
                    require_profiler_settings, timing_contract, validate_collection, write_json)
from profiling import sample_evidence, validate_captures
from cases import PRIMITIVES, context_id
from policy import VERSION, digest
from refinement import audit_holdout, observations, propose_intervals, refinement_plan
from screening import threshold_guidance


def paired_stats(samples, mode, metric, relative_margin, absolute_margin):
    grouped = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        grouped[(sample["session"], sample["round"])][sample["mode"]].append(sample[metric])
    sessions = defaultdict(list)
    plain, other = [], []
    for (session, _), values in grouped.items():
        if "plain" not in values or mode not in values:
            continue
        p, s = statistics.mean(values["plain"]), statistics.mean(values[mode])
        plain.append(p)
        other.append(s)
        margin = max(relative_margin * p, absolute_margin)
        sessions[session].append((p - s, p - s - margin))
    if not sessions:
        return dict(decision="NO_PAIRS")
    gains = [statistics.median(pair[0] for pair in values) for values in sessions.values()]
    excess = [statistics.median(pair[1] for pair in values) for values in sessions.values()]
    # Resample session summaries, not individual correlated calls. Sessions still
    # share one process/device/time window and are not independent experiments.
    # This is an empirical bound, not a simultaneous guarantee over unseen shapes.
    rng = random.Random(50315)
    bootstrap = sorted(statistics.mean(rng.choices(excess, k=len(excess))) for _ in range(2000))
    lower = bootstrap[max(0, math.floor(0.01 * len(bootstrap)))]
    enough = len(sessions) >= 3 and min(map(len, sessions.values())) >= 5
    decision = "WIN" if enough and min(excess) > 0 and lower > 0 else (
        "REGRESSION" if max(gains) < 0 else "UNCERTAIN")
    p, s = statistics.median(plain), statistics.median(other)
    return dict(decision=decision, plain_us=p, strategy_us=s, speedup=p / s, gain_us=p - s,
                gain_pct=100 * (p - s) / p, session_gains_us=gains,
                min_session_excess_us=min(excess), lower99_excess_us=lower,
                sessions=len(sessions), min_paired_rounds=min(map(len, sessions.values())))


def evidence_status(result, probe, mode):
    if probe["status"] != "PASS":
        return probe["status"]
    settings = result["settings"]
    expected_pairs = {(session, round_id, name): 2 for session in range(settings["sessions"])
                      for round_id in range(settings["rounds"]) for name in settings["modes"]}
    actual_pairs = Counter((s["session"], s["round"], s["mode"]) for s in probe.get("samples", []))
    if actual_pairs != expected_pairs:
        return "INCOMPLETE_PAIRS"
    invalid = sample_evidence(probe, settings)
    if invalid:
        return invalid
    strategies = result.get("strategies", {})
    if "plain" not in strategies or mode not in strategies:
        return "MISSING_STRATEGY"
    if any(name not in probe.get("metadata", {}) for name in ("plain", mode)):
        return "MISSING_METADATA"
    plain_compile = probe["metadata"]["plain"]["active_compile_id"]
    strategy_compile = probe["metadata"][mode]["active_compile_id"]
    if any(s["compile_id"] == plain_compile for s in strategies["plain"]["splits"]):
        return "BASELINE_SPLIT"
    splits = [s for s in strategies[mode]["splits"] if s["compile_id"] == strategy_compile]
    decisions = [d for d in strategies[mode]["decisions"] if d["compile_id"] == strategy_compile and d["depth"] == 0]
    if (result["case"]["op"] in {op for op, _ in PRIMITIVES} and probe["R"] > 1 and not decisions):
        return "MISSING_REDUCTION_DECISION"
    for selected_mode in ("plain", mode):
        info = probe["metadata"][selected_mode]
        launches = info["launches"]
        if not launches or not info["selected"] or not strategies[selected_mode]["sources"]:
            return "MISSING_CODEGEN_EVIDENCE"
        if any(launch.get("fallback") for launch in launches):
            return "FALLBACK"
        if any("grid" not in launch for launch in launches):
            return "MISSING_GRID"
    if result["case"]["op"] in ("prod", "any", "all", "xor_sum"):
        patterns = {check["pattern"] for check in probe.get("correctness_patterns", [])}
        if not {"identity", "tail", "partition_edges"} <= patterns:
            return "MISSING_CORRECTNESS_PATTERNS"
    if not splits:
        requested = any(d["S"].lstrip("-").isdigit() and int(d["S"]) > 1 for d in decisions)
        return "SPLIT_NOT_REALIZED" if requested else "NO_SPLIT"
    if mode == "candidate" or result["settings"]["policy"] == "table":
        if not any(d["compile_id"] == plain_compile for d in strategies["plain"]["decisions"]):
            return "MISSING_BASELINE_DECISION"
        expected = sorted(int(d["S"]) for d in decisions if int(d["S"]) > 1)
        actual = sorted(int(s["arguments"]["split"]) for s in splits)
        if expected != actual or any(s["depth"] > 0 for s in splits):
            return "UNEXPECTED_SPLIT_STRUCTURE"
    if len(probe["metadata"][mode]["launches"]) < 2:
        return "MISSING_SPLIT_LAUNCH"
    return "VERIFIED_SPLIT"


def summarize(directory, relative_margin=0.08, absolute_margin=0.2, metric=METRIC):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    require_profiler_settings(manifest["settings"])
    if metric != METRIC or manifest.get("schema") != "npu-split-manifest-v3":
        raise ValueError("only profiler kernel duration manifests can be analyzed; recollect legacy Event data")
    if "collection" in manifest:
        validate_collection(manifest)
    complete = collection_complete(manifest)
    screening = any(case["role"] == "screening" for case in manifest["cases"])
    rows, results, failures = [], [], []
    for case_index, case in enumerate(manifest["cases"], 1):
        if case_index % 100 == 0:
            print(f"Analyzing {case_index}/{len(manifest['cases'])} cases", flush=True)
        case_dir = directory / "cases" / case["id"]
        path = case_dir / "result.json"
        result = read_json(path) if path.exists() else None
        if (case_dir / "error.json").exists() or result is None or result.get("status") != "PASS":
            failures.append(dict(case_id=case["id"], status=(result or {}).get("status", "MISSING_OR_ERROR")))
        if result is None:
            continue
        if (result.get("environment") != manifest["environment"]
                or result.get("source_hashes") != manifest["source_hashes"]
                or result.get("table_digest") != digest(manifest.get("table", {}))
                or measurement_settings(result.get("settings", {})) != measurement_settings(manifest["settings"]) or result.get("case") != case):
            failures.append(dict(case_id=case["id"], status="RESULT_IDENTITY_MISMATCH"))
            continue
        expected_probes = {(index, cache, execution) for index, _ in enumerate(case.get("runtime_shapes", [[case["X"], case["R"]]]))
                           for cache in manifest["settings"]["cache_modes"] for execution in manifest["settings"]["executions"]}
        actual_probes = {(p["probe_id"], p["cache"], p.get("execution", "")) for p in result["probes"]}
        if actual_probes != expected_probes or len(result["probes"]) != len(expected_probes):
            failures.append(dict(case_id=case["id"], status="INCOMPLETE_PROBES"))
        if result.get("schema") != "npu-split-result-v3":
            failures.append(dict(case_id=case["id"], status="INVALID_RESULT_SCHEMA"))
            continue
        invalid_profiles = False
        for probe in result["probes"]:
            if probe.get("status") == "PASS":
                try:
                    validate_captures(probe, manifest["settings"], case_dir)
                except (ValueError, KeyError, OSError) as exc:
                    failures.append(dict(case_id=case["id"], status="INVALID_PROFILER_EVIDENCE", error=str(exc)))
                    invalid_profiles = True
        if invalid_profiles:
            continue
        results.append(result)
        for probe in result["probes"]:
            for mode in manifest["settings"]["modes"]:
                if mode == "plain":
                    continue
                row = dict(case_id=case["id"], op=case["op"], dtype=case["dtype"], layout=case["layout"],
                           role=case["role"], X=probe["X"], R=probe["R"], probe_id=probe["probe_id"],
                           metric=METRIC, timing_method=TIMING_METHOD,
                           cache=probe["cache"], execution=probe.get("execution", ""), mode=mode, context=context_id(case))
                row["physical_device"] = result.get("device_assignment", {}).get("visible_devices", "")
                row["hostname"] = result.get("device_assignment", {}).get("hostname", "")
                metadata = probe.get("metadata", {})
                active = metadata.get(mode, {}).get("active_compile_id")
                decisions = [d for d in result.get("strategies", {}).get(mode, {}).get("decisions", []) if d["compile_id"] == active]
                row["ir_decisions"] = decisions
                if case["role"] == "control":
                    targets = {"argmax": ("argmax", "argmax_value", "argmax_with_value"),
                               "argmin": ("argmin", "argmin_value", "argmin_with_value"),
                               "var_mean": ("welford_reduce", "welford_combine"), "cumsum": ("scan",)}[case["op"]]
                    seen = [d for d in decisions if d["meta"].get("op") in targets]
                    row["control_status"] = ("PATH_NOT_EXERCISED" if not seen else
                        "UNEXPECTED_SPECIAL_SPLIT" if any(int(d["S"]) != 1 for d in seen) else "VERIFIED_SPECIAL_UNSPLIT")
                row["requested_S"] = [d["S"] for d in decisions if not d["depth"]]
                row["actual_S"] = [s["arguments"].get("split") for s in result.get("strategies", {}).get(mode, {}).get("splits", [])
                                   if s["compile_id"] == active]
                row["tiling_signature"] = {name: [{k: launch.get(k) for k in ("selected_config", "runtime_blocks")}
                                               for launch in metadata.get(name, {}).get("launches", [])] for name in ("plain", mode)}
                for name, label in (("plain", "plain"), (mode, "strategy")):
                    launches = metadata.get(name, {}).get("launches", [])
                    row[label + "_grids"] = [launch.get("grid") for launch in launches]
                    row[label + "_programs"] = [launch.get("programs") for launch in launches]
                    row[label + "_compile_id"] = metadata.get(name, {}).get("active_compile_id")
                row["input_working_set_bytes"] = probe.get("input_working_set_bytes")
                status = evidence_status(result, probe, mode)
                row["evidence"] = status
                valid_times = all(math.isfinite(s.get(metric, 0)) and s.get(metric, 0) > 0 for s in probe.get("samples", []))
                stats = paired_stats(probe.get("samples", []), mode, metric, relative_margin, absolute_margin) if valid_times else dict(decision="INVALID_TIMING")
                row.update(stats)
                row["decision"] = stats["decision"] if status == "VERIFIED_SPLIT" else status
                if probe.get("reused_seed_graph"):
                    row["same_seed_graph"] = all(probe["reused_seed_graph"].values())
                rows.append(row)

    families = observations(results, rows)
    # A range can omit entire consumers/dtypes. Only the full collection can
    # establish context coverage and propose follow-up shapes or intervals.
    refinement = (refinement_plan(manifest, families) if complete else
                  dict(pending=[], blocked=[], status="WAITING_FOR_COLLECTION_MERGE"))
    guidance = threshold_guidance(manifest, rows, complete, failures) if screening else None
    proposals = propose_intervals(families, manifest["settings"].get("max_r_gap", 1.25))
    needs_refinement = bool(refinement["pending"] or refinement["blocked"])
    write_json(directory / "refinement.json", refinement)
    next_cases = [*manifest["cases"], *(item["case"] for item in refinement["pending"]),
                  *(guidance["next_cases"] if guidance else [])]
    write_json(directory / "next_cases.json", dict(cases=list({case["id"]: case for case in next_cases}.values())))
    if guidance:
        write_json(directory / "threshold-guidance.json", guidance)

    # Incomplete data must never silently widen a gate. Fail-closed globally;
    # rerun the failed/skipped cases (or explicitly create a smaller new manifest).
    table = dict(schema="npu-split-table-v3", status=("INCOMPLETE" if failures else "PARTIAL_COLLECTION" if not complete else "SCREENING_ONLY" if screening else "NEEDS_REFINEMENT" if needs_refinement else
                 "DRAFT_REQUIRES_HOLDOUT" if proposals else "NO_ADMISSIBLE_INTERVAL"),
                 environment=manifest["environment"],
                 suite_source_hashes=manifest["source_hashes"], max_autotune=manifest["settings"]["max_autotune"],
                 timing_contract=timing_contract(manifest["settings"]),
                 selector={"version": VERSION, "params": manifest["settings"]["selector"]},
                 rules=proposals if complete and not failures and not needs_refinement and not screening else [], provisional_rules=proposals,
                 collection_complete=complete,
                 refinement_pending=len(refinement["pending"]), refinement_blocked=len(refinement["blocked"]),
                 blocked_by_incomplete_cases=len(failures),
                 criteria=dict(metric=metric, relative_margin=relative_margin, absolute_margin_us=absolute_margin),
                 domain=dict(executions=manifest["settings"]["executions"], cache_modes=manifest["settings"]["cache_modes"],
                             contexts=sorted({c["op"] for c in manifest["cases"]}), dynamic=False),
                 warning="Finite measured intervals only. Validate frozen rules on independent shapes and fusion contexts; not a production guarantee.")
    write_json(directory / "thresholds.proposed.json", table)
    coverage = [dict(op=op, dtype=dtype,
                     planned_cases=sum(c["op"] == op and c["dtype"] == dtype for c in manifest["cases"]),
                     decisions=dict(Counter(r["decision"] for r in rows if r["op"] == op and r["dtype"] == dtype)))
                for op, dtype in PRIMITIVES]
    write_json(directory / "coverage.json", dict(primitive_dtype_pairs=coverage, families=len(families),
               note="A planned type is not a verified split; inspect each evidence/decision count."))
    if manifest["settings"]["policy"] == "table" and "policy" in manifest["settings"]["modes"]:
        audit = (audit_holdout(manifest["table"], manifest, rows, failures) if complete else
                 dict(passed=False, status="WAITING_FOR_COLLECTION_MERGE", table_digest=digest(manifest["table"])))
        write_json(directory / "holdout.audit.json", audit)
    write_json(directory / "summary.json", dict(rows=rows, failures=failures, table=table))
    fields = ["case_id", "role", "op", "dtype", "layout", "X", "R", "mode", "execution", "cache", "probe_id", "metric", "timing_method",
              "evidence", "decision", "plain_us", "strategy_us", "speedup", "gain_pct", "gain_us",
              "min_session_excess_us", "lower99_excess_us", "same_seed_graph"]
    fields += ["hostname", "physical_device", "requested_S", "actual_S", "plain_grids", "strategy_grids", "plain_programs", "strategy_programs",
               "plain_compile_id", "strategy_compile_id", "input_working_set_bytes", "tiling_signature", "control_status"]
    with (directory / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()} for row in rows)
    counts = Counter(row["decision"] for row in rows)
    lines = ["# NPU split calibration", "", f"Metric: `{metric}`. Relative margin {relative_margin:.1%}; absolute margin {absolute_margin:g} us.",
             "", "Metric is the sum of kernel_details.csv device task durations / forward count, including every partial/combine kernel. Host time, inter-task gaps and Event intervals are excluded. No NPU Graph is used.",
             "", f"Cases: {len(manifest['cases'])}; excluded by planned memory budget: {len(manifest.get('excluded_cases', []))}; incomplete/error cases: {len(failures)}; proposed intervals: {len(table['rules'])}.",
             "", "| Decision | Count |", "|---|---:|"]
    lines += [f"| {key} | {value} |" for key, value in sorted(counts.items())]
    if guidance:
        lines += ["", "## Quick threshold guidance", "",
                  "Screening anchors only; no deployable thresholds. Both cache conditions must WIN. "
                  "Transition midpoints are in next_cases.json; repeated --resume --extend narrows brackets without rerunning completed cases.",
                  "", "| op / dtype / layout / X | Observed winning anchors | Transition brackets |",
                  "|---|---|---|"]
        for group in guidance["groups"]:
            wins = "; ".join(", ".join(map(str, window["tested_r"])) for window in group["observed_win_windows"]) or group["status"]
            brackets = "; ".join(f"{b['lo']} ({b['left']}) .. {b['hi']} ({b['right']})" for b in group["transition_brackets"]) or "none observed"
            lines.append(f"| {group['op']} / {group['dtype']} / {group['layout']} / {group['X']} | {wins} | {brackets} |")
    lines += ["", f"Admission table status: `{table['status']}`; pending R/context probes: {len(refinement['pending'])}; memory-blocked probes: {len(refinement['blocked'])}.",
              f"Collection domain complete: `{complete}`. Partial ranges cannot export admission rules; merge them first.",
              "", "Primitive/dtype completeness is in coverage.json. S, grids, programs and selected tiling are also included in summary.csv.",
              "", "Use campaign.py to append pending probes without rerunning completed cases. Freeze only after refinement has completed."]
    if failures:
        lines += ["", "## Incomplete cases", "", "| Case | Status |", "|---|---|"]
        lines += [f"| {item['case_id']} | {item['status']} |" for item in failures[:50]]
        lines += ["", "Full failures are in summary.json; inspect cases/<id>/worker.log and error.json."]
    lines += ["", "## Slowest verified split observations", "", "| Case | Mode / execution / cache | Gain | Plain us | Split us |", "|---|---|---:|---:|---:|"]
    measured = sorted((r for r in rows if r["evidence"] == "VERIFIED_SPLIT" and "gain_pct" in r), key=lambda r: r["gain_pct"])
    lines += [f"| {r['case_id']} | {r['mode']} / {r['execution']} / {r['cache']} | {r['gain_pct']:.2f}% | {r['plain_us']:.3f} | {r['strategy_us']:.3f} |" for r in measured[:30]]
    lines += ["", "## Interpretation", "",
              "- NO_SPLIT is not a split win. Missing launch/grid evidence and recompilation are not admitted.",
              "- Every observed family context must be measured and win at EVERY admitted calibration point. Missing contexts trigger follow-up probes.",
              "- At least three passing R points and bounded spacing are required. Pending refinement or incomplete cases suppress exported rules.",
              "- Rules are bounded by measured R values, with exact X/dtype/op/read-layout keys; no extrapolation above the measured maximum.",
              "- Planned memory exclusions are recorded in manifest.json and are outside the calibrated domain.",
              "- Fusion context is not part of num_splits' key. Passing these graphs does not prove arbitrary consumers are safe.",
              "- Dynamic probes report whether the original seed graph was reused; the experimental admission table always rejects dynamic IR.",
              "- Every frozen rule/context requires fresh interior shapes, endpoints and outside controls in all execution/cache modes; see holdout.audit.json.",
              "- Sessions share a process; repeat the run on another occasion/device for independent environment checks."]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{directory / 'report.md'}: {dict(counts)}; incomplete={len(failures)}", flush=True)
    return rows, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--relative-margin", type=float, default=0.08)
    parser.add_argument("--absolute-margin-us", type=float, default=0.2)
    parser.add_argument("--metric", choices=(METRIC,), default=METRIC)
    args = parser.parse_args()
    summarize(args.directory, args.relative_margin, args.absolute_margin_us, args.metric)


if __name__ == "__main__":
    main()
