"""Evidence-driven R refinement and per-rule holdout; never search S values."""

from __future__ import annotations

from collections import defaultdict
import json
import math
import re

from cases import context_id, context_spec, estimated_working_bytes, from_spec, make_case
from policy import KEY_FIELDS, SUPPORTED, digest, family, family_id, split_boundaries, split_count


def observations(results, rows):
    lookup = {(r["case_id"], r["probe_id"], r["cache"], r["execution"]): r
              for r in rows if r["mode"] == "candidate"}
    families = {}
    for result in results:
        case = result["case"]
        if case["role"] != "calibration" or case.get("dynamic_axes"):
            continue
        ctx = context_id(case)
        for probe in result.get("probes", []):
            row = lookup.get((case["id"], probe["probe_id"], probe["cache"], probe.get("execution", "")))
            if row is None:
                continue
            for decision in row.get("ir_decisions", []):
                meta = decision["meta"]
                if (decision["depth"] or not meta.get("hinted") or meta.get("layout") == "unknown"
                        or meta.get("op") not in SUPPORTED or not all(k in meta for k in KEY_FIELDS)
                        or meta["X"] != case["X"] or meta["R"] != case["R"]):
                    continue  # cannot safely map this IR dimension back to the input case
                key = family_id(meta)
                item = families.setdefault(key, dict(key=family(meta), contexts={}, templates={}, points={}))
                item["contexts"][ctx] = context_spec(case)
                item["templates"][ctx] = case
                bucket = item["points"].setdefault(meta["R"], {}).setdefault(ctx, [])
                if row not in bucket:
                    bucket.append(row)
    return families


def winning(rows):
    return bool(rows) and all(row["decision"] == "WIN" for row in rows)


def launch_signature(rows):
    # Grid changes naturally with shape. Compare tiling/runtime blocks, not
    # kernel names or grid, to find changes in the selected implementation.
    return tuple(sorted({json.dumps(row.get("tiling_signature", {}), sort_keys=True) for row in rows}))


def reduction_tiles(rows):
    found = set()

    def visit(value):
        if isinstance(value, dict):
            for name, item in value.items():
                if re.match(r"^R\w*BLOCK", name.upper()) and isinstance(item, int) and not isinstance(item, bool) and item > 1:
                    found.add(item)
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    for row in rows:
        visit(row.get("tiling_signature", {}))
    return found


def refinement_plan(manifest, families):
    settings = manifest["settings"]
    ratio = settings.get("max_r_gap", 1.25)
    boundary_ratio = settings.get("boundary_r_gap", 1.05)
    known = {c["id"] for c in manifest["cases"]}
    pending, blocked = {}, {}

    def request(template, r, reason):
        case = from_spec(template, R=int(r), role="calibration")
        if r < 2 or case["id"] in known:
            return
        estimate = estimated_working_bytes(case, case["X"], case["R"], settings)
        destination = blocked if estimate > settings["max_working_mib"] * 1024**2 else pending
        item = destination.setdefault(case["id"], dict(case=case, reasons=[], estimated_bytes=estimate))
        if reason not in item["reasons"]:
            item["reasons"].append(reason)

    for item in families.values():
        points = item["points"]
        # Every observed consumer of a family must be tested at each promising
        # R, not merely somewhere else in the calibration matrix.
        for r, contexts in points.items():
            if any(winning(rows) for rows in contexts.values()):
                for ctx, template in item["templates"].items():
                    if ctx not in contexts:
                        request(template, r, "missing_context_at_promising_R")
        for ctx, template in item["templates"].items():
            series = {r: contexts[ctx] for r, contexts in points.items() if ctx in contexts}
            ordered = sorted(series)
            for lo, hi in zip(ordered, ordered[1:]):
                left, right = winning(series[lo]), winning(series[hi])
                if not (left or right):
                    continue  # no admitted interval is inferred here
                changed = left != right or launch_signature(series[lo]) != launch_signature(series[hi])
                target = boundary_ratio if changed else ratio
                if hi > lo + 1 and hi / lo > target:
                    mid = max(lo + 1, min(hi - 1, math.isqrt(lo * hi)))
                    request(template, mid, "gain_or_tiling_transition" if changed else "winning_interval_interior")
                for center in split_boundaries(template["X"], manifest["environment"]["vector_cores"], settings["selector"]):
                    if lo <= center <= hi:
                        for r in (center - 1, center, center + 1):
                            if ordered[0] <= r <= ordered[-1]:
                                request(template, r, "fixed_selector_S_transition")
            # Measured R tile candidates give extra local boundary probes. This
            # is not an exhaustive enumeration of compiler tiling transitions.
            for r, rows in series.items():
                if not winning(rows) or not any(row["case_id"] in manifest.get("initial_case_ids", []) for row in rows):
                    continue
                # Only seed observations nominate neighboring tile boundaries;
                # applying this recursively would walk the whole integer grid.
                s = split_count(template["X"], r, manifest["environment"]["vector_cores"], settings["selector"])
                for tile in reduction_tiles(rows):
                    quantum = s * tile
                    for center in ((r // quantum) * quantum, (r // quantum + 1) * quantum):
                        for neighbor in (center - 1, center, center + 1):
                            if ordered[0] <= neighbor <= ordered[-1] and split_count(
                                    template["X"], neighbor, manifest["environment"]["vector_cores"], settings["selector"]) == s:
                                request(template, neighbor, "selected_R_tile_neighbor")
    order = lambda item: (item["case"]["op"], item["case"]["dtype"], item["case"]["layout"], item["case"]["X"], item["case"]["R"])
    return dict(pending=sorted(pending.values(), key=order), blocked=sorted(blocked.values(), key=order),
                max_r_gap=ratio, boundary_r_gap=boundary_ratio)


def propose_intervals(families, max_r_gap):
    proposals = []
    for item in families.values():
        required = set(item["contexts"])
        run = []

        def flush():
            if len(run) < 3 or run[-1] - run[0] + 1 - len(run) < 2:
                return
            identity = dict(key=item["key"], r_min=run[0], r_max=run[-1])
            proposals.append(dict(identity, rule_id=digest(identity)[:16], tested_r=run.copy(),
                                  contexts=item["contexts"], observations=sum(
                                      len(rows) for r in run for rows in item["points"][r].values())))

        for r, contexts in sorted(item["points"].items()):
            passes = set(contexts) == required and all(winning(rows) for rows in contexts.values())
            if not passes or (run and r / run[-1] > max_r_gap):
                flush()
                run = []
            if passes:
                run.append(r)
        flush()
    return sorted(proposals, key=lambda rule: (json.dumps(rule["key"], sort_keys=True), rule["r_min"]))


def make_holdout_plan(table, calibration):
    """New interior R in EVERY interval/context, endpoints and outside controls."""
    measured = defaultdict(set)
    for case in calibration["cases"]:
        measured[(context_id(case), case["X"])].add(case["R"])
    cases, expectations, gaps = {}, [], []
    for rule in table["rules"]:
        lo, hi, x = rule["r_min"], rule["r_max"], rule["key"]["X"]
        for ctx, spec in rule["contexts"].items():
            used = measured[(ctx, x)]
            fresh = set()
            for a, b in zip(rule["tested_r"], rule["tested_r"][1:]):
                mid = math.isqrt(a * b)
                choices = (mid, mid + 1, a + 1, b - 1)
                candidate = next((r for r in choices if a < r < b and r not in used), None)
                if candidate is not None:
                    fresh.add(candidate)
                elif b > a + 1:
                    gaps.append(dict(rule_id=rule["rule_id"], context=ctx, reason="no_fresh_R_in_sample_gap", lo=a, hi=b))
            # Ensure at least two independent shapes even for narrow intervals.
            for r in (lo + 1, hi - 1, (2 * lo + hi) // 3, (lo + 2 * hi) // 3):
                if lo < r < hi and r not in used:
                    fresh.add(r)
            if len(fresh) < 2:
                gaps.append(dict(rule_id=rule["rule_id"], context=ctx, reason="fewer_than_two_fresh_interior_R"))
            for r in sorted(fresh | {max(2, lo - 1), lo, hi, hi + 1}):
                case = make_case(x=x, r=r, role="holdout", **spec)
                cases[case["id"]] = case
                expectations.append(dict(rule_id=rule["rule_id"], context=ctx, case_id=case["id"], R=r,
                                         kind="fresh_interior" if r in fresh else "endpoint" if r in (lo, hi) else "outside"))
    return dict(schema="npu-split-holdout-plan-v2", table_digest=digest(table), cases=list(cases.values()),
                expectations=expectations, gaps=gaps)


def audit_holdout(table, manifest, rows, failures):
    checks = []
    matrix = {(execution, cache) for execution in table["domain"]["executions"] for cache in table["domain"]["cache_modes"]}
    for rule in table.get("rules", []):
        lo, hi, x = rule["r_min"], rule["r_max"], rule["key"]["X"]
        for ctx in rule["contexts"]:
            eligible = [row for row in rows if row["mode"] == "policy" and row["role"] == "holdout"
                        and row["X"] == x and row.get("context") == ctx]
            fresh = sorted({row["R"] for row in eligible if lo < row["R"] < hi and row["R"] not in rule["tested_r"]})
            issues = []
            if len(fresh) < 2:
                issues.append("missing_two_fresh_interior_shapes")
            for a, b in zip(rule["tested_r"], rule["tested_r"][1:]):
                if b > a + 1 and not any(a < r < b for r in fresh):
                    issues.append(f"gap={a}..{b}:missing_fresh_interior_shape")
            for r in sorted(set(fresh) | {max(2, lo - 1), lo, hi, hi + 1}):
                at_r = [row for row in eligible if row["R"] == r]
                if {(row["execution"], row["cache"]) for row in at_r} != matrix:
                    issues.append(f"R={r}:missing_execution_or_cache")
                    continue
                expected_split = any(other["key"] == rule["key"] and other["r_min"] <= r <= other["r_max"]
                                     for other in table["rules"])
                for row in at_r:
                    decisions = [d for d in row.get("ir_decisions", []) if not d["depth"]
                                 and all(d["meta"].get(k) == v for k, v in rule["key"].items()) and d["meta"].get("R") == r]
                    actual_split = any(int(d["S"]) > 1 for d in decisions)
                    if not decisions or actual_split != expected_split:
                        issues.append(f"R={r}:admission_not_observed")
                    if expected_split and row["decision"] != "WIN":
                        issues.append(f"R={r}:{row['decision']}")
                    if not expected_split and row["evidence"] not in ("VERIFIED_SPLIT", "NO_SPLIT"):
                        issues.append(f"R={r}:{row['evidence']}")
            checks.append(dict(rule_id=rule["rule_id"], context=ctx, fresh_R=fresh, passed=not issues, issues=sorted(set(issues))))
    admitted = [row for row in rows if row["mode"] == "policy" and row["decision"] != "NO_SPLIT"]
    passed = (bool(checks) and bool(admitted) and not failures and not manifest.get("excluded_cases")
              and all(check["passed"] for check in checks) and all(row["decision"] == "WIN" for row in admitted))
    return dict(passed=passed, status="VALIDATED_MEASURED_DOMAIN" if passed else "HOLDOUT_INCOMPLETE_OR_FAILED",
                table_digest=digest(table), checks=checks, incomplete_cases=len(failures),
                planned_exclusions=len(manifest.get("excluded_cases", [])),
                warning="This certifies only this finite measurement domain, not arbitrary shapes or future fusion contexts.")
