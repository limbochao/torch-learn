"""Small, explicit threshold brackets for quick screening; never an admission table."""

from collections import defaultdict
import math

from cases import from_spec


def threshold_guidance(manifest, rows, complete, failures):
    groups = {}
    for case in manifest["cases"]:
        if case["role"] != "screening":
            continue
        key = (case["op"], case["dtype"], case["layout"], case["X"])
        item = groups.setdefault(key, dict(template=case, cases={}, rows=defaultdict(list)))
        item["cases"][case["R"]] = case
    for row in rows:
        key = (row["op"], row["dtype"], row["layout"], row["X"])
        if row["role"] == "screening" and key in groups:
            groups[key]["rows"][row["R"]].append(row)
    expected = {(mode, cache, execution) for mode in manifest["settings"]["modes"] if mode != "plain"
                for cache in manifest["settings"]["cache_modes"] for execution in manifest["settings"]["executions"]}
    failed_ids = {failure["case_id"] for failure in failures}
    ratio = manifest["settings"].get("boundary_r_gap", 1.05)
    known = {case["id"] for case in manifest["cases"]}
    pending, report = {}, []
    for key, group in sorted(groups.items()):
        points = []
        for r, case in sorted(group["cases"].items()):
            values = group["rows"][r]
            valid = (case["id"] not in failed_ids and len(values) == len(expected)
                     and {(row["mode"], row["cache"], row["execution"]) for row in values} == expected)
            decisions = {row["decision"] for row in values}
            verdict = ("INCOMPLETE" if not valid else "WIN" if decisions == {"WIN"} else
                       "INVALID_EVIDENCE" if not decisions <= {"WIN", "REGRESSION", "UNCERTAIN", "NO_SPLIT"} else
                       "NO_SPLIT" if decisions == {"NO_SPLIT"} else
                       "REGRESSION" if "REGRESSION" in decisions else
                       "UNCERTAIN" if decisions <= {"WIN", "UNCERTAIN", "NO_SPLIT"} else "INVALID_EVIDENCE")
            points.append(dict(R=r, verdict=verdict,
                               min_gain_pct=min((v["gain_pct"] for v in values if "gain_pct" in v), default=None),
                               requested_S=values[0].get("requested_S", []) if values else []))
        windows, transitions, run = [], [], []
        for point in [*points, dict(R=None, verdict="END")]:
            if point["verdict"] == "WIN":
                run.append(point["R"])
            elif run:
                windows.append(dict(first_observed_win=run[0], last_observed_win=run[-1], tested_r=run))
                run = []
        for left, right in zip(points, points[1:]):
            if (left["verdict"] in ("INCOMPLETE", "INVALID_EVIDENCE") or
                    right["verdict"] in ("INCOMPLETE", "INVALID_EVIDENCE") or
                    (left["verdict"] == "WIN") == (right["verdict"] == "WIN")):
                continue
            lo, hi = left["R"], right["R"]
            midpoint = math.isqrt(lo * hi) if hi > lo + 1 and hi / lo > ratio else None
            transitions.append(dict(lo=lo, hi=hi, left=left["verdict"], right=right["verdict"], next_R=midpoint))
            if midpoint is not None and complete:
                case = from_spec(group["template"], R=midpoint)
                if case["id"] not in known:
                    pending[case["id"]] = case
        report.append(dict(op=key[0], dtype=key[1], layout=key[2], X=key[3], points=points,
                           observed_win_windows=windows, transition_brackets=transitions,
                           status="INCOMPLETE" if any(p["verdict"] in ("INCOMPLETE", "INVALID_EVIDENCE") for p in points)
                           else "OBSERVED_WINS" if windows else "NO_OBSERVED_WIN"))
    usable = complete and not failures and all(group["status"] != "INCOMPLETE" for group in report)
    return dict(status="SCREENING_ONLY" if usable else "INCOMPLETE_SCREENING",
                collection_complete=complete, groups=report, next_cases=list(pending.values()),
                note="Observed anchors and transition brackets only. No monotonicity, unmeasured interval, layout or X admission is inferred.")
