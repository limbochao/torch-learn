#!/usr/bin/env python3
"""Run smoke, calibration, additive R refinement, frozen per-rule holdout."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from common import HERE, external_output, read_json, source_hashes, write_json
from policy import digest
from refinement import make_holdout_plan
from run import normalize_args, parser


def arguments(args):
    names = ("lowering_profile", "execution", "cache", "rotate_buffers", "sessions", "rounds", "repeats", "warmup",
             "seed", "max_working_mib", "timeout", "outputs_per_program", "target_waves", "min_chunk", "max_s",
             "relative_margin", "absolute_margin_us", "metric", "max_r_gap", "boundary_r_gap")
    values = [part for name in names for part in ("--" + name.replace("_", "-"), str(getattr(args, name)))]
    values += ["--max-autotune" if args.max_autotune else "--no-max-autotune"]
    if args.cores is not None:
        values += ["--cores", str(args.cores)]
    if args.visible_devices:
        values += ["--devices", args.visible_devices]
    elif args.device != 0:
        values += ["--device", str(args.device)]  # Legacy logical-device selection.
    values += ["--schedule", args.schedule]
    return values


def main():
    p = parser()
    p.description = __doc__
    p.add_argument("--max-refine-rounds", type=int, default=12, help="stop with pending work if this budget is exhausted; may increase on resume")
    p.add_argument("--with-dynamic", action="store_true", help="also collect dynamic compiled-graph reuse diagnostics after static validation")
    p.add_argument("--with-controls", action="store_true", help="also collect excluded-type controls")
    p.add_argument("--collect-all", action="store_true", help="collect all initial/static/dynamic/control cases despite individual failures; failed evidence still prevents validation")
    args = normalize_args(p, p.parse_args())
    if args.max_refine_rounds < 1:
        p.error("max-refine-rounds must be positive")
    if args.modes != "plain,candidate" or args.policy != "installed" or args.table or args.extend:
        p.error("campaign manages strategy/table/extension stages; use run.py for individual custom comparisons")
    if args.shards != 1 or args.shard_index or (args.start_percent, args.end_percent) != ("0", "100"):
        p.error("a campaign needs a single complete manifest; use run.py for percentage ranges and merge.py to combine them")
    root = external_output(args.out)
    identity = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ("resume", "plan", "out", "max_refine_rounds", "max_working_mib")}
    identity["source_hashes"] = source_hashes()
    if args.cases_json:
        identity["custom_cases"] = read_json(args.cases_json)
    identity_path = root / "campaign.json"
    if root.exists() and any(root.iterdir()):
        if not args.resume or not identity_path.exists() or read_json(identity_path) != identity:
            p.error("campaign output/config/source mismatch; use an empty directory or identical --resume")
    root.mkdir(parents=True, exist_ok=True)
    if not identity_path.exists():
        write_json(identity_path, identity)
    common = arguments(args)
    filters = []
    for name in ("ops", "dtypes", "layouts", "match", "limit", "x", "r", "around"):
        value = getattr(args, name)
        if value:
            filters += ["--" + name, *map(str, value if isinstance(value, list) else [value])]
    selection = ["--cases-json", str(args.cases_json)] if args.cases_json else ["--preset", args.preset]
    state_path = root / "campaign.state.json"
    state = read_json(state_path) if state_path.exists() else dict(refinement_rounds=0, history=[])

    def stage(name, out, options, *, base=None, extend=False, collect_errors=False):
        state.update(stage=name, status="RUNNING")
        write_json(state_path, state)
        command = [sys.executable, "-B", str(HERE / "run.py"), *(common if base is None else base), "--out", str(out), *options]
        if (out / "manifest.json").exists():
            command += ["--resume"] + (["--extend"] if extend else [])
        print(f"\n[{name}] {out}", flush=True)
        # Inherit terminal output, while each worker keeps its full private log.
        code = subprocess.call(command)
        state["history"].append(dict(stage=name, returncode=code, utc=datetime.now(timezone.utc).isoformat()))
        state["status"] = "PASS" if code == 0 else "NEEDS_ATTENTION"
        write_json(state_path, state)
        if code and not (collect_errors and code == 1):
            raise SystemExit(code)
        return code

    if args.plan:
        stage("initial-plan", root / "plan", [*selection, *filters, "--plan"])
        print("This is the initial matrix. Actual results determine additional R/context probes and per-rule holdout.", flush=True)
        return

    # Smoke covers all primitive/dtype capabilities selected by broad filters;
    # X/R/layout restrictions on calibration must not accidentally empty smoke.
    smoke_filters = [part for name in ("ops", "dtypes") if getattr(args, name)
                     for part in ("--" + name, getattr(args, name))]
    smoke_code = stage("smoke", root / "smoke", ["--preset", "smoke", *smoke_filters,
                       *([] if args.collect_all else ["--fail-fast"])], extend=True, collect_errors=args.collect_all)
    smoke_rows = read_json(root / "smoke" / "summary.json")["rows"]
    smoke_bad = smoke_code or any(row["decision"] not in ("WIN", "REGRESSION", "UNCERTAIN", "NO_SPLIT") for row in smoke_rows)
    if smoke_bad and not args.collect_all:
        print("Smoke has missing split/codegen/measurement evidence; inspect its report before calibration.", flush=True)
        raise SystemExit(1)

    calibration = root / "calibration"
    calibration_code = stage("calibration", calibration, [*selection, *filters], extend=True, collect_errors=args.collect_all)
    diagnostic_codes = []
    if args.collect_all and args.with_dynamic:
        diagnostic_codes.append(stage("dynamic", root / "dynamic", ["--preset", "dynamic", *smoke_filters],
                                      extend=True, collect_errors=True))
    if args.with_controls:
        diagnostic_codes.append(stage("controls", root / "controls", ["--preset", "controls"],
                                      extend=True, collect_errors=args.collect_all))
    if smoke_bad or calibration_code or any(diagnostic_codes):
        state.update(status="INCOMPLETE", stage="collection-finished-with-failures")
        write_json(state_path, state)
        print("All requested initial collection stages finished; failures remain, so no policy is validated.", flush=True)
        raise SystemExit(1)
    while True:
        plan = read_json(calibration / "refinement.json")
        if plan["blocked"]:
            print("Required refinement exceeds the declared memory domain; inspect refinement.json. No admission table exported.", flush=True)
            raise SystemExit(3)
        if not plan["pending"]:
            break
        if state["refinement_rounds"] >= args.max_refine_rounds:
            print(f"Refinement budget reached with {len(plan['pending'])} pending cases. Resume with a larger --max-refine-rounds.", flush=True)
            raise SystemExit(3)
        state["refinement_rounds"] += 1
        write_json(state_path, state)
        stage(f"refinement-{state['refinement_rounds']}", calibration,
              ["--cases-json", str(calibration / "next_cases.json")], extend=True)

    table = read_json(calibration / "thresholds.proposed.json")
    if not table["rules"] or table["status"] != "DRAFT_REQUIRES_HOLDOUT":
        print("No admissible interval established. Review the complete calibration report; S=1 remains the default.", flush=True)
        raise SystemExit(2)
    table_id = digest(table)[:16]
    frozen = root / "tables" / f"{table_id}.json"
    if not frozen.exists():
        write_json(frozen, table)
    write_json(root / "frozen-policy.json", table)
    holdout_plan = make_holdout_plan(table, read_json(calibration / "manifest.json"))
    plan_path = root / "tables" / f"{table_id}.holdout.json"
    write_json(plan_path, holdout_plan)
    if holdout_plan["gaps"]:
        print(f"Holdout cannot cover every interval: {plan_path}", flush=True)
        raise SystemExit(3)
    holdout_common = common.copy()
    holdout_common[holdout_common.index("--seed") + 1] = str(args.seed + 100000)
    holdout = root / "holdout" / table_id
    stage("holdout", holdout, ["--cases-json", str(plan_path), "--modes", "plain,policy", "--policy", "table", "--table", str(frozen)], base=holdout_common, extend=True)
    audit = read_json(holdout / "holdout.audit.json")
    write_json(root / "validated-policy.json", dict(table, status=audit["status"], validation=audit))
    if args.with_dynamic and not args.collect_all:
        stage("dynamic", root / "dynamic", ["--preset", "dynamic", *smoke_filters], extend=True)
    state.update(status="COMPLETE", stage="complete", validated_table=str(root / "validated-policy.json"))
    write_json(state_path, state)
    print(f"Validated the finite measured domain: {root / 'validated-policy.json'}", flush=True)


if __name__ == "__main__":
    main()
