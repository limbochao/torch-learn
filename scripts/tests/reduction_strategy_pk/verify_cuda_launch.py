"""Compare actual CUDA profiler launch parameters with recorded compiler configs."""

import argparse
import json
import math
from pathlib import Path

from common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CUDA run directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.input / "run.json").read_text())
    if manifest["backend"] != "cuda":
        raise ValueError("Expected a CUDA run")
    errors, checked, cooperative = [], 0, 0
    skipped = []
    seen = set()
    for case in manifest["cases"]:
        directory = args.input / "cases" / case["id"]
        result = json.loads((directory / "result.json").read_text())
        if result["status"] != "PASS":
            skipped.append(dict(case=case["id"], status=result["status"]))
            continue
        selected = {item["strategy"]: item for item in result["selected"]}
        for repeat in range(3):
            profile = json.loads((directory / f"confirm-{repeat}" / "device_events.json").read_text())
            for group in profile["ranges"]:
                strategy = group["label"].rsplit(":", 1)[0]
                configs = {item["kernel_name"]: item["selected_config"]
                           for item in selected[strategy]["selected_configs"]}
                actual_names = set()
                for event in group["events"]:
                    name = event["name"]
                    if not name.startswith("triton_"):
                        continue
                    actual_names.add(name)
                    config = configs.get(name)
                    if config is None:
                        errors.append(dict(case=case["id"], strategy=strategy, error="Unknown timed kernel", name=name))
                        continue
                    checked += 1
                    launch = event["args"]
                    if launch.get("block") != [config["num_warps"] * 32, 1, 1]:
                        errors.append(dict(case=case["id"], strategy=strategy,
                                           error="Timed thread block differs", actual=launch.get("block"),
                                           recorded_warps=config["num_warps"]))
                    if "RSPLIT" in config:
                        cooperative += 1
                        expected = [config["RSPLIT"], math.ceil(case["X"] / config.get("XBLOCK", 1)), 1]
                        if launch.get("grid") != expected:
                            errors.append(dict(case=case["id"], strategy=strategy,
                                               error="Timed cooperative grid differs", actual=launch.get("grid"),
                                               recorded=expected))
                    seen.add((case["id"], strategy, name))
                if actual_names != set(configs):
                    errors.append(dict(case=case["id"], strategy=strategy,
                                       error="Timed/recorded kernel coverage differs"))
    report = dict(cases=len(manifest["cases"]), checked_device_kernel_events=checked,
                  checked_cooperative_grid_events=cooperative, unique_case_strategy_kernels=len(seen),
                  source="Actual confirmation device_events derived from profiler trace, compared with recorded selected configs",
                  scope="All timed Triton kernel names and thread block sizes; RSPLIT/XBLOCK cooperative grids. "
                  "Internal loop tile sizes are not observable from trace launch parameters.", errors=errors)
    if skipped:
        report["skipped_nonpassing_cases"] = skipped
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
