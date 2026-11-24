"""Fingerprint the actual CPU input bytes for every shape/dtype and validation seed."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Frozen worker source directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case-ids", default="", help="Fingerprint only the declared subset; default is the full matrix")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    common = importlib.import_module("common")
    import torch

    # Match the worker, including its CPU threading configuration.
    torch.set_num_threads(4)
    selected = common.cases()
    if args.case_ids:
        ids = args.case_ids.split(",")
        selected = [case for case in selected if case["id"] in ids]
        if len(selected) != len(ids):
            raise ValueError("Unknown or duplicate fingerprint case ID")
    unique = {}
    for case in selected:
        unique.setdefault((case["X"], case["R"], case["dtype"]), case)
    records = {}
    for (x, r, dtype), case in sorted(unique.items()):
        for seed in (213, 214, 215):
            values = common.cpu_inputs(case, seed)
            digests = [hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
                       for value in values]
            records[f"x{x}-r{r}-{dtype}-seed{seed}"] = dict(X=x, R=r, dtype=dtype, seed=seed,
                                                          input_sha256=digests)
    report = dict(torch=torch.__version__, torch_commit=torch.version.git_version,
                  machine=platform.machine(), python=platform.python_version(),
                  cpu_capability=torch.backends.cpu.get_cpu_capability(), threads=4,
                  source_common_sha256=hashlib.sha256((args.source / "common.py").read_bytes()).hexdigest(),
                  declared_cases=len(selected), distinct_shapes_dtypes=len(unique),
                  records=records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
