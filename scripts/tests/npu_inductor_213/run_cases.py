"""Run the requested upstream Inductor cases through transfer_to_npu."""

import argparse
import json
import os
from pathlib import Path
import sys


class Results:
    def __init__(self, cases, output):
        self.cases = cases
        self.output = output
        self.data = {"selected": {}, "reports": [], "missing": []}

    def save(self):
        self.output.write_text(json.dumps(self.data, indent=2, ensure_ascii=False))

    def pytest_collection_modifyitems(self, session, config, items):
        selected, deselected = [], []
        for item in items:
            filename = Path(str(item.path)).name
            names = self.cases.get(filename, [])
            matches = [name for name in names if item.originalname == name or item.name == name
                       or item.name.startswith(name + "_") or item.name.startswith(name + "[")]
            if not matches or getattr(item.cls, "device_type", None) != "npu":
                deselected.append(item)
                continue
            name = max(matches, key=len)
            key = filename + "::" + name
            self.data["selected"].setdefault(key, []).append(item.nodeid)
            selected.append(item)
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)
        self.data["missing"] = [filename + "::" + name for filename, names in self.cases.items()
                                for name in names if filename + "::" + name not in self.data["selected"]]
        self.save()

    def pytest_runtest_logreport(self, report):
        self.data["reports"].append({
            "nodeid": report.nodeid, "when": report.when, "outcome": report.outcome,
            "duration": report.duration, "longrepr": str(report.longrepr) if report.longrepr else "",
            "wasxfail": getattr(report, "wasxfail", None),
        })
        self.save()

    def pytest_collectreport(self, report):
        if report.failed:
            self.data.setdefault("collection_errors", []).append(str(report.longrepr))
            self.save()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("cases.tsv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--file")
    parser.add_argument("--case")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    cases = {}
    for line in args.manifest.read_text().splitlines()[1:]:
        filename, name = line.split()
        if (not args.file or filename == args.file) and (not args.case or name == args.case):
            cases.setdefault(filename, []).append(name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(args.case_dir)
    sys.path.insert(0, str(args.case_dir))
    sys.path.insert(0, str(args.case_dir.parent))

    import torch
    import torch_npu
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
    import torch_npu._inductor  # noqa: F401
    import pytest
    from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, HAS_TRITON

    recorder = Results(cases, args.output)
    recorder.data["environment"] = {
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "device": torch.npu.get_device_name(), "GPU_TYPE": GPU_TYPE,
        "HAS_GPU": bool(HAS_GPU), "HAS_TRITON": bool(HAS_TRITON),
    }
    print(json.dumps(recorder.data["environment"]), flush=True)
    options = list(cases) + ["-q", "-ra", "--tb=short", "-p", "no:cacheprovider"]
    if args.collect_only:
        options.append("--collect-only")
    code = pytest.main(options, plugins=[recorder])
    recorder.data["exitcode"] = int(code)
    recorder.save()
    return int(code)


if __name__ == "__main__":
    sys.exit(main())
