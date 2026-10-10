"""Run a selected NPU launcher test suite with durable per-phase results."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


class Results:
    def __init__(self, path):
        self.path = path

    def pytest_runtest_logreport(self, report):
        entry = {
            "nodeid": report.nodeid,
            "when": report.when,
            "outcome": report.outcome,
            "duration": report.duration,
            "longrepr": str(report.longrepr) if report.longrepr else "",
            "sections": list(report.sections),
        }
        with self.path.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--select", default="")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    # Test imports load torch_npu._inductor before per-compile options are applied.
    os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")
    import pytest
    import torch
    import torch_npu
    import triton

    torch.npu.set_device(args.device)
    # Initialize the shared driver utility in the persistent outer cache, before
    # test fixtures create and remove their individual Triton cache directories.
    target = triton.runtime.driver.active.get_current_target()
    binding = getattr(torch_npu._C, "_StaticNpuLauncher", None)
    metadata = {
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "torch_npu": torch_npu.__version__,
        "torch_npu_git": torch_npu.version.git_version,
        "triton": triton.__version__,
        "device_index": args.device,
        "device_name": torch.npu.get_device_name(args.device),
        "startup_backend": os.environ["TORCHINDUCTOR_NPU_BACKEND"],
        "triton_target": str(target),
        "binding_present": binding is not None,
        "binding_supported": binding._is_supported() if binding is not None else False,
        "suite": str(args.suite),
        "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
    }
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)
    if not metadata["binding_supported"]:
        raise RuntimeError("NPU static launcher binding is unavailable")
    selected = str(args.suite.resolve()) + args.select
    code = pytest.main([
        "-vv", "--tb=short", "-o", "faulthandler_timeout=180",
        "--rootdir", str(args.suite.parent.resolve()),
        "--junitxml=" + str((args.output / "results.xml").resolve()), selected,
    ], plugins=[Results(args.output / "phases.jsonl")])
    (args.output / "exit.json").write_text(json.dumps({"exit_code": int(code)}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
