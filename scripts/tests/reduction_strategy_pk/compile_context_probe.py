"""Check launch configuration recording across initial compile and later invocation."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--backend", choices=("cuda", "npu"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--case-ids", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    run = importlib.import_module("run")
    common = importlib.import_module("common")
    from autotune_tiling import BestTilingRecorder
    import torch
    from torch._dynamo.utils import counters
    from torch._inductor.utils import run_and_get_code

    if args.backend == "npu":
        import torch_npu  # noqa: F401

    api = getattr(torch, args.backend)
    api.set_device(0)
    environment = run.environment(args.backend, args.device)
    if args.backend == "cuda":
        api.set_per_process_memory_fraction(.20, 0)
    torch.set_num_threads(4)
    ids = args.case_ids.split(",")
    cases = [case for case in common.cases() if case["id"] in ids]
    if len(cases) != len(ids):
        raise ValueError("Unknown or duplicate case ID")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for case in cases:
        torch._dynamo.reset()
        counters.clear()
        directory = output / case["id"]
        values = common.cpu_inputs(case, 213)
        inputs = common.device_inputs(case, values, args.backend)
        expected = common.reference(case, values)
        first, invoke = run.compile_strategy(case, "plain", inputs, args.backend, directory / "initial", expected)
        initial_graphs = counters["stats"]["unique_graphs"]
        recorder = BestTilingRecorder(args.backend)
        recorder.install()
        recorder.start_capture()
        try:
            actual, sources = run_and_get_code(invoke)
            api.synchronize()
            later = recorder.stop_capture()
        finally:
            recorder.uninstall()
        for index, text in enumerate(sources):
            (directory / f"later_output_code_{index}.py").write_text(text)
        record = dict(case=case, grad_enabled_after_initial_compile=torch.is_grad_enabled(),
                      initial_unique_graphs=initial_graphs, later_unique_graphs=counters["stats"]["unique_graphs"],
                      initial_selected_configs=first["selected_configs"], later_selected_configs=later,
                      configuration_equal=first["selected_configs"] == later,
                      accuracy=common.compare(actual, expected, case["op"]))
        records.append(record)
        common.write_json(directory / "context.json", record)
    report = dict(environment=environment, records=records, performance="Not measured by this diagnostic",
                  frozen_source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in source.glob("*.py")})
    common.write_json(output / "context.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
