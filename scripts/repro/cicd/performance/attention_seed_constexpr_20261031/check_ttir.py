"""Replay captured attention TTIR checks without modifying installed compiler code."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import traceback


CASES = ("original", "constant_bound", "runtime_i32")


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one metadata field: {old}")
    return source.replace(old, new, 1)


def variant_source(source, case):
    if case == "constant_bound":
        return replace_once(source, "'constants': {}", "'constants': {'load_seed_offset': 1}")
    if case == "runtime_i32":
        return replace_once(source, "'load_seed_offset': 'constexpr'", "'load_seed_offset': 'i32'")
    return source


def check_case(source, case, output_dir):
    module_path = output_dir / f"{case}_ttir_check.py"
    module_path.write_text(variant_source(source, case))
    # Triton inspects a real source file when the generated @triton.jit is imported.
    spec = importlib.util.spec_from_file_location(f"attention_seed_{case}", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # Import/configuration errors are setup failures, never an expected original failure.
    spec.loader.exec_module(module)
    kernel = module.triton_
    config = str(kernel._select_ttir_test_config())
    try:
        kernel._triton_make_ttir()
    except Exception as error:
        detail = traceback.format_exc()
        (output_dir / f"{case}_traceback.txt").write_text(detail)
        expected = case == "original" and "'NoneType' object has no attribute 'type'" in str(error)
        return {"status": "failed", "expected": expected, "config": config, "error": str(error)}
    for index, ttir in enumerate(kernel.compile_results):
        (output_dir / f"{case}_{index}.ttir").write_text(str(ttir))
    return {"status": "passed", "expected": case != "original", "config": config}


def main():
    repo_root = Path(__file__).resolve().parents[5]
    default_artifact = (
        repo_root / "artifacts/cicd/performance/attention-seed-constexpr-20261031"
        / "failed_first_attention_checker.py"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=default_artifact)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = args.artifact.read_text()
    cases = CASES if args.case == "all" else (args.case,)
    results = {case: check_case(source, case, output_dir) for case in cases}
    (output_dir / "ttir_check_results.json").write_text(json.dumps(results, indent=2) + "\n")
    for case, result in results.items():
        print(f"{case}: {result['status']}, expected={result['expected']}")
    return 0 if all(result["expected"] for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
