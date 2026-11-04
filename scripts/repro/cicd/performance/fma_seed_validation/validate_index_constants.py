"""Run the index-codegen metadata regression against exact before/after methods.

This overlays only codegen_kernel in the current process. It does not install a
framework build, execute a device kernel, or measure performance.
"""

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import textwrap
import unittest


def method_source(path):
    source = path.read_text()
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == "NPUIndexTritonKernel")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == "codegen_kernel")
    return textwrap.dedent("".join(source.splitlines(keepends=True)[method.lineno - 1:method.end_lineno]))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--test-support-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import torch_npu
    from torch._inductor.codegen.triton_utils import equal_1_arg_indices
    from torch_npu._inductor.codegen import triton as codegen

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.test_support_dir.resolve()))
    test_class = runpy.run_path(str(args.test_file))["TestIndexTritonConstants"]
    installed_source = Path(codegen.__file__)
    installed_sha = sha256(installed_source)
    original = codegen.NPUIndexTritonKernel.codegen_kernel
    report = {
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "test_sha256": sha256(args.test_file), "installed_codegen_sha256": installed_sha,
        "validation_scope": "process-local codegen method; metadata only; no device launch",
        "variants": {},
    }
    try:
        for label, path in (("before", args.before), ("after", args.after)):
            namespace = dict(vars(codegen), equal_1_arg_indices=equal_1_arg_indices)
            exec(compile(method_source(path), f"<master-{label}-codegen>", "exec"), namespace)
            codegen.NPUIndexTritonKernel.codegen_kernel = namespace["codegen_kernel"]
            stream = io.StringIO()
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_class)
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
            (args.output_dir / f"{label}.txt").write_text(stream.getvalue())
            report["variants"][label] = {
                "source_sha256": sha256(path), "tests_run": result.testsRun,
                "failures": len(result.failures), "errors": len(result.errors),
                "skipped": len(result.skipped), "passed": result.wasSuccessful(),
            }
    finally:
        codegen.NPUIndexTritonKernel.codegen_kernel = original
        report["installed_source_unchanged"] = sha256(installed_source) == installed_sha
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    before, after = report["variants"]["before"], report["variants"]["after"]
    assert before["failures"] == 3 and before["errors"] == 0, before
    assert after["passed"] and after["tests_run"] == 3 and after["skipped"] == 0, after
    assert report["installed_source_unchanged"]


if __name__ == "__main__":
    main()
