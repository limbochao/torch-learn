"""Prepare an NPU copy of two upstream static-launcher test classes."""

import argparse
import ast
import difflib
import hashlib
import json
import textwrap
from pathlib import Path


ADAPTER = '''
def _make_launcher(self, compiled_kernel):
    device = torch.npu.current_device()
    result = NPUStaticArtifactAdapter.from_compiled_kernel(
        compiled_kernel, {"device": device}, {}
    )
    self.assertIsInstance(result, NPUStaticallyLaunchedTritonKernel)
    self.assertTrue(result.npubin_raw)
    with tempfile.NamedTemporaryFile(suffix=".npubin", delete=False) as tmp_file:
        self.tmp_files.append(tmp_file)
        result.reload_npubin_from_raw(tmp_file.name)
    result.npubin_raw = None
    result.load_kernel(device)
    self.addCleanup(result.close)
    return result
'''

EXTRA_IMPORTS = '''
import functools
import unittest
import torch_npu
from torch_npu._inductor.triton_experimental.static_launcher import (
    NPUStaticallyLaunchedTritonKernel,
    NPUStaticTritonCompileResult,
)
from torch_npu._inductor.triton_experimental.static_launcher.adapter import NPUStaticArtifactAdapter
from torch_npu._inductor.triton_experimental.npu_triton_helpers import libdevice

GPU_TYPE = "npu"
HAS_XPU_AND_TRITON = False


def _npu_compile(*args, **kwargs):
    kwargs["options"] = {**kwargs.get("options", {}), "npu_backend": "triton_experimental"}
    return torch.compile(*args, **kwargs)


'''

OBSERVER = '''

def _observe_static_test(method):
    @functools.wraps(method)
    def checked(self):
        make_launcher = NPUStaticTritonCompileResult.make_launcher
        run = NPUStaticallyLaunchedTritonKernel.run
        with (
            mock.patch.object(NPUStaticTritonCompileResult, "make_launcher",
                              autospec=True, side_effect=make_launcher) as made,
            mock.patch.object(NPUStaticallyLaunchedTritonKernel, "run",
                              autospec=True, side_effect=run) as launched,
        ):
            method(self)
            if method.__name__ not in ("test_disable_static_triton_launcher", "test_incompatible_code"):
                self.assertGreater(made.call_count, 0, "NPU static compile result was never used")
                self.assertGreater(launched.call_count, 0, "NPU static kernel was never launched")
            if method.__name__ == "test_disable_static_triton_launcher":
                self.assertEqual(made.call_count, 0)
                self.assertEqual(launched.call_count, 0)
            print("STATIC_EVIDENCE", method.__name__, made.call_count, launched.call_count, flush=True)
    return checked


for _name, _method in list(vars(TestStaticTritonCompileResult).items()):
    if _name.startswith("test_"):
        setattr(TestStaticTritonCompileResult, _name, _observe_static_test(_method))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = args.source.read_text()
    lines = original.splitlines(keepends=True)
    tree = ast.parse(original)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    preamble = "".join(lines[: classes["TestStaticTritonLauncherUnit"].lineno - 1])
    selected = []
    for name in ("TestStaticTritonLauncher", "TestStaticTritonCompileResult"):
        node = classes[name]
        start = min([node.lineno] + [item.lineno for item in node.decorator_list]) - 1
        selected.append("".join(lines[start : node.end_lineno]) + "\n")
    baseline = preamble + "\n\n".join(selected)
    direct, compiled = selected
    node = classes["TestStaticTritonLauncher"]
    for method in node.body:
        if not isinstance(method, ast.FunctionDef):
            continue
        old = "".join(lines[method.lineno - 1 : method.end_lineno])
        if method.name == "_make_launcher":
            direct = direct.replace(old, textwrap.indent(ADAPTER.strip(), "    ") + "\n")
        elif method.name == "write_cubin_to_tmp":
            direct = direct.replace(old, "")
    for old, new in {
        'launcher.arg_tys, "Oi"': 'launcher.arg_kinds, ("tensor", "i32")',
        'launcher.arg_tys, "OBHIK"': 'launcher.arg_kinds, ("tensor", "u8", "u16", "u32", "u64")',
        'launcher.arg_tys, "Obhil"': 'launcher.arg_kinds, ("tensor", "i8", "i16", "i32", "i64")',
        'launcher.arg_tys, "O"': 'launcher.arg_kinds, ("tensor",)',
    }.items():
        direct = direct.replace(old, new)
    # Preserve unsigned scalar signatures; the output tensor stores their small sum as int64.
    direct = direct.replace("dtype=torch.uint64", "dtype=torch.int64")
    # The upstream implicit-constant example allocates four elements but passes
    # a reduction length of 128. Preserve specialization of xnumel=1 while
    # keeping both ordinary and static launches inside the input allocation.
    direct = direct.replace("arg0, arg1, 1, 128, XBLOCK", "arg0, arg1, 1, arg0.numel(), XBLOCK")
    direct = direct.replace("stream, arg0, arg2, 128)", "stream, arg0, arg2, arg0.numel())")
    for name, reason in {
        "test_high_shared_mem": "CUDA shared-memory mutation has no equivalent NPU resource contract",
        "test_too_high_shared_mem": "CUDA shared-memory OOM assertion is device-specific",
        "test_launcher_keeps_module_owner_alive_until_release":
            "CUDA module destructor API; NPU ownership is tested by the existing NPU suite",
    }.items():
        direct = direct.replace(f"    def {name}(", f"    @unittest.skip({reason!r})\n    def {name}(")
    compiled = compiled.replace("torch.compile", "_npu_compile")
    compiled = compiled.replace("dtype=torch.float64", "dtype=torch.float32")
    compiled = compiled.replace(
        "CannotStaticallyLaunchKernel: User defined triton kernel",
        "CannotStaticallyLaunchNPUKernel: user-defined Triton kernel",
    )
    compiled = compiled.replace(
        "torch._inductor.runtime.triton_heuristics.StaticTritonCompileResult.make_launcher",
        "torch_npu._inductor.triton_experimental.static_launcher.compile_result."
        "NPUStaticTritonCompileResult.make_launcher",
    )
    compiled = (
        '@torch._inductor.config.patch({"compile_threads": 1, "cpp_wrapper": False, '
        '"triton.store_cubin": False})\n' + compiled
    )
    generated = preamble + EXTRA_IMPORTS + direct + "\n\n" + compiled + OBSERVER
    generated = generated.replace(
        "@requires_gpu_and_triton", '@unittest.skipUnless(torch.npu.is_available(), "requires NPU")'
    )
    ast.parse(generated)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated)
    args.output.with_suffix(".patch").write_text("".join(difflib.unified_diff(
        baseline.splitlines(keepends=True), generated.splitlines(keepends=True),
        fromfile="upstream-selected-classes.py", tofile=args.output.name,
    )))
    args.output.with_suffix(".manifest.json").write_text(json.dumps({
        "source": str(args.source),
        "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "generated_sha256": hashlib.sha256(generated.encode()).hexdigest(),
        "classes": ["TestStaticTritonLauncher", "TestStaticTritonCompileResult"],
        "exclusions": "CUDA/XPU-specific units and FastCudaLauncher classes are not selected",
    }, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
