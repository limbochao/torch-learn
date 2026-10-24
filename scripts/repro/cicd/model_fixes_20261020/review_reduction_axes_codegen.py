"""Inspect actual reduction emitter methods without importing torch or using a device.

This checks source-level DSL strings and CSE metadata, not Triton compilation.
"""

import argparse
import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace


class Axis:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class Variable:
    def __init__(self, name, dtype, shape):
        self.name, self.dtype, self.shape = name, dtype, shape
        self.mask_vars = set()

    def __str__(self):
        return self.name


class RecordingCSE:
    def __init__(self):
        self.reduction_cache = {}
        self.lines = []
        self.counter = 0

    def newvar(self, dtype, shape):
        self.counter += 1
        return Variable(f"tmp{self.counter}", dtype, shape)

    def generate(self, buffer, expression, dtype, shape):
        variable = self.newvar(dtype, shape)
        self.lines.append((expression, shape))
        return variable

    def namedvar(self, name, dtype, shape):
        return Variable(name, dtype, shape)


class Buffer:
    def __init__(self):
        self.lines = []

    def writeline(self, line):
        self.lines.append(line)


def load_classes(repo, revision):
    selected = {
        "triton_tensor_ndim", "dense_size_list", "is_contiguous_reduction",
        "dense_size_str", "reduction_resize", "reduction_dim", "reduction",
    }
    classes = []
    for filename, class_name in (
        ("kernel_analysis.py", "ReductionAnalysis"),
        ("triton.py", "NPUIndexTritonKernel"),
    ):
        path = f"torch_npu/_inductor/codegen/{filename}"
        source = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{revision}:{path}"], text=True
        )
        tree = ast.parse(source)
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        if class_name == "NPUIndexTritonKernel":
            node.bases = []
            node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in selected]
        classes.append(node)
    namespace = {
        "ir": SimpleNamespace(
            Reduction=SimpleNamespace(default_value=lambda *args: 0, default_accumulator=lambda *args: 0),
            get_reduction_combine_fn=lambda *args: lambda left, right: f"({left} + {right})",
        ),
        "triton_acc_type": lambda dtype: "tl.float32",
        "upcast_acc_dtype": lambda dtype: dtype,
        "constant_repr": repr,
        "is_welford_reduction": lambda reduction_type: False,
        "torch": SimpleNamespace(bool=object()),
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future] + classes, type_ignores=[]))
    exec(compile(module, f"<{revision}-source>", "exec"), namespace)
    return namespace


def inspect(namespace, names, persistent=True):
    kernel = namespace["NPUIndexTritonKernel"]()
    axes = [Axis(name) for name in names]
    reduction_axes = [axis for axis in axes if axis.name.startswith("r")]
    kernel.golden_var_list = list(reversed(axes))
    kernel.tiling_axis = axes
    kernel.numof_reduction_axis = lambda: len(reduction_axes)
    kernel.reduction_axis_list = lambda: reduction_axes
    kernel.parse_golden_from_load_store_index = lambda: list(kernel.golden_var_list)
    kernel.inside_reduction = True
    kernel.find_reduction_node = lambda: object()
    kernel.vectorized_welford_axis = None
    kernel.persistent_reduction = persistent
    kernel.reduce_analysis = None
    if len(reduction_axes) == 1:
        analysis = object.__new__(namespace["ReductionAnalysis"])
        analysis.kernel = kernel
        analysis.contiguous_reduction = False
        analysis.reduced_dim = next(i for i, axis in enumerate(axes) if axis in reduction_axes)
        kernel.reduce_analysis = analysis
    kernel.sorted_axis = []
    kernel._load_mask = None
    kernel.load_store_indexing = [" + ".join(names)]
    kernel.range_tree_nodes = dict.fromkeys(axes)
    kernel.filter_masks = lambda masks: None
    kernel.cse = RecordingCSE()
    kernel.compute = Buffer()
    kernel.prefix = Buffer()
    kernel.post_loop_combine = Buffer()
    kernel._map_tuple_or_scalar = lambda fn, value: tuple(map(fn, value)) if isinstance(value, tuple) else fn(value)
    kernel.outside_loop_vars = set()
    kernel.reduction_result_vars = set()
    namespace["V"] = SimpleNamespace(kernel=kernel)
    dtype = type("FloatDType", (), {"is_floating_point": True})()
    value = Variable("input", dtype, tuple(f"{name.upper()}BLOCK_SUB" for name in names))
    result = kernel.reduction(dtype, dtype, "sum", value)
    return {
        "axes": names,
        "persistent": persistent,
        "packed_shape": kernel.dense_size_str(),
        "reduction_dim": kernel.reduction_dim(),
        "result_metadata": result.shape,
        "emitted_final": kernel.cse.lines[-1][0] if persistent else kernel.post_loop_combine.lines[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    revision = "66bed811b6a06b2847f86980611f89d8ebdb06c7"
    for rev in (f"{revision}^", revision):
        namespace = load_classes(args.repo, rev)
        for names in (("r0", "r1"), ("r0", "r1", "x2"), ("r0", "x1", "r2"), ("r0", "x1")):
            print(json.dumps({"revision": rev, **inspect(namespace, names)}, ensure_ascii=False))
        print(json.dumps({"revision": rev, **inspect(namespace, ("r0", "r1"), persistent=False)}))


if __name__ == "__main__":
    main()
