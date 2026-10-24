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
        self.variables = []
        self.statements = []
        self.named_variables = []

    def newvar(self, dtype, shape):
        self.counter += 1
        variable = Variable(f"tmp{self.counter}", dtype, shape)
        self.variables.append(variable)
        return variable

    def generate(self, buffer, expression, dtype, shape):
        variable = self.newvar(dtype, shape)
        self.lines.append((expression, shape))
        self.statements.append({"name": str(variable), "expression": str(expression), "shape": shape})
        return variable

    def namedvar(self, name, dtype, shape):
        variable = Variable(name, dtype, shape)
        self.named_variables.append({"name": name, "shape": shape})
        return variable


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


def inspect(namespace, names, persistent=True, parsed_names=None, broadcast=False, reduction_type="sum"):
    kernel = namespace["NPUIndexTritonKernel"]()
    axes = [Axis(name) for name in names]
    reduction_axes = [axis for axis in axes if axis.name.startswith("r")]
    kernel.golden_var_list = list(reversed(axes))
    kernel.tiling_axis = axes
    kernel.numof_reduction_axis = lambda: len(reduction_axes)
    kernel.reduction_axis_list = lambda: reduction_axes
    parsed_axes = [Axis(name) for name in reversed(parsed_names)] if parsed_names else kernel.golden_var_list
    kernel.parse_golden_from_load_store_index = lambda: list(parsed_axes)
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
    input_names = [name for name in names if not name.startswith("r")] if broadcast else names
    kernel.load_store_indexing = [" + ".join(input_names)]
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
    value_shape = tuple(
        "1" if broadcast and name.startswith("r") else f"{name.upper()}BLOCK_SUB" for name in names
    )
    value = Variable("input", dtype, value_shape)
    result = kernel.reduction(dtype, dtype, reduction_type, value)
    generated_names = {statement["name"] for statement in kernel.cse.statements}
    result_holder = next(variable for variable in kernel.cse.variables if str(variable) not in generated_names)
    analysis = kernel.reduce_analysis
    return {
        "axes": names,
        "persistent": persistent,
        "reduction_type": reduction_type,
        "broadcast": broadcast,
        "parsed_value_axes": parsed_names or names,
        "input_metadata": value.shape,
        "helper_returns": {
            "analysis_dense_size_list": analysis.dense_size_list(),
            "kernel_dense_size_list": kernel.dense_size_list(),
            "dense_reduction_list": analysis.dense_reduction_list(),
            "dense_post_reduction_list": analysis.dense_post_reduction_list(),
            "is_contiguous_reduction": kernel.is_contiguous_reduction(),
            "triton_tensor_ndim": kernel.triton_tensor_ndim(),
            "reduction_resize_s": kernel.reduction_resize("s", kernel.reduction_dim()),
        },
        "packed_shape": kernel.dense_size_str(),
        "reduction_dim": kernel.reduction_dim(),
        "initial_result_metadata": result_holder.shape,
        "result_metadata": result.shape,
        "emitted_final": kernel.cse.lines[-1][0] if persistent else kernel.post_loop_combine.lines[-1],
        "statements": kernel.cse.statements,
        "accumulators": kernel.cse.named_variables,
        "prefix": kernel.prefix.lines,
        "post_loop_combine": kernel.post_loop_combine.lines,
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
        cases = (
            (("x1", "r0", "r2"), {}),
            (("r2", "r3", "x5"), {"parsed_names": ("r2", "x4", "r3", "x5")}),
            (("r0", "x1", "r2"), {"broadcast": True}),
            (("r0", "x1", "r2"), {"reduction_type": "max"}),
            (("r0", "x1", "r2"), {"persistent": False}),
        )
        for names, options in cases:
            print(json.dumps({"revision": rev, **inspect(namespace, names, **options)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
