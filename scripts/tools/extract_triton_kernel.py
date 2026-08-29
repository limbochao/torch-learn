#!/usr/bin/env python3
"""Extract one standalone Triton kernel from an Inductor output_code.py."""

import argparse
import ast
import json
import keyword
import operator
import pprint
import re
import textwrap
from pathlib import Path


KERNEL_MARKER = "# kernel path:"
GRAPH_FRAGMENT_MARKER = "# Graph fragment:"
GRAPH_LINE_PREFIX = "#   "
ARG_NAME = re.compile(r"arg\d+_\d+")
NODE_REFERENCE = re.compile(r"%([A-Za-z_]\w*)")
PLACEHOLDER = re.compile(
    r"^%([A-Za-z_]\w*)\s*(?::.*?)?\s*=\s*PlaceHolder\[target=.*\]$"
)
CALL_FUNCTION = re.compile(
    r"^%([A-Za-z_]\w*)\s*(?::.*?)?\s*=\s*"
    r"call_function\[target=([^]]+)\]\(args = (.*), kwargs = (.*)\)$"
)
RETURN = re.compile(r"^return(?:\s+(.*))?$")
DEVICE_LITERAL = re.compile(r"(?:cpu|cuda|npu|xpu|mtia)(?::\d+)?")
TENSOR_METADATA = re.compile(
    r':\s*Tensor\s+"(?P<dtype>[^"\[]+)\[(?P<shape>[^]]*)\]'
    r'\[(?P<stride>[^]]*)\](?P<device>[^"\s]+)"'
)
TORCH_DTYPES = {
    "b8": "torch.bool",
    "f16": "torch.float16",
    "f32": "torch.float32",
    "f64": "torch.float64",
    "bf16": "torch.bfloat16",
    "i8": "torch.int8",
    "i16": "torch.int16",
    "i32": "torch.int32",
    "i64": "torch.int64",
    "u8": "torch.uint8",
}
SAFE_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
SAFE_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
SAFE_DIMENSION_FUNCTIONS = {"max", "min"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract a Triton definition and the arguments used by its first "
            "launch from an Inductor output_code.py"
        )
    )
    parser.add_argument("source", type=Path, help="Inductor output_code.py")
    parser.add_argument("kernel_name", help="generated Triton kernel variable name")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "output path (default: <kernel_name>.py, or "
            "<kernel_name>_case.py with --only-eager)"
        ),
    )
    eager_output = parser.add_mutually_exclusive_group()
    eager_output.add_argument(
        "--include-eager",
        action="store_true",
        help=(
            "generate eager_forward(...) from the kernel's Graph fragment "
            "metadata"
        ),
    )
    eager_output.add_argument(
        "--only-eager",
        action="store_true",
        help="generate only an eager benchmark case for the kernel",
    )
    parser.add_argument(
        "--manual-tiling",
        "--manual_tiling",
        action="store_true",
        help="emit a direct @triton.jit launch with explicit tiling",
    )
    parser.add_argument(
        "--tiling-json",
        "--tiling_json",
        help="tiling record as JSON text or a path to a JSON file",
    )
    return parser.parse_args()


BACKEND_LAUNCH_OPTIONS = {
    "compile_mode",
    "multibuffer",
    "num_ctas",
    "num_stages",
    "num_warps",
}


def assigned_names(node):
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {
            target.id
            for target in targets
            if isinstance(target, ast.Name)
        }
    return set()


def loaded_names(node):
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def is_kernel_definition(node, kernel_name):
    if not isinstance(node, ast.Assign) or assigned_names(node) != {kernel_name}:
        return False
    value = node.value
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "triton"
    )


def is_kernel_launch(node, kernel_name):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == kernel_name
    )


def source_segment(source, node):
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise ValueError(f"cannot read source at line {node.lineno}")
    return textwrap.dedent(segment).rstrip()


def load_tiling_record(value):
    if value is None:
        return None
    try:
        if value.lstrip().startswith("{"):
            content = value
        else:
            content = Path(value).read_text(encoding="utf-8")
        record = json.loads(content)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read tiling JSON: {error}") from error
    if not isinstance(record, dict):
        raise ValueError("tiling JSON must contain an object")
    return record


def triton_source(definition):
    if len(definition.value.args) < 2:
        raise ValueError("async_compile.triton definition has no source argument")
    source_arg = definition.value.args[1]
    if not isinstance(source_arg, ast.Constant) or not isinstance(source_arg.value, str):
        raise ValueError("async_compile.triton source must be a string literal")
    return source_arg.value


def manual_kernel_source(definition, kernel_name):
    source = triton_source(definition)
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == kernel_name
    ]
    if len(functions) != 1:
        raise ValueError(
            f"expected one JIT definition for {kernel_name!r}, found {len(functions)}"
        )
    function = functions[0]
    jit_decorators = [
        decorator
        for decorator in function.decorator_list
        if isinstance(decorator, ast.Attribute)
        and isinstance(decorator.value, ast.Name)
        and decorator.value.id == "triton"
        and decorator.attr == "jit"
    ]
    if len(jit_decorators) != 1:
        raise ValueError(f"{kernel_name!r} does not have exactly one @triton.jit")
    imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    import_text = "\n".join(source_segment(source, node) for node in imports)
    function_text = source_segment(source, function)
    return f"{import_text}\n\n@triton.jit\n{function_text}", function


def constexpr_names(function):
    names = []
    for arg in function.args.args:
        annotation = arg.annotation
        if (
            isinstance(annotation, ast.Attribute)
            and isinstance(annotation.value, ast.Name)
            and annotation.value.id == "tl"
            and annotation.attr == "constexpr"
        ):
            names.append(arg.arg)
    return tuple(names)


def manual_tiling_config(function, record):
    parameters = tuple(arg.arg for arg in function.args.args)
    tiling_names = tuple(name for name in parameters if name.isupper())
    config = {name: 1 for name in tiling_names}
    backend = {
        "compile_mode": "unstructured_in_simt",
        "multibuffer": False,
        "num_ctas": 1,
        "num_stages": 2,
        "num_warps": 1,
    }
    if record is None:
        return config, backend

    selected = record.get("selected_config", record)
    runtime_blocks = record.get("runtime_blocks", {})
    if not isinstance(selected, dict) or not isinstance(runtime_blocks, dict):
        raise ValueError("tiling record config and runtime_blocks must be objects")
    for name in tiling_names:
        if name in selected:
            config[name] = selected[name]
        if name in runtime_blocks:
            config[name] = runtime_blocks[name]
    for name in BACKEND_LAUNCH_OPTIONS:
        if name in selected:
            backend[name] = selected[name]
    return config, backend


def manual_launch(launch, function, record):
    parameters = tuple(arg.arg for arg in function.args.args)
    config, backend = manual_tiling_config(function, record)
    positional = [ast.unparse(arg) for arg in launch.args]
    supplied_names = parameters[: len(positional)]
    missing_required = [
        name for name in parameters[len(positional) :] if name not in config
    ]
    if missing_required:
        raise ValueError(
            "manual launch cannot supply kernel parameters "
            f"{missing_required}; the original .run call has too few arguments"
        )

    numel_expressions = {
        name.removesuffix("_numel"): expression
        for name, expression in zip(supplied_names, positional)
        if name.endswith("_numel")
    }
    split_axes = []
    selected = record.get("selected_config", record) if record else {}
    raw_split_axes = selected.get("split_axis") if isinstance(selected, dict) else None
    axis_names = tuple(numel_expressions)
    if isinstance(raw_split_axes, (list, tuple)):
        for axis_index in raw_split_axes:
            if isinstance(axis_index, int) and 0 <= axis_index < len(axis_names):
                split_axes.append(axis_names[axis_index])
    if not split_axes:
        split_axes = [
            axis
            for axis in axis_names
            if f"{axis.upper()}BLOCK" in config
            and f"{axis.upper()}BLOCK" not in constexpr_names(function)
        ]
    grid = []
    for axis in split_axes[:3]:
        block_name = f"{axis.upper()}BLOCK"
        block = int(config[block_name])
        grid.append(f"triton.cdiv({numel_expressions[axis]}, {block})")
    grid.extend("1" for _ in range(3 - len(grid)))

    keywords = [f"{name}={config[name]!r}" for name in config]
    keywords.extend(f"{name}={backend[name]!r}" for name in sorted(backend))
    arguments = positional + keywords
    body = ",\n".join(f"    {argument}" for argument in arguments)
    return (
        f"{launch.func.value.id}[({', '.join(grid)})](\n"
        f"{body},\n"
        ")"
    )


def find_parent_function(tree, target):
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    current = target
    while current in parent:
        current = parent[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
    raise ValueError(f"kernel launch at line {target.lineno} is not inside a function")


def simple_assignments(scope, before_line):
    assignments = {}
    for node in ast.walk(scope):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if node.lineno >= before_line:
            continue
        for name in assigned_names(node):
            assignments.setdefault(name, []).append(node)
    for nodes in assignments.values():
        nodes.sort(key=lambda item: item.lineno)
    return assignments


def benchmark_assignments(tree):
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "benchmark_compiled_module"
        ):
            return simple_assignments(node, float("inf"))
    return {}


def latest_assignment(assignments, name, before_line):
    candidates = [node for node in assignments.get(name, ()) if node.lineno < before_line]
    return candidates[-1] if candidates else None


def collect_dependencies(launch, local_assignments, benchmark):
    selected_local = {}
    selected_inputs = {}
    pending = [(name, launch.lineno) for name in loaded_names(launch)]
    seen = set()

    while pending:
        name, before_line = pending.pop()
        if name in seen or name == launch.func.value.id:
            continue
        seen.add(name)

        assignment = latest_assignment(local_assignments, name, before_line)
        if assignment is not None:
            selected_local[assignment.lineno] = assignment
            pending.extend(
                (dependency, assignment.lineno)
                for dependency in loaded_names(assignment)
            )
            continue

        assignment = latest_assignment(benchmark, name, float("inf"))
        if assignment is not None:
            selected_inputs[assignment.lineno] = assignment
            pending.extend(
                (dependency, assignment.lineno)
                for dependency in loaded_names(assignment)
            )
        elif ARG_NAME.fullmatch(name):
            raise ValueError(f"cannot find benchmark input construction for {name}")

    return (
        [selected_inputs[line] for line in sorted(selected_inputs)],
        [selected_local[line] for line in sorted(selected_local)],
    )


def kernel_block(lines, definition):
    start = definition.lineno - 1
    while start > 0 and not lines[start].lstrip().startswith(KERNEL_MARKER):
        start -= 1
    if not lines[start].lstrip().startswith(KERNEL_MARKER):
        start = definition.lineno - 1
    return "".join(lines[start : definition.end_lineno]).rstrip()


def graph_fragment(lines, definition):
    marker = None
    for index in range(definition.lineno - 2, -1, -1):
        stripped = lines[index].lstrip()
        if stripped.startswith(GRAPH_FRAGMENT_MARKER):
            marker = index
            break
        if stripped.startswith(KERNEL_MARKER):
            break
    if marker is None:
        raise ValueError(
            f"kernel at line {definition.lineno} has no Graph fragment metadata"
        )

    fragment = []
    for line in lines[marker + 1 : definition.lineno - 1]:
        stripped = line.lstrip()
        if stripped.startswith("# SchedulerNodes:"):
            break
        if stripped.startswith(GRAPH_LINE_PREFIX):
            fragment.append(stripped[len(GRAPH_LINE_PREFIX) :].rstrip())
    if not fragment:
        raise ValueError(
            f"kernel at line {definition.lineno} has an empty Graph fragment"
        )
    return fragment


def split_top_level(text, delimiter):
    parts = []
    start = 0
    stack = []
    quote = None
    escaped = False
    pairs = {"(": ")", "[": "]", "{": "}"}

    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
        elif char == delimiter and not stack:
            parts.append(text[start:index].strip())
            start = index + 1

    if quote is not None or stack:
        raise ValueError(f"unbalanced Graph fragment expression: {text!r}")
    parts.append(text[start:].strip())
    return parts


def replace_node_references(expression):
    return NODE_REFERENCE.sub(r"\1", expression)


def render_kwargs(mapping):
    if mapping == "{}":
        return []
    if not mapping.startswith("{") or not mapping.endswith("}"):
        raise ValueError(f"unsupported Graph fragment kwargs: {mapping!r}")

    entries = []
    for item in split_top_level(mapping[1:-1], ","):
        if not item:
            continue
        key_value = split_top_level(item, ":")
        if len(key_value) < 2:
            raise ValueError(f"unsupported Graph fragment kwarg: {item!r}")
        name, value = key_value[0], ":".join(key_value[1:]).strip()
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(f"invalid Graph fragment kwarg name: {name!r}")
        value = replace_node_references(value)
        if DEVICE_LITERAL.fullmatch(value):
            value = f"torch.device({value!r})"
        entries.append(f"{name}={value}")
    return entries


def render_call(target, args, kwargs):
    if not args.startswith("(") or not args.endswith(")"):
        raise ValueError(f"unsupported Graph fragment args: {args!r}")
    arguments = [
        replace_node_references(item)
        for item in split_top_level(args[1:-1], ",")
        if item
    ]
    arguments.extend(render_kwargs(kwargs))
    return f"{target}({', '.join(arguments)})"


def tensor_metadata(line):
    match = TENSOR_METADATA.search(line)
    if match is None:
        raise ValueError(f"cannot parse Graph fragment tensor metadata: {line!r}")
    dtype = match.group("dtype")
    if dtype not in TORCH_DTYPES:
        raise ValueError(f"unsupported Graph fragment tensor dtype: {dtype!r}")

    shape = [item for item in split_top_level(match.group("shape"), ",") if item]
    stride = [item for item in split_top_level(match.group("stride"), ",") if item]
    if len(shape) != len(stride):
        raise ValueError(f"tensor shape and stride rank differ: {line!r}")

    symbolic_dims = []
    for dim, expression in enumerate(shape):
        if expression_symbols(expression):
            symbolic_dims.append(dim)
    return {
        "shape": shape,
        "stride": stride,
        "device": match.group("device"),
        "dtype": TORCH_DTYPES[dtype],
        "symbolic_dims": symbolic_dims,
    }


def expression_symbols(expression):
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"invalid symbolic dimension expression: {expression!r}") from error
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    unsupported = called_names - SAFE_DIMENSION_FUNCTIONS
    if unsupported:
        raise ValueError(
            f"unsupported symbolic dimension function(s): {sorted(unsupported)}"
        )
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id not in called_names
    }


def render_eager_forward(lines, definition):
    inputs = []
    input_metadata = {}
    statements = []
    return_expression = None

    for line in graph_fragment(lines, definition):
        placeholder = PLACEHOLDER.match(line)
        if placeholder:
            name = placeholder.group(1)
            if name not in inputs:
                inputs.append(name)
                input_metadata[name] = tensor_metadata(line)
            continue

        call = CALL_FUNCTION.match(line)
        if call:
            name, target, args, kwargs = call.groups()
            if not name.isidentifier() or keyword.iskeyword(name):
                raise ValueError(f"invalid Graph fragment node name: {name!r}")
            statements.append(f"{name} = {render_call(target, args, kwargs)}")
            continue

        returned = RETURN.match(line)
        if returned:
            return_expression = replace_node_references(returned.group(1) or "None")
            continue

        raise ValueError(f"unsupported Graph fragment line: {line!r}")

    if return_expression is None:
        raise ValueError("Graph fragment has no return value")
    signature = ", ".join(inputs)
    body = statements + [f"return {return_expression}"]
    function = [
        "# Eager reference reconstructed from Inductor Graph fragment metadata.",
        f"def eager_forward({signature}):",
        textwrap.indent("\n".join(body), "    "),
    ]
    result = "\n".join(function)
    ast.parse(result)
    return result, inputs, input_metadata


def render_compiled_eager(inputs, input_metadata):
    signature = ", ".join(inputs)
    body = []
    for name in inputs:
        body.extend(
            f"torch._dynamo.mark_dynamic({name}, {dim})"
            for dim in input_metadata[name]["symbolic_dims"]
        )
    body.extend(
        [
            "compiled_eager_forward = torch.compile(eager_forward, dynamic=None)",
            f"return compiled_eager_forward({signature})",
        ]
    )
    function = [
        "# Compile and run the eager reference with the extracted inputs.",
        f"def run_compiled_eager({signature}):",
        textwrap.indent("\n".join(body), "    "),
    ]
    result = "\n".join(function)
    ast.parse(result)
    return result


def tuple_expression(items):
    suffix = "," if len(items) == 1 else ""
    return f"({', '.join(items)}{suffix})"


def render_eager_inputs(inputs, input_metadata):
    statements = []
    argument_names = []
    for index, name in enumerate(inputs):
        metadata = input_metadata[name]
        argument_name = f"eager_input_{index}"
        argument_names.append(argument_name)
        statements.append(
            f"{argument_name} = rand_strided("
            f"{tuple_expression(metadata['shape'])}, "
            f"{tuple_expression(metadata['stride'])}, "
            f"device={metadata['device']!r}, dtype={metadata['dtype']})"
        )
    result = "\n".join(statements)
    ast.parse(result)
    return result, argument_names


def assignment_value(node):
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        return node.value
    return None


def all_simple_assignments(tree):
    assignments = {}
    for node in ast.walk(tree):
        for name in assigned_names(node):
            assignments.setdefault(name, []).append(node)
    for nodes in assignments.values():
        nodes.sort(key=lambda item: item.lineno)
    return assignments


def evaluate_integer_expression(node, resolve_name):
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return node.value
    if isinstance(node, ast.Name):
        return resolve_name(node.id)
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_UNARY_OPERATORS:
        return SAFE_UNARY_OPERATORS[type(node.op)](
            evaluate_integer_expression(node.operand, resolve_name)
        )
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_BINARY_OPERATORS:
        return SAFE_BINARY_OPERATORS[type(node.op)](
            evaluate_integer_expression(node.left, resolve_name),
            evaluate_integer_expression(node.right, resolve_name),
        )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in SAFE_DIMENSION_FUNCTIONS
        and not node.keywords
    ):
        values = [evaluate_integer_expression(arg, resolve_name) for arg in node.args]
        return max(values) if node.func.id == "max" else min(values)
    raise ValueError("expression is not a supported integer expression")


def resolve_symbol_values(tree, symbols):
    assignments = all_simple_assignments(tree)
    resolved = {}

    def resolve(name, resolving=()):
        if name in resolved:
            return resolved[name]
        if name in resolving:
            raise ValueError(f"cyclic assignment while resolving {name!r}")

        values = set()
        for assignment in assignments.get(name, ()):
            value = assignment_value(assignment)
            if value is None:
                continue
            try:
                candidate = evaluate_integer_expression(
                    value,
                    lambda dependency: resolve(dependency, resolving + (name,)),
                )
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            if candidate > 0:
                values.add(candidate)
        if len(values) != 1:
            detail = "no value" if not values else f"ambiguous values {sorted(values)}"
            raise ValueError(f"cannot resolve symbolic dimension {name!r}: {detail}")
        resolved[name] = values.pop()
        return resolved[name]

    return {name: resolve(name) for name in sorted(symbols)}


def sample_bindings(baseline):
    samples = [dict(baseline)]
    for name, value in baseline.items():
        for candidate in (value - 1, value + 1):
            if candidate <= 0:
                continue
            binding = dict(baseline)
            binding[name] = candidate
            if binding not in samples:
                samples.append(binding)
    keys = tuple(sorted(baseline))
    return sorted(samples, key=lambda binding: tuple(binding[key] for key in keys))


def render_binding_list(name, bindings):
    items = [
        textwrap.indent(
            pprint.pformat(binding, width=84, sort_dicts=True), "    "
        )
        + ","
        for binding in bindings
    ]
    return f"{name} = [\n" + "\n".join(items) + "\n]"


def render_eager_case(kernel_name, eager_forward, inputs, input_metadata, tree):
    symbols = set()
    for metadata in input_metadata.values():
        for expression in metadata["shape"] + metadata["stride"]:
            symbols.update(expression_symbols(expression))
    baseline = resolve_symbol_values(tree, symbols)
    samples = sample_bindings(baseline)

    setup = [f"{name} = binding[{name!r}]" for name in sorted(symbols)]
    for name in inputs:
        metadata = input_metadata[name]
        setup.append(
            f"{name} = rand_strided(\n"
            f"    {tuple_expression(metadata['shape'])},\n"
            f"    {tuple_expression(metadata['stride'])},\n"
            f"    device=device,\n"
            f"    dtype={metadata['dtype']},\n"
            ")"
        )
    setup.append(f"return {tuple_expression(inputs)}, {{}}")

    dynamic_dims = {
        f"args[{index}]": tuple(input_metadata[name]["symbolic_dims"])
        for index, name in enumerate(inputs)
        if input_metadata[name]["symbolic_dims"]
    }
    sections = [
        "import torch\nfrom torch._dynamo.testing import rand_strided",
        eager_forward,
        render_binding_list("SAMPLE_BINDINGS", samples),
        render_binding_list("COMPILE_BINDINGS", [baseline]),
        "def make_inputs(binding, device):\n"
        + textwrap.indent("\n".join(setup), "    "),
        "DYNAMIC_DIMS = "
        + pprint.pformat(dynamic_dims, width=88, sort_dicts=True),
        "CASE = {\n"
        f"    'name': {kernel_name!r},\n"
        "    'forward': eager_forward,\n"
        "    'make_inputs': make_inputs,\n"
        "    'sample_bindings': SAMPLE_BINDINGS,\n"
        "    'compile_bindings': COMPILE_BINDINGS,\n"
        "    'dynamic_dims': DYNAMIC_DIMS,\n"
        "}",
    ]
    result = "\n\n\n".join(sections)
    result = "\n".join(line.rstrip() for line in result.splitlines()) + "\n"
    ast.parse(result)
    return result


def common_header(lines, tree):
    for index, line in enumerate(lines):
        if line.lstrip().startswith(KERNEL_MARKER):
            return "".join(lines[:index]).rstrip()
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "triton"
    ]
    if not definitions:
        raise ValueError("source contains no async_compile.triton(...) definition")
    return "".join(lines[: min(node.lineno for node in definitions) - 1]).rstrip()


def manual_common_header(lines, tree):
    header = common_header(lines, tree)
    omitted = (
        "from torch._inductor.async_compile import AsyncCompile",
        "async_compile = AsyncCompile()",
    )
    return "\n".join(
        line for line in header.splitlines() if line.strip() not in omitted
    ).rstrip()


def kernel_device(definition):
    for keyword in definition.value.keywords:
        if keyword.arg == "device_str" and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


def device_setup(device, input_nodes):
    if device not in {"npu", "cuda"}:
        return []
    input_text = "\n".join(ast.unparse(node) for node in input_nodes)
    match = re.search(rf"device=['\"]{device}:(\d+)", input_text)
    index = match.group(1) if match else "0"
    return [f"torch.{device}.set_device({index})"]


def render_extraction(
    source,
    tree,
    kernel_name,
    include_eager=False,
    only_eager=False,
    manual_tiling=False,
    tiling_record=None,
):
    lines = source.splitlines(keepends=True)
    definitions = [
        node for node in tree.body if is_kernel_definition(node, kernel_name)
    ]
    if len(definitions) != 1:
        raise ValueError(
            f"expected one definition for {kernel_name!r}, found {len(definitions)}"
        )
    definition = definitions[0]

    launches = sorted(
        (node for node in ast.walk(tree) if is_kernel_launch(node, kernel_name)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    if not launches:
        raise ValueError(f"found no {kernel_name}.run(...) call")
    launch = launches[0]
    if only_eager:
        inputs, locals_ = [], []
    else:
        scope = find_parent_function(tree, launch)
        inputs, locals_ = collect_dependencies(
            launch,
            simple_assignments(scope, launch.lineno),
            benchmark_assignments(tree),
        )

    if only_eager:
        lines = source.splitlines(keepends=True)
        eager_forward, eager_inputs, input_metadata = render_eager_forward(
            lines, definition
        )
        return (
            render_eager_case(
                kernel_name, eager_forward, eager_inputs, input_metadata, tree
            ),
            launch,
            len(launches),
        )

    if manual_tiling:
        definition_text, jit_function = manual_kernel_source(definition, kernel_name)
    else:
        definition_text = kernel_block(lines, definition)
    sections = [
        manual_common_header(lines, tree) if manual_tiling else common_header(lines, tree),
        definition_text,
    ]
    if include_eager:
        eager_forward, eager_inputs, input_metadata = render_eager_forward(
            lines, definition
        )
        sections.extend(
            [
                eager_forward,
                render_compiled_eager(eager_inputs, input_metadata),
            ]
        )
    if not manual_tiling:
        sections.append("async_compile.wait(globals())\ndel async_compile")
    if inputs or include_eager:
        input_construction = "\n".join(
            source_segment(source, node) for node in inputs
        )
        sections.append(
            "from torch._dynamo.testing import rand_strided\n\n"
            + input_construction
        )
    setup = device_setup(kernel_device(definition), inputs)
    body = setup + [source_segment(source, node) for node in locals_]
    if include_eager:
        eager_construction, eager_arguments = render_eager_inputs(
            eager_inputs, input_metadata
        )
        body.append(eager_construction)
        body.append(f"eager_result = run_compiled_eager({', '.join(eager_arguments)})")
    body.append(
        manual_launch(launch, jit_function, tiling_record)
        if manual_tiling
        else source_segment(source, launch)
    )
    sections.append("\n".join(body))
    extracted = "\n\n\n".join(sections).rstrip()
    extracted = "\n".join(line.rstrip() for line in extracted.splitlines()) + "\n"
    return extracted, launch, len(launches)


def main():
    args = parse_args()
    if not args.kernel_name.isidentifier():
        raise SystemExit(f"invalid Python kernel name: {args.kernel_name!r}")
    source_path = args.source.resolve()
    default_name = (
        f"{args.kernel_name}_case.py" if args.only_eager else f"{args.kernel_name}.py"
    )
    output_path = (args.output or Path(default_name)).resolve()
    try:
        if args.tiling_json and not args.manual_tiling:
            raise ValueError("--tiling-json requires --manual-tiling")
        if args.manual_tiling and args.only_eager:
            raise ValueError("--manual-tiling cannot be combined with --only-eager")
        tiling_record = load_tiling_record(args.tiling_json)
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
        extracted, launch, launch_count = render_extraction(
            source,
            tree,
            args.kernel_name,
            include_eager=args.include_eager,
            only_eager=args.only_eager,
            manual_tiling=args.manual_tiling,
            tiling_record=tiling_record,
        )
    except (OSError, SyntaxError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(extracted, encoding="utf-8")
    print(
        f"extracted {args.kernel_name}: first launch at line {launch.lineno} "
        f"({launch_count} launch(es) found) -> {output_path}"
    )


if __name__ == "__main__":
    main()
