"""Source-extracted checks only: synthetic environment, no lowering/compiler/device."""
import argparse
import ast
from functools import lru_cache
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, required=True, help="Git repository containing both fixed revisions")
parser.add_argument("--output", type=Path, help="Optional JSON output with all observations")
args = parser.parse_args()
repo = args.repo
base = "0c3f556423d8a5aa1fec77d93ee17ccb6327aef1"
candidate = "7c7c20c21af250cca6344ec554e1e604f84e103f"
probe_path = Path(__file__).with_name("review_reduction_axes_codegen.py")


def source_at(revision, path):
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{path}"], text=True)


spec = importlib.util.spec_from_file_location('probe', probe_path)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class Reduction:
    default_value = staticmethod(lambda *args: 0)
    default_accumulator = staticmethod(lambda *args: 0)


@lru_cache(maxsize=None)
def load(working, types=('sum',)):
    ns = p.load_classes(repo, base)
    ns['ir'].Reduction = Reduction
    ns['EnableReduction'], ns['DisableReduction'] = object(), object()
    if working:
        analysis_tree = ast.parse(source_at(candidate, 'torch_npu/_inductor/codegen/kernel_analysis.py'))
        analysis_node = next(
            n for n in analysis_tree.body if isinstance(n, ast.ClassDef) and n.name == 'ReductionAnalysis'
        )
        tree = ast.parse(source_at(candidate, 'torch_npu/_inductor/codegen/triton.py'))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NPUIndexTritonKernel')
        selected = {
            'triton_tensor_ndim', 'dense_size_list', 'is_contiguous_reduction',
            'dense_size_str', 'reduction_resize', 'reduction_dim', 'reduction',
            '_use_contiguous_sum_layout', '_iter_schedule_nodes',
        }
        node.bases = []
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in selected]
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        module = ast.fix_missing_locations(ast.Module(body=[future, analysis_node, node], type_ignores=[]))
        exec(compile(module, '<candidate-source>', 'exec'), ns)

    def initialize(kernel):
        ns['_last_kernel'] = kernel
        reductions = []
        for kind in types:
            node = Reduction()
            node.reduction_type = kind
            reductions.append(SimpleNamespace(node=SimpleNamespace(data=node)))
        kernel.node_schedule = reductions
        kernel.features = SimpleNamespace(node_schedule=reductions)

    ns['NPUIndexTritonKernel'].__init__ = initialize
    return ns


# Execute the seven exact unit-test methods from the new test file. Only the
# unavailable import environment (sympy/torch/IR classes) is substituted.
ns = load(True)
ns.update(SimpleNamespace=SimpleNamespace, TestCase=unittest.TestCase,
          sympy=SimpleNamespace(Symbol=p.Axis),
          triton_codegen=SimpleNamespace(NPUIndexTritonKernel=ns['NPUIndexTritonKernel']))
tests = ast.parse(source_at(candidate, 'test/_inductor/test_contiguous_sum_layout.py'))
node = next(n for n in tests.body if isinstance(n, ast.ClassDef) and n.name == 'TestContiguousSumLayout')
module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
exec(compile(module, '<source-extracted-tests>', 'exec'), ns)
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ns[node.name]))
assert result.wasSuccessful()

records = []
non_target = 0
corrected = 0
for names in list(itertools.permutations(('r0', 'r1', 'x2', 'x3'))) + [
    ('r0', 'r1'), ('r0', 'x1'), ('x0', 'r1'), ('r0', 'r1', 'x2'), ('x0', 'r1', 'r2'),
    ('r0', 'r1', 'r2', 'x3'),
]:
    parsed_cases = [names]
    if len([n for n in names if n.startswith('r')]) > 1:
        i = next(i for i, n in enumerate(names) if n.startswith('r'))
        parsed_cases.append(names[:i + 1] + ('x9',) + names[i + 1:])
    for parsed, persistent, broadcast, types in itertools.product(
        parsed_cases, (True, False), (False, True), (('sum',), ('max',), ('sum', 'max')),
    ):
        options = dict(parsed_names=parsed, persistent=persistent, broadcast=broadcast, reduction_type=types[0])
        old = p.inspect(load(False, types), names, **options)
        new = p.inspect(load(True, types), names, **options)
        red_positions = [i for i, n in enumerate(names) if n.startswith('r')]
        target = types == ('sum',) and 1 < len(red_positions) < len(names) and (
            red_positions[-1] - red_positions[0] + 1 == len(red_positions)
        )
        if not target:
            assert old == new, (names, options, old, new)
            non_target += 1
        else:
            shape = ['1' if name.startswith('r') else f'{name.upper()}BLOCK_SUB' for name in names]
            assert tuple(new['result_metadata']) == tuple(shape), (names, options, new)
            assert new['helper_returns']['triton_tensor_ndim'] == len(names) - len(red_positions) + 1
            assert new['reduction_dim'] == len(names) - len(red_positions)
            assert new['helper_returns']['reduction_resize_s'] == f"s.reshape({', '.join(shape)})"
            if not persistent:
                packed_shape = tuple(new['helper_returns']['dense_post_reduction_list'])
                assert tuple(new['accumulators'][0]['shape']) == packed_shape
            corrected += 1
        outside = []
        for working in (False, True):
            kernel = load(working, types)['_last_kernel']
            kernel.inside_reduction = False
            outside.append((kernel.dense_size_list(), kernel.dense_size_str(), kernel.triton_tensor_ndim()))
        if not target:
            assert outside[0] == outside[1], (names, types, outside)
        else:
            assert outside[1][0] == new['helper_returns']['dense_post_reduction_list'], (names, outside)
            assert outside[1][1] == new['packed_shape'], (names, outside)
        records.append(dict(axes=names, persistent=persistent, broadcast=broadcast,
                            reduction_types=types, parsed_axes=parsed, target=target,
                            old=old, new=new, outside_reduction_helpers=outside))
summary = dict(
    base=base, candidate_commit=candidate,
    scope=(
        'Source-extracted emitter checks with synthetic CSE, dtype, schedule and axis objects; '
        'masks empty; no real graph lowering, Triton compilation or device execution.'
    ),
    source_extracted_unit_methods=result.testsRun,
    comparisons=len(records), non_target_identical=non_target, target_shape_checks=corrected,
    outside_reduction_helper_comparisons=len(records),
    npu_tests='Not run: local host has no torch/NPU; eight device cases added to test file.',
)
if args.output is not None:
    args.output.write_text(json.dumps(dict(summary=summary, records=records), indent=2) + '\n')
print(json.dumps(summary, indent=2))
