"""Prepare auditable test overlays and narrow NPU Inductor fixes without installing them."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import textwrap


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after, 1)


def edit_method(text, name, edit):
    nodes = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, name
    node = nodes[0]
    lines = text.splitlines(keepends=True)
    before = ''.join(lines[node.lineno - 1:node.end_lineno])
    after = edit(before)
    assert before != after, name
    lines[node.lineno - 1:node.end_lineno] = [after]
    return ''.join(lines)


def prepare_tests(original, manifest):
    updated = dict(original)
    for filename in ['test_torchinductor.py', 'test_compile_subprocess.py', 'test_torchinductor_dynamic_shapes.py']:
        content = updated[filename]
        for item in manifest['changes']:
            tree = ast.parse(content)
            nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == item['method']]
            if not nodes:
                continue
            assert len(nodes) == 1
            node = nodes[0]
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            lines = content.splitlines(keepends=True)
            segment = ''.join(lines[start:node.end_lineno])
            segment = replace_once(segment, item['before'], item['after'])
            lines[start:node.end_lineno] = [segment]
            content = ''.join(lines)
        updated[filename] = ('from skip_validation_support import npu_large_tensor_test, '
                             'npu_triton_templates_available\n' + content)

    name = 'test_inplacing_pass.py'
    updated[name] = replace_once(updated[name], 'const = torch.tensor(0.0)\ndevice = GPU_TYPE',
                                'device = GPU_TYPE\nconst = torch.tensor(0.0, device=device)')
    name = 'test_torchinductor.py'
    updated[name] = edit_method(updated[name], 'define_custom_op_for_test', lambda text: replace_once(
        text, '        libtest.impl(id_, fn, "MPS")',
        '        libtest.impl(id_, fn, "MPS")\n        libtest.impl(id_, fn, "PrivateUse1")'))
    updated[name] = edit_method(updated[name], 'test_bmm_dot_shape_int_preserves_eager_error', lambda text: text.replace(
        '        with self.assertRaisesRegex(NotImplementedError, msg):',
        '        with self.assertRaisesRegex(error_type, msg):').replace(
        '        msg = \'baddbmm_cuda" not implemented for\'',
        '        msg = \'baddbmm_cuda" not implemented for\'\n'
        '        error_type = NotImplementedError\n'
        '        if torch.device(self.device).type == "npu":\n'
        '            error_type = RuntimeError\n'
        '            msg = r"Tensor matA not implemented for DT_INT64"'))
    for method, operand in [('test_mm_mixed_dtype', 'matB'), ('test_linear_mixed_dtype', 'matA')]:
        updated[name] = edit_method(updated[name], method, lambda text, operand=operand: replace_once(
            text, '        msg = "expected .* and .* to have the same dtype, but got: .* != .*"',
            '        msg = "expected .* and .* to have the same dtype, but got: .* != .*"\n'
            '        if torch.device(self.device).type == "npu":\n'
            f'            msg = r"Tensor {operand} not implemented for DT_INT64"'))
    name = 'test_torchinductor_dynamic_shapes.py'
    updated[name] = edit_method(updated[name], 'test_constant_fold_uniform_value_dynamic',
                                lambda text: text.replace('run(source_codes[0])', 'run("\\n".join(source_codes))'))
    name = 'test_utils.py'
    tree = ast.parse(updated[name])
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'test_flops_fx')
    lines = updated[name].splitlines(keepends=True)
    method = ''.join(lines[node.lineno - 1:node.end_lineno])
    lines[node.lineno - 1:node.end_lineno] = []
    content = ''.join(lines)
    content = content.replace('\n\nif __name__ == "__main__":',
                              '\n\nclass TestFlopsHost(TestCase):\n' + method + '\n\nif __name__ == "__main__":')
    updated[name] = content
    return updated


def prepare_inductor(original):
    updated = dict(original)
    name = 'torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py'
    updated[name] = edit_method(updated[name], 'dtype_optimal_pass', lambda text: replace_once(
        replace_once(text, '        if node.op == "call_method":',
                     '        if node.op == "call_method" and node.target == "to" and node.args:'),
        '            input_dtype = input_fake.dtype if input_fake is not None else None',
        '            input_dtype = input_fake.dtype if isinstance(input_fake, torch.Tensor) else None'))
    name = 'torch_npu/_inductor/runtime/triton_heuristics.py'
    updated[name] = replace_once(updated[name],
        "        return benchmarker.benchmark_gpu(kernel_call, rep=1, device_type='npu')",
        '        # Pointwise autotuning preserves results across tile configurations.\n'
        '        # Keep custom kernels and reductions subject to the deterministic guard.\n'
        '        vetted = not self.custom_kernel and self.heuristic_type == HeuristicType.POINTWISE\n'
        "        return benchmarker.benchmark_gpu(\n"
        "            kernel_call, rep=1, device_type='npu', is_vetted_benchmarking=vetted\n"
        '        )')
    return updated


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--test-dir', type=Path, required=True)
    parser.add_argument('--site-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    artifacts = repo / 'artifacts'
    previous = artifacts / 'npu-inductor-pr47384-restored-driver-20261009'
    preflight = json.loads((previous / 'preflight.json').read_text())
    manifest = json.loads((artifacts / 'npu-inductor-213-batches-20261008/skip-validation/manifest.json').read_text())
    root = args.output_dir
    assert not root.exists(), root
    overlay = root / 'test/inductor'
    overlay.mkdir(parents=True)
    originals = {}
    for p in args.test_dir.glob('*.py'):
        if p.name in preflight['test_hashes']:
            assert hashlib.sha256(p.read_bytes()).hexdigest() == preflight['test_hashes'][p.name], p
        originals[p.name] = p.read_text(encoding='utf-8-sig')
        shutil.copy2(p, overlay / p.name)
    shutil.copytree(args.test_dir / 'extension_backends', overlay / 'extension_backends')
    tests = prepare_tests(originals, manifest)
    sources = {name: (args.site_dir / name).read_text(encoding='utf-8-sig') for name in [
        'torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py',
        'torch_npu/_inductor/runtime/triton_heuristics.py',
    ]}
    for name in sources:
        assert hashlib.sha256((args.site_dir / name).read_bytes()).hexdigest() == preflight['source_hashes'][name]
    candidates = prepare_inductor(sources)
    patches = root / 'patches'
    patches.mkdir()
    changes = []
    for domain, before, after in [('test', originals, tests), ('site', sources, candidates)]:
        for name, content in after.items():
            if content == before[name]:
                continue
            compile(content, name, 'exec')
            target = overlay / name if domain == 'test' else root / 'candidate' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            patch_name = name.replace('/', '__') + '.patch'
            patch = ''.join(difflib.unified_diff(before[name].splitlines(keepends=True),
                content.splitlines(keepends=True), fromfile='a/' + name, tofile='b/' + name))
            (patches / patch_name).write_text(patch)
            changes.append({'domain': domain, 'file': name, 'patch': 'patches/' + patch_name,
                            'before_sha256': hashlib.sha256(before[name].encode()).hexdigest(),
                            'after_sha256': hashlib.sha256(content.encode()).hexdigest()})
    for name in ['run_batch.py', 'skip_validation_support.py', 'run_conv_precision_validation.py']:
        shutil.copy2(Path(__file__).parent / name, overlay / name)
    (root / 'changes.json').write_text(json.dumps(changes, indent=2) + '\n')
    print(json.dumps(changes, indent=2))


if __name__ == '__main__':
    main()
