"""Prepare auditable test overlays and narrow NPU Inductor fixes without installing them."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after, 1)


def edit_method(text, name, edit, with_decorators=False):
    nodes = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, name
    node = nodes[0]
    lines = text.splitlines(keepends=True)
    start = min([node.lineno] + [d.lineno for d in node.decorator_list]) if with_decorators else node.lineno
    before = ''.join(lines[start - 1:node.end_lineno])
    after = edit(before)
    assert before != after, name
    lines[start - 1:node.end_lineno] = [after]
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
    updated[name] = edit_method(updated[name], 'test_bmm_dot_shape_int_preserves_eager_error', lambda text:
        text.replace(
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
    updated[name] = edit_method(updated[name], 'test_linear_mixed_dtype', lambda text: replace_once(
        text, '        with self.assertRaisesRegex(RuntimeError, msg):\n            with torch.no_grad():',
        '        if torch.device(self.device).type == "npu" and not config.cpp_wrapper:\n'
        '            msg = r"expected scalar type torch.int64 but found torch.float32"\n'
        '        with self.assertRaisesRegex(RuntimeError, msg):\n            with torch.no_grad():'))
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
    name = 'test_benchmarking.py'
    for method in ['test_benchmark_smoke', 'test_benchmark_gpu_smoke']:
        def edit_smoke(text, method=method):
            start = text.index('    @decorateIf(')
            end = text.index('    @parametrize(', start)
            text = text[:start] + text[end:]
            if method == 'test_benchmark_smoke':
                target = '        timing = benchmarker.benchmark(fn, fn_args, fn_kwargs)'
                guard = ('        if benchmarker_cls is Benchmarker and device != "cpu":\n'
                         '            with self.assertRaises(NotImplementedError):\n'
                         '                benchmarker.benchmark(fn, fn_args, fn_kwargs)\n'
                         '            return\n')
            else:
                target = '        timing = benchmarker.benchmark_gpu(_callable)'
                guard = ('        if benchmarker_cls is Benchmarker:\n'
                         '            with self.assertRaises(NotImplementedError):\n'
                         '                benchmarker.benchmark_gpu(_callable)\n'
                         '            return\n')
            return replace_once(text, target, guard + target)
        updated[name] = edit_method(updated[name], method, edit_smoke, with_decorators=True)
    for suffix in ['no_devices', 'many_devices']:
        def edit_infer(text, suffix=suffix):
            text = text.replace('    @unittest.expectedFailure\n', '')
            if suffix == 'no_devices':
                return replace_once(text, '        benchmarker.benchmark(fn, (), {})',
                    '        with self.assertRaisesRegex(ValueError, "with no device types"):\n'
                    '            benchmarker.benchmark(fn, (), {})')
            text = text.replace('(fn, cpu_args, cpu_kwargs), _ = self.make_sum("cpu")',
                                '(fn, cpu_args, cpu_kwargs), _ = self.make_params("cpu")')
            text = text.replace('(_, gpu_args, gpu_kwargs), _ = self.make_sum(GPU_TYPE)',
                                '(_, gpu_args, gpu_kwargs), _ = self.make_params(GPU_TYPE)')
            return replace_once(text, '        benchmarker.benchmark(fn, many_devices_args, many_devices_kwargs)',
                '        with self.assertRaisesRegex(ValueError, "with multiple device types"):\n'
                '            benchmarker.benchmark(fn, many_devices_args, many_devices_kwargs)')
        updated[name] = edit_method(updated[name], 'test_benchmark_safely_infers_device_' + suffix,
                                   edit_infer, with_decorators=True)
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
    name = 'torch/_inductor/lowering.py'
    updated[name] = edit_method(updated[name], 'force_fallback', lambda text: text.replace(
        'op: torch._ops.OpOverload', 'op: torch._ops.OpOverload | torch._ops.HigherOrderOperator').replace(
        'assert isinstance(op, torch._ops.OpOverload), (',
        'assert isinstance(op, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)), (').replace(
        'Only OpOverload to make the clean up easier',
        'Only individual overloads or higher-order operators can be restored independently'))
    name = 'torch_npu/_inductor/kernel/bmm.py'
    updated[name] = replace_once(updated[name],
        '    def tuned_bmm(mat1, mat2, *, layout=None):\n',
        '    def tuned_bmm(mat1, mat2, *, layout=None):\n'
        '        sizevars = V.graph.sizevars\n'
        '        dtype = mat1.get_dtype()\n'
        '        # Match the upstream tiny-dot decomposition for NPU floating types.\n'
        '        # Hints on M/N select the optimization without specializing symbols;\n'
        '        # K must use a static bound so cached graphs remain valid.\n'
        '        if (\n'
        '            mat1.get_device().type == mat2.get_device().type == "npu"\n'
        '            and dtype == mat2.get_dtype()\n'
        '            and dtype in (torch.float16, torch.bfloat16, torch.float32)\n'
        '            and sizevars.optimization_hint(mat1.get_size()[1], fallback=2) == 1\n'
        '            and sizevars.optimization_hint(mat2.get_size()[2], fallback=2) == 1\n'
        '            and not sizevars.statically_known_gt(mat1.get_size()[2], 32)\n'
        '        ):\n'
        '            return L.sum_(L.mul(L.unsqueeze(mat1, -1), L.unsqueeze(mat2, 1)), axis=2)\n\n')
    return updated


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--test-dir', type=Path, required=True)
    parser.add_argument('--site-dir', type=Path, required=True)
    parser.add_argument('--torch-dir', type=Path, help='Torch source root; defaults to site-dir')
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
        'torch_npu/_inductor/kernel/bmm.py',
    ]}
    for name in sources:
        if name in preflight['source_hashes']:
            assert hashlib.sha256((args.site_dir / name).read_bytes()).hexdigest() == preflight['source_hashes'][name]
    name = 'torch/_inductor/lowering.py'
    sources[name] = ((args.torch_dir or args.site_dir) / name).read_text()
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
