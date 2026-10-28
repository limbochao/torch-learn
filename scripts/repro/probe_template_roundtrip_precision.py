"""Direct fixed-template precision controls for boundary cases 18 and 21.

Case 18 checks whether one generated BMM's BF16 and FP32 outputs agree after
BF16 rounding. Case 21 compares the captured fused GEMM with the captured
unfused GEMM followed by the same relu/sub epilogue evaluated on CPU.
The saved DSL is used verbatim; Inductor and its autotuner are not invoked.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def load_template(path, output, label):
    text = path.read_text()
    tree = ast.parse(text)
    sources = [text]
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'triton' and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)):
            sources.append(node.args[1].value)
    for source in sources:
        for node in ast.parse(source).body:
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                        and decorator.func.attr == 'template'):
                    keywords = {item.arg: item.value for item in decorator.keywords}
                    meta = {ast.literal_eval(key): value for key, value in
                            zip(keywords['triton_meta'].keys, keywords['triton_meta'].values)}
                    signature = ast.literal_eval(meta['signature'])
                    options = {key: ast.literal_eval(keywords[key]) for key in ('num_warps', 'num_stages')}
                    options.update(ast.literal_eval(meta['npu_compile_options'])
                                   if 'npu_compile_options' in meta else {})
                    options.update(debug=False, compile_mode='simd_simt_template')
                    attrs = ast.literal_eval(meta['configs'])[0]
                    definition = ast.get_source_segment(source, node)
                    extracted = (
                        'import triton\nimport triton.language as tl\n'
                        'from torch_npu._inductor.runtime import triton_helpers\n'
                        'from torch_npu._inductor.runtime.triton_helpers import extension\n\n'
                        '@triton.jit\n' + definition + '\n'
                    )
                    destination = output / f'{label}-direct.py'
                    destination.write_text(extracted)
                    spec = importlib.util.spec_from_file_location(f'roundtrip_{label}', destination)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[spec.name] = module
                    spec.loader.exec_module(module)
                    return getattr(module, node.name), signature, attrs, options, {
                        'source': path.name, 'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
                        'dsl_sha256': hashlib.sha256(definition.encode()).hexdigest(),
                    }
    raise ValueError(f'No template definition in {path}')


def compile_template(path, output, label, fp_fusion_off):
    import triton
    from triton.compiler.compiler import ASTSource
    from triton.runtime import driver

    function, signature, attrs, options, record = load_template(path, output, label)
    if fp_fusion_off:
        options['enable_fp_fusion'] = False
    binary = triton.compile(ASTSource(function, signature, {}, attrs),
                            target=driver.active.get_current_target(), options=options)
    for kind, value in binary.asm.items():
        if isinstance(value, str):
            (output / f'{label}.{kind}').write_text(value)
    record.update(signature=signature, options=options)
    print(json.dumps({'compiled': label, **record}), flush=True)
    return binary, record


def compare(actual, expected):
    import torch

    actual, expected = actual.cpu(), expected.cpu()
    return {'equal': torch.equal(actual, expected),
            'mismatched': int((actual != expected).sum()), 'elements': actual.numel(),
            'max_abs': float((actual.float() - expected.float()).abs().max())}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--case', type=int, choices=(18, 21), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fp-fusion-off', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2] / 'artifacts/npu-inductor-boundary-20261028/precision-roundtrip'

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    torch.manual_seed(1234)
    stream = torch.npu.current_stream().npu_stream
    if args.case == 18:
        binary, record = compile_template(root / 'case18-bmm-candidate.py', args.output, 'bmm',
                                          args.fp_fusion_off)
        left = torch.randn((32, 512, 512), device='npu', dtype=torch.bfloat16)
        right = torch.randn((32, 512, 1490), device='npu', dtype=torch.bfloat16)
        low = torch.empty((32, 512, 1490), device='npu', dtype=torch.bfloat16)
        high = torch.empty_like(low, dtype=torch.float32)
        binary[(24, 32, 1)](left, right, low, high, stream=stream)
        torch.npu.synchronize()
        result = {'case': 18, 'compiled': [record], 'roundtrip': compare(high, low.cpu().float())}
    else:
        fused, fused_record = compile_template(root / 'case21-fused.py', args.output, 'fused',
                                               args.fp_fusion_off)
        unfused, unfused_record = compile_template(root / 'case21-unfused.py', args.output, 'unfused',
                                                   args.fp_fusion_off)
        bias = torch.randn((128,), device='npu', dtype=torch.bfloat16)
        x = torch.randn((128, 128), device='npu', dtype=torch.bfloat16)
        fused_out, gemm_out = torch.empty_like(x), torch.empty_like(x)
        fused[(1, 1, 1)](bias, x, fused_out, stream=stream)
        unfused[(1, 1, 1)](bias, x, gemm_out, stream=stream)
        torch.npu.synchronize()
        expected = (gemm_out.cpu().float().relu() - 1).to(torch.bfloat16)
        result = {'case': 21, 'compiled': [fused_record, unfused_record],
                  'fused_vs_materialized': compare(fused_out, expected)}
    result['environment'] = {'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
                             'device': torch.npu.get_device_name(0), 'seed': 1234,
                             'fp_fusion_off': args.fp_fusion_off}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
