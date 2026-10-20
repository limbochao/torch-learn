"""Validate a diagnostic tile/index variant against window reverse + LayerNorm gradients."""

import importlib.util
import json
import tempfile
from pathlib import Path

import torch
import torch_npu


def main():
    source = Path(__file__).resolve().parent / 'training_accuracy_kernel/compile_repro.py'
    text = source.read_text()
    original = '3584 * r2 + 25088 * r4 + 50176 * (r2 // 7)'
    replacement = '3584 * (r2 % 7) + 25088 * r4 + 50176 * (r2 // 7)'
    if text.count(original) != 1:
        raise RuntimeError('The captured address formula has changed; inspect the source before replaying')
    with tempfile.TemporaryDirectory(prefix='swin-training-variant-') as directory:
        variant = Path(directory) / 'candidate.py'
        variant.write_text(text.replace(original, replacement))
        spec = importlib.util.spec_from_file_location('swin_training_variant', variant)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.manual_seed(0)
        device = 'npu:0'
        gradient = torch.randn((8, 2, 2, 7, 7, 512), device=device)
        activation = torch.randn((8, 14, 14, 512), device=device)
        mean = torch.randn((8, 14, 14, 1), device=device)
        rstd = torch.rand((8, 14, 14, 1), device=device) + 0.5
        weight_gradient = torch.empty((512,), device=device)
        bias_gradient = torch.empty_like(weight_gradient)
        kernel = getattr(module, module._STANDALONE_FN_NAME)
        kernel[(32,)](
            gradient, activation, mean, rstd, weight_gradient, bias_gradient,
            512, 8, 7, 2, 14, 16, X0BLOCK_SUB=4, **module._STANDALONE_OPTIONS,
        )
        torch.npu.synchronize()
        reversed_gradient = gradient.permute(0, 1, 3, 2, 4, 5).reshape(8, 14, 14, 512)
        expected_weight = (reversed_gradient * (activation - mean) * rstd).sum((0, 1, 2))
        expected_bias = reversed_gradient.sum((0, 1, 2))
        errors = []
        for actual, expected in [(weight_gradient, expected_weight), (bias_gradient, expected_bias)]:
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-5)
            errors.append((actual - expected).abs().max().item())
        print(json.dumps({
            'status': 'PASS', 'seed': 0, 'X0BLOCK_SUB': 4, 'runtime_X0BLOCK': 16, 'grid': [32],
            'address_edit': '3584*r2 -> 3584*(r2%7)',
            'max_abs_error_weight_bias': errors, 'atol': 2e-4, 'rtol': 2e-5,
            'note': 'Two diagnostic edits; the unmodified kernel was not launched. This is not a full-model fix.',
        }), flush=True)


if __name__ == '__main__':
    main()
