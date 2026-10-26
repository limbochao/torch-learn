"""Replay the original or simplified DINOv2 reduction with explicit tiling.

Examples on Ascend A5 with PyTorch, torch_npu and Triton-Ascend installed:
    python run.py
    python run.py --case small_x --check
    python run.py --kernel minimal --case all

Inputs are synthetic and retain the original model shapes and strides.
--check exits with an error if any selected configuration produces wrong output.
"""
import argparse
import json

import torch
import torch_npu
import triton

from minimal_kernel import masked_sum
from original_kernel import triton_red_fused_native_layer_norm_backward_select_backward_0


CASES = {
    'original': (32, 64, 'simt_only'),
    'small_x': (8, 64, 'simt_only'),
    'small_r': (32, 16, 'simt_only'),
    'template': (32, 64, 'simd_simt_template'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', choices=('original', 'minimal'), default='original')
    parser.add_argument('--case', choices=(*CASES, 'all'), default='original')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    torch.npu.set_device(0)
    grad = torch.full((16, 768), 1 / (16 * 768), device='npu', dtype=torch.float32)
    beta = torch.empty((768,), device='npu', dtype=torch.float32)
    if args.kernel == 'original':
        value = torch.ones((16, 1370, 768), device='npu', dtype=torch.float32)
        mean = torch.zeros((16, 1370, 1), device='npu', dtype=torch.float32)
        rstd = torch.ones((16, 1370, 1), device='npu', dtype=torch.float32)
        gamma = torch.empty_like(beta)
    # CPU reference: 16 equal contributions. No NPU reduction in the reference.
    expected = torch.full((768,), 1 / 768, dtype=torch.float32)
    failed = []
    cases = CASES if args.case == 'all' else {args.case: CASES[args.case]}
    for name, (xb, rb, mode) in cases.items():
        beta.fill_(float('nan'))
        launch = dict(num_warps=8, num_stages=2, compile_mode=mode, multibuffer=False)
        grid = (triton.cdiv(768, xb),)
        if args.kernel == 'original':
            gamma.fill_(float('nan'))
            triton_red_fused_native_layer_norm_backward_select_backward_0[grid](
                grad, value, mean, rstd, gamma, beta, 768, 21920,
                XBLOCK=xb, R0_BLOCK=rb, **launch,
            )
            outputs = {'gamma': gamma, 'beta': beta}
        else:
            masked_sum[grid](grad, beta, XBLOCK=xb, RBLOCK=rb, **launch)
            outputs = {'beta': beta}
        torch.npu.synchronize()
        report = {
            'kernel': args.kernel, 'case': name, 'input_kind': 'synthetic',
            'XBLOCK': xb, 'R0_BLOCK': rb, 'grid': list(grid), **launch,
            'expected': 1 / 768, 'outputs': {},
        }
        for label, output in outputs.items():
            actual = output.cpu()
            passed = torch.allclose(actual, expected, atol=1e-8, rtol=1e-5)
            report['outputs'][label] = {
                'unique': actual.unique().tolist(),
                'max_abs': (actual - expected).abs().max().item(),
                'ratio_to_expected': actual[0].item() * 768,
                'passed': passed,
            }
            if not passed:
                failed.append(f'{name}.{label}')
        print(json.dumps(report), flush=True)
    if args.check and failed:
        raise AssertionError('incorrect outputs: ' + ', '.join(failed))


if __name__ == '__main__':
    main()
