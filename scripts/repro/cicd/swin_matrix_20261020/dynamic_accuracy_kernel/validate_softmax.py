"""Compare the captured softmax DSL with a diagnostic reduction-mask edit."""

import importlib.util
import json
from pathlib import Path

import torch
import torch_npu


def main():
    root = Path(__file__).resolve().parent
    results = []
    for variant in ['direct', 'masked']:
        path = root / f'softmax13_{variant}.py'
        spec = importlib.util.spec_from_file_location(f'softmax_{variant}', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for kind in ['zeros', 'random']:
            torch.manual_seed(0)
            shape = (512, 4, 49, 49)
            logits = torch.zeros(shape, device='npu')
            bias = torch.zeros((169, 4), device='npu')
            if kind == 'random':
                logits.normal_(0, .05)
                bias.normal_(0, .05)
            indices = torch.zeros((49, 49), dtype=torch.int64, device='npu')
            expected = (logits + bias[indices].permute(2, 0, 1)).softmax(-1)
            module.triton_per_fused__softmax_add_clone_index_matm_13[(32,)](
                logits, indices, bias, 512, 4, 49, 49, 16,
                Z0BLOCK_SUB=1, X2BLOCK_SUB=1, num_warps=1, num_stages=2,
                compile_mode='simd_simt_template', multibuffer=False,
            )
            torch.npu.synchronize()
            row = {
                'variant': variant, 'input': kind,
                'allclose': torch.allclose(logits, expected, atol=2e-6, rtol=2e-5),
                'max_abs': (logits - expected).abs().max().item(),
                'row_sum_min': logits.sum(-1).min().item(),
                'row_sum_max': logits.sum(-1).max().item(),
                'first_value': logits.flatten()[0].item(),
                'reference_first_value': expected.flatten()[0].item(),
            }
            results.append(row)
    print(json.dumps(results, indent=2), flush=True)
    assert not results[0]['allclose'] and not results[1]['allclose']
    assert results[2]['allclose'] and results[3]['allclose']


if __name__ == '__main__':
    main()
