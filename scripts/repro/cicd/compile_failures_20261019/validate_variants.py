import importlib.util
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu

ROOT = Path(__file__).resolve().parent
torch.manual_seed(0)
torch.npu.set_device(0)
results = []

def load(model, filename):
    path = ROOT / model / filename
    spec = importlib.util.spec_from_file_location(model + '_validation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def densenet():
    m = load('phlippe_densenet', 'compile_repro.py')
    x = torch.randn(128, 64, 32, 32, device='npu')
    mean = torch.randn(64, device='npu')
    var = torch.rand(64, device='npu') + 0.5
    weight = torch.randn(64, device='npu')
    bias = torch.randn(64, device='npu')
    pooled = torch.empty((128, 64, 16, 16), device='npu')
    normalized = torch.empty_like(pooled)
    backing = torch.full((128, 80, 16, 16), float('nan'), device='npu')
    joined = backing[:, 16:, :, :]
    assert joined.stride() == (20480, 256, 16, 1)
    fn = getattr(m, m._STANDALONE_FN_NAME)
    fn[(1,)](x, mean, var, weight, bias, pooled, normalized, joined,
              131072, 16, 128, 1024, Y0BLOCK_SUB=128, **m._STANDALONE_OPTIONS)
    torch.npu.synchronize()
    expected_pool = F.avg_pool2d(x, 2, 2)
    expected_norm = torch.relu((expected_pool - mean[None, :, None, None])
                               / torch.sqrt(var[None, :, None, None] + 1e-5)
                               * weight[None, :, None, None] + bias[None, :, None, None])
    for actual, expected in [(pooled, expected_pool), (normalized, expected_norm), (joined, expected_pool)]:
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert torch.isnan(backing[:, :16]).all()
    print('phlippe_densenet tile128 check=passed', flush=True)
    results.append({'model': 'phlippe_densenet', 'variant': 'tile128', 'check': 'passed',
                    'input_shape': list(x.shape), 'input_stride': list(x.stride()),
                    'output_alias_stride': list(joined.stride())})


def albert():
    m = load('AlbertForMaskedLM', 'defer_loads.py')
    # The original kernel loads per-row mean/rstd, and column-major logical reductions of NCHW buffers.
    scalars = {2, 3}
    for start in range(4, 81, 7):
        scalars.update((start + 5, start + 6))
    inputs = [torch.randn((2048,) if i in scalars else (2048, 4096), device='npu')
              for i in range(81)]
    out0 = torch.empty(4096, device='npu')
    out1 = torch.empty_like(out0)
    fn = getattr(m, m._STANDALONE_FN_NAME)
    fn[(64,)](out0, out1, *inputs, 4096, 2048, 64,
               X0BLOCK_SUB=1, **m._STANDALONE_OPTIONS)
    torch.npu.synchronize()
    weighted = [(inputs[0] * (inputs[1] - inputs[2][:, None]) * inputs[3][:, None]).sum(0)]
    summed = [inputs[0].sum(0)]
    for start in range(4, 81, 7):
        dy = inputs[start] + inputs[start + 1] + inputs[start + 2] + inputs[start + 3]
        weighted.append((dy * ((inputs[start + 4] - inputs[start + 5][:, None])
                               * inputs[start + 6][:, None])).sum(0))
        summed.append(dy.sum(0))
    order = [0, 1, 2, 3, 4, 5, 7, 9, 11, 6, 8, 10]
    expected0, expected1 = weighted[0], summed[0]
    for index in order[1:]:
        expected0 = expected0 + weighted[index]
        expected1 = expected1 + summed[index]
    max_abs = [float((out0 - expected0).abs().max()), float((out1 - expected1).abs().max())]
    torch.testing.assert_close(out0, expected0, atol=2e-3, rtol=1e-3)
    torch.testing.assert_close(out1, expected1, atol=2e-3, rtol=1e-3)
    print('AlbertForMaskedLM defer_loads check=passed', max_abs, flush=True)
    results.append({'model': 'AlbertForMaskedLM', 'variant': 'defer_loads', 'check': 'passed',
                    'matrix_shape': [2048, 4096], 'matrix_stride': [4096, 1],
                    'max_abs': max_abs, 'atol': 2e-3, 'rtol': 1e-3})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate diagnostic DSL variants against eager.')
    parser.add_argument('--model', choices=['all', 'albert', 'densenet'], default='all')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.model in ('all', 'densenet'):
        densenet()
    if args.model in ('all', 'albert'):
        albert()
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + '\n')
