"""Numerical regressions for the codegen fixes; run in the NPU benchmark environment."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from torch.utils import _pytree


def embedding_norm(ids, table, position):
    value = table[ids] + position
    mean = value.mean(-1, keepdim=True)
    variance = ((value - mean) ** 2).mean(-1, keepdim=True)
    return (value - mean) * torch.rsqrt(variance + 1e-5)


def identities(value):
    return ((value + 1) ** 2).sum(-1), (value + 2).amin(-1), (-value - 2).amax(-1)


def product(value):
    return (value + 0.99).prod(-1)


def arg_extremes(value):
    return (-value - 2).argmax(-1), (value + 2).argmin(-1)


def integer_extremes(value):
    return (value + 3).amin(-1), (-value - 3).amax(-1)


def bias_gradient(value):
    return value.reshape(4, 12, 128, 64).permute(0, 2, 1, 3).reshape(512, 768).sum(0)


def separated_axes(value):
    return (value + 0.5).sum((0, 2)), (value - 1).sum((1, 3))


def negative_index(table, indices):
    return table[indices] + 1


parser = argparse.ArgumentParser()
parser.add_argument('--group', choices=['mask', 'axes', 'negative'], required=True)
parser.add_argument('--dynamic', action='store_true')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.manual_seed(0)
results = []


def check(name, function, inputs, atol=2e-4, rtol=2e-4):
    torch._dynamo.reset()
    expected = _pytree.tree_leaves(function(*inputs))
    compiled = torch.compile(function, backend='inductor', dynamic=args.dynamic, fullgraph=True)
    actual = _pytree.tree_leaves(compiled(*inputs))
    torch.npu.synchronize()
    differences = [(a.float() - b.float()).abs().max().item() for a, b in zip(actual, expected)]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    results.append(dict(name=name, dynamic=args.dynamic, status='PASS', max_abs=differences))
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results[-1]), flush=True)


if args.group == 'mask':
    for width in (197, 768):
        ids = torch.zeros(16, device='npu', dtype=torch.int64)
        table = torch.randn((4, width), device='npu') * 0.1 + 1
        position = torch.randn((16, width), device='npu') * 0.1 + 2
        check(f'embedding_norm_{width}', embedding_norm, (ids, table, position))
    for width in (31, 197, 256):
        value = torch.rand((8, width), device='npu') * 0.001
        check(f'identities_{width}', identities, (value,))
        check(f'product_{width}', product, (value,))
    value = torch.rand((8, 197), device='npu') * 0.1
    check('arg_extremes_197', arg_extremes, (value,), atol=0, rtol=0)
    for dtype in (torch.int32, torch.int64):
        value = torch.randint(1, 100, (8, 197), device='npu', dtype=dtype)
        check(f'integer_extremes_{dtype}', integer_extremes, (value,), atol=0, rtol=0)
elif args.group == 'axes':
    value = torch.randn((48, 128, 64), device='npu')
    check('bert_bias_gradient', bias_gradient, (value,), atol=2e-4, rtol=1e-3)
    value = torch.randn((4, 7, 8, 11), device='npu')
    check('separated_axes', separated_axes, (value,))
else:
    table = torch.randn((8, 16), device='npu')
    indices = torch.tensor([-8, -3, -1, 0, 4, 7], device='npu')
    check('negative_index', negative_index, (table, indices))
    print(json.dumps({key: os.environ.get(key) for key in (
        'INDUCTOR_ASCEND_DUMP_FX_GRAPH', 'INDUCTOR_ASCEND_CHECK_ACCURACY')}), flush=True)
