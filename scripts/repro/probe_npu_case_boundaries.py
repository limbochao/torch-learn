"""Separate public eager/AOT behavior from Inductor for three boundary failures."""

import argparse
import json
import traceback

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu  # noqa: F401
from torch_npu.utils import _dynamo

_dynamo.use_jit_script = True
_dynamo.patch_has_triton()


def attempt(name, function):
    try:
        result = function()
        print(json.dumps({'probe': name, 'status': 'passed', 'result': result}), flush=True)
    except Exception as error:
        print(json.dumps({'probe': name, 'status': 'failed', 'error': str(error)}), flush=True)
        traceback.print_exc(limit=12)


def rightshift():
    a = torch.arange(128, dtype=torch.uint8).reshape(1, 4, 1, 32)
    b = torch.tensor([0, 1], dtype=torch.uint8).reshape(1, 1, 2, 1)
    expected = a >> b
    actual = a.npu() >> b.npu()
    torch.testing.assert_close(actual.cpu(), expected)
    return {'expected_shape': list(expected.shape), 'actual_shape': list(actual.shape)}


def floorzero():
    result = []
    for dtype in [torch.int32, torch.int64]:
        for dividend in [0, 1, -1, 5, -5]:
            a = torch.full((8,), dividend, dtype=dtype, device='npu')
            b = torch.zeros_like(a)
            eager = torch.floor_divide(a, b)
            compiled = torch.compile(torch.floor_divide, fullgraph=True)(a, b)
            result.append({'dtype': str(dtype), 'dividend': dividend,
                           'eager': eager.cpu().tolist(), 'compiled': compiled.cpu().tolist()})
    return result


def gather(backend):
    torch.manual_seed(1234)
    a = torch.randn(4, 5, 10, 6, device='npu', requires_grad=True)
    b = torch.randint(5, (4, 5, 10, 1), device='npu', dtype=torch.int64)

    def fn(x, index):
        return torch.gather(x, 1, index, sparse_grad=True)

    function = fn if backend == 'eager' else torch.compile(fn, backend=backend, fullgraph=True)
    output = function(a, b)
    output.sum().backward()
    torch.npu.synchronize()
    return {'backend': backend, 'shape': list(output.shape), 'gradient_layout': str(a.grad.layout)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('probe', choices=['rightshift', 'floorzero', 'gather'])
    parser.add_argument('--backend', default='inductor',
                        choices=['eager', 'aot_eager', 'aot_eager_decomp_partition', 'inductor'])
    args = parser.parse_args()
    torch.npu.set_device(0)
    if args.probe == 'gather':
        attempt('gather_' + args.backend, lambda: gather(args.backend))
    else:
        attempt(args.probe, globals()[args.probe])
