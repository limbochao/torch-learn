"""Check that the NPU tiny-dot repair reuses a graph across the declared K range."""
import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
from torch._dynamo.testing import CompileCounterWithBackend


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert Path('/.dockerenv').exists()
    torch.npu.set_device(args.device)
    torch.manual_seed(20261009)
    records = []
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        torch._dynamo.reset()
        counter = CompileCounterWithBackend('inductor')

        def fn(a, b):
            return torch.bmm(a, b)

        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        for k in (3, 32, 33, 64):
            a = torch.randn(4, 1, k, device='npu', dtype=dtype)
            b = torch.randn(4, k, 1, device='npu', dtype=dtype)
            torch._dynamo.mark_dynamic(a, 2, min=1, max=64)
            torch._dynamo.mark_dynamic(b, 1, min=1, max=64)
            expected = fn(a, b)
            actual = compiled(a, b)
            torch.testing.assert_close(actual, expected)
            assert counter.frame_count == 1, (dtype, k, counter.frame_count)
            records.append({'dtype': str(dtype), 'k': k, 'frames': counter.frame_count, 'check': 'passed'})
            args.output.write_text(json.dumps(records, indent=2) + '\n')
    print('BMM_DYNAMIC_REUSE_PASSED', len(records), flush=True)


if __name__ == '__main__':
    main()
