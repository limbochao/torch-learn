"""Run the exact four-element eager expression from boundary case 34."""

import json
import os
from pathlib import Path
import sys

import torch
import torch_npu  # noqa: F401


def main():
    destination = Path(sys.argv[1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert torch.npu.device_count() == 2
    torch.manual_seed(0)
    x, y = (torch.randn(4).to('npu:0') for _ in range(2))
    result = {'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
              'task_queue_enable': os.environ.get('TASK_QUEUE_ENABLE', 'default'),
              'shape': list(x.shape), 'dtype': str(x.dtype), 'stages': []}

    def fn(x, y):
        r = torch.ops.aten.div(x, y)
        r = r.to('npu:1')
        return 2 * r

    try:
        output = fn(x, y)
        result['stages'].append('eager_returned')
        torch.npu.synchronize(0)
        torch.npu.synchronize(1)
        result['stages'].append('synchronized')
        torch.testing.assert_close(output.cpu(), 2 * (x.cpu() / y.cpu()))
        result['stages'].append('checked')
    except RuntimeError as error:
        result['error'] = str(error)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
