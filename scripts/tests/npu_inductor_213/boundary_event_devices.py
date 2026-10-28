"""Log the event devices around a candidate without changing event behavior."""

import json
import os
from pathlib import Path


def pytest_configure():
    import torch
    from torch_npu._inductor.runtime import triton_heuristics as heuristics

    def measure(kernel_call_fn):
        before = torch.npu.current_device()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        kernel_call_fn()
        after = torch.npu.current_device()
        end.record()
        torch.npu.synchronize()
        record = {'before': before, 'after': after,
                  'start_device': str(start.device), 'end_device': str(end.device)}
        path = Path(os.environ['BOUNDARY_ARTIFACT_DIR'])
        path.mkdir(parents=True, exist_ok=True)
        with (path / 'event-devices.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        return start.elapsed_time(end)

    heuristics._measure_prerun_ms = measure
