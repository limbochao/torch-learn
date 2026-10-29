"""Log the event devices around a candidate without changing event behavior."""

import json
import os
from pathlib import Path


def pytest_configure():
    import torch
    import ctypes
    runtime = ctypes.CDLL("libascendcl.so")

    def actual_device():
        value = ctypes.c_int(-1)
        return {"returncode": runtime.aclrtGetDevice(ctypes.byref(value)), "device": value.value}

    from torch_npu._inductor.runtime import triton_heuristics as heuristics

    def measure(kernel_call_fn):
        actual_before = actual_device()
        before = torch.npu.current_device()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        kernel_call_fn()
        after = torch.npu.current_device()
        end.record()
        torch.npu.synchronize()
        runtime.aclrtEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
        raw_ms = ctypes.c_float()
        raw_status = runtime.aclrtEventElapsedTime(ctypes.byref(raw_ms), ctypes.c_void_p(start.npu_event),
                                                  ctypes.c_void_p(end.npu_event))
        record = {'before': before, 'after': after,
                  'start_device': str(start.device), 'end_device': str(end.device),
                  'actual_before': actual_before, 'actual_after_sync': actual_device(),
                  'raw_elapsed_status': raw_status, 'raw_elapsed_ms': raw_ms.value,
                  'start_complete': start.query(), 'end_complete': end.query()}
        path = Path(os.environ['BOUNDARY_ARTIFACT_DIR'])
        path.mkdir(parents=True, exist_ok=True)
        with (path / 'event-devices.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        return start.elapsed_time(end)

    heuristics._measure_prerun_ms = measure
