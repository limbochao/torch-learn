"""Separate eager NPU event timing from Inductor in the two-device case."""

import ctypes
import json
from pathlib import Path
import sys

import torch
import torch_npu  # noqa: F401


def main():
    destination = Path(sys.argv[1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    runtime = ctypes.CDLL('libascendcl.so')
    runtime.aclrtEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
    assert torch.npu.device_count() == 2
    records = []
    x = torch.ones(1024, device='npu:0')
    for stage in ('before_peer', 'after_peer', 'after_copy'):
        if stage == 'after_peer':
            peer = torch.ones(1024, device='npu:1')
        if stage == 'after_copy':
            peer.copy_(x)
        torch.npu.set_device(0)
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        x = x / 2
        end.record()
        torch.npu.synchronize()
        device = ctypes.c_int(-1)
        runtime.aclrtGetDevice(ctypes.byref(device))
        elapsed = ctypes.c_float()
        status = runtime.aclrtEventElapsedTime(ctypes.byref(elapsed), ctypes.c_void_p(start.npu_event),
                                              ctypes.c_void_p(end.npu_event))
        record = {'stage': stage, 'torch_device': torch.npu.current_device(), 'cann_device': device.value,
                  'start_device': str(start.device), 'end_device': str(end.device),
                  'raw_status': status, 'raw_ms': elapsed.value}
        try:
            record['torch_ms'] = start.elapsed_time(end)
        except RuntimeError as error:
            record['torch_error'] = str(error)
        records.append(record)
        destination.write_text(json.dumps(records, indent=2) + '\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
