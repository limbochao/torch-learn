"""Process-local helpers for temporary validation of NPU skip conditions.

Copy alongside the temporary upstream tests. This does not change installed
PyTorch/torch_npu modules or the source tests in their original directory.
"""

from functools import wraps
import gc
import json
import os
from pathlib import Path
import unittest


def record_observation(event):
    destination = os.environ.get("NPU_SKIP_OBSERVATIONS")
    if destination:
        with Path(destination).open("a") as stream:
            stream.write(json.dumps(event) + "\n")


def npu_triton_templates_available():
    import torch
    from torch._inductor.codegen.common import BackendFeature, has_backend_feature
    from torch.utils._triton import has_triton

    available = bool(
        torch.npu.is_available()
        and has_triton()
        and has_backend_feature(torch.device("npu"), BackendFeature.TRITON_TEMPLATES)
    )
    record_observation({"kind": "template_capability", "available": available})
    return available


def npu_large_tensor_test(size, *, inductor=True):
    import torch
    from torch.testing._internal.common_device_type import largeTensorTest

    size_bytes = int(size[:-2]) * 1024**3 if isinstance(size, str) else size

    def decorate(fn):
        original_guard = largeTensorTest(size, inductor=inductor)(fn)

        @wraps(fn)
        def guarded(self, *args, **kwargs):
            device = torch.device(self.device)
            if device.type != "npu":
                return original_guard(self, *args, **kwargs)
            if device.index is None:
                device = torch.device("npu", torch.npu.current_device())
            gc.collect()
            torch.npu.empty_cache()
            free, total = torch.npu.mem_get_info(device)
            fraction = torch.npu.get_per_process_memory_fraction(device)
            reserved = torch.npu.memory_reserved(device)
            available = min(free, max(0, int(total * fraction) - reserved))
            required = size_bytes * (2 if inductor and torch._inductor.config.cpp_wrapper else 1)
            record_observation({
                "kind": "memory_guard", "test": fn.__name__, "device": str(device),
                "free_bytes": free, "total_bytes": total, "process_fraction": fraction,
                "reserved_bytes": reserved, "available_bytes": available, "required_bytes": required,
            })
            if available < required:
                raise unittest.SkipTest(f"Insufficient NPU memory: {available} available, {required} required")
            return fn(self, *args, **kwargs)

        return guarded

    return decorate
