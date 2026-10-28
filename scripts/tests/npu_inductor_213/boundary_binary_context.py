"""Candidate: preserve the caller's CANN context across Triton binary loading."""

import ctypes
import json
import os
from pathlib import Path


def pytest_configure():
    from triton.runtime import driver

    runtime = ctypes.CDLL('libascendcl.so')
    original = driver.active.utils.load_binary

    def load_binary(*args, **kwargs):
        context = ctypes.c_void_p()
        status = runtime.aclrtGetCurrentContext(ctypes.byref(context))
        try:
            return original(*args, **kwargs)
        finally:
            if status == 0 and context.value:
                restore = runtime.aclrtSetCurrentContext(context)
                if restore:
                    raise RuntimeError(f'aclrtSetCurrentContext failed: {restore}')
                directory = os.environ.get('BOUNDARY_ARTIFACT_DIR')
                if directory:
                    path = Path(directory)
                    path.mkdir(parents=True, exist_ok=True)
                    with (path / 'binary-context-restored.jsonl').open('a') as stream:
                        stream.write(json.dumps({'restored': True}) + '\n')

    driver.active.utils.load_binary = load_binary
