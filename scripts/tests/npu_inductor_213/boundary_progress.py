"""Opt-in pytest progress for long test methods; never changes inputs or assertions."""

import faulthandler
import os
from pathlib import Path
import time


_stack_file = None


def record(*values):
    line = ' '.join(map(str, values))
    print(line, flush=True)
    directory = os.environ.get('BOUNDARY_ARTIFACT_DIR')
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with (path / 'progress.log').open('a') as stream:
            stream.write(line + '\n')


def pytest_runtest_call(item):
    global _stack_file
    directory = os.environ.get('BOUNDARY_ARTIFACT_DIR')
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        _stack_file = (path / 'stacks.log').open('a')
        faulthandler.dump_traceback_later(120, repeat=True, file=_stack_file)
    else:
        faulthandler.dump_traceback_later(120, repeat=True)
    instance = getattr(item, 'instance', None)
    if instance is None or not hasattr(instance, 'common'):
        return
    original = instance.common
    sequence = 0

    def logged(*args, **kwargs):
        nonlocal sequence
        sequence += 1
        tensors = args[1] if len(args) > 1 else ()
        shapes = [(tuple(x.shape), tuple(x.stride()), str(x.dtype))
                  for x in tensors if hasattr(x, 'shape')]
        started = time.monotonic()
        record('COMMON_START', sequence, getattr(args[0], '__name__', '?'), shapes)
        result = original(*args, **kwargs)
        record('COMMON_END', sequence, round(time.monotonic() - started, 3))
        return result

    instance.common = logged


def pytest_runtest_teardown():
    global _stack_file
    faulthandler.cancel_dump_traceback_later()
    if _stack_file:
        _stack_file.close()
        _stack_file = None
