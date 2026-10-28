"""Opt-in pytest progress for long test methods; never changes inputs or assertions."""

import faulthandler
import time


def pytest_runtest_call(item):
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
        print('COMMON_START', sequence, getattr(args[0], '__name__', '?'), shapes, flush=True)
        result = original(*args, **kwargs)
        print('COMMON_END', sequence, round(time.monotonic() - started, 3), flush=True)
        return result

    instance.common = logged


def pytest_runtest_teardown():
    faulthandler.cancel_dump_traceback_later()
