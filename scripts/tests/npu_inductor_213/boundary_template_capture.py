"""Capture one-candidate template launches omitted by the tuner's prerun sweep.

Use with the kernel skill's trace_compile.py and TRITON_TRACE_PRERUN_INPUTS=metadata.
This wraps the template benchmark callable, adding one synchronization on its
first invocation. It preserves source, layout, scalars, grid and compile config.
"""

import itertools
import json
import os
from pathlib import Path
import threading
import traceback

import torch

_ids = itertools.count(100000)
_lock = threading.Lock()


def encode(value):
    if isinstance(value, torch.Tensor):
        return {'kind': 'tensor', 'dtype': str(value.dtype), 'device': str(value.device),
                'shape': tuple(value.shape), 'stride': tuple(value.stride()),
                'storage_offset': value.storage_offset(), 'requires_grad': value.requires_grad}
    if isinstance(value, (tuple, list)):
        return {'kind': 'tuple' if isinstance(value, tuple) else 'list', 'items': [encode(x) for x in value]}
    if isinstance(value, dict):
        return {'kind': 'dict', 'items': [(encode(k), encode(v)) for k, v in value.items()]}
    if value is None or isinstance(value, (int, float, bool, str)):
        return {'kind': 'literal', 'value': value}
    raise TypeError(type(value))


def pytest_configure():
    from torch_npu._inductor.select_algorithm import _NPUTritonBenchmarkRequestMixin as Request

    original = Request.make_run_fn
    root = Path(os.environ['TRITON_TRACE_CAPTURE_DIR'])
    journal = root / 'journal.jsonl'

    def persist(record):
        with _lock, journal.open('a') as stream:
            stream.write(json.dumps(record, default=repr) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def make_run_fn(self, *inputs, **kwargs):
        function = original(self, *inputs, **kwargs)
        if 'triton_bmm_batched' not in self.kernel_name:
            return function
        captured = False

        def run():
            nonlocal captured
            if captured:
                return function()
            captured = True
            records = [json.loads(line) for line in journal.read_text().splitlines()]
            record = next(r for r in reversed(records)
                          if r.get('event') == 'PASS' and r.get('fn_name') == self.kernel_name)
            record = dict(record)
            record['id'] = next(_ids)
            record['candidate_id'] = 'single-template'
            record['variant_id'] = 0
            record['runtime_blocks'] = []
            record['input_group_id'] = record['id']
            record['input_capture_mode'] = 'metadata'
            record['capture_hook'] = 'template benchmark callable; synchronize after first launch'
            arguments = tuple(function.args)
            launch_kwargs = {k: v for k, v in function.keywords.items()
                             if k not in ('stream', 'benchmark_run', 'warmup')}
            output = root / 'prerun_inputs' / f'template-{record["id"]}.pt'
            output.parent.mkdir(exist_ok=True)
            torch.save({'format_version': 3, 'capture_mode': 'metadata',
                        'args': encode(arguments), 'kwargs': encode(launch_kwargs)}, output)
            record['input_artifact'] = str(output)
            persist({**record, 'event': 'PRERUN_READY'})
            try:
                result = function()
                torch.npu.synchronize()
            except BaseException as error:
                persist({**record, 'event': 'PRERUN_PYTHON_EXCEPTION',
                         'error_type': type(error).__name__, 'error': repr(error),
                         'traceback': traceback.format_exc()})
                raise
            persist({**record, 'event': 'PRERUN_PASS'})
            return result

        return run

    Request.make_run_fn = make_run_fn
