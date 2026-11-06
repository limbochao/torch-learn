"""Temporary NPU-only selection and result recording for the approved 2.10 cases."""
import ast
import json
import os
import shutil
import unittest
from pathlib import Path

import torch
import torch_npu

torch.npu.set_device(0)
from torch_npu.contrib import transfer_to_npu
from torch_npu.utils import _dynamo
_dynamo.patch_has_triton()
torch.cuda.get_device_capability = lambda *args, **kwargs: (10, 0)
import torch_npu.testing
import torch_npu._inductor

import torch.testing._internal.triton_utils as triton_utils

# Test gating only: CUDA-spelled upstream tests are routed to the NPU by transfer_to_npu.
triton_utils.requires_gpu_and_triton = unittest.skipUnless(_dynamo.has_triton(), 'requires Ascend Triton')
triton_utils.requires_cuda_and_triton = triton_utils.requires_gpu_and_triton
if not hasattr(triton_utils, 'requires_xpu_and_triton'):
    triton_utils.requires_xpu_and_triton = unittest.skip('XPU-only test excluded from NPU validation')

import torch.testing._internal.common_dtype as common_dtype
if not hasattr(common_dtype, 'highest_precision_float'):
    common_dtype.highest_precision_float = lambda device: (
        torch.float32 if torch.device(device).type == 'mps' else torch.float64
    )

import torch.testing._internal.common_utils as common_utils
if not hasattr(common_utils, 'IS_CPU_EXT_SVE_SUPPORTED'):
    common_utils.IS_CPU_EXT_SVE_SUPPORTED = False  # unused CPU-only test gating

# Capture temporary generated code before unittest.tearDown removes a fresh cache.
_original_call_test_method = unittest.TestCase._callTestMethod


def _call_test_method_with_failure_capture(self, method):
    try:
        return _original_call_test_method(self, method)
    except unittest.SkipTest:
        raise
    except Exception:
        cache = Path(os.environ.get('TORCHINDUCTOR_CACHE_DIR', '/nonexistent'))
        evidence = Path(os.environ['COMBO_SELECTED_MANIFEST']).parent / 'evidence'
        target = evidence / 'generated_on_failure' / (type(self).__name__ + '.' + self._testMethodName)
        try:
            for source in cache.rglob('*.py'):
                data = source.read_bytes()
                if b'triton_meta' not in data and b'async_compile' not in data:
                    continue
                destination = target / source.relative_to(cache)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        except Exception as capture_error:
            print('Failure artifact capture error:', repr(capture_error), flush=True)
        raise


unittest.TestCase._callTestMethod = _call_test_method_with_failure_capture

SELECTED = json.loads(Path(os.environ['COMBO_SELECTED_MANIFEST']).read_text())
CATALOG = {}
SOURCE_METHODS = {}


def source_method_names(path, cls):
    key = str(path), cls
    if key not in SOURCE_METHODS:
        tree = ast.parse(Path(str(path)).read_text())
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls]
        SOURCE_METHODS[key] = [n.name for c in classes for n in c.body
                               if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')]
    return SOURCE_METHODS[key]


def pytest_collection_modifyitems(config, items):
    kept, dropped = [], []
    for item in items:
        file = Path(str(item.fspath)).name
        cls = item.cls.__name__ if item.cls else ''
        matches = [r for r in SELECTED if r['file'] == file and
                   (cls == r['class'] or cls in (r['class'] + 'NPU', r['class'] + 'PRIVATEUSE1') or
                    (r['class'] == 'CommonTemplate' and cls == 'GPUTests')) and
                   (item.name == r['name'] or item.name.startswith(r['name'] + '_'))]
        if not matches:
            dropped.append(item)
            continue
        chosen = max(matches, key=lambda r: len(r['name']))
        all_names = source_method_names(item.fspath, chosen['class'])
        actual = max((name for name in all_names if item.name == name or item.name.startswith(name + '_')),
                     key=len, default=None)
        if actual != chosen['name']:
            dropped.append(item)
            continue
        item.user_properties.append(('source_case', chosen['name']))
        item.user_properties.append(('source_file', file))
        CATALOG[item.nodeid] = {**chosen, 'nodeid': item.nodeid}
        kept.append(item)
    items[:] = kept
    config.hook.pytest_deselected(items=dropped)
    target = os.environ.get('COMBO_COLLECTION_JSON')
    if target:
        Path(target).write_text(json.dumps(list(CATALOG.values()), ensure_ascii=False, indent=2))


def pytest_runtest_logreport(report):
    target = os.environ.get('COMBO_RESULTS_JSONL')
    if target:
        record = {'nodeid': report.nodeid, 'phase': report.when, 'outcome': report.outcome,
                  'duration': report.duration, 'properties': report.user_properties}
        if report.failed or report.skipped:
            record['longrepr'] = str(report.longrepr)
        with open(target, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
