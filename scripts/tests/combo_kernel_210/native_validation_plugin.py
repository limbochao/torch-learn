"""Select native 2.10 source methods and record results without changing test assertions."""

import ast
import json
import os
import shutil
import unittest
from pathlib import Path

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch_npu.utils import _dynamo

_dynamo.use_jit_script = True
# transfer_to_npu disables Triton discovery; restore the NPU probe before test gating is imported.
_dynamo.patch_has_triton()
torch.cuda.get_device_capability = lambda *args, **kwargs: (10, 0)
import torch_npu.testing
import torch_npu._inductor
from torch.testing._internal import inductor_utils
from torch.testing._internal.common_device_type import get_desired_device_type_test_bases

torch.npu.set_device(0)
# Upstream is_gpu() excludes the NPU device-test base even when HAS_GPU_AND_TRITON is true.
if inductor_utils.GPU_TYPE == "npu":
    npu_requested = any(getattr(base, "device_type", "") == "npu"
                        for base in get_desired_device_type_test_bases(allow_xpu=True))
    assert npu_requested and torch.npu.is_available() and inductor_utils.HAS_GPU_AND_TRITON
    inductor_utils.RUN_GPU = True

SELECTED = json.loads(Path(os.environ["COMBO_SELECTED_MANIFEST"]).read_text())
SOURCE_METHODS = {}
_original_call_test_method = unittest.TestCase._callTestMethod


def _capture_failure(self, method):
    try:
        return _original_call_test_method(self, method)
    except unittest.SkipTest:
        raise
    except Exception:
        cache = Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/nonexistent"))
        evidence = Path(os.environ["COMBO_SELECTED_MANIFEST"]).parent / "evidence"
        target = evidence / "generated_on_failure" / (type(self).__name__ + "." + self._testMethodName)
        try:
            for path in cache.rglob("*.py"):
                data = path.read_bytes()
                if b"triton_meta" not in data and b"async_compile" not in data:
                    continue
                destination = target / path.relative_to(cache)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        except Exception as error:
            print("Failure artifact capture error:", repr(error), flush=True)
        raise


unittest.TestCase._callTestMethod = _capture_failure


def source_names(path, cls):
    key = str(path), cls
    if key not in SOURCE_METHODS:
        tree = ast.parse(Path(str(path)).read_text())
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == cls]
        SOURCE_METHODS[key] = [node.name for cls_node in classes for node in cls_node.body
                               if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]
    return SOURCE_METHODS[key]


def pytest_collection_modifyitems(config, items):
    kept, dropped, catalog = [], [], []
    for item in items:
        file = Path(str(item.fspath)).name
        cls = item.cls.__name__ if item.cls else ""
        candidates = [row for row in SELECTED if row["file"] == file and (
            cls in (row["class"], row["class"] + "NPU", row["class"] + "PRIVATEUSE1") or
            (row["class"] == "CommonTemplate" and cls == "GPUTests")) and (
            item.name == row["name"] or item.name.startswith(row["name"] + "_"))]
        if not candidates:
            dropped.append(item)
            continue
        selected = max(candidates, key=lambda row: len(row["name"]))
        actual = max((name for name in source_names(item.fspath, selected["class"])
                      if item.name == name or item.name.startswith(name + "_")), key=len, default=None)
        if actual != selected["name"]:
            dropped.append(item)
            continue
        item.user_properties.extend([("source_case", selected["name"]), ("source_file", file)])
        kept.append(item)
        catalog.append({**selected, "nodeid": item.nodeid})
    items[:] = kept
    config.hook.pytest_deselected(items=dropped)
    if target := os.environ.get("COMBO_COLLECTION_JSON"):
        Path(target).write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")


def pytest_runtest_logreport(report):
    if target := os.environ.get("COMBO_RESULTS_JSONL"):
        record = {"nodeid": report.nodeid, "phase": report.when, "outcome": report.outcome,
                  "duration": report.duration, "properties": report.user_properties}
        if report.failed or report.skipped:
            record["longrepr"] = str(report.longrepr)
        with open(target, "a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
