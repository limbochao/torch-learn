"""Record selected source imports, including the same bootstrap used by child interpreters."""

import hashlib
import importlib
from pathlib import Path
import runpy

import torch_npu
from torch_npu.contrib import transfer_to_npu  # noqa: F401
from torch_npu.utils import _dynamo

_dynamo.use_jit_script = True
_dynamo.patch_has_triton()

import sitecustomize

for name, relative in sitecustomize.modules.items():
    module = importlib.import_module(name)
    path = Path(module.__file__)
    assert path.resolve() == (Path(sitecustomize.root) / relative).resolve()
    print('CANDIDATE_SOURCE', name, hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)

runpy.run_path(str(Path(__file__).with_name('run_batch.py')), run_name='__main__')
