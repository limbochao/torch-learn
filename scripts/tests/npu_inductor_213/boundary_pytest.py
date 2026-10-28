"""Match the NPU test preamble before importing the shared pytest recorder."""

import faulthandler
from pathlib import Path
import runpy

faulthandler.dump_traceback_later(120, repeat=True)

import torch_npu
from torch_npu.contrib import transfer_to_npu  # noqa: F401
from torch_npu.utils import _dynamo

_dynamo.use_jit_script = True
_dynamo.patch_has_triton()

runpy.run_path(str(Path(__file__).with_name('run_batch.py')), run_name='__main__')
