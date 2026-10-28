"""Load selected candidate modules without relocating torch or its shared libraries."""

import importlib.abc
import importlib.util
import os
from pathlib import Path
import sys


root = os.environ.get('BOUNDARY_CANDIDATE_DIR')
modules = {'torch._inductor.cudagraph_utils': 'torch/_inductor/cudagraph_utils.py'}
if os.environ.get('BOUNDARY_NATIVE_DRIVER') == '1':
    modules['torch._inductor.runtime.triton_helpers'] = 'torch/_inductor/runtime/triton_helpers.py'
if os.environ.get('BOUNDARY_FUSION_METADATA') == '1':
    modules['torch_npu._inductor.codegen.scheduling'] = 'torch_npu/_inductor/codegen/scheduling.py'
if os.environ.get('BOUNDARY_BMM_MACRO') == '1':
    modules['torch_npu._inductor.kernel.bmm'] = 'torch_npu/_inductor/kernel/bmm.py'
if os.environ.get('BOUNDARY_HOP_FALLBACK') == '1':
    modules['torch._inductor.lowering'] = 'torch/_inductor/lowering.py'

if os.environ.get('BOUNDARY_GATHER_META') == '1':
    modules['torch_npu.op_plugin.meta._meta_registrations'] = 'torch_npu/op_plugin/meta/_meta_registrations.py'


class CandidateFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in modules:
            candidate = Path(root) / modules[fullname]
            if not candidate.is_file():
                raise ImportError(f'Missing candidate source for {fullname}')
            return importlib.util.spec_from_file_location(fullname, candidate)
        return None


if root:
    sys.meta_path.insert(0, CandidateFinder())
