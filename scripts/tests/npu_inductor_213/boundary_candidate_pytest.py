"""Run exact candidate function bodies in their installed module namespaces.

This avoids relocating torch and its extension libraries through a package overlay.
The original installed files remain untouched; the selected candidate source and
function hashes are recorded in the test log.
"""

import ast
import __future__
import hashlib
import importlib
import os
from pathlib import Path
import runpy

import torch_npu
from torch_npu.contrib import transfer_to_npu  # noqa: F401
from torch_npu.utils import _dynamo

_dynamo.use_jit_script = True
_dynamo.patch_has_triton()


def replace_body(relative, module_name, function, class_name=None):
    path = Path(os.environ['BOUNDARY_CANDIDATE_DIR']) / relative
    source = path.read_text()
    tree = ast.parse(source)
    parent = tree if class_name is None else next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    node = next(node for node in parent.body if isinstance(node, ast.FunctionDef) and node.name == function)
    module = importlib.import_module(module_name)
    namespace = dict(vars(module))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    target = module if class_name is None else getattr(module, class_name)
    setattr(target, function, namespace[function])
    print('CANDIDATE', module_name, class_name, function, hashlib.sha256(source.encode()).hexdigest(), flush=True)


replace_body('torch/_inductor/cudagraph_utils.py', 'torch._inductor.cudagraph_utils',
             'check_caching_allocator_for_cudagraphs')
if os.environ.get('BOUNDARY_FUSION_METADATA') == '1':
    replace_body('torch_npu/_inductor/codegen/scheduling.py', 'torch_npu._inductor.codegen.scheduling',
                 'codegen_template', 'NPUTritonScheduling')

runpy.run_path(str(Path(__file__).with_name('run_batch.py')), run_name='__main__')
