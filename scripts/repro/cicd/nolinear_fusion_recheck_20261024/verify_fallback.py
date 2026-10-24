"""Run unchanged regression tests with extracted backend source and CPU symbolic support.

Device-dependent NPU imports are stubbed. Production methods are compiled from
source unchanged; community gates/scheduling come from the v2.13 checkout.
This verifies scheduler protocols, not NPU kernel generation/execution.
"""
import argparse
import ast
import subprocess
import logging
import sys
import types
import unittest
from pathlib import Path

import torch
import torch._inductor.codegen.simd as simd
from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.scheduler import BaseSchedulerNode, BaseScheduling, FusedSchedulerNode, Scheduler, SchedulerNode
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--npu-repo', type=Path, required=True, help='torch_npu source checkout at 9fe2e0f635')
parser.add_argument('--torch-repo', type=Path, required=True, help='community PyTorch v2.13.0 source checkout')
args = parser.parse_args()
root = args.npu_repo.resolve()
upstream = args.torch_repo.resolve()
for label, repo in (('torch_npu', root), ('community', upstream)):
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    print(f'{label} source HEAD: {revision}', flush=True)


def extract(path, name, scope, cls=None):
    body = ast.parse(path.read_text()).body
    if cls:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(
        n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
    )
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.Module(body=[future, node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
    return scope[name]


# The v2.13 compatibility gate and schedule are used verbatim, rather than
# silently substituting the older installed CPU torch implementation.
for name in ('group_fn', 'can_fuse', 'generate_node_schedule'):
    method = extract(upstream / 'torch/_inductor/codegen/simd.py', name, dict(vars(simd)), 'SIMDScheduling')
    setattr(TritonScheduling, name, method)
    if name == 'can_fuse':
        TritonScheduling.can_fuse_vertical = method
        TritonScheduling.can_fuse_horizontal = method

for name in ('torch_npu', 'torch_npu._inductor', 'torch_npu._inductor.codegen'):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
config = types.ModuleType('torch_npu._inductor.config')
config.enable_welford = False
sys.modules[config.__name__] = config
sys.modules['torch_npu._inductor'].config = config

features = dict(vars(simd))
features.update(vars(__import__('typing')))
features['functools'] = __import__('functools')
features.update({'Iterable': __import__('typing').Iterable, 'sympy_product': simd.sympy_product})
numel_list = extract(root / 'torch_npu/_inductor/codegen/npu_kernel_features.py', 'NumelList', features)

scope = dict(vars(simd))
scope.update(TritonScheduling=TritonScheduling, NPUTritonKernel=object, NPUIndexTritonKernel=object,
             npu_config=config, NumelList=numel_list)
scheduling = types.ModuleType('torch_npu._inductor.codegen.scheduling')
for name in ('NPUNoLinearTritonScheduling',):
    setattr(scheduling, name, extract(root / 'torch_npu/_inductor/codegen/scheduling.py', name, scope))
# Only index group construction is needed: index codegen failure is injected.
class IndexScheduling(TritonScheduling):
    pass
scope['flatten'] = extract(root / 'torch_npu/_inductor/codegen/triton.py', 'flatten', dict(vars(simd)))
IndexScheduling.group_fn = extract(root / 'torch_npu/_inductor/codegen/scheduling.py', 'group_fn', scope,
                                  'NPUTritonScheduling')
scheduling.NPUTritonScheduling = IndexScheduling
sys.modules[scheduling.__name__] = scheduling

scope = dict(CUDACombinedScheduling=CUDACombinedScheduling, TritonScheduling=TritonScheduling,
             BaseSchedulerNode=BaseSchedulerNode, BaseScheduling=BaseScheduling, FusedSchedulerNode=FusedSchedulerNode,
             Scheduler=Scheduler, SchedulerNode=SchedulerNode, V=V, OrderedSet=OrderedSet,
             npu_config=config, log=logging.getLogger('source-test'))
combined = types.ModuleType('torch_npu._inductor.codegen.npu_combined_scheduling')
combined.NPUCombinedScheduling = extract(root / 'torch_npu/_inductor/codegen/npu_combined_scheduling.py',
                                        'NPUCombinedScheduling', scope)
sys.modules[combined.__name__] = combined

path = root / 'test/_inductor/test_nolinear_fallback_fusion.py'
test_module = types.ModuleType('fallback_tests')
exec(compile(path.read_text(), str(path), 'exec'), test_module.__dict__)
print('Support:', torch.__version__, '; backend gates/schedule: upstream v2.13 source', flush=True)
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_module))
raise SystemExit(not result.wasSuccessful())
