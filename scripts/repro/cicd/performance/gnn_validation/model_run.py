"""Process-local GNN lowering / Reduction.num_splits experiments using the real runner."""

import argparse
import atexit
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=['baseline', 'split', 'scatter', 'both', 'pool-fallback'], required=True)
    parser.add_argument('--split-relu-only', action='store_true')
    parser.add_argument('--split-min-reduction', type=int, default=8192)
    args, argv = parser.parse_known_args()
    if argv[:1] == ['--']:
        argv = argv[1:]
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') in ('1', '2', '3')
    args.output.mkdir(parents=True, exist_ok=True)
    changes = {'variant': args.variant, 'removed': [], 'split_events': []}
    if args.variant in ('scatter', 'both'):
        class Loader(importlib.machinery.SourceFileLoader):
            def get_code(self, fullname):
                source = self.get_source(fullname)
                lines = source.splitlines(keepends=True)
                entries = {'aten.scatter_add,', 'aten.scatter_add.default,', 'aten.scatter_add.out,',
                           'aten.scatter_add_,', 'aten.scatter_add_.default,'}
                changes['removed'] = [line.strip() for line in lines if line.strip() in entries]
                assert len(changes['removed']) == 5, changes
                patched = ''.join(line for line in lines if line.strip() not in entries)
                return self.source_to_code(patched, self.path)

        class Finder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname != 'torch_npu._inductor.lowering_fallback_list':
                    return None
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                spec.loader = Loader(fullname, spec.origin)
                return spec

        sys.meta_path.insert(0, Finder())

    import torch
    import torch_npu
    import triton
    from torch._inductor import ir, lowering
    from torch._inductor.virtualized import V
    from torch._inductor.runtime.hints import ReductionHint
    from torch_npu._inductor import ir as npu_ir

    npu_root = Path(torch_npu.__file__).parent
    files = [npu_root / '_inductor' / name for name in (
        'ir.py', 'config.py', 'lowering.py', 'lowering_fallback_list.py', 'codegen/triton.py',
        'codegen/scheduling.py', 'runtime/triton_heuristics.py')]
    changes['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    changes['harness_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    changes['run_environment'] = {name: os.environ.get(name) for name in (
        'ASCEND_RT_VISIBLE_DEVICES', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR',
        'TORCHINDUCTOR_NPU_BACKEND', 'TORCHINDUCTOR_COMPILE_THREADS', 'TORCHNPU_PRECOMPILE_THREADS')}
    compiler = Path(triton.__file__).parent / 'backends/ascend/compiler.py'
    changes['triton_compiler_sha256'] = hashlib.sha256(compiler.read_bytes()).hexdigest()
    changes['versions'] = {m.__name__: m.__version__ for m in (torch, torch_npu, triton)}
    changes['argv'] = argv
    changes['scatter_lowering'] = inspect.getsource(lowering.lowerings[torch.ops.aten.scatter_add.default])
    changes['split_reductions_config'] = torch._inductor.config.split_reductions
    original = ir.Reduction.num_splits
    changes['original_num_splits'] = inspect.getsource(original)
    changes['split_min_reduction'] = args.split_min_reduction
    if args.variant == 'pool-fallback':
        # Implementation-selection control, not a production fallback policy.
        op = torch.ops.aten.avg_pool2d.default
        changes['original_pool_lowering'] = inspect.getsource(lowering.lowerings[op])
        lowering.lowerings[op] = lowering.fallback_handler(op, add_to_fallback_set=False)
        changes['pool_lowering'] = inspect.getsource(lowering.lowerings[op])
    if args.variant in ('split', 'both'):
        def num_splits(device, dst_dtype, src_dtype, inner_fn, ranges, reduction_ranges,
                       reduction_type, reduction_numel, input_node=None):
            from sympy import prod
            n = V.graph.sizevars.optimization_hint(prod(ranges))
            r = V.graph.sizevars.optimization_hint(reduction_numel)
            node = V.graph.current_node
            allowed = True
            if args.split_relu_only:
                producer = node.args[0] if node is not None and node.args else None
                allowed = getattr(producer, 'target', None) == torch.ops.aten.where.self
            # Experimental outer-reduction policy. No changes to other reductions.
            if (device.type == 'npu' and reduction_type == 'sum' and 1 < n <= 128
                    and r >= args.split_min_reduction and allowed):
                split = min(256, (int(r) + 511) // 512)
                changes['split_events'].append({'node': str(node), 'producer': str(node.args[0]),
                                                'n': int(n), 'r': int(r), 'split': split})
                return ReductionHint.DEFAULT, split
            return original(device, dst_dtype, src_dtype, inner_fn, ranges, reduction_ranges,
                            reduction_type, reduction_numel, input_node)
        ir.Reduction.num_splits = num_splits
        npu_ir.num_splits = num_splits

    def save():
        changes['source_unchanged'] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha
                                          for p, sha in changes['source_sha256'].items())
        (args.output / 'environment.json').write_text(json.dumps(changes, indent=2) + '\n')

    atexit.register(save)
    torch._inductor.config.compile_threads = 1
    torch._inductor.config.trace.enabled = True
    torch.manual_seed(1337)
    helper = Path(__file__).parents[1] / 'fma_seed_validation/run.py'
    profile_helper = runpy.run_path(str(helper))
    changes['profiler'] = profile_helper['install_performance_profiler'](args.runner, args.output)
    save()
    sys.path.insert(0, str(args.runner.parent))
    sys.argv = [str(args.runner), *argv]
    runpy.run_path(str(args.runner), run_name='__main__')


if __name__ == '__main__':
    main()
