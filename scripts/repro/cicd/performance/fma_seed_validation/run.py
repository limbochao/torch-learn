"""Run the original TorchBench runner with process-local FMA/constexpr experiments."""

import argparse
import atexit
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import logging
import os
from pathlib import Path
import runpy
import sys
import textwrap


def install_fma_list_hook():
    """Remove exactly two entries before any NPU lowering import can consume them."""
    state = {}

    class Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            source = self.get_source(fullname)
            lines = source.splitlines(keepends=True)
            removed = [line.strip().rstrip(',') for line in lines
                       if line.strip() in ('prims.fma,', 'prims.fma.default,')]
            assert removed == ['prims.fma', 'prims.fma.default'], removed
            patched = ''.join(line for line in lines
                              if line.strip() not in ('prims.fma,', 'prims.fma.default,'))
            state.update(removed_entries=removed, original_sha256=hashlib.sha256(source.encode()).hexdigest(),
                         patched_sha256=hashlib.sha256(patched.encode()).hexdigest())
            return self.source_to_code(patched, self.path)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != 'torch_npu._inductor.lowering_fallback_list':
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            spec.loader = Loader(fullname, spec.origin)
            return spec

    finder = Finder()
    sys.meta_path.insert(0, finder)
    return state, finder


def install_performance_profiler(runner, output):
    """Profile the runner's already-warmed eager/compiled functions with identical inputs."""
    sys.path.insert(0, str(runner.parent))
    import common

    method = common.BenchmarkRunner.run_performance_test
    source = inspect.getsource(method)
    needle = '            compilation_time = dynamo_latency - eager_latency'
    assert source.count(needle) == 1 and 'super()' not in source
    block = '''            if self.args.enable_profiler is not None:
                for phase, profile_fn in (("eager", self.model_iter_fn),
                                          ("dynamo", optimized_model_iter_fn)):
                    profile_fn(model, example_inputs)
                    synchronize()
                    profiler = NPUProfiler(
                        enable=True, level=f"level{self.args.enable_profiler}", warmup=10,
                        active=max(1, self.args.iterations - 10),
                        save_path=os.path.join(self.args.prof_output_path, name, phase),
                    )
                    profiler.start()
                    try:
                        for _ in range(self.args.iterations):
                            profile_fn(model, example_inputs)
                            synchronize()
                            profiler.step()
                    finally:
                        profiler.stop()
'''
    patched = source.replace(needle, block + needle)
    (output / 'run_performance_test_patched.py').write_text(patched)
    namespace = {}
    exec(compile(textwrap.dedent(patched), '<process-local-device-profiler>', 'exec'), method.__globals__, namespace)
    common.BenchmarkRunner.run_performance_test = namespace[method.__name__]
    return {'original_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'patched_sha256': hashlib.sha256(patched.encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--variant', choices=['baseline', 'no_fma_fallback', 'bind_constexpr'], required=True)
    parser.add_argument('--check-setup-only', action='store_true')
    args, runner_args = parser.parse_known_args()
    if runner_args[:1] == ['--']:
        runner_args = runner_args[1:]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    hook, finder = install_fma_list_hook() if args.variant == 'no_fma_fallback' else ({}, None)

    import torch
    import torch_npu
    import triton
    import transformers

    logging.getLogger('torch_npu._inductor.codegen.npu_combined_scheduling').setLevel(logging.DEBUG)

    from torch._inductor import inductor_prims  # noqa: F401: register prims.fma
    from torch_npu._inductor import lowering_fallback_list

    npu_root = Path(torch_npu.__file__).parent
    runner = args.runner.resolve()
    files = [npu_root / '_inductor' / name for name in (
        'lowering_fallback_list.py', 'lowering.py', 'decomposition.py', 'codegen/triton.py',
    )] + [runner, runner.parent / 'common.py']
    metadata = {
        'variant': args.variant,
        'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'modules': {m.__name__: {'version': m.__version__, 'file': m.__file__}
                    for m in (torch, torch_npu, triton, transformers)},
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'runner': str(args.runner.resolve()),
        'runner_sha256': hashlib.sha256(args.runner.read_bytes()).hexdigest(),
        'runner_argv': runner_args,
        'environment': {k: os.environ.get(k) for k in (
            'ASCEND_RT_VISIBLE_DEVICES', 'TORCHINDUCTOR_NPU_BACKEND', 'TORCHINDUCTOR_CACHE_DIR',
            'TRITON_CACHE_DIR', 'TORCHINDUCTOR_COMPILE_THREADS', 'TORCHNPU_PRECOMPILE_THREADS',
        )},
    }
    fma_ops = {torch.ops.prims.fma, torch.ops.prims.fma.default}
    loaded_entries = [str(op) for op in lowering_fallback_list.FALLBACK_LIST if op in fma_ops]
    metadata['fma_fallback_before'] = hook.get('removed_entries', loaded_entries)
    if args.variant == 'no_fma_fallback':
        assert hook and not loaded_entries, 'The import-time list-only patch did not run'
        metadata['fma_import_hook'] = hook
        sys.meta_path.remove(finder)
    metadata['fma_fallback_after'] = loaded_entries

    if args.variant == 'bind_constexpr':
        from torch_npu._inductor.codegen.triton import NPUIndexTritonKernel

        method = NPUIndexTritonKernel.codegen_kernel
        original = inspect.getsource(method)
        (output / 'codegen_kernel_original.py').write_text(original)
        needle = '        self.triton_meta = triton_meta'
        assert original.count(needle) == 1
        assert 'equal_1_arg_indices(signature)' not in original, 'Fix already present; baseline must be revised'
        patched = original.replace(needle, '''        from torch._inductor.codegen.triton_utils import equal_1_arg_indices
        for arg_num in equal_1_arg_indices(signature):
            triton_meta["constants"][signature[arg_num].name] = 1
        self.triton_meta = triton_meta''')
        (output / 'codegen_kernel_patched.py').write_text(patched)
        namespace = {}
        exec(compile(textwrap.dedent(patched), '<process-local-constexpr-fix>', 'exec'), method.__globals__, namespace)
        NPUIndexTritonKernel.codegen_kernel = namespace[method.__name__]
        metadata['patched_method_sha256'] = hashlib.sha256(patched.encode()).hexdigest()

    def finish():
        from torch._inductor import lowering

        fn = lowering.lowerings.get(torch.ops.prims.fma.default)
        metadata['fma_lowering'] = {'module': getattr(fn, '__module__', None),
                                    'qualname': getattr(fn, '__qualname__', None)}
        metadata['source_unchanged'] = all(
            hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha
            for p, sha in metadata['source_sha256'].items()
        )
        (output / 'environment.json').write_text(json.dumps(metadata, indent=2) + '\n')

    atexit.register(finish)
    (output / 'environment.json').write_text(json.dumps(metadata, indent=2) + '\n')
    torch._inductor.config.compile_threads = 1
    torch._inductor.config.trace.enabled = True
    torch.manual_seed(1337)
    sys.path.insert(0, str(runner.parent))
    metadata['performance_profiler_patch'] = install_performance_profiler(runner, output)
    (output / 'environment.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if args.check_setup_only:
        print('SETUP_OK', args.variant, metadata['fma_fallback_after'], flush=True)
        return
    sys.argv = [str(runner), *runner_args]
    print('EXPERIMENT', args.variant, 'FMA', metadata['fma_fallback_after'], flush=True)
    runpy.run_path(str(runner), run_name='__main__')


if __name__ == '__main__':
    main()
