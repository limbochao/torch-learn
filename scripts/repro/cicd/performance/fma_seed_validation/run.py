"""Run the original TorchBench runner with process-local FMA/constexpr experiments."""

import argparse
import atexit
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import runpy
import sys
import textwrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--variant', choices=['baseline', 'no_fma_fallback', 'bind_constexpr'], required=True)
    args, runner_args = parser.parse_known_args()
    if runner_args[:1] == ['--']:
        runner_args = runner_args[1:]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.getLogger('torch_npu._inductor.codegen.npu_combined_scheduling').setLevel(logging.DEBUG)

    import torch
    import torch_npu
    import triton
    import transformers

    from torch._inductor import inductor_prims  # noqa: F401: register prims.fma
    from torch_npu._inductor import lowering_fallback_list

    npu_root = Path(torch_npu.__file__).parent
    files = [npu_root / '_inductor' / name for name in (
        'lowering_fallback_list.py', 'lowering.py', 'decomposition.py', 'codegen/triton.py',
    )]
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
    metadata['fma_fallback_before'] = [str(op) for op in lowering_fallback_list.FALLBACK_LIST if op in fma_ops]
    if args.variant == 'no_fma_fallback':
        if 'torch_npu._inductor.lowering' in sys.modules:
            raise RuntimeError('NPU lowering already imported; cannot isolate a list-only change')
        lowering_fallback_list.FALLBACK_LIST[:] = [
            op for op in lowering_fallback_list.FALLBACK_LIST if op not in fma_ops
        ]
        assert len(metadata['fma_fallback_before']) == 2
    metadata['fma_fallback_after'] = [str(op) for op in lowering_fallback_list.FALLBACK_LIST if op in fma_ops]

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
    runner = args.runner.resolve()
    sys.path.insert(0, str(runner.parent))
    sys.argv = [str(runner), *runner_args]
    print('EXPERIMENT', args.variant, 'FMA', metadata['fma_fallback_after'], flush=True)
    runpy.run_path(str(runner), run_name='__main__')


if __name__ == '__main__':
    main()
