"""Replay the captured training candidate directly in CANN BiSheng."""

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import triton


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--optimization', choices=['O0', 'O2'], default='O2')
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--trace-passes', action='store_true')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    compiler = shutil.which('bisheng')
    if compiler is None:
        parser.error('bisheng is unavailable; load the CANN environment first')
    backend = Path(triton.__file__).resolve().parent / 'backends/ascend'
    libraries = [backend / 'lib/libdevice.10.bc', backend / 'bishengir/lib/meta_op.aiv.c310.O0.bc']
    for library in libraries:
        if not library.is_file():
            parser.error(f'missing library: {library.name}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='swin-llvm-') as temporary:
        command = [
            compiler, str(args.ir.resolve()), '-o', str(Path(temporary) / 'kernel.o'),
            '--cce-aicore-arch=dav-c310-vec', '--cce-aicore-only', '-' + args.optimization,
            '-cce-bitcode-is-aicore', '-Wno-override-module',
            '-mllvm', '--cce-vf-auto-sync=global',
            '-mllvm', '-cce-aicore-dcci-insert-for-scalar=false',
            '-mllvm', '-cce-vf-aa-between-iters=true',
            '-mllvm', '-cce-vf-enable-loop-fusion=false',
            '-cce-link-aicore-ll-module', str(libraries[0]), '--cce-long-scbz=true',
            '--cce-aicore-input-parameter-size=1536',
            '-cce-link-aicore-ll-module', str(libraries[1]),
        ]
        if args.trace_passes:
            command.extend(['-mllvm', '-debug-pass=Executions'])
        started = time.monotonic()
        with (args.output_dir / 'compiler.txt').open('wb') as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                status, code = 'COMPLETED', result.returncode
            except subprocess.TimeoutExpired:
                status, code = 'PROBE_TIMEOUT', 124
        summary = {
            'status': status, 'exit_code': code, 'elapsed_seconds': time.monotonic() - started,
            'timeout_seconds': args.timeout, 'optimization': args.optimization,
            'trace_passes': args.trace_passes,
        }
        (args.output_dir / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
