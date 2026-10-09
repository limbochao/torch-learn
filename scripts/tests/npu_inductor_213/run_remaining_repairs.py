"""Run selected original assertions from the repair overlay, with per-case evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--round', default='first')
    parser.add_argument('rows', type=int, nargs='+')
    args = parser.parse_args()
    assert Path('/.dockerenv').exists()
    root = args.root.resolve()
    work = root / 'test/inductor'
    assert work.exists()
    repo = Path(__file__).resolve().parents[3]
    plan = json.loads((repo / 'artifacts/npu-inductor-pr47384-restored-driver-20261009/plan.json').read_text())
    cases = {row: case for batch in plan for row, case in zip(batch['rows'], batch['cases'], strict=True)}
    results = root / args.round
    results.mkdir(exist_ok=True)
    site = Path(sys.executable).resolve().parents[1] / 'lib/python3.11/site-packages'
    changes = json.loads((root / 'changes.json').read_text())
    env = os.environ.copy()
    env.update(TORCH_TRANSFER_TO_NPU='1', TORCH_NPU_DEVICE_CAPABILITY='8.0',
               TORCHINDUCTOR_COMPILE_THREADS='4', TORCHNPU_PRECOMPILE_THREADS='4',
               NPU_TRANSFER_PR_HEAD='16437be0ffc9d67c8b27855a234243e35385c58b',
               PYTORCH_TEST_WITH_SLOW='1', TMPDIR=str(root),
               PATH=str(Path(sys.executable).parent) + ':' + env['PATH'],
               NPU_SKIP_ORIGINAL_TEST_DIR=str(root.parent),
               NPU_SKIP_OBSERVATIONS=str(results / f'observations-{args.device}.jsonl'))
    for row in args.rows:
        output = results / f'case-{row}.json'
        assert not output.exists(), output
        for item in changes:
            if item['domain'] == 'site':
                assert hashlib.sha256((site / item['file']).read_bytes()).hexdigest() == item['after_sha256']
        cache = results / f'cache-{row}'
        cache.mkdir(exist_ok=True)
        env['TORCHINDUCTOR_CACHE_DIR'] = str(cache)
        script = 'run_conv_precision_validation.py' if row in (83, 201) else 'run_batch.py'
        command = [sys.executable, script, '--case-names', '--precompile-workers', '4',
                   '--device-index', str(args.device), '--output', str(output), cases[row]]
        invocation = {'row': row, 'case': cases[row], 'device': args.device, 'start': time.time(),
                      'script': script, 'changes': changes, 'timeout_seconds': 1800}
        print('START', row, cases[row], flush=True)
        with output.with_suffix('.log').open('w') as log:
            process = subprocess.Popen(command, cwd=work, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                invocation['returncode'] = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                invocation['timeout'] = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        invocation['end'] = time.time()
        output.with_name(f'case-{row}-invocation.json').write_text(json.dumps(invocation, indent=2) + '\n')
        data = json.loads(output.read_text()) if output.exists() else {}
        print('END', row, data.get('summary', 'incomplete'), flush=True)
    print('QUEUE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
