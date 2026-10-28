"""Reproduce the requested boundary cases in an already configured NPU container."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=420)
    parser.add_argument('--physical-devices', default='3')
    parser.add_argument('--rows', type=int, nargs='*')
    parser.add_argument('--precompile-workers', type=int, default=2)
    parser.add_argument('--compile-threads', type=int, default=2)
    args = parser.parse_args()
    assert Path('/.dockerenv').exists(), 'Run inside the target Docker container'
    assert Path.cwd().name == 'inductor_npu', 'Run from the authorized test directory'
    devices = [int(value) for value in args.physical_devices.split(',')]
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == args.physical_devices
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [line.split('\t')[:2] for line in args.manifest.read_text().splitlines() if line]
    for row, (filename, case) in enumerate(cases, 1):
        if args.rows and row not in args.rows:
            continue
        output = args.output / f'case-{row:02}.json'
        invocation = args.output / f'case-{row:02}-invocation.json'
        if invocation.exists():
            continue
        now = datetime.now(ZoneInfo('Asia/Shanghai'))
        allowed = {1, 2, 3}
        if 9 <= now.hour < 22 and (22 * 3600 - now.hour * 3600 - now.minute * 60 - now.second) > args.timeout + 30:
            allowed.update({4, 5, 7})
        assert set(devices) <= allowed, 'Device restriction window must cover the complete case timeout'
        usage = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=True)
        (args.output / f'case-{row:02}-device.txt').write_text(usage.stdout)
        env = os.environ.copy()
        cache = args.output / f'cache-{row:02}'
        cache.mkdir(exist_ok=True)
        env.update(TORCH_TRANSFER_TO_NPU='1', TORCH_NPU_DEVICE_CAPABILITY='8.0',
                   TORCHINDUCTOR_COMPILE_THREADS=str(args.compile_threads),
                   TORCHNPU_PRECOMPILE_THREADS=str(args.precompile_workers),
                   TORCHINDUCTOR_CACHE_DIR=str(cache), TMPDIR=str(args.output),
                   TRITON_CACHE_DIR=str(args.output / 'triton_cache'),
                   BOUNDARY_ARTIFACT_DIR=str(args.output / f'generated-{row:02}'),
                   PYTHONUNBUFFERED='1')
        command = [sys.executable, str(args.runner), '--case-names', '--precompile-workers', str(args.precompile_workers),
                   '--output', str(output), f'{filename}::{case}']
        record = {'row': row, 'file': filename, 'case': case, 'command': command,
                  'start': now.isoformat(), 'physical_devices': devices, 'timeout_seconds': args.timeout}
        print('START', row, filename, case, flush=True)
        started = time.monotonic()
        with output.with_suffix('.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                record['returncode'] = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                record['timeout'] = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                record['returncode'] = process.returncode
        record['seconds'] = time.monotonic() - started
        invocation.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        result = json.loads(output.read_text()) if output.exists() else {}
        print('END', row, result.get('summary', 'incomplete'), round(record['seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
