import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import time
import sys

parser = argparse.ArgumentParser()
parser.add_argument('plan', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--cards', default='0,1,2')
parser.add_argument('--case-dir', type=Path, required=True)
parser.add_argument('--python', default=sys.executable)
args = parser.parse_args()
ROOT = args.output.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
PLUGIN_ROOT = Path(__file__).resolve().parent
CASE = args.case_dir.resolve()
PYTHON = args.python
SPECIAL = {113, 229, 230, 231}
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)
PLAN = json.loads(args.plan.read_text())
(ROOT / 'plan.json').write_text(json.dumps(PLAN, indent=2))


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(row, card):
    name = f"case-{row['row']:03d}"
    dest = RESULTS / name
    if dest.with_suffix('.invocation.json').exists():
        return
    env = os.environ.copy()
    env.update(TORCH_TRANSFER_TO_NPU='1', TORCH_NPU_DEVICE_CAPABILITY='8.0',
               TORCHINDUCTOR_COMPILE_THREADS='2', TORCHNPU_PRECOMPILE_THREADS='2', OMP_NUM_THREADS='2',
               TORCHINDUCTOR_CACHE_DIR=str(ROOT / 'cache' / name),
               RECHECK_JSON=str(dest.with_suffix('.json')),
               PYTHONPATH=str(PLUGIN_ROOT) + os.pathsep + env.get('PYTHONPATH', ''))
    env.update(row.get('extra_env', {}))
    if card == 'full':
        env.pop('ASCEND_RT_VISIBLE_DEVICES', None)
    else:
        env['ASCEND_RT_VISIBLE_DEVICES'] = str(card)
    nodes = [n.removeprefix('test/inductor/') for n in row['nodes']]
    selection = nodes or [row['file'], '-k', row['test']]
    cmd = [PYTHON, '-m', 'pytest', '-vs', '-p', 'isolated_plugin',
           '--junitxml=' + str(dest.with_suffix('.xml')), *selection]
    record = {'row': row['row'], 'card': card, 'command': cmd, 'cwd': str(CASE),
              'start': stamp(), 'timeout_seconds': 900,
              'environment': {k: env[k] for k in ('TORCH_TRANSFER_TO_NPU', 'TORCH_NPU_DEVICE_CAPABILITY',
                             'TORCHINDUCTOR_COMPILE_THREADS', 'TORCHNPU_PRECOMPILE_THREADS', 'OMP_NUM_THREADS',
                             'TORCHINDUCTOR_CACHE_DIR', 'RECHECK_JSON', 'ASCEND_RT_VISIBLE_DEVICES') if k in env},
              'nodes': nodes}
    record['environment'].update(row.get('extra_env', {}))
    start = time.monotonic()
    with dest.with_suffix('.log').open('w') as log:
        process = subprocess.Popen(cmd, cwd=CASE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        record['pid'] = process.pid
        dest.with_suffix('.running.json').write_text(json.dumps(record, indent=2))
        try:
            record['returncode'] = process.wait(timeout=900)
            record['timed_out'] = False
        except subprocess.TimeoutExpired:
            record['timed_out'] = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            record['returncode'] = process.returncode
    record.update(end=stamp(), duration=time.monotonic() - start)
    dest.with_suffix('.invocation.json').write_text(json.dumps(record, indent=2))
    dest.with_suffix('.running.json').unlink()
    print(json.dumps({'row': row['row'], 'card': card, 'rc': record['returncode'],
                      'seconds': round(record['duration'], 1)}), flush=True)


def worker(card, jobs):
    while True:
        try:
            row = jobs.get_nowait()
        except queue.Empty:
            break
        run(row, card)


if __name__ == '__main__':
    jobs = queue.Queue()
    for row in PLAN:
        if row['row'] not in SPECIAL:
            jobs.put(row)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.cards.split(','))) as pool:
        futures = [pool.submit(worker, card, jobs) for card in args.cards.split(',')]
        for future in futures:
            future.result()
    for row in PLAN:
        if row['row'] in SPECIAL:
            run(row, 'full')
    (ROOT / 'complete.json').write_text(json.dumps({'end': stamp(), 'rows': len(PLAN)}))
