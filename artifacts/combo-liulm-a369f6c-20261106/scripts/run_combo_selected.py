"""Run approved source methods in isolated pytest processes on allowed physical NPUs selected for this validation."""
import concurrent.futures
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'Community-Use-Case-Testing/v2.13.0/inductor_npu'
OUT = ROOT / 'evidence'
COLLECTION = json.loads((OUT / 'collection.json').read_text())
GROUPS = defaultdict(list)
for row in COLLECTION:
    GROUPS[(row['file'], row['class'], row['name'])].append(row['nodeid'])
TASKS = list(GROUPS.items())
excluded = set(os.environ.get('COMBO_EXCLUDE_METHODS', '').split(','))
TASKS = [task for task in TASKS if task[0][2] not in excluded]
if len(sys.argv) > 1:
    TASKS = [task for task in TASKS if task[0][2] in sys.argv[1:]]


def append_result(record):
    with (OUT / 'jobs.jsonl').open('a') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def run_one(task, physical_device):
    hour = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).hour
    if physical_device in {4, 5} and not 9 <= hour < 22:
        raise RuntimeError("Selected physical device is reserved in this time window")
    (file, cls, name), nodes = task
    stem = file.removesuffix('.py') + '.' + name
    log_path = OUT / (stem + '.log')
    jsonl_path = OUT / (stem + '.jsonl')
    env = dict(os.environ)
    env.update(ASCEND_RT_VISIBLE_DEVICES=str(physical_device), PYTHONPATH=str(ROOT),
               COMBO_SELECTED_MANIFEST=str(ROOT / 'combo_npu_a5_selected.json'),
               COMBO_RESULTS_JSONL=str(jsonl_path),
               TORCHINDUCTOR_CACHE_DIR=str(ROOT / 'cache' / f'device{physical_device}'),
               TORCHINDUCTOR_COMPILE_THREADS='1', OMP_NUM_THREADS='1',
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONUNBUFFERED='1')
    env.pop('COMBO_COLLECTION_JSON', None)
    cmd = [sys.executable, '-m', 'pytest', '-p', 'combo_validation_plugin',
           '-q', '-rA', '--tb=short', '--disable-warnings', *nodes,
           '--junitxml=' + str(OUT / (stem + '.xml'))]
    started = time.time()
    record = {'file': file, 'class': cls, 'name': name, 'nodes': nodes,
              'physical_device': physical_device, 'started': datetime.datetime.now().isoformat(),
              'command': cmd, 'log': str(log_path), 'jsonl': str(jsonl_path)}
    append_result({**record, 'event': 'start'})
    print('START', physical_device, name, len(nodes), flush=True)
    with log_path.open('w') as log:
        process = subprocess.Popen(cmd, cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            record['returncode'] = process.wait(timeout=2400)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            record['returncode'] = 124
    record.update(event='finish', elapsed=time.time()-started,
                  finished=datetime.datetime.now().isoformat())
    append_result(record)
    print('FINISH', physical_device, name, record['returncode'], round(record['elapsed'], 1), flush=True)
    return record


def worker(physical_device, tasks):
    wait_pid = os.environ.get('COMBO_WAIT_DEVICE1_PID') if physical_device == 1 else None
    if wait_pid:
        print('WAIT_PILOT', physical_device, wait_pid, flush=True)
        while Path('/proc', wait_pid).exists():
            time.sleep(2)
    results = []
    for task in tasks:
        results.append(run_one(task, physical_device))
    return results

# Largest parameter groups first, balanced by estimated instance count, and isolated per source method.
devices = [int(device) for device in os.environ.get('COMBO_PHYSICAL_DEVICES', '1,3,4,5').split(',')]
assert devices and len(set(devices)) == len(devices) and set(devices) <= {1, 3, 4, 5}, devices
lanes = {device: [] for device in devices}
loads = {device: 0 for device in devices}
for task in sorted(TASKS, key=lambda item: len(item[1]), reverse=True):
    lane = min(loads, key=loads.get)
    lanes[lane].append(task)
    loads[lane] += len(task[1]) + 3
with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
    futures = [pool.submit(worker, lane, tasks) for lane, tasks in lanes.items()]
    results = [record for future in futures for record in future.result()]
summary_path = OUT / os.environ.get('COMBO_SUMMARY_NAME', 'run_summary.json')
summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
print('ALL_FINISHED', len(results), flush=True)
