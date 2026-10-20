"""Run two independent queues, one model per NPU, with durable per-case status."""
import concurrent.futures
import csv
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ.get('MATRIX_DIR', 'four_model_matrix_20261020')).resolve()
root.mkdir(parents=True, exist_ok=True)
bench = Path(os.environ['TORCHBENCH_RUNNER_DIR']).resolve()
devices = [int(value) for value in os.environ.get('MATRIX_DEVICES', '0,1').split(',')]
if len(devices) != 2 or devices[0] == devices[1]:
    raise ValueError('MATRIX_DEVICES requires two distinct visible NPU indices')
models = [('AlbertForMaskedLM', 'huggingface'), ('GPT2ForSequenceClassification', 'huggingface'),
          ('nfnet_l0', 'timm_models'), ('BERT_pytorch', 'torchbench')]
cases = [(shape, mode, metric) for shape in ('static', 'dynamic')
         for mode in ('inference', 'training') for metric in ('performance', 'accuracy')]


def run_queue(entries, device):
    for model, suite in entries:
        for shape, mode, metric in cases:
            mode_key = "train" if mode == "training" else "infer"
            metric_key = "perf" if metric == "performance" else "acc"
            key = f"{shape[0]}_{mode_key}_{metric_key}"
            folder = root / 'cases' / model / key
            folder.mkdir(parents=True, exist_ok=True)
            state = folder / 'status.json'
            if state.exists() and json.loads(state.read_text()).get('status') in ('PASS', 'FAIL'):
                continue
            command = [sys.executable, str(Path(__file__).with_name('run_case.py')), str(bench/f'{suite}.py'),
                       f'--{metric}', f'--{mode}', '--float32', '--backend', 'inductor',
                       '--npu-backend', 'triton', '--disable-aclgraph', '--only', model,
                       '--output', str(folder/'benchmark.csv')]
            command += ['--repeat', '1', '--iterations', '1'] if metric == 'performance' else ['--iterations', '5']
            if shape == 'dynamic':
                command += ['--dynamic-shapes']
            env = os.environ.copy()
            env.update(ASCEND_RT_VISIBLE_DEVICES=str(device), TORCHINDUCTOR_COMPILE_THREADS='1',
                       TORCH_COMPILE_DEBUG='1', PYTHONUNBUFFERED='1', CASE_DIR=str(folder),
                       TORCH_COMPILE_DEBUG_DIR=str(folder/'debug'),
                       TORCHINDUCTOR_CACHE_DIR=str(root/'inductor_cache'/model))
            row = dict(model=model, suite=suite, case=key, shape=shape, mode=mode, metric=metric,
                       device=device, compile_threads=1, command=command,
                       started=datetime.datetime.now().astimezone().isoformat(),
                       status='RUNNING')
            with (folder/'run.txt').open('w') as log:
                process = subprocess.Popen(command, cwd=bench, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                row['pid'] = process.pid
                state.write_text(json.dumps(row, indent=2))
                print('START', model, key, 'pid', process.pid, flush=True)
                row['exit_code'] = process.wait()
            row['finished'] = datetime.datetime.now().astimezone().isoformat()
            result = folder/'benchmark.csv'
            row['benchmark'] = list(csv.DictReader(result.open())) if result.exists() else []
            row['status'] = 'FAIL'
            if row['exit_code'] == 0 and row['benchmark']:
                record = row['benchmark'][-1]
                if metric == 'performance':
                    passed = float(record.get('speedup') or 0) > 0
                else:
                    passed = record.get('accuracy') in ('pass', 'pass_accuracy')
                if passed:
                    row['status'] = 'PASS'
            state.write_text(json.dumps(row, indent=2))
            print('END', model, key, row['status'], row['exit_code'], flush=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(run_queue, models[::2], devices[0]), pool.submit(run_queue, models[1::2], devices[1])]
    for future in futures:
        future.result()
