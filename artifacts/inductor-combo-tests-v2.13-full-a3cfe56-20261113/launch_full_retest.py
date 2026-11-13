from collections import defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    assert root.parent == Path('/home/l30023782/PTA_213') and Path('/.dockerenv').is_file()
    assert os.getenv('CONDA_DEFAULT_ENV') == 'pta_213_case'
    context = json.loads((root / 'execution-context.json').read_text())
    expected = json.loads((root / 'validation-source-sha256.json').read_text())
    before = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in expected}
    assert before == expected, 'Staged test source differs from the prepared snapshot'
    save(root / 'test-source-before.json', before)
    env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES='1', ASCEND_DEVICE_ID='0', DEVICE_ID='0',
               TORCHINDUCTOR_COMPILE_THREADS='1', OMP_NUM_THREADS='2')
    for key in ('ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_DEVICE_ID', 'DEVICE_ID',
                'TORCHINDUCTOR_COMPILE_THREADS', 'OMP_NUM_THREADS'):
        os.environ[key] = env[key]
    import torch
    import torch_npu
    import triton

    state = {'stage': 'smoke', 'started': time.time(), 'test_commit': context['test_commit']}
    save(root / 'launch-stage.json', state)
    smokes = []
    smoke = (
        "import torch,torch_npu,json; torch.npu.set_device(0); "
        "x=torch.arange(512,device='npu',dtype=torch.float32); "
        "value=(x+1).sum().item(); torch.npu.synchronize(); assert value==131328.0; "
        "print(json.dumps({'logical_card':0,'device_count':torch.npu.device_count(),'result':value}))"
    )
    for card in (1, 2, 3):
        result = subprocess.run([sys.executable, '-c', smoke], cwd=root,
                                env=dict(env, ASCEND_RT_VISIBLE_DEVICES=str(card)),
                                capture_output=True, text=True, timeout=120)
        smokes.append({'physical_card': card, 'returncode': result.returncode,
                       'stdout': result.stdout, 'stderr': result.stderr})
        save(root / 'card-smoke.json', smokes)
        assert result.returncode == 0, smokes[-1]
    metadata = {
        'date': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'docker': True,
        'python': sys.version, 'executable': sys.executable, 'conda_environment': os.getenv('CONDA_DEFAULT_ENV'),
        'torch': torch.__version__, 'torch_git': torch.version.git_version,
        'torch_npu': torch_npu.__version__, 'torch_npu_git': torch_npu.version.git_version,
        'triton': triton.__version__, 'card_smoke': smokes, 'compile_threads': 1, 'omp_num_threads': 2,
        'overlay_applied': False, 'foreach_functional': True, 'test_commit': context['test_commit'],
        'main_base': context['main_base'], 'hardware_skip_patch': 'hardware-skip.patch',
        'reusing_old_results': False, 'sampling': False,
    }
    save(root / 'environment.json', metadata)
    with (root / 'npu-smi-before.log').open('w') as stream:
        subprocess.run(['npu-smi', 'info'], cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
    base = root / 'v2.13.0/inductor_npu'
    command = [sys.executable, '-u', str(root / 'run_plain_pytest.py'), '--test-root', str(base),
               '--manifest', str(root / 'manifest.json'), '--output', str(root / 'full-collection'),
               '--cache-root', str(root / 'cache-collection'), '--test-commit', context['test_commit'],
               '--devices', '1', '--collect-only', '--foreach-functional']
    state['stage'] = 'collecting'
    save(root / 'launch-stage.json', state)
    with (root / 'full-collection.run.log').open('w') as stream:
        subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    jobs = json.loads((root / 'full-collection/collection.json').read_text())
    nodes = [node for job in jobs for node in job['nodes']]
    assert len(jobs) == 105 and len(nodes) == len(set(nodes))
    overlap = json.loads((root / 'overlap60-jobs.json').read_text())
    overlap_keys = {(row['file'], row['class'], row['name']) for row in overlap}
    collected_keys = {(row['file'], row['class'], row['name']) for row in jobs}
    assert len(overlap_keys) == 60 and overlap_keys <= collected_keys
    overlap_nodes = {node for job in jobs if (job['file'], job['class'], job['name']) in overlap_keys
                     for node in job['nodes']}
    assert len(overlap_nodes) == 413
    save(root / 'collection-scope.json', {
        'table_rows': 116, 'methods': len(jobs), 'instances': len(nodes),
        'overlap_methods': 60, 'overlap_instances': len(overlap_nodes),
        'outside_methods': len(jobs) - 60, 'outside_instances': len(nodes) - len(overlap_nodes),
        'overlap_keys': sorted(overlap_keys),
    })
    installed = json.loads((root / 'full-collection/installed_sha256.json').read_text())
    verified = json.loads((root / 'verified-daily-installed-sha256.json').read_text())
    save(root / 'installed-source-before.json', {
        'files': len(installed), 'matches_previous_daily': installed == verified,
        'changed_files': sorted(name for name in set(installed) | set(verified)
                                if installed.get(name) != verified.get(name)),
    })
    weights = json.loads((root / 'prior-timing.json').read_text())
    buckets = [{'weight': 0, 'rows': defaultdict(list)} for _ in range(6)]
    instances = [(max(weights.get(node, 30), 10), job, node) for job in jobs for node in job['nodes']]
    for weight, job, node in sorted(instances, key=lambda row: row[0], reverse=True):
        bucket = min(buckets, key=lambda row: row['weight'])
        bucket['rows'][job['file'], job['class'], job['name']].append(node)
        bucket['weight'] += weight
    (root / 'results').mkdir()
    launches = []
    for index, bucket in enumerate(buckets):
        name, device = f'shard-{index + 1:02d}', (1, 2, 3)[index % 3]
        manifest = [{'file': key[0], 'class': key[1], 'name': key[2], 'only_nodes': sorted(values)}
                    for key, values in bucket['rows'].items()]
        save(root / (name + '.json'), manifest)
        triton_cache = root / 'cache-triton' / name
        triton_cache.mkdir(parents=True)
        worker_env = dict(env, ASCEND_RT_VISIBLE_DEVICES=str(device), TRITON_CACHE_DIR=str(triton_cache))
        command = [sys.executable, '-u', str(root / 'run_plain_pytest.py'), '--test-root', str(base),
                   '--manifest', str(root / (name + '.json')), '--output', str(root / 'results' / name),
                   '--cache-root', str(root / 'cache-inductor' / name),
                   '--test-commit', context['test_commit'], '--devices', str(device), '--foreach-functional']
        with (root / (name + '.run.log')).open('w') as stream:
            process = subprocess.Popen(command, cwd=root, env=worker_env, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        launches.append({'name': name, 'device': device, 'pid': process.pid, 'command': command,
                         'methods': len(manifest), 'instances': sum(len(row['only_nodes']) for row in manifest),
                         'estimated_seconds': round(bucket['weight'], 2), 'started': time.time()})
    save(root / 'launch.json', launches)
    save(root / 'shards.json', [{'name': row['name'], 'device': row['device']} for row in launches])
    state.update(stage='launched', methods=len(jobs), instances=len(nodes), workers=len(launches))
    save(root / 'launch-stage.json', state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
