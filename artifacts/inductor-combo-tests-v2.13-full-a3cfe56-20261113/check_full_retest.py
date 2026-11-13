from collections import Counter
import json
import os
from pathlib import Path
import time


root = Path(__file__).resolve().parent
assert root.parent == Path('/home/l30023782/PTA_213') and Path('/.dockerenv').is_file()
result = {'checked_at': time.time()}
for name in ('launch-stage.json', 'collection-scope.json'):
    if (root / name).exists():
        result[name] = json.loads((root / name).read_text())
launches = json.loads((root / 'launch.json').read_text()) if (root / 'launch.json').exists() else []
result['workers'] = []
reports = {}
running = []
for launch in launches:
    directory = root / 'results' / launch['name']
    state = json.loads((directory / 'status.json').read_text()) if (directory / 'status.json').exists() else {}
    jobs = json.loads((directory / 'jobs.json').read_text()) if (directory / 'jobs.json').exists() else []
    alive = (Path('/proc') / str(launch['pid'])).exists()
    result['workers'].append({'name': launch['name'], 'pid': launch['pid'], 'alive': alive,
                              'stage': state.get('stage'), 'completed': state.get('completed', 0),
                              'methods': state.get('methods'), 'instances': state.get('instances'),
                              'job_status': dict(Counter(job['status'] for job in jobs)),
                              'error': state.get('error')})
    for job in jobs:
        if job['status'] == 'RUNNING':
            stem = f"{job['index']:02d}-{job['name']}"
            log = directory / (stem + '.log')
            running.append({'shard': launch['name'], 'class': job['class'], 'name': job['name'],
                            'elapsed': round(time.time() - job['started'], 1),
                            'log_size': log.stat().st_size if log.exists() else None,
                            'log_tail': log.read_text(errors='replace')[-500:] if log.exists() else None})
        for node, report in job.get('reports', {}).items():
            assert node not in reports, node
            reports[node] = report
result['instance_results'] = dict(Counter(row['status'] for row in reports.values()))
result['completed_instances'] = len(reports)
result['running'] = running
result['outside_nonpass'] = []
overlap = {(row['file'], row['class'], row['name'])
           for row in json.loads((root / 'overlap60-jobs.json').read_text())}
for launch in launches:
    path = root / 'results' / launch['name'] / 'jobs.json'
    if not path.exists():
        continue
    for job in json.loads(path.read_text()):
        if (job['file'], job['class'], job['name']) in overlap or job['status'] in ('PENDING', 'RUNNING', 'PASS'):
            continue
        result['outside_nonpass'].append({'file': job['file'], 'class': job['class'], 'name': job['name'],
                                          'status': job['status'], 'nodes': job['nodes'],
                                          'first_detail': next((row['detail'][:1500] for row in job['reports'].values()
                                                                if row['status'] != 'PASS'), '')})
print(json.dumps(result, ensure_ascii=False))
