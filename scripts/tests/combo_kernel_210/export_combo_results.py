"""Aggregate recorded pytest phases against the validated parameterized collection."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'evidence'
expected = json.loads((OUT / 'collection.json').read_text())
node_map = {row['nodeid']: row for row in expected}
assert len(node_map) == len(expected), 'Duplicate collected node IDs'
reports = {}
for path in sorted(OUT.glob('test_*.jsonl')):
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record['phase'] == 'call' or record['outcome'] != 'passed':
            reports[record['nodeid']] = record
unexpected = set(reports) - set(node_map)
assert not unexpected, unexpected

jobs = {}
for line in (OUT / 'jobs.jsonl').read_text().splitlines():
    job = json.loads(line)
    jobs[job['file'], job['name']] = job

methods = []
for method in json.loads((ROOT / 'combo_npu_a5_selected.json').read_text()):
    nodes = [row['nodeid'] for row in expected
             if row['file'] == method['file'] and row['name'] == method['name']]
    assert nodes, method
    results = [reports[node] for node in nodes if node in reports]
    job = jobs.get((method['file'], method['name']))
    methods.append({
        **method,
        'job_started': job is not None,
        'job_finished': bool(job and job['event'] == 'finish'),
        'returncode': job.get('returncode') if job else None,
        'expected_instances': len(nodes),
        'completed_instances': len(results),
        'counts': dict(Counter(record['outcome'] for record in results)),
        'failures': [record for record in results if record['outcome'] != 'passed'],
    })
summary = {
    'selected_methods': len(methods),
    'expected_instances': len(expected),
    'completed_instances': len(reports),
    'instance_counts': dict(Counter(record['outcome'] for record in reports.values())),
    'methods': methods,
}
(OUT / 'functional_results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps({key: value for key, value in summary.items() if key != 'methods'}))
