import json
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'evidence'
records = []
for path in OUT.glob('test_*.jsonl'):
    for line in path.read_text().splitlines():
        try: records.append(json.loads(line))
        except json.JSONDecodeError: pass
nodes = {}
for r in records:
    if r['phase'] == 'call' or r['outcome'] != 'passed':
        nodes[r['nodeid']] = r
print('INSTANCE_RESULTS', dict(Counter(r['outcome'] for r in nodes.values())), 'TOTAL', len(nodes))
starts, finishes = [], []
p = OUT / 'jobs.jsonl'
if p.exists():
    for line in p.read_text().splitlines():
        try: row=json.loads(line)
        except json.JSONDecodeError: continue
        (starts if row['event']=='start' else finishes).append(row)
print('METHOD_JOBS', 'started',len(starts),'finished',len(finishes))
for row in finishes[-7:]:
    print('FINISHED',row['name'],row['returncode'],round(row['elapsed'],1))
for r in list(nodes.values()):
    if r['outcome']=='failed':
        s = r.get('longrepr','').splitlines()
        errors = [line for line in s if line.startswith('E ') or 'Error:' in line]
        print('FAIL',r['nodeid'], '\n'.join(errors[-5:])[:1400])
