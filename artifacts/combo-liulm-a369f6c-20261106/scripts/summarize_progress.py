from pathlib import Path
import json,collections
root=Path(__file__).resolve().parent
out=root/'evidence'
jobs={}
p=out/'jobs.jsonl'
if p.exists():
    for line in p.read_text().splitlines():
        row=json.loads(line);jobs[row['file'],row['name']]=row
counts=collections.Counter();failure=[];per_method=[]
for p in out.glob('test_*.jsonl'):
    reports=[json.loads(line) for line in p.read_text().splitlines()]
    calls=[r for r in reports if r['phase']=='call']
    counts.update(r['outcome'] for r in calls)
    per_method.append({'method':p.stem,'counts':dict(collections.Counter(r['outcome'] for r in calls))})
    for r in reports:
        if r['outcome']=='failed':
            reason='\n'.join(l for l in r.get('longrepr','').splitlines() if l.startswith('E '))[:1400]
            failure.append({'nodeid':r['nodeid'],'phase':r['phase'],'reason':reason})
result={'methods_finished':sum(r['event']=='finish' for r in jobs.values()),'methods_started':len(jobs),'instances_executed':sum(counts.values()),'instance_counts':dict(counts),'running':[{'file':r['file'],'name':r['name'],'physical_device':r['physical_device']} for r in jobs.values() if r['event']=='start'],'method_results':per_method,'failures':failure,'driver_done':(out/'run_summary.json').exists()}
(out/'progress.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
