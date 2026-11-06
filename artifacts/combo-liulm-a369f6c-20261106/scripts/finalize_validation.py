from pathlib import Path
import time,json,subprocess,collections,hashlib,importlib.util
root=Path(__file__).resolve().parent
out=root/'evidence'
while not (out/'run_summary.json').exists():
    time.sleep(20)
try:
    subprocess.run(['python','export_combo_results.py'],cwd=root,check=True)
    subprocess.run(['python','capture_combo_evidence.py'],cwd=root,check=True)
    expected=json.loads((out/'collection.json').read_text())
    nodes={r['nodeid']:{} for r in expected}
    for path in out.glob('test_*.jsonl'):
        for line in path.read_text().splitlines():
            row=json.loads(line)
            assert row['nodeid'] in nodes,row['nodeid']
            nodes[row['nodeid']][row['phase']]=row
    outcomes={}
    for node,phases in nodes.items():
        assert 'setup' in phases and 'teardown' in phases,(node,list(phases))
        statuses=[r['outcome'] for r in phases.values()]
        if 'failed' in statuses:outcomes[node]='failed'
        elif 'skipped' in statuses:outcomes[node]='skipped'
        else:
            assert set(phases)=={'setup','call','teardown'} and all(s=='passed' for s in statuses),node
            outcomes[node]='passed'
    assert len(nodes)==409
    results=json.loads((out/'functional_results.json').read_text())
    assert results['selected_methods']==62 and results['completed_instances']==409
    assert all(r['job_finished'] and r['completed_instances']==r['expected_instances'] for r in results['methods'])
    assert dict(collections.Counter(outcomes.values()))==results['instance_counts']
    expected_hash=json.loads((root/'expected_source_sha256.json').read_text())
    actual={}
    for relative in expected_hash:
        pkg,rest=relative.split('/',1)
        p=Path(importlib.util.find_spec(pkg).origin).parent/rest
        actual[relative]=hashlib.sha256(p.read_bytes()).hexdigest()
    assert actual==expected_hash,'Installed source changed during validation'
    finished={'all_expected_instances_reported':True,'method_count':62,'instance_count':409,'instance_counts':results['instance_counts'],'all_pytest_phases_checked':True,'installed_source_unchanged':True,'post_run_source_sha256':actual}
    (out/'integrity_checks.json').write_text(json.dumps(finished,indent=2)+'\n')
    print('FINAL_INTEGRITY_PASSED',flush=True)
except Exception:
    import traceback
    (out/'finalization_error.txt').write_text(traceback.format_exc())
    raise
