import collections,datetime,hashlib,json,os,subprocess,sys,sysconfig,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
status={}
def save(**kw):
 status.update(kw,time=datetime.datetime.now().isoformat());(ROOT/'pipeline_status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)
def command(args,cwd,env,log):
 with log.open('w') as f:
  process=subprocess.run(args,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
 assert process.returncode==0,(args,process.returncode,str(log))
def run_phase(phase, resume=False):
 p=ROOT/phase;out=p/'evidence';devices=os.environ['COMBO_PHYSICAL_DEVICES']
 env=dict(os.environ,COMBO_WITH_F3='1' if phase=='with_fixes' else '0',PYTHONPATH=str(p),ASCEND_RT_VISIBLE_DEVICES=devices.split(',')[0],COMBO_SELECTED_MANIFEST=str(p/'combo_npu_a5_selected.json'),COMBO_COLLECTION_JSON=str(out/'collection.json'),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TORCHINDUCTOR_COMPILE_THREADS='1',OMP_NUM_THREADS='1')
 if not resume:
  save(phase=phase,state='collecting')
  command([sys.executable,'capture_stage_environment.py'],p,env,out/'environment_capture.log')
  base=p/'Community-Use-Case-Testing/v2.13.0/inductor_npu'
  files=sorted({m['file'] for m in json.loads((p/'combo_npu_a5_selected.json').read_text())})
  command([sys.executable,'-m','pytest','-p','combo_validation_plugin','--collect-only','-q',*files],base,env,out/'collection.log')
  actual=json.loads((out/'collection.json').read_text());expected=json.loads((out/'expected_collection.json').read_text())
  assert {x['nodeid'] for x in actual}=={x['nodeid'] for x in expected}
  (out/'collection_equivalence.json').write_text(json.dumps({'same_as_expected':True,'expected_instances':len(expected),'actual_instances':len(actual)},indent=2)+'\n')
  save(phase=phase,state='testing',expected_instances=len(actual),expected_methods=len(json.loads((p/"combo_npu_a5_selected.json").read_text())))
  command([sys.executable,'-u','run_combo_selected.py'],p,env,out/'driver.log')
 actual=json.loads((out/'collection.json').read_text())
 save(phase=phase,state='finalizing')
 command([sys.executable,'export_combo_results.py'],p,env,out/'export.log')
 command([sys.executable,'capture_combo_evidence.py'],p,env,out/'capture.log')
 result=json.loads((out/'functional_results.json').read_text())
 assert result['completed_instances']==len(actual)
 assert all(m['job_finished'] and m['completed_instances']==m['expected_instances'] for m in result['methods'])
 assert all(m['returncode'] in (0,1) for m in result['methods'])
 nodes=collections.defaultdict(dict)
 for f in out.glob('test_*.jsonl'):
  for line in f.read_text().splitlines():
   r=json.loads(line);nodes[r['nodeid']][r['phase']]=r['outcome']
 assert set(nodes)=={x['nodeid'] for x in actual}
 assert all({'setup','teardown'} <= set(r) and ('call' in r or r['setup']!='passed') for r in nodes.values())
 outcomes=['failed' if 'failed' in r.values() else 'skipped' if 'skipped' in r.values() else 'passed' for r in nodes.values()]
 assert dict(collections.Counter(outcomes))==result['instance_counts']
 source=json.loads((p/'expected_source_sha256.json').read_text());site=Path(sysconfig.get_paths()['purelib'])
 post={f:hashlib.sha256((site/f).read_bytes()).hexdigest() for f in source}
 assert source==post
 (out/'integrity_checks.json').write_text(json.dumps({'all_expected_instances_reported':True,'method_count':len(result['methods']),'instance_count':len(nodes),'instance_counts':result['instance_counts'],'all_pytest_phases_checked':True,'installed_source_unchanged':True,'post_run_source_sha256':post},indent=2)+'\n')
 save(phase=phase,state='complete',completed_instances=len(nodes),instance_counts=result['instance_counts'])
 return result
try:
 import time
 save(phase='failed_first',state='testing',expected_instances=15,expected_methods=13,execution_order=['failed_first','with_fixes','docs_push','full'])
 while not (ROOT/'failed_first/evidence/run_summary.json').exists():
  time.sleep(5)
 first=run_phase('failed_first',resume=True)
 fixed=run_phase('with_fixes')
 save(phase='focused_complete',state='awaiting_docs_push',results={name:json.loads((ROOT/name/'evidence/functional_results.json').read_text())['instance_counts'] for name in ['failed_first','with_fixes']})
except BaseException:
 save(state='error',error=traceback.format_exc());raise
