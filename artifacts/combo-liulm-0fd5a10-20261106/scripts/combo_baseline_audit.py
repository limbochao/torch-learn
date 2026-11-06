"""Observe combo codegen routes; optionally apply the existing F3 method only."""
import functools,hashlib,json,os,traceback
from pathlib import Path

def install():
    import torch_npu._inductor.codegen.scheduling as linear
    import torch_npu._inductor.codegen.npu_combined_scheduling as combined
    import torch_npu._inductor.codegen.triton_combo_kernel as combo
    from torch_npu._inductor.codegen.npu_kernel_features import NumelList
    root=Path(os.environ['COMBO_SELECTED_MANIFEST']).parent
    report=Path(os.environ.get('COMBO_RESULTS_JSONL',str(root/'evidence/collection-audit.jsonl')))
    output=report.with_name('audit.'+report.stem+'.jsonl')
    def emit(record):
        record['test']=os.environ.get('PYTEST_CURRENT_TEST','collection')
        with output.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    if os.environ.get('COMBO_WITH_F3')=='1':
        source=(root/'f3_horizontal_partition.py').read_text()
        namespace=dict(vars(combo),NumelList=NumelList)
        exec(compile(source,str(root/'f3_horizontal_partition.py'),'exec'),namespace)
        combo.NPUComboKernel.horizontal_partition=staticmethod(namespace['horizontal_partition'])
        emit({'event':'existing_fix_loaded','commit':'e578ec524d592475fd52702b975ce03d1a95bbf2','method':'horizontal_partition','source_sha256':hashlib.sha256(source.encode()).hexdigest()})
    attempts=[]
    original=linear.NPUTritonScheduling.codegen_combo_kernel
    @functools.wraps(original)
    def observe_linear(self,*args,**kwargs):
        event={'event':'linear_combo_attempt'};attempts.append(event)
        try:
            result=original(self,*args,**kwargs)
        except Exception as error:
            event.update(outcome='failed',exception_type=type(error).__name__,exception=str(error),traceback=traceback.format_exc())
            emit(dict(event));raise
        event['outcome']='passed';emit(dict(event));return result
    linear.NPUTritonScheduling.codegen_combo_kernel=observe_linear
    outer=combined.NPUCombinedScheduling.codegen_combo_kernel
    @functools.wraps(outer)
    def observe_route(self,*args,**kwargs):
        start=len(attempts)
        try:result=outer(self,*args,**kwargs)
        except Exception as error:
            emit({'event':'combo_route','outcome':'failed','linear_attempts':attempts[start:],'exception_type':type(error).__name__,'exception':str(error)});raise
        recent=attempts[start:]
        route=('linear_combo' if any(x.get('outcome')=='passed' for x in recent) else 'ordinary_kernels_after_error' if recent else 'ordinary_kernels_no_linear_attempt')
        emit({'event':'combo_route','outcome':'passed','route':route,'linear_attempts':recent})
        return result
    combined.NPUCombinedScheduling.codegen_combo_kernel=observe_route
    check=linear.NPUTritonScheduling.make_ttir_for_check
    @functools.wraps(check)
    def observe_source(self,src,*args,**kwargs):
        if attempts and attempts[-1].get('outcome') is None:
            digest=hashlib.sha256(src.encode()).hexdigest()
            target=root/'evidence/combo_sources'/output.stem/digest
            target=target.with_suffix('.py');target.parent.mkdir(parents=True,exist_ok=True);target.write_text(src)
        return check(self,src,*args,**kwargs)
    linear.NPUTritonScheduling.make_ttir_for_check=observe_source
