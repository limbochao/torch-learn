"""Validate the current Albert diagnostic variants against explicit eager parameter gradients."""
import importlib.util
import json
from pathlib import Path
import torch
import torch_npu
root=Path(__file__).resolve().parent/'AlbertForMaskedLM'
rows=[]
for case,variant,length,rblock in [('s_train_perf','deferred',2048,None),('s_train_acc','tile1',512,None),
                                   ('d_train_perf','tile1',2048,128),('d_train_acc','tile1',512,512)]:
    path=root/f'{case}_{variant}.py'
    spec=importlib.util.spec_from_file_location(case,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    name=next(n for n in vars(mod) if n.startswith('triton_'))
    scalars={2,3}
    for start in range(4,81,7):scalars.update((start+5,start+6))
    torch.manual_seed(0)
    inputs=[torch.randn((length,) if i in scalars else (length,4096),device='npu') for i in range(81)]
    out0=torch.empty(4096,device='npu');out1=torch.empty_like(out0)
    kwargs={'X0BLOCK_SUB':1,'num_warps':1,'num_stages':2,
            'compile_mode':'simd_simt_template','multibuffer':False}
    if rblock is not None:kwargs['R1BLOCK_SUB']=rblock
    getattr(mod,name)[(64,)](out0,out1,*inputs,4096,length,64,**kwargs)
    torch.npu.synchronize()
    weighted=[(inputs[0]*((inputs[1]-inputs[2][:,None])*inputs[3][:,None])).sum(0)]
    summed=[inputs[0].sum(0)]
    for start in range(4,81,7):
        dy=inputs[start]+inputs[start+1]+inputs[start+2]+inputs[start+3]
        weighted.append((dy*((inputs[start+4]-inputs[start+5][:,None])*inputs[start+6][:,None])).sum(0))
        summed.append(dy.sum(0))
    expected0,expected1=weighted[0],summed[0]
    for index in [1,2,3,4,5,7,9,11,6,8,10]:
        expected0=expected0+weighted[index];expected1=expected1+summed[index]
    errors=[(out0-expected0).abs().max().item(),(out1-expected1).abs().max().item()]
    torch.testing.assert_close(out0,expected0,atol=2e-3,rtol=1e-3)
    torch.testing.assert_close(out1,expected1,atol=2e-3,rtol=1e-3)
    row={'case':case,'variant':variant,'length':length,'X0BLOCK_SUB':1,'R1BLOCK_SUB':rblock,
         'allclose':True,'max_abs':errors,'atol':2e-3,'rtol':1e-3}
    rows.append(row);print(json.dumps(row),flush=True)
    Path('albert_numeric_validation.json').write_text(json.dumps(rows,indent=2))
    del inputs,out0,out1,weighted,summed,expected0,expected1
