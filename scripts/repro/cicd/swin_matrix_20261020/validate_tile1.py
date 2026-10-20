"""Diagnostic: preserve generated DSL and compile options; change only Y3BLOCK_SUB to 1."""
import importlib.util,json
from pathlib import Path
import torch
import torch_npu
root=Path(__file__).resolve().parent
p=root/'static_inference_kernel/compile_repro.py'
spec=importlib.util.spec_from_file_location('swin_repro',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.manual_seed(0)
torch.npu.set_device(0)
x=torch.randn(8,128,56,56,device='npu')
add_bias=torch.randn(128,device='npu')
windows=torch.randn(8,8,8,7,7,128,device='npu')
mean=torch.randn(8,56,56,1,device='npu')
var_sum=(torch.rand_like(mean)+0.5)*128
weight=torch.randn(128,device='npu')
bias=torch.randn(128,device='npu')
out=torch.full((8,56,56,128),float('nan'),device='npu')
fn=getattr(m,m._STANDALONE_FN_NAME)
fn[(8,56)](x,add_bias,windows,mean,var_sum,weight,bias,out,8,7,448,128,1,8,
           Y1BLOCK_SUB=1,Y3BLOCK_SUB=1,**m._STANDALONE_OPTIONS)
torch.npu.synchronize()
reversed_windows=windows.permute(0,1,3,2,4,5).reshape(8,56,56,128)
expected=(((x.permute(0,2,3,1)+(add_bias+reversed_windows))-mean)
          *torch.rsqrt(var_sum/128+1e-5))*weight+bias
error=float((out-expected).abs().max())
torch.testing.assert_close(out,expected,atol=2e-5,rtol=2e-5)
assert torch.isfinite(out).all()
result={'variant':'Y1BLOCK_SUB=1,Y3BLOCK_SUB=1','check':'PASS','max_abs':error,
        'atol':2e-5,'rtol':2e-5,'seed':0,'input_shape':list(x.shape),
        'input_stride':list(x.stride()),'window_shape':list(windows.shape),'grid':[8,56],
        'runtime_blocks':[1,8]}
print(json.dumps(result),flush=True)
