"""A masked indirect load must return `other=0` for all 256 padded lanes."""
import json
from pathlib import Path
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def masked_gather(ids, table, out, nrows, width, XBLOCK, B:tl.constexpr):
    row0 = tl.program_id(0) * XBLOCK
    r = tl.arange(0, B)[None,:]
    for step in range(XBLOCK):
        row = row0 + step + tl.arange(0, 1)[:,None]
        active = row < nrows
        token = tl.load(ids + row, active, other=0)
        value = tl.load(table + token * 768 + r, active & (r < width), other=0.0)
        tl.store(out + row * B + r, value, active)

if __name__=='__main__':
    torch.manual_seed(0)
    ids=torch.zeros(32,device='npu',dtype=torch.int64)
    table=torch.full((4096,),9.,device='npu')
    table[:768]=3.
    out=torch.empty((32,1024),device='npu')
    expected=torch.zeros_like(out);expected[:,:768]=3.
    results=[]
    for mode in ('simd_simt_template','simt_only'):
        kernel=masked_gather[(32,)](ids,table,out,32,768,1,B=1024,
            num_warps=1,num_stages=2,compile_mode=mode,multibuffer=False)
        torch.npu.synchronize()
        record={'mode':mode,'valid_correct':bool((out[:,:768]==3).all()),
                'tail_min':out[:,768:].min().item(),'tail_max':out[:,768:].max().item(),
                'allclose':torch.equal(out,expected)}
        print(json.dumps(record),flush=True);results.append(record)
        for kind,data in kernel.asm.items():
            if isinstance(data,str): Path(__file__).with_name(f'masked_gather_{mode}.{kind}.txt').write_text(data)
    Path(__file__).with_name('masked_gather_results.json').write_text(json.dumps(results,indent=2))
