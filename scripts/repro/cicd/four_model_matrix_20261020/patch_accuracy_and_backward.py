"""Diagnostic only: inject explicit mean/variance masks before Triton compilation."""
import runpy
import sys
from torch._inductor.async_compile import AsyncCompile

original=AsyncCompile.triton

def patched(self,kernel_name,source,*args,**kwargs):
    edits=[]
    if 'embedding' in kernel_name:
        for axis,value,square,dim in [('r1','tmp4','tmp10',1),('r1','tmp8','tmp14',1),('r2','tmp8','tmp14',2)]:
            if f'tl.sum({value}, {dim})' not in source or f'tl.sum({square}, {dim})' not in source:
                continue
            if f'{axis.upper()}BLOCK_SUB: tl.constexpr = 1024' not in source: continue
            source=source.replace(f'tl.sum({value}, {dim})',f'tl.sum(tl.where({axis}_mask, {value}, 0.0), {dim})')
            source=source.replace(f'tl.sum({square}, {dim})',f'tl.sum(tl.where({axis}_mask, {square}, 0.0), {dim})')
            edits.append((axis,value,square))
    if 'clone_matmul_backward_sum' in kernel_name and 'R2BLOCK_SUB* R3BLOCK_SUB* X5BLOCK_SUB' in source:
        source=source.replace('tl.reshape(tmp0, [R2BLOCK_SUB* R3BLOCK_SUB* X5BLOCK_SUB])',
                              'tl.reshape(tmp0, [R2BLOCK_SUB* R3BLOCK_SUB, X5BLOCK_SUB])')
        source=source.replace('triton_helpers.promote_to_tensor(tl.sum(tmp1, 0))','tl.sum(tmp1, 0)')
        edits.append('preserve_x5')
    if edits: print('DIAGNOSTIC_MASKS',kernel_name,edits,flush=True)
    return original(self,kernel_name,source,*args,**kwargs)

AsyncCompile.triton=patched
target=sys.argv[1]
sys.argv=sys.argv[1:]
runpy.run_path(target,run_name='__main__')
