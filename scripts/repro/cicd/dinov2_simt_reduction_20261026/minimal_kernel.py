"""Beta-gradient-only reduction, simplified from original_kernel.py."""
import triton
import triton.language as tl

@triton.jit
def masked_sum(g,o,XBLOCK:tl.constexpr,RBLOCK:tl.constexpr):
    x=tl.program_id(0)*XBLOCK+tl.arange(0,XBLOCK)[:,None]
    rb=tl.arange(0,RBLOCK)[None,:]
    acc=tl.full((XBLOCK,RBLOCK),0,tl.float32)
    for offset in tl.range(0,21920,RBLOCK):
        r=offset+rb
        value=tl.load(g+x+768*(r//1370),(x<768)&(r<21920),other=0)
        value=tl.where(r%1370==0,value,0)
        acc=tl.where((x<768)&(r<21920),acc+value,acc)
    value=tl.sum(acc,1)[:,None]
    tl.store(o+x,value,x<768)
