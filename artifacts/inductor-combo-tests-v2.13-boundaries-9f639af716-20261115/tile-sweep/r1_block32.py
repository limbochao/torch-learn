
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

import torch
import torch_npu
if not torch_npu.npu.is_initialized() and torch_npu.npu._is_in_bad_fork():
    torch_npu.npu._initialized = True
from torch_npu._inductor.runtime import triton_heuristics as triton_heuristics
from torch_npu._inductor.runtime import triton_helpers
from torch_npu._inductor.runtime.triton_helpers import libdevice, extension, math as tl_math
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_sum_0(in_ptr0, out_ptr0, xnumel, r0_numel, r1_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr, R1_BLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 32
    r1_numel = 64
    rnumel = r0_numel * r1_numel
    RBLOCK: tl.constexpr = R0_BLOCK*R1_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :, None]
    r1_base = tl.arange(0, R1_BLOCK)[None, None, :]
    rbase = r1_base + r0_base*r1_numel
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK, R1_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        for r1_offset in tl.range(0, r1_numel, R1_BLOCK):
            r1_index = r1_offset + r1_base
            r1_mask = r1_index < r1_numel
            roffset = r1_offset + r0_offset*r1_numel
            rindex = r1_index + r0_index*r1_numel
            r0_1 = r0_index
            r1_2 = r1_index
            tmp0 = tl.load(in_ptr0 + (r1_2 + 64*x0 + 1024*r0_1), r0_mask & r1_mask & xmask, eviction_policy='evict_first', other=0.0)
            tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK, R1_BLOCK])
            tmp3 = _tmp2 + tmp1
            _tmp2 = tl.where(r0_mask & r1_mask & xmask, tmp3, _tmp2)
    tmp4 = tl.reshape(_tmp2, [XBLOCK, RBLOCK])
    tmp2 = tl.sum(tmp4, 1)[:, None, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
