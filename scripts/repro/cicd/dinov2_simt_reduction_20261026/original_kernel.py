"""Original DINOv2 final LayerNorm parameter-gradient DSL, without Inductor autotune.

The @triton.jit function below is copied verbatim from the generated kernel.
Use run.py to launch it with an explicit configuration and synthetic inputs.
"""
import triton
import triton.language as tl

@triton.jit
def triton_red_fused_native_layer_norm_backward_select_backward_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 768
    r0_numel = 21920
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp16 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = (r0_index % 1370)
        r0_2 = r0_index // 1370
        r0_3 = r0_index
        tmp3 = tl.load(in_ptr0 + (x0 + 768*r0_2), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr1 + (x0 + 768*r0_3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r0_3), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr3 + (r0_3), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp0 = (r0_1).to(tl.int32)
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = tl.full([1, 1], 0.0, tl.float32)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp8 = tmp6 - tmp7
        tmp10 = tmp8 * tmp9
        tmp11 = tmp5 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask & xmask, tmp14, _tmp13)
        tmp15 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(r0_mask & xmask, tmp17, _tmp16)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
