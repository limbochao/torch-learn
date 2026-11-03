"""Two-stage diagnostic equivalent of the original reduction, not a framework fix."""

import triton
import triton.language as tl


@triton.jit
def partial(A, B, C, P, Q, CHUNK: tl.constexpr, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[None, :]
    part = tl.program_id(1)
    rbase = tl.arange(0, RBLOCK)[:, None]
    acc0 = tl.full([RBLOCK, XBLOCK], 0, tl.float32)
    acc1 = tl.full([RBLOCK, XBLOCK], 0, tl.float32)
    for offset in range(tl.cdiv(CHUNK, RBLOCK)):
        r = part * CHUNK + offset * RBLOCK + rbase
        mask = (x < 128) & (r < tl.minimum((part + 1) * CHUNK, 16384))
        a = tl.load(A + r * 128 + x, mask, other=0)
        b = tl.load(B + r * 128 + x, mask, other=0)
        c = tl.load(C + r * 128 + x, mask, other=0)
        value = a + b
        acc0 = tl.where(mask, acc0 + value, acc0)
        acc1 = tl.where(mask, acc1 + value * c, acc1)
    tl.store(P + part * 128 + x, tl.sum(acc0, 0)[None, :], x < 128)
    tl.store(Q + part * 128 + x, tl.sum(acc1, 0)[None, :], x < 128)


@triton.jit
def finish(P, Q, O0, O1, PARTS: tl.constexpr, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[None, :]
    r = tl.arange(0, RBLOCK)[:, None]
    mask = (x < 128) & (r < PARTS)
    a = tl.load(P + r * 128 + x, mask, other=0)
    b = tl.load(Q + r * 128 + x, mask, other=0)
    tl.store(O0 + x, tl.sum(a, 0)[None, :], x < 128)
    tl.store(O1 + x, tl.sum(b, 0)[None, :], x < 128)
