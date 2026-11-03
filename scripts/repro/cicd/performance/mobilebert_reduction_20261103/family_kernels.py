"""Equivalent two-stage diagnostics for five distinct MobileBERT reductions."""

import triton
import triton.language as tl


@triton.jit
def partial(
    A,
    B,
    C,
    D,
    E,
    F,
    P,
    Q,
    WIDTH: tl.constexpr,
    KIND: tl.constexpr,
    CHUNK: tl.constexpr,
    XB: tl.constexpr,
    RB: tl.constexpr,
):
    x = tl.program_id(0) * XB + tl.arange(0, XB)[None, :]
    part = tl.program_id(1)
    acc = tl.full([RB, XB], 0, tl.float32)
    acc2 = tl.full([RB, XB], 0, tl.float32)
    if KIND == 13:
        scale = tl.load(E + x, x < WIDTH, other=0)
        bias = tl.load(F + x, x < WIDTH, other=0)
    for offset in range(tl.cdiv(CHUNK, RB)):
        r = part * CHUNK + offset * RB + tl.arange(0, RB)[:, None]
        mask = (x < WIDTH) & (r < tl.minimum((part + 1) * CHUNK, 16384))
        address = r * WIDTH + x
        value = tl.load(A + address, mask, other=0)
        if KIND == 0 or KIND == 13 or KIND == 20:
            value = value + tl.load(B + address, mask, other=0)
        if KIND == 20:
            value = value + tl.load(C + address, mask, other=0)
            value = value + tl.load(D + address, mask, other=0)
        acc = tl.where(mask, acc + value, acc)
        if KIND == 7:
            h = tl.load(B + address, mask, other=0)
        elif KIND == 13:
            h = tl.load(C + address, mask, other=0)
            other = tl.load(D + address, mask, other=0)
            h = h + (other * scale + bias)
        elif KIND == 20:
            h = tl.load(E + address, mask, other=0)
        if KIND == 7 or KIND == 13 or KIND == 20:
            acc2 = tl.where(mask, acc2 + value * h, acc2)
    tl.store(P + part * WIDTH + x, tl.sum(acc, 0)[None, :], x < WIDTH)
    if KIND == 7 or KIND == 13 or KIND == 20:
        tl.store(Q + part * WIDTH + x, tl.sum(acc2, 0)[None, :], x < WIDTH)


@triton.jit
def finish(
    P, Q, O, O2, WIDTH: tl.constexpr, DUAL: tl.constexpr, PARTS: tl.constexpr, XB: tl.constexpr, RB: tl.constexpr
):
    x = tl.program_id(0) * XB + tl.arange(0, XB)[None, :]
    r = tl.arange(0, RB)[:, None]
    mask = (x < WIDTH) & (r < PARTS)
    value = tl.load(P + r * WIDTH + x, mask, other=0)
    tl.store(O + x, tl.sum(value, 0)[None, :], x < WIDTH)
    if DUAL:
        value2 = tl.load(Q + r * WIDTH + x, mask, other=0)
        tl.store(O2 + x, tl.sum(value2, 0)[None, :], x < WIDTH)
