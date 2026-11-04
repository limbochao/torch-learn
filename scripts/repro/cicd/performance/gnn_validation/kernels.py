"""Diagnostic DSL: split long outer reductions and fuse GCN message scattering."""

import triton
import triton.language as tl


@triton.jit
def partial(A, G, O, P, ROWS: tl.constexpr, RELU: tl.constexpr,
            CHUNK: tl.constexpr, XB: tl.constexpr, RB: tl.constexpr):
    x = tl.program_id(0) * XB + tl.arange(0, XB)[None, :]
    part = tl.program_id(1)
    acc = tl.full((RB, XB), 0, tl.float32)
    for offset in range(tl.cdiv(CHUNK, RB)):
        r = part * CHUNK + offset * RB + tl.arange(0, RB)[:, None]
        mask = (x < 64) & (r < tl.minimum((part + 1) * CHUNK, ROWS))
        v = tl.load(A + r * 64 + x, mask, other=0)
        if RELU:
            g = tl.load(G + r * 64 + x, mask, other=0)
            v = tl.where(v <= 0, 0., g)
            tl.store(O + r * 64 + x, v, mask)
        acc += tl.where(mask, v, 0.)
    tl.store(P + part * 64 + x, tl.sum(acc, 0)[None, :], x < 64)


@triton.jit
def finish(P, O, PARTS: tl.constexpr, XB: tl.constexpr, RB: tl.constexpr):
    x = tl.program_id(0) * XB + tl.arange(0, XB)[None, :]
    r = tl.arange(0, RB)[:, None]
    v = tl.load(P + r * 64 + x, (x < 64) & (r < PARTS), other=0)
    tl.store(O + x, tl.sum(v, 0)[None, :], x < 64)


@triton.jit
def scatter_message(X, E, W, O, EDGES: tl.constexpr, BLOCK: tl.constexpr):
    for block in range(tl.program_id(0), tl.cdiv(EDGES * 64, BLOCK), tl.num_programs(0)):
        i = block * BLOCK + tl.arange(0, BLOCK)
        edge = i // 64
        c = i % 64
        mask = edge < EDGES
        src = tl.load(E + edge, mask, other=0)
        dst = tl.load(E + EDGES + edge, mask, other=0)
        weight = tl.load(W + edge, mask, other=0)
        val = tl.load(X + src * 64 + c, mask, other=0)
        tl.atomic_add(O + dst * 64 + c, weight * val, mask, sem="relaxed")
