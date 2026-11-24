"""A5 SIMD algorithm prototypes; split and cooperative share all math helpers."""

import triton
import triton.language as tl
from triton.language.extra.cann import extension


@triton.jit
def x_grid_barrier_debug(sem):
    tl.debug_barrier()
    extension.sync_block_all("all_vector", 0)
    tl.debug_barrier()


@triton.jit
def local_stats(A, B, row, shard, R: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr,
                SX: tl.constexpr, SR: tl.constexpr, OP: tl.constexpr, INTERLEAVED: tl.constexpr):
    chunk = tl.cdiv(R, S * BLOCK) * BLOCK
    start = shard * BLOCK if INTERLEAVED else shard * chunk
    stop = R if INTERLEAVED else tl.minimum((shard + 1) * chunk, R)
    step = S * BLOCK if INTERLEAVED else BLOCK
    acc = tl.full((BLOCK,), 0, tl.float32)
    maxima = tl.full((BLOCK,), float("-inf"), tl.float32)
    maximum = tl.full((), float("-inf"), tl.float32)
    exp_sum = tl.full((), 0, tl.float32)
    for begin in range(start, stop, step):
        offsets = begin + tl.arange(0, BLOCK)
        a = tl.load(A + row * SX + offsets * SR, offsets < R, other=0).to(tl.float32)
        b = tl.load(B + row * SX + offsets * SR, offsets < R, other=0).to(tl.float32)
        z = a + b
        if OP == 3:
            z = tl.where(offsets < R, z, float("-inf"))
            new_maximum = tl.maximum(maximum, tl.max(z, 0))
            exp_sum = exp_sum * tl.exp(maximum - new_maximum) + tl.sum(tl.exp(z - new_maximum), 0)
            maximum = new_maximum
        else:
            if OP == 2:
                acc += z * z
            elif OP != 1:
                acc += z
            if OP == 1 or OP == 4:
                maxima = tl.maximum(maxima, tl.where(offsets < R, z, float("-inf")))
    if OP == 3:
        return maximum, exp_sum
    else:
        return tl.max(maxima, 0), tl.sum(acc, 0)


@triton.jit
def read_stats(W, row, S: tl.constexpr, P: tl.constexpr, X: tl.constexpr, OP: tl.constexpr):
    peers = tl.arange(0, P)
    sums = tl.load(W + row * S + peers, peers < S, other=0)
    if OP == 1 or OP == 3 or OP == 4:
        maxima = tl.load(W + X * S + row * S + peers, peers < S, other=float("-inf"))
        maximum = tl.max(maxima, 0)
    else:
        maximum = tl.full((), 0, tl.float32)
    if OP == 3:
        total = tl.sum(sums * tl.exp(maxima - maximum), 0)
    else:
        total = tl.sum(sums, 0)
    return maximum, total


@triton.jit
def emit(A, B, Out, MaxOut, row, shard, maximum, total, R: tl.constexpr, E: tl.constexpr,
         BLOCK: tl.constexpr, SX: tl.constexpr, SR: tl.constexpr, OP: tl.constexpr):
    if OP == 0:
        tl.store(Out + row, total, shard == 0)
    elif OP == 1:
        tl.store(Out + row, maximum, shard == 0)
    elif OP == 4:
        tl.store(Out + row, total, shard == 0)
        tl.store(MaxOut + row, maximum, shard == 0)
    else:
        chunk = tl.cdiv(R, E * BLOCK) * BLOCK
        for begin in range(shard * chunk, tl.minimum((shard + 1) * chunk, R), BLOCK):
            offsets = begin + tl.arange(0, BLOCK)
            a = tl.load(A + row * SX + offsets * SR, offsets < R, other=0).to(tl.float32)
            b = tl.load(B + row * SX + offsets * SR, offsets < R, other=0).to(tl.float32)
            z = a + b
            value = z * tl.rsqrt(total / R + 1e-5) if OP == 2 else tl.exp(z - maximum) / total
            tl.store(Out + row * R + offsets, value, offsets < R)


@triton.jit
def partial(A, B, W, R: tl.constexpr, S: tl.constexpr, X: tl.constexpr, BLOCK: tl.constexpr,
            SX: tl.constexpr, SR: tl.constexpr, OP: tl.constexpr, INTERLEAVED: tl.constexpr):
    shard, row = tl.program_id(0), tl.program_id(1)
    maximum, total = local_stats(A, B, row, shard, R, S, BLOCK, SX, SR, OP, INTERLEAVED)
    tl.store(W + row * S + shard, total)
    if OP == 1 or OP == 3 or OP == 4:
        tl.store(W + X * S + row * S + shard, maximum)


@triton.jit
def finish(A, B, W, Out, MaxOut, R: tl.constexpr, S: tl.constexpr, P: tl.constexpr, X: tl.constexpr,
           E: tl.constexpr, BLOCK: tl.constexpr, SX: tl.constexpr, SR: tl.constexpr, OP: tl.constexpr):
    shard, row = tl.program_id(0), tl.program_id(1)
    maximum, total = read_stats(W, row, S, P, X, OP)
    emit(A, B, Out, MaxOut, row, shard, maximum, total, R, E, BLOCK, SX, SR, OP)


@triton.jit
def cooperative(A, B, W, Out, MaxOut, R: tl.constexpr, S: tl.constexpr, P: tl.constexpr, X: tl.constexpr,
                BLOCK: tl.constexpr, SX: tl.constexpr, SR: tl.constexpr, OP: tl.constexpr):
    shard, row = tl.program_id(0), tl.program_id(1)
    maximum, total = local_stats(A, B, row, shard, R, S, BLOCK, SX, SR, OP, False)
    tl.store(W + row * S + shard, total)
    if OP == 1 or OP == 3 or OP == 4:
        tl.store(W + X * S + row * S + shard, maximum)
    x_grid_barrier_debug(W)
    maximum, total = read_stats(W, row, S, P, X, OP)
    emit(A, B, Out, MaxOut, row, shard, maximum, total, R, S, BLOCK, SX, SR, OP)


@triton.jit
def exchange(W, Out, epoch, S: tl.constexpr, P: tl.constexpr, X: tl.constexpr):
    shard, row = tl.program_id(0), tl.program_id(1)
    peers = tl.arange(0, P)
    for phase in range(4):
        value = shard + 1 + row * 100 + phase * 10000 + epoch * 1000
        tl.store(W + row * S + shard, value)
        x_grid_barrier_debug(W)
        values = tl.load(W + row * S + peers, peers < S, other=0)
        tl.store(Out + (phase * X + row) * S + shard, tl.sum(values, 0))
        x_grid_barrier_debug(W)
