    @triton.jit
    def reduce(A, Bptr, W, Out, Sem, R: tl.constexpr, S: tl.constexpr, P: tl.constexpr,
               Y: tl.constexpr, BLOCK: tl.constexpr, KIND: tl.constexpr, OP: tl.constexpr):
        shard = tl.program_id(0)
        row = tl.program_id(1) + Y * tl.program_id(2)
        chunk = tl.cdiv(R, S * BLOCK) * BLOCK
        if OP == 2:
            acc = tl.full((BLOCK,), float("-inf"), tl.float32)
        else:
            acc = tl.full((BLOCK,), 0, tl.float32)
        for begin in range(shard * chunk, tl.minimum((shard + 1) * chunk, R), BLOCK):
            offsets = begin + tl.arange(0, BLOCK)
            a = tl.load(A + row * R + offsets, offsets < R, other=0).to(tl.float32)
            b = tl.load(Bptr + row * R + offsets, offsets < R, other=0).to(tl.float32)
            value = a + b
            if OP == 2:
                acc = tl.maximum(acc, tl.where(offsets < R, value, float("-inf")))
            else:
                acc += value
        if OP == 2:
            partial = tl.max(acc, 0)
        else:
            partial = tl.sum(acc, 0)
        tl.store(W + row * S + shard, partial)
        barrier(Sem.to(tl.pointer_type(tl.uint32)) + row, KIND)
        peers = tl.arange(0, P)
        if OP == 2:
            values = tl.load(W + row * S + peers, peers < S, other=float("-inf"))
            result = tl.max(values, 0)
        else:
            values = tl.load(W + row * S + peers, peers < S, other=0)
            result = tl.sum(values, 0)
            if OP == 1:
                result /= R
        # Keep the same all-read combine algorithm for all three implementations.
        tl.store(Out + row, result, shard == 0)
