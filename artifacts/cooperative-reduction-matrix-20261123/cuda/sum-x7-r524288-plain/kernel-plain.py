    @triton.jit
    def plain(A, Bptr, Out, R: tl.constexpr, BLOCK: tl.constexpr, OP: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        if OP == 2:
            acc = tl.full((BLOCK,), float("-inf"), tl.float32)
        else:
            acc = tl.full((BLOCK,), 0, tl.float32)
        for begin in range(0, R, BLOCK):
            index = begin + offsets
            a = tl.load(A + row * R + index, index < R, other=0).to(tl.float32)
            b = tl.load(Bptr + row * R + index, index < R, other=0).to(tl.float32)
            value = a + b
            if OP == 2:
                acc = tl.maximum(acc, tl.where(index < R, value, float("-inf")))
            else:
                acc += value
        result = tl.max(acc, 0) if OP == 2 else tl.sum(acc, 0)
        tl.store(Out + row, result)
