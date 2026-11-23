    @triton.jit
    def exchange(W, Out, Sem, epoch, S: tl.constexpr, P: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr,
                 PHASES: tl.constexpr, KIND: tl.constexpr, SKEW: tl.constexpr):
        shard = tl.program_id(0)
        row = tl.program_id(1) + Y * tl.program_id(2)
        sem = Sem.to(tl.pointer_type(tl.uint32)) + row
        peers = tl.arange(0, P)
        for phase in range(PHASES):
            extra = 0
            if SKEW:
                for step in range((shard % 4) * 8):
                    extra += step
            value = shard + 1 + row * 1000 + phase * 10000 + epoch * 1000 + extra
            tl.store(W + row * S + shard, value)
            barrier(sem, KIND)
            values = tl.load(W + row * S + peers, peers < S, other=0)
            tl.store(Out + (phase * Y * Z + row) * S + shard, tl.sum(values, 0))
            # Protect readers before the next phase overwrites their workspace.
            if PHASES > 1:
                barrier(sem, KIND)
